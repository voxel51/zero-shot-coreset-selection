from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fiftyone.train.core.model import embeddings
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import LinearProbe, MLPProbe, Model


class FSLDataset(Dataset):
    """Dataset for Few-Shot Learning with support set. Keeps all data on GPU."""

    def __init__(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        device: torch.device = None,
    ):
        """
        Args:
            embeddings: Input embeddings (N, D)
            labels: Class labels (N,)
            device: Device to store tensors on (defaults to CUDA if available)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to tensors and move to device - data stays on GPU throughout training
        if isinstance(embeddings, np.ndarray):
            self.embeddings = torch.from_numpy(embeddings).float().to(device)
        else:
            self.embeddings = embeddings.float().to(device)

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long().to(device)
        else:
            self.labels = labels.long().to(device)

        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Data is already on GPU, no transfer needed
        return {"inputs": self.embeddings[idx], "labels": self.labels[idx]}


class WarmupStepLR(_LRScheduler):
    """Learning rate scheduler with warmup followed by step decay (WSL)."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of steps for linear warmup
            step_size: Period of learning rate decay after warmup
            gamma: Multiplicative factor of learning rate decay
            last_epoch: The index of last epoch
        """
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Step decay after warmup
            steps_after_warmup = self.last_epoch - self.warmup_steps
            decay_steps = steps_after_warmup // self.step_size
            return [base_lr * (self.gamma**decay_steps) for base_lr in self.base_lrs]


class WarmupSteadyDecayLR(_LRScheduler):
    """Learning rate scheduler with warmup, steady, and linear decay phases (WSD) for step-based training."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        decay_fraction: float = 0.1,
        min_lr_fraction: float = 0.01,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of steps for linear warmup
            total_steps: Total number of training steps
            decay_fraction: Fraction of total steps to use for decay phase (default 0.1 = last 10%)
            min_lr_fraction: Minimum learning rate as fraction of base_lr at end of decay
            last_epoch: The index of last epoch/step
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_fraction = decay_fraction
        self.min_lr_fraction = min_lr_fraction

        # Calculate decay phase boundaries
        self.decay_steps = int(total_steps * decay_fraction)
        self.steady_end = total_steps - self.decay_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Use last_epoch as step counter (it gets incremented on each scheduler.step() call)
        current_step = self.last_epoch

        if current_step < self.warmup_steps:
            # Linear warmup phase
            scale = (current_step + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        elif current_step < self.steady_end:
            # Steady phase - maintain base learning rate
            return self.base_lrs
        else:
            # Linear decay phase
            steps_into_decay = current_step - self.steady_end
            decay_progress = min(steps_into_decay / self.decay_steps, 1.0)
            # Linear interpolation from 1.0 to min_lr_fraction
            scale = 1.0 - decay_progress * (1.0 - self.min_lr_fraction)
            return [base_lr * scale for base_lr in self.base_lrs]


class Trainer:
    """Step-based training loop for FSL models."""

    def __init__(
        self,
        model: Model,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_type: str = "wsd",
        scheduler_kwargs: Dict = None,
    ):
        """
        Args:
            model: Model instance (LinearProbe or MLPProbe)
            device: Device to run training on
            learning_rate: Initial/maximum learning rate
            weight_decay: L2 regularization strength
            scheduler_type: Type of LR scheduler ('wsd', 'wsl', 'cosine', 'constant')
            scheduler_kwargs: Keyword arguments for the scheduler. Common options:
                - For 'wsd': warmup_steps, total_steps, decay_fraction (default 0.1), min_lr_fraction (default 0.01)
                - For 'wsl': warmup_steps, step_size, gamma
                - For 'cosine': T_max, eta_min
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Setup learning rate scheduler
        self.scheduler = None
        scheduler_kwargs = scheduler_kwargs or {}

        if scheduler_type == "wsd":
            # Default values for WSD scheduler
            wsd_defaults = {
                "warmup_steps": 100,
                "total_steps": 1000,
                "decay_fraction": 0.1,
                "min_lr_fraction": 0.01,
            }
            wsd_kwargs = {**wsd_defaults, **scheduler_kwargs}
            self.scheduler = WarmupSteadyDecayLR(self.optimizer, **wsd_kwargs)
        elif scheduler_type == "wsl":
            # Default values for WSL scheduler (legacy)
            wsl_defaults = {"warmup_steps": 100, "step_size": 500, "gamma": 0.1}
            wsl_kwargs = {**wsl_defaults, **scheduler_kwargs}
            self.scheduler = WarmupStepLR(self.optimizer, **wsl_kwargs)
        elif scheduler_type == "cosine":
            # Cosine annealing scheduler
            cosine_defaults = {"T_max": 1000, "eta_min": 1e-6}
            cosine_kwargs = {**cosine_defaults, **scheduler_kwargs}
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **cosine_kwargs
            )
        # 'constant' means no scheduler

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }
        self.global_step = 0

    def compute_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute classification accuracy."""
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass - batch is already on GPU
        loss = self.model.forward_loss(batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        # Compute metrics
        with torch.no_grad():
            outputs = self.model(batch)
            acc = self.compute_accuracy(outputs, batch["labels"])

        self.global_step += 1
        return loss.item(), acc

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model on entire validation set."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Batch is already on GPU from dataset
                loss = self.model.forward_loss(batch)
                outputs = self.model(batch)
                acc = self.compute_accuracy(outputs, batch["labels"])

                total_loss += loss.item()
                total_acc += acc
                num_batches += 1

        return total_loss / num_batches, total_acc / num_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_steps: int = 1000,
        val_interval: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model for a fixed number of steps.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            num_steps: Total number of training steps
            val_interval: Validate every N steps
            early_stopping_patience: Patience for early stopping (in validation intervals)
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # Create infinite iterator over training data
        train_iter = iter(train_loader)

        # Progress bar
        pbar = tqdm(range(num_steps), desc="Training", disable=not verbose)

        for step in pbar:
            # Get next batch (cycle through dataset if needed)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Training step
            train_loss, train_acc = self.train_step(batch)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["learning_rates"].append(current_lr)

            # Validation at intervals
            if val_loader is not None and (step + 1) % val_interval == 0:
                val_loss, val_acc = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"\nEarly stopping at step {step+1}")
                        break

                # Update progress bar
                if verbose:
                    pbar.set_postfix(
                        {
                            "loss": f"{train_loss:.4f}",
                            "acc": f"{train_acc:.3f}",
                            "val_loss": f"{val_loss:.4f}",
                            "val_acc": f"{val_acc:.3f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )
            else:
                # Update progress bar with training metrics only
                if verbose:
                    pbar.set_postfix(
                        {
                            "loss": f"{train_loss:.4f}",
                            "acc": f"{train_acc:.3f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )

        # Restore best model if we did validation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"Restored best model with val_loss: {best_val_loss:.4f}")

        return self.history

    def predict_proba(self, embeddings: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get probabilities for classification.

        Args:
            embeddings: Input embeddings (N, D)

        Returns:
            Class probabilities (N, K)
        """
        self.model.eval()

        # Convert to tensor and move to device
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        embeddings = embeddings.to(self.device)

        with torch.no_grad():
            # Create batch format
            if len(embeddings.shape) == 1:
                embeddings = embeddings.unsqueeze(0)

            batch = {"inputs": embeddings}
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def predict(
        self, embeddings: Union[np.ndarray, torch.Tensor], threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            embeddings: Input embeddings (N, D)
            threshold: For binary tasks (K=2), threshold for positive class

        Returns:
            Predictions (N,) — class indices
        """
        probs = self.predict_proba(embeddings)
        K = probs.shape[1]
        if K == 2:
            return (probs[:, 1] >= threshold).astype(int)
        return np.argmax(probs, axis=1)

    def predict_scores(self, embeddings: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get raw scores/logits for classification.

        Args:
            embeddings: Input embeddings (N, D)

        Returns:
            Logits per class (N, K) for multi-class; for binary (K=2), returns margin (pos - neg)
        """
        self.model.eval()

        # Convert to tensor and move to device
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        embeddings = embeddings.to(self.device)

        with torch.no_grad():
            # Create batch format
            if len(embeddings.shape) == 1:
                embeddings = embeddings.unsqueeze(0)

            batch = {"inputs": embeddings}
            outputs = self.model(batch)

            K = outputs.shape[1]
            if K == 2:
                scores = outputs[:, 1] - outputs[:, 0]
                return scores.cpu().numpy()
            return outputs.cpu().numpy()


def prepare_support_set(
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare binary classification support set with automatic train/val split.

    Args:
        positive_embeddings: Positive class embeddings (N_pos, D)
        negative_embeddings: Negative class embeddings (N_neg, D)
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        train_embeddings, train_labels, val_embeddings, val_labels
        Labels: 1 for positive class, 0 for negative class
    """
    np.random.seed(random_seed)

    # Binary classification: combine positive (label=1) and negative (label=0) examples
    all_embeddings = np.vstack([positive_embeddings, negative_embeddings])
    all_labels = np.hstack(
        [
            np.ones(len(positive_embeddings)),  # Positive class = 1
            np.zeros(len(negative_embeddings)),  # Negative class = 0
        ]
    )

    # Shuffle data
    indices = np.random.permutation(len(all_embeddings))
    all_embeddings = all_embeddings[indices]
    all_labels = all_labels[indices]

    # Split into train/val
    val_size = int(len(all_embeddings) * val_split)

    train_embeddings = all_embeddings[val_size:]
    train_labels = all_labels[val_size:]
    val_embeddings = all_embeddings[:val_size]
    val_labels = all_labels[:val_size]

    return train_embeddings, train_labels, val_embeddings, val_labels


def prepare_multiclass_set(
    class_embeddings: Union[
        Dict[Union[int, str], np.ndarray], Tuple[np.ndarray, ...], list
    ],
    val_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare multi-class classification dataset with automatic train/val split.

    Args:
        class_embeddings: Either a dict of label->embeddings or a list/tuple of per-class embeddings.
            - Dict keys can be ints or strings. Strings will be mapped to integer labels [0..K-1]
            - For list/tuple, classes are assigned labels [0..K-1] by position
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        train_embeddings, train_labels, val_embeddings, val_labels
    """
    np.random.seed(random_seed)

    # Normalize input into iterable of (label, embeddings)
    pairs: list[Tuple[int, np.ndarray]] = []
    if isinstance(class_embeddings, dict):
        # Map non-integer labels to consecutive ints (preserve insertion order)
        label_map: Dict[Union[int, str], int] = {}
        next_label = 0
        for k in class_embeddings.keys():
            if isinstance(k, int):
                label_map[k] = k
            else:
                label_map[k] = next_label
                next_label += 1
        for k, emb in class_embeddings.items():
            pairs.append((label_map[k], emb))
    else:
        # list/tuple of per-class embeddings
        for i, emb in enumerate(class_embeddings):
            pairs.append((i, emb))

    # Stack all embeddings and build labels
    all_embeddings = []
    all_labels = []
    for lbl, emb in pairs:
        all_embeddings.append(emb)
        all_labels.append(np.full(len(emb), lbl, dtype=np.int64))

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.hstack(all_labels)

    # Shuffle
    indices = np.random.permutation(len(all_embeddings))
    all_embeddings = all_embeddings[indices]
    all_labels = all_labels[indices]

    # Split
    val_size = int(len(all_embeddings) * val_split)
    train_embeddings = all_embeddings[val_size:]
    train_labels = all_labels[val_size:]
    val_embeddings = all_embeddings[:val_size]
    val_labels = all_labels[:val_size]

    return train_embeddings, train_labels, val_embeddings, val_labels


def train_model(
    positive_embeddings: Union[
        np.ndarray, Dict[Union[int, str], np.ndarray], Tuple[np.ndarray, ...], list
    ],
    negative_embeddings: Optional[np.ndarray] = None,
    model_type: str = "linear",
    hidden_dim: int = 256,
    batch_size: int = 32,
    num_steps: int = 1000,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler_type: str = "wsd",
    scheduler_kwargs: Dict = None,
    val_split: float = 0.2,
    val_interval: int = 50,
    early_stopping_patience: int = 10,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    random_seed: int = 42,
    num_classes: Optional[int] = None,
) -> Tuple[Model, Trainer, Dict]:
    """
    Train a classification model for Few-Shot Learning.

    Args:
        positive_embeddings: Positive class embeddings (N_pos, D) or dict mapping IDs to embeddings
        negative_embeddings: Negative class embeddings (N_neg, D) or dict mapping IDs to embeddings
        model_type: 'linear' or 'mlp'
        hidden_dim: Hidden dimension for MLP
        batch_size: Batch size for training
        num_steps: Total number of training steps
        learning_rate: Initial/maximum learning rate
        weight_decay: L2 regularization
        scheduler_type: LR scheduler type ('wsd', 'wsl', 'cosine', 'constant')
        scheduler_kwargs: Keyword arguments for the scheduler. Common options:
            - For 'wsd': warmup_steps, total_steps (defaults to num_steps), decay_fraction (default 0.1), min_lr_fraction (default 0.01)
            - For 'wsl': warmup_steps, step_size, gamma
            - For 'cosine': T_max (defaults to num_steps), eta_min
        val_split: Validation split fraction
        val_interval: Validate every N steps
        early_stopping_patience: Early stopping patience (in validation intervals)
        device: Device to use (auto-detect if None)
        verbose: Print progress
        random_seed: Random seed
        num_classes: If provided, sets the number of classes (model output dim)
            explicitly rather than inferring from `train_labels`. If the training
            data does not contain samples for all classes in `num_classes`, training
            proceeds gracefully on the available classes and the unused class logits
            simply won't be supervised.

    Returns:
        Trained model, trainer instance, and training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Using device: {device}")

    # Prepare data: support binary (pos/neg) and multi-class via dict/list
    if negative_embeddings is None and isinstance(
        positive_embeddings, (dict, list, tuple)
    ):
        train_emb, train_labels, val_emb, val_labels = prepare_multiclass_set(
            positive_embeddings, val_split, random_seed
        )
    else:
        # Backward-compatible binary path
        # If dicts were passed here by mistake, convert to arrays
        if isinstance(positive_embeddings, dict):
            positive_embeddings = np.array(list(positive_embeddings.values()))
        if isinstance(negative_embeddings, dict):
            negative_embeddings = np.array(list(negative_embeddings.values()))
        train_emb, train_labels, val_emb, val_labels = prepare_support_set(
            positive_embeddings, negative_embeddings, val_split, random_seed
        )

    # Set dimensions
    input_dim = train_emb.shape[1]
    # Determine number of classes
    if num_classes is not None:
        output_dim = int(num_classes)
        present_classes = np.unique(train_labels)
        # Informational note if some classes are missing from training labels
        if verbose and len(present_classes) < output_dim:
            missing = sorted(set(range(output_dim)) - set(present_classes.tolist()))
            print(
                f"Note: only {len(present_classes)} of {output_dim} classes present in training data; missing labels: {missing}."
            )
    else:
        # Fallback to dynamic inference from labels
        output_dim = int(len(np.unique(train_labels)))

    if verbose:
        print(f"Training set: {len(train_emb)} samples")
        print(f"Validation set: {len(val_emb)} samples")
        print(f"Input dimension: {input_dim}")
        print(f"{output_dim}-class classification task")
        print(f"Training for {num_steps} steps with batch size {batch_size}")

    # Create model
    if model_type == "linear":
        model = LinearProbe(input_dim, output_dim)
    elif model_type == "mlp":
        model = MLPProbe(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create datasets - data stays on GPU throughout
    train_dataset = FSLDataset(train_emb, train_labels, device)
    if len(val_emb) > 0:
        val_dataset = FSLDataset(val_emb, val_labels, device)

    # Create dataloaders with pin_memory for faster GPU transfer (though data is already on GPU)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    if len(val_emb) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    # Prepare scheduler kwargs with defaults
    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    # Set default total_steps for schedulers that need it
    if scheduler_type == "wsd" and "total_steps" not in scheduler_kwargs:
        scheduler_kwargs["total_steps"] = num_steps
    elif scheduler_type == "cosine" and "T_max" not in scheduler_kwargs:
        scheduler_kwargs["T_max"] = num_steps

    # Create trainer
    trainer = Trainer(
        model, device, learning_rate, weight_decay, scheduler_type, scheduler_kwargs
    )

    # Train model
    history = trainer.fit(
        train_loader,
        val_loader,
        num_steps=num_steps,
        val_interval=val_interval,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
    )

    return model, trainer, history


def load_model(checkpoint_path: str, device: Optional[torch.device] = None) -> Model:
    """
    Load a trained binary classification model from checkpoint.

    Args:
        checkpoint_path: Path to saved model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate model architecture (binary classification: output_dim = 2)
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint.get(
        "output_dim", 2
    )  # Default to 2 for binary classification

    if checkpoint["model_type"] == "linear":
        model = LinearProbe(input_dim, output_dim)
    else:
        model = MLPProbe(input_dim, checkpoint["hidden_dim"], output_dim)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def _get_embeddings_and_labels(
    dataset_name: str = "cifar100",
    split: str = "train",
    embeddings_path: str = None,
    labels_path: str = None,
    num_samples: int = None,
) -> Tuple[np.ndarray, np.ndarray]:

    if embeddings_path is None or labels_path is None:
        import fiftyone.zoo as foz

        dataset = foz.load_zoo_dataset(
            dataset_name, max_samples=num_samples, split=split
        )

        if embeddings_path is None:
            model = foz.load_zoo_model("clip-vit-base32-torch")
            embeddings = dataset.compute_embeddings(
                model, batch_size=16, num_workers=8, progress=True
            )
            embeddings = np.array(embeddings)
            np.save(
                f"./data/embeddings_clip_{dataset_name}_{split}_full.npy", embeddings
            )
        else:
            embeddings = np.load(embeddings_path)

        if labels_path is None:
            classes = sorted({s.ground_truth.label for s in dataset})
            label_to_index = {c: i for i, c in enumerate(classes)}

            labels_int = np.array(
                [label_to_index[s.ground_truth.label] for s in dataset], dtype=np.int64
            )
            np.save(f"./data/labels_{dataset_name}_{split}_full.npy", labels_int)
        else:
            labels_int = np.load(labels_path)

    else:
        embeddings = np.load(embeddings_path)
        labels_int = np.load(labels_path)

    return embeddings, labels_int


def train_mlp_from_embeddings_and_labels(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Model, Trainer, Dict]:
    """
    Train an MLP on CIFAR-10 embeddings using labels from FiftyOne.

    Args:
        embeddings: Array of shape (N, D) with CIFAR-10 embeddings.
        labels: Array of shape (N,) with integer class labels corresponding to embeddings.
        num_classes: Total number of classes in the dataset (e.g., 10 for CIFAR-10).
        checkpoint_path: Optional path to save a checkpoint.

    Returns:
        (model, trainer, history)
    """

    if len(labels) != embeddings.shape[0]:
        raise ValueError(
            f"Label count ({len(labels)}) does not match embeddings count ({embeddings.shape[0]}). "
        )

    class_dict = {i: embeddings[labels == i] for i in range(num_classes)}

    model, trainer, history = train_model(
        class_dict,
        None,
        model_type="mlp",
        val_split=0.0,
        num_classes=num_classes,
    )

    if checkpoint_path:
        checkpoint = {
            "model_type": "mlp",
            "input_dim": embeddings.shape[1],
            "output_dim": num_classes,
            "model_state_dict": model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

    return model, trainer, history


def _random_train_set(
    embeddings, subset_size, rng: Optional[np.random.Generator] = None
):
    """
    Returns a random subset of indices for the given embeddings.

    Uses a dedicated NumPy Generator to avoid being affected by any global
    np.random.seed() calls made elsewhere (e.g., by other libraries).

    Args:
        embeddings: Array-like of embeddings to sample from
        subset_size: Fraction of the dataset to sample (0,1]
        rng: Optional np.random.Generator. If None, a fresh generator is created.

    Returns:
        numpy.ndarray of selected indices (without replacement)
    """
    if rng is None:
        rng = np.random.default_rng()
    num_samples = int(len(embeddings) * subset_size)
    return rng.choice(len(embeddings), size=num_samples, replace=False)


def _zcore_train_set(embeddings, subset_size=0.3):

    from zcore import zcore_scores

    scores = zcore_scores(embeddings)
    num_samples = int(len(embeddings) * subset_size)
    return np.argsort(scores)[-num_samples:]


def _do_pca(embeddings, n_components):
    from zcore import pca_reduction

    return pca_reduction(embeddings, n_components=n_components)


def _class_imbalanced_train_set(embeddings, labels):
    """
    Out of the total number of classes, randomly select 10% and included them in full.
    For the remaining 90% of classes, only include 10% of their samples.
    """

    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    num_imbalanced_classes = int(num_classes * 0.1)

    # Randomly select 10% of classes to include in full
    imbalanced_classes = np.random.choice(
        unique_classes, size=num_imbalanced_classes, replace=False
    )

    # Create a mask for the imbalanced classes
    imbalanced_mask = np.isin(labels, imbalanced_classes)

    # For the remaining 90% of classes, randomly select 10% of their samples
    balanced_mask = ~imbalanced_mask
    balanced_indices = np.where(balanced_mask)[0]
    selected_balanced_indices = np.random.choice(
        balanced_indices, size=int(len(balanced_indices) * 0.1), replace=False
    )

    # Combine the indices for the imbalanced and balanced classes
    final_indices = np.concatenate(
        [np.where(imbalanced_mask)[0], selected_balanced_indices]
    )

    np.save("./data/cifar100_imbalanced_labels.npy", labels[final_indices])
    np.save("./data/cifar100_imbalanced_clip_embeddings.npy", embeddings[final_indices])

    return embeddings[final_indices], labels[final_indices]


def _are_all_classes_present(labels):
    unique_classes = np.unique(labels)
    return len(unique_classes) == 100


def compare_random_vs_zcore():
    # embeddings_train, labels_train = _get_embeddings_and_labels("cifar100", embeddings_path="/tmp/embeddings_cifar100_train.npy")
    # embeddings_train, labels_train = _class_imbalanced_train_set(embeddings_train, labels_train)

    subset_size = 0.1
    num_classes = 100

    embeddings_train = np.load("./data/cifar100_imbalanced_clip_embeddings.npy")
    labels_train = np.load("./data/cifar100_imbalanced_labels.npy")

    # Use a stable local RNG that won't be affected by global reseeding
    rng = np.random.default_rng()

    rand_indices = _random_train_set(embeddings_train, subset_size=subset_size, rng=rng)
    embeddings_train_rand, labels_train_rand = (
        embeddings_train[rand_indices],
        labels_train[rand_indices],
    )
    print(_are_all_classes_present(labels_train_rand))

    model_rand, trainer_rand, history_rand = train_mlp_from_embeddings_and_labels(
        embeddings_train_rand, labels_train_rand, num_classes=num_classes
    )

    # reduced_embeddings_train = _do_pca(embeddings_train, n_components=128)
    # zcore_indices = _zcore_train_set(reduced_embeddings_train, subset_size=subset_size)
    zcore_indices = _zcore_train_set(embeddings_train, subset_size=subset_size)
    embeddings_train_zcore, labels_train_zcore = (
        embeddings_train[zcore_indices],
        labels_train[zcore_indices],
    )

    print(
        f"_are_all_classes_present(labels_train_zcore): {_are_all_classes_present(labels_train_zcore)}"
    )

    model_zcore, trainer_zcore, history_zcore = train_mlp_from_embeddings_and_labels(
        embeddings_train_zcore, labels_train_zcore, num_classes=num_classes
    )

    embeddings_val, labels_val = _get_embeddings_and_labels(
        "cifar100", split="test", embeddings_path="/tmp/embeddings_cifar100_test.npy"
    )
    val_loss_rand, val_acc_rand = trainer_rand.validate(
        DataLoader(
            FSLDataset(embeddings_val, labels_val, device=trainer_rand.device),
            batch_size=64,
        )
    )
    print(
        f"Validation Loss random: {val_loss_rand:.4f}, Validation Accuracy random: {val_acc_rand:.4f}"
    )

    val_loss_zcore, val_acc_zcore = trainer_zcore.validate(
        DataLoader(
            FSLDataset(embeddings_val, labels_val, device=trainer_zcore.device),
            batch_size=64,
        )
    )
    print(
        f"Validation Loss zcore: {val_loss_zcore:.4f}, Validation Accuracy zcore: {val_acc_zcore:.4f}"
    )


def _normalize_embeddings(embeddings):

    maxes = np.max(embeddings, axis=0, keepdims=True)
    mins = np.min(embeddings, axis=0, keepdims=True)
    return (embeddings - mins) / (maxes - mins + 1e-10)


def _do_10_runs():
    subset_size = 0.1
    num_classes = 100

    embeddings_train = np.load("./data/cifar100_imbalanced_clip_embeddings.npy")
    # embeddings_normalized = _normalize_embeddings(embeddings_train)
    embeddings_normalized = embeddings_train
    labels_train = np.load("./data/cifar100_imbalanced_labels.npy")

    prev_indices = None

    # Use a dedicated RNG to decouple from any global np.random.seed() calls
    rng = np.random.default_rng()

    overlaps = []
    accuracies = []

    embeddings_val, labels_val = _get_embeddings_and_labels(
        "cifar100",
        split="test",
        embeddings_path="./data/embeddings_clip_cifar100_test_full.npy",
        labels_path="./data/labels_cifar100_test_full.npy",
    )

    for _ in range(1):

        # reduced_embeddings_train = _do_pca(embeddings_train, n_components=128)
        # curr_indices = _zcore_train_set(reduced_embeddings_train, subset_size=subset_size)
        curr_indices = _zcore_train_set(embeddings_normalized, subset_size=subset_size)
        # curr_indices = _random_train_set(embeddings_train, subset_size=subset_size, rng=rng)

        # Compare current coreset to previous (sentinel pattern)
        if prev_indices is not None:
            set_prev = set(prev_indices)
            set_curr = set(curr_indices)
            intersection = set_prev.intersection(set_curr)
            overlap = len(intersection) / len(set_prev) if len(set_prev) > 0 else 0.0
            overlaps.append(overlap)

        prev_indices = curr_indices

        embeddings_train_zcore, labels_train_zcore = (
            embeddings_train[curr_indices],
            labels_train[curr_indices],
        )
        print(
            f"_are_all_classes_present(labels_train_zcore): {_are_all_classes_present(labels_train_zcore)}"
        )
        model_zcore, trainer_zcore, history_zcore = (
            train_mlp_from_embeddings_and_labels(
                embeddings_train_zcore, labels_train_zcore, num_classes=num_classes
            )
        )
        val_loss_zcore, val_acc_zcore = trainer_zcore.validate(
            DataLoader(
                FSLDataset(embeddings_val, labels_val, device=trainer_zcore.device),
                batch_size=64,
            )
        )
        print(
            f"Validation Loss zcore: {val_loss_zcore:.4f}, Validation Accuracy zcore: {val_acc_zcore:.4f}"
        )
        accuracies.append(val_acc_zcore)

    print()
    print("Average overlap between consecutive runs:", np.mean(overlaps))
    print("All overlaps:", overlaps)
    print("Average validation accuracy:", np.mean(accuracies))
    print("All validation accuracies:", accuracies)


if __name__ == "__main__":
    # compare_random_vs_zcore()
    _do_10_runs()
