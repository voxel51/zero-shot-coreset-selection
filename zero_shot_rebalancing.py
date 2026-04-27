import contextlib

import numpy as np
import fiftyone.zoo as foz
import hashlib
import os
import contextlib

from train import train_mlp_from_embeddings_and_labels, FSLDataset
from torch.utils.data import DataLoader
import time

from boundary_central import (_balancedness, 
                              ideal_train_set, 
                              random_train_set,
                              _get_zcore_like_scores,
                              score_based_selection,
                              boundaryness_centrality_scores,
                              select_balanced_boundary_central)

from openimages import _alignment, coreset_with_target_alignment

TEMPLATES_20 = [
    "a photo of a {}",
    "a photo of the {}",
    "a blurry photo of a {}",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
    "a bright photo of a {}",
    "a dark photo of a {}",
    "a low resolution photo of a {}",
    "a high resolution photo of a {}",
    "a photo of a small {}",
    "a photo of a big {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a photo of one {}",
    "a photo of many {}",
    "a photo of a {} in the scene",
    "a photo of a {} in the wild",
    "a good photo of a {}",
    "a bad photo of a {}",
    "a jpeg photo of a {}",
]


def _l2norm(x, axis=-1, eps=1e-12):
    x = np.asarray(x)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def clip_text_ensemble_features(
    model,
    classes,
    templates,
):
    """
    Returns: (n_classes, dim) normalized text features,
    computed by averaging normalized text embeddings across templates.
    """

    # Prompts ordered as: for each template, all classes
    # prompts = [t.format(c) for t in templates for c in classes]

    # FiftyOne TorchCLIPModel.embed_prompts() expects full prompt strings
    # text = model.embed_prompts(prompts)              # (T*C, dim)
    # text = _l2norm(text, axis=1)

    # T = len(templates)
    C = len(classes)
    # text = text.reshape(T, C, -1)                    # (T, C, dim)

    text_sum = None

    # Embed one template at a time: prompts = [t.format(c) for c in classes]
    for t in templates:
        print(t)
        prompts_t = [t.format(c) for c in classes]   # (C,)
        emb_t = model.embed_prompts(prompts_t)       # (C, dim)
        emb_t = _l2norm(emb_t, axis=1)

        if text_sum is None:
            text_sum = emb_t.astype(np.float32, copy=False)
        else:
            text_sum += emb_t

    # text_avg = text.mean(axis=0)    
    text_avg = text_sum / float(len(templates))                 # (C, dim)
    text_avg = _l2norm(text_avg, axis=1)

    return text_avg


def clip_prompt_ensemble_predict(
    image_embeddings,
    text_features,
):
    """
    Returns:
      pred_idxs: (n_samples,) indices into `classes`
      scores: (n_samples, n_classes) cosine-sim scores
    """
    # Load model only for text embedding
    # model = foz.load_zoo_model("clip-vit-base32-torch")

    # text_features = clip_text_ensemble_features(
    #     model, classes, templates, cache_dir=cache_dir
    # )  # (C, dim)

    img = _l2norm(image_embeddings, axis=1)          # (N, dim)
    scores = img @ text_features.T                   # (N, C)
    pred_idxs = scores.argmax(axis=1)
    return pred_idxs, scores



def _prepare_data_cifar100():
    embeddings = np.load("./data/clip_embeddings_cifar100_train_sweepImbalanced.npy")
    labels = np.load("./data/labels_cifar100_train_sweepImbalanced.npy")

    # embeddings = np.load("./data/clip_embeddings_cifar100_train_balanced.npy")
    # labels = np.load("./data/labels_cifar100_train_balanced.npy")

    # idxs = np.load("./data/idxs_cifar100_train_sweepImbalanced.npy")

    dataset_full = foz.load_zoo_dataset("cifar100", split="train")
    classes = sorted({s.ground_truth.label for s in dataset_full})
    classes_clean = [c.replace("_", " ") for c in classes]
    # label_to_index = {c: i for i, c in enumerate(classes_clean)}
    # index_to_label = {i: c for c, i in label_to_index.items()}
    
    model = foz.load_zoo_model("clip-vit-base32-torch", classes=classes_clean)

    # ids = dataset_full.values("id")
    # sweep_ids = [ids[i] for i in idxs]
    # dataset = dataset_full.select(sweep_ids, ordered=True)

    # dataset.apply_model(model, "zero_shot_labels", batch_size=32, num_workers=4)

    # pred_labels = dataset.values("zero_shot_labels")
    # pred_labels_list = [entry["label"] for entry in pred_labels]
    # pred_labels_idxs = [label_to_index[c] for c in pred_labels_list]

    text_features = clip_text_ensemble_features(
        model, classes_clean, TEMPLATES_20,
    )  # (C, dim)

    pred_labels_idxs, scores = clip_prompt_ensemble_predict(embeddings, text_features)

    # Match CLIP logits: exp(logit_scale) * cosine_similarity
    logit_scale = float(model._model.logit_scale.exp().detach().cpu().item())
    logits = scores * logit_scale  # (N, C)

    N, C = logits.shape
    p_max = np.empty(N, dtype=np.float32)
    margin = np.empty(N, dtype=np.float32)
    entropy = np.empty(N, dtype=np.float32)

    chunk = 4096
    for i in range(0, N, chunk):
        lb = logits[i:i+chunk].astype(np.float32, copy=False)  # (B, C)

        # top-1 / top-2 logit margin (no softmax needed)
        top2 = np.partition(lb, -2, axis=1)[:, -2:]           # (B, 2) unsorted
        m1 = top2.max(axis=1)
        m2 = top2.min(axis=1)
        margin[i:i+chunk] = m1 - m2

        # stable softmax for p_max + entropy
        lb = lb - lb.max(axis=1, keepdims=True)
        exp_lb = np.exp(lb)
        Z = exp_lb.sum(axis=1, keepdims=True)
        p = exp_lb / (Z + 1e-12)

        p_max[i:i+chunk] = p.max(axis=1)
        entropy[i:i+chunk] = -(p * np.log(p + 1e-12)).sum(axis=1)

    np.save("./data/clip_pred_labels_cifar100_train_sweepImbalanced.npy", pred_labels_idxs)
    np.save("./data/clip_pred_pmax_cifar100_train_sweepImbalanced.npy", p_max)
    np.save("./data/clip_pred_margin_cifar100_train_sweepImbalanced.npy", margin)
    np.save("./data/clip_pred_entropy_cifar100_train_sweepImbalanced.npy", entropy)


    print(np.mean(pred_labels_idxs == labels))

    # np.save("./data/clip_pred_labels_cifar100_train_balanced.npy", pred_labels_idxs)


def _prepare_data_openimages():

    embeddings = np.load("./data/openimages_v7/clip_embeddings_train200000SingleLabel.npy")
    labels = np.load("./data/openimages_v7/train_label_200000SingleLabel.npy")

    classes = np.load("./data/openimages_v7/common_classes_200000SingleLabel.npy", allow_pickle=True).tolist()


    # idxs = np.load("./data/idxs_cifar100_train_sweepImbalanced.npy")

    # label_to_index = {c: i for i, c in enumerate(classes_clean)}
    # index_to_label = {i: c for c, i in label_to_index.items()}
    
    model = foz.load_zoo_model("clip-vit-base32-torch", classes=classes)

    # ids = dataset_full.values("id")
    # sweep_ids = [ids[i] for i in idxs]
    # dataset = dataset_full.select(sweep_ids, ordered=True)

    # dataset.apply_model(model, "zero_shot_labels", batch_size=32, num_workers=4)

    # pred_labels = dataset.values("zero_shot_labels")
    # pred_labels_list = [entry["label"] for entry in pred_labels]
    # pred_labels_idxs = [label_to_index[c] for c in pred_labels_list]

    text_features = clip_text_ensemble_features(
        model, classes, TEMPLATES_20,
    )  # (C, dim)


    pred_labels_idxs, scores = clip_prompt_ensemble_predict(embeddings, text_features)

    # Match CLIP logits: exp(logit_scale) * cosine_similarity
    logit_scale = float(model._model.logit_scale.exp().detach().cpu().item())
    logits = scores * logit_scale  # (N, C)

    N, C = logits.shape
    p_max = np.empty(N, dtype=np.float32)
    margin = np.empty(N, dtype=np.float32)
    entropy = np.empty(N, dtype=np.float32)

    chunk = 4096
    for i in range(0, N, chunk):
        lb = logits[i:i+chunk].astype(np.float32, copy=False)  # (B, C)

        # top-1 / top-2 logit margin (no softmax needed)
        top2 = np.partition(lb, -2, axis=1)[:, -2:]           # (B, 2) unsorted
        m1 = top2.max(axis=1)
        m2 = top2.min(axis=1)
        margin[i:i+chunk] = m1 - m2

        # stable softmax for p_max + entropy
        lb = lb - lb.max(axis=1, keepdims=True)
        exp_lb = np.exp(lb)
        Z = exp_lb.sum(axis=1, keepdims=True)
        p = exp_lb / (Z + 1e-12)

        p_max[i:i+chunk] = p.max(axis=1)
        entropy[i:i+chunk] = -(p * np.log(p + 1e-12)).sum(axis=1)

    np.save("./data/openimages_v7/clip_pred_labels_train200000SingleLabel.npy", pred_labels_idxs)
    np.save("./data/openimages_v7/clip_pred_pmax_train200000SingleLabel.npy", p_max)
    np.save("./data/openimages_v7/clip_pred_margin_train200000SingleLabel.npy", margin)
    np.save("./data/openimages_v7/clip_pred_entropy_train200000SingleLabel.npy", entropy)

    print(np.mean(pred_labels_idxs == labels))

    



def rebalance_with_scores(
    zcore_like_scores,
    labels,
    subset_size: float = 0.1,
    seed: int | None = None,
    shuffle_output: bool = True,
) -> np.ndarray:
    """
    Like ideal_train_set(labels, subset_size), but within each class selects
    samples in descending zcore_like_scores order (random tie-break).

    Args:
        zcore_like_scores: (N,) array, higher = higher priority
        labels: (N,) array-like class labels (e.g. zero_shot_labels)
        subset_size: fraction of N to select
        seed: RNG seed for remainder distribution + tie-breaks
        shuffle_output: if True, shuffles the returned indices

    Returns:
        chosen: (K,) int array of selected indices
    """
    labels = np.asarray(labels)
    scores = np.asarray(zcore_like_scores)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and zcore_like_scores must be 1D arrays.")
    if len(labels) != len(scores):
        raise ValueError("labels and zcore_like_scores must have the same length.")
    if not (0 < subset_size <= 1):
        raise ValueError("subset_size must be in (0, 1].")

    n = len(labels)
    rng = np.random.default_rng(seed)

    classes, y_inv = np.unique(labels, return_inverse=True)
    C = len(classes)

    # Group indices by class; within each class sort by score desc (random tie-break)
    by_class = []
    for c in range(C):
        idxs = np.flatnonzero(y_inv == c)
        if idxs.size == 0:
            by_class.append(idxs)
            continue

        # random tie-break, but always "score first"
        tie = rng.random(idxs.size)
        order = np.lexsort((tie, -scores[idxs]))  # last key is primary
        by_class.append(idxs[order])

    target_total = int(round(subset_size * n))
    target_total = max(1, min(target_total, n))

    base = target_total // C
    rem = target_total % C

    # Same "spread remainder randomly" behavior as ideal_train_set
    order = rng.permutation(C)
    desired = np.full(C, base, dtype=int)
    desired[order[:rem]] += 1

    chosen = []
    chosen_counts = np.zeros(C, dtype=int)

    # First pass: take up to desired from each class (best scores first)
    for c in range(C):
        take = min(desired[c], len(by_class[c]))
        if take:
            chosen.append(by_class[c][:take])
            chosen_counts[c] += take
            by_class[c] = by_class[c][take:]  # remove taken

    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)

    # Redistribute any shortfall, keeping counts as balanced as possible
    short = target_total - len(chosen)
    if short > 0:
        while short > 0:
            candidates = [c for c in range(C) if len(by_class[c]) > 0]
            if not candidates:
                break

            min_count = min(chosen_counts[c] for c in candidates)
            under = [c for c in candidates if chosen_counts[c] == min_count]
            c = int(rng.choice(under))

            chosen = np.append(chosen, by_class[c][0])  # next-best remaining in class c
            by_class[c] = by_class[c][1:]
            chosen_counts[c] += 1
            short -= 1

    if shuffle_output and chosen.size:
        rng.shuffle(chosen)

    return chosen




def main():

    cifar100 = True
    openimages = False

    if cifar100: 

        num_classes = 100
        imbalanced_baseset = True

        embeddings_val = np.load("./data/embeddings_clip_cifar100_test_full.npy")
        labels_val = np.load("./data/labels_cifar100_test_full.npy")

        if imbalanced_baseset:

            embeddings = np.load("./data/clip_embeddings_cifar100_train_sweepImbalanced.npy")
            labels = np.load("./data/labels_cifar100_train_sweepImbalanced.npy")
            zero_shot_labels = np.load("./data/clip_pred_labels_cifar100_train_sweepImbalanced.npy")

            p_max = np.load("./data/clip_pred_pmax_cifar100_train_sweepImbalanced.npy")
            entropy = np.load("./data/clip_pred_entropy_cifar100_train_sweepImbalanced.npy")
            margin = np.load("./data/clip_pred_margin_cifar100_train_sweepImbalanced.npy")


        else:
            # Create balanced baseset
            embeddings = np.load("./data/clip_embeddings_cifar100_train_balanced.npy")
            labels = np.load("./data/labels_cifar100_train_balanced.npy")
            zero_shot_labels = np.load("./data/clip_pred_labels_cifar100_train_balanced.npy")

            p_max = np.load("./data/clip_pred_pmax_cifar100_train_balanced.npy")
            entropy = np.load("./data/clip_pred_entropy_cifar100_train_balanced.npy")
            margin = np.load("./data/clip_pred_margin_cifar100_train_balanced.npy")


    
    elif openimages:

        num_classes = 335

        labels_val = np.load("./data/openimages_v7/val_label_20000SingleLabel.npy")
        labels = np.load("./data/openimages_v7/train_label_200000SingleLabel.npy")

        embeddings = np.load("./data/openimages_v7/clip_embeddings_train200000SingleLabel.npy")
        embeddings_val = np.load("./data/openimages_v7/clip_embeddings_val20000SingleLabel.npy")   

        zero_shot_labels = np.load("./data/openimages_v7/clip_pred_labels_train200000SingleLabel.npy")

        margin = np.load("./data/openimages_v7/clip_pred_margin_train200000SingleLabel.npy")
        entropy = np.load("./data/openimages_v7/clip_pred_entropy_train200000SingleLabel.npy")
        p_max = np.load("./data/openimages_v7/clip_pred_pmax_train200000SingleLabel.npy")




    print("counts in baseset:", np.bincount(labels))
    print()

    # sanity check, compare accuracy of zero-shot labels vs true labels
    accuracy = np.mean(zero_shot_labels == labels)
    print(f"Zero-shot label accuracy: {accuracy:.4f}")


    # for frac_boundary in [0.1, 0.3, 0.5]:
    for frac_boundary in [0.3]:
        # print(f"\n--- frac_boundary: {frac_boundary} ---")

        for subset_size in [0.1 ,0.3 ,0.5, 0.7]:
        #for subset_size in [0.5]:
            print(f"\n--- Subset size: {subset_size} ---")

            balancednesses = [[], [], [], [], [], []]  # ideal, random, zcore-like, zero-shot, rebalance, balanced boundary-central
            accuracies = [[], [], [], [], [], [], []]  # ideal, random, zcore-like, zero-shot, rebalance, balanced boundary-central
            runtimes = [[], []] # zcore, zero-shot + bc

            alignments = [[], [], [], [], [], []]


            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                for _ in range(5):

                    # # Oracle
                    print()
                    print("Oracle")
                    ideal_idxs = ideal_train_set(labels, subset_size=subset_size)
                    ideal_idxs = coreset_with_target_alignment(
                        labels_base=labels,
                        labels_val=labels_val,
                        subset_size=subset_size,
                        target_alignment=1.0,
                        seed=None,
                    )
                    embeddings_train = embeddings[ideal_idxs]
                    labels_train = labels[ideal_idxs]

                    #print("counts in oracle set:", np.bincount(labels_train))

                    _, trainer, _ = (
                        train_mlp_from_embeddings_and_labels(
                            embeddings_train, labels_train, num_classes=num_classes
                        )
                    )
                    _, val_acc = trainer.validate(
                        DataLoader(
                            FSLDataset(embeddings_val, labels_val, device=trainer.device),
                            batch_size=64,
                        )
                    )
                    accuracies[0].append(val_acc)
                    # alignments[0].append(_alignment(labels_train, labels_val, num_classes=num_classes))


                    # Random
                    print()
                    print("Random")
                    random_idxs = random_train_set(labels, subset_size=subset_size)
                    embeddings_train = embeddings[random_idxs]
                    labels_train = labels[random_idxs]
                    # print("counts in random set:", np.bincount(labels_train))

                    _, trainer, _ = (
                        train_mlp_from_embeddings_and_labels(
                            embeddings_train, labels_train, num_classes=num_classes
                        )
                    )
                    _, val_acc = trainer.validate(
                        DataLoader(
                            FSLDataset(embeddings_val, labels_val, device=trainer.device),
                            batch_size=64,
                        )
                    )
                    accuracies[1].append(val_acc)
                    # alignments[1].append(_alignment(labels_train, labels_val, num_classes=num_classes))

                    # Zcore
                    print()
                    print("Zcore")
                    t1 = time.time()
                    zcore_like_scores = _get_zcore_like_scores(embeddings)
                    zcore_idxs = score_based_selection(zcore_like_scores, subset_size=subset_size)
                    # t2 = time.time()
                    # runtimes[0].append(t2 - t1)
                    embeddings_train = embeddings[zcore_idxs]
                    labels_train = labels[zcore_idxs]
                    # print("counts in zcore set:", np.bincount(labels[zcore_idxs]))

                    _, trainer, _ = (
                        train_mlp_from_embeddings_and_labels(
                            embeddings_train, labels_train, num_classes=num_classes, scores=zcore_like_scores[zcore_idxs]
                        )
                    )
                    _, val_acc = trainer.validate(
                        DataLoader(
                            FSLDataset(embeddings_val, labels_val, device=trainer.device),
                            batch_size=64,
                        )
                    )
                    # print(f"ZCore val accuracy: {val_acc:.4f}")
                    accuracies[2].append(val_acc)
                    # alignments[2].append(_alignment(labels_train, labels_val, num_classes=num_classes))                 

                    # Zero-shot random
                    print()
                    print("Zero-shot random")
                    zero_shot_idxs = ideal_train_set(zero_shot_labels, subset_size=subset_size)

                    zero_shot_idxs = coreset_with_target_alignment(
                        labels_base=zero_shot_labels,
                        labels_val=labels_val,
                        subset_size=subset_size,
                        target_alignment=1.0,
                        seed=None,
                    )

                    embeddings_train = embeddings[zero_shot_idxs]
                    labels_train = labels[zero_shot_idxs]

                    _, trainer, _ = (
                        train_mlp_from_embeddings_and_labels(
                            embeddings_train, labels_train, num_classes=num_classes
                        )
                    )
                    _, val_acc = trainer.validate(
                        DataLoader(
                            FSLDataset(embeddings_val, labels_val, device=trainer.device),
                            batch_size=64,
                        )
                    )
                    accuracies[3].append(val_acc)
                    # alignments[3].append(_alignment(labels_train, labels_val, num_classes=num_classes))

                    # Zero-shot + zcore
                    print()
                    print("Zero-shot + ZCore")
                    rebalance_idxs = rebalance_with_scores(zcore_like_scores, zero_shot_labels, subset_size=subset_size, seed=0)
                    zs_zcore_idxs = coreset_with_target_alignment(
                        labels_base=zero_shot_labels,
                        labels_val=labels_val,
                        subset_size=subset_size,
                        target_alignment=1.0,
                        scores=zcore_like_scores,
                        fraction_pick_low_scores=0.0,
                        seed=None,
                    )
                    embeddings_train = embeddings[zs_zcore_idxs]
                    labels_train = labels[zs_zcore_idxs]

                    _, trainer, _ = (
                        train_mlp_from_embeddings_and_labels(
                            embeddings_train, labels_train, num_classes=num_classes,
                        )
                    )
                    _, val_acc = trainer.validate(
                        DataLoader(
                            FSLDataset(embeddings_val, labels_val, device=trainer.device),
                            batch_size=64,
                        )
                    )
                    accuracies[4].append(val_acc)
                    # alignments[4].append(_alignment(labels_train, labels_val, num_classes=num_classes))
                    # balancednesses[0].append(_balancedness(labels[ideal_idxs]))
                    
                    # balancednesses[1].append(_balancedness(labels[random_idxs]))
                    
                    # balancednesses[2].append(_balancedness(labels[zcore_idxs]))
                    
                    # balancednesses[3].append(_balancedness(labels[zero_shot_idxs]))

                    # balancednesses[4].append(_balancedness(labels[rebalance_idxs]))


                    # Zero-shot + BC
                    print()
                    print("Zero-shot + BC")
                    # t3 = time.time()
                    boundary_scores, centrality_scores, cluster_labels = boundaryness_centrality_scores(
                        embeddings,
                        labels=zero_shot_labels,
                    )
                    boundary_central_idxs = select_balanced_boundary_central(
                        cluster_labels, boundary_scores, centrality_scores, subset_size, embeddings, val_labels=labels_val, frac_boundary=frac_boundary
                    )
                    # t4 = time.time()
                    # runtimes[1].append(t4 - t3)
                    # print("counts in zero-shot + BC set:", np.bincount(labels[boundary_central_idxs]))
                    embeddings_train = embeddings[boundary_central_idxs]
                    labels_train = labels[boundary_central_idxs]

                    _, trainer, _ = (
                        train_mlp_from_embeddings_and_labels(
                            embeddings_train, labels_train, num_classes=num_classes
                        )
                    )
                    _, val_acc = trainer.validate(
                        DataLoader(
                            FSLDataset(embeddings_val, labels_val, device=trainer.device),
                            batch_size=64,
                        )
                    )
                    accuracies[5].append(val_acc)

                    # Zero-shot + "Keep Easy"
                    keep_easy_idxs = coreset_with_target_alignment(
                        labels_base=zero_shot_labels,
                        labels_val=labels_val,
                        subset_size=subset_size,
                        target_alignment=1.0,
                        scores=p_max,
                        #fraction_pick_low_scores=0.3,
                        fraction_pick_low_scores=0.0,
                        seed=None,
                    )

                    # t4 = time.time()
                    # runtimes[1].append(t4 - t3)
                    # print("counts in zero-shot + BC set:", np.bincount(labels[boundary_central_idxs]))
                    embeddings_train = embeddings[keep_easy_idxs]
                    labels_train = labels[keep_easy_idxs]

                    _, trainer, _ = (
                        train_mlp_from_embeddings_and_labels(
                            embeddings_train, labels_train, num_classes=num_classes
                        )
                    )
                    _, val_acc = trainer.validate(
                        DataLoader(
                            FSLDataset(embeddings_val, labels_val, device=trainer.device),
                            batch_size=64,
                        )
                    )
                    accuracies[6].append(val_acc)

                # balancednesses[5].append(_balancedness(labels[boundary_central_idxs]))
                # alignments[5].append(_alignment(labels[boundary_central_idxs], labels_val, num_classes=num_classes))



            #print("\n=== Balancedness Results ===")
            # for method, scores in zip(["Ideal", "Random", "ZCore", "Zero-Shot", "Rebalance"], balancednesses):
            #     avg_score = np.mean(scores)
            #     print(f"{method}: Average Balancedness = {avg_score:.4f}")

            #print(f"Balanced Boundary-Central: Average Balancedness = {np.mean(balancednesses[5]):.4f}")

            print("\n=== Accuracy Results ===")
            for method, scores in zip(["Ideal", "Random", "ZCore", "Zero-Shot Random", "Zero-Shot + ZCore", "Zero-Shot + BC", "Zero-Shot + Keep Easy (entropy)"], accuracies):
                avg_score = np.mean(scores)
                print(f"{method}: Average Accuracy = {avg_score:.4f}")

            # print("zcore:")
            # print(np.mean(accuracies[2]))
            # print("zero-shot + zcore:")
            # print(np.mean(accuracies[4]))
            # print("zero-shot + BC:")
            # print(np.mean(accuracies[5]))
            # print("zero-shot + BC alignment:")
            # print(np.mean(alignments))

            # print("\n=== Runtime Results ===")
            # print(f"ZCore selection time: {np.mean(runtimes[0]):.4f} seconds")
            # print(f"Zero-Shot + BC selection time: {np.mean(runtimes[1]):.4f} seconds")

            # print("\n=== Alignment Results ===")
            # for method, scores in zip(["Ideal", "Random", "ZCore", "Zero-Shot Random", "Zero-Shot + ZCore", "Zero-Shot + BC"], alignments):
            #     avg_score = np.mean(scores)
            #     print(f"{method}: Average Alignment = {avg_score:.4f}")

        



def zero_shot_runtime_analysis():
    import time 

    embeddings = np.load("./data/clip_embeddings_cifar100_train_sweepImbalanced.npy")

    dataset_full = foz.load_zoo_dataset("cifar100", split="train")
    classes = sorted({s.ground_truth.label for s in dataset_full})
    classes_clean = [c.replace("_", " ") for c in classes]
    
    model = foz.load_zoo_model("clip-vit-base32-torch", classes=classes_clean)

    runtimes = [[], []]
    for _ in range(5):
        t1 = time.time()
        text_features = clip_text_ensemble_features(
            model, classes_clean, TEMPLATES_20,
        )  # (C, dim)
        t2 = time.time()
        runtimes[0].append(t2 - t1)

        t3 = time.time()
        pred_labels_idxs, _ = clip_prompt_ensemble_predict(embeddings, text_features)
        t4 = time.time()
        runtimes[1].append(t4 - t3)
    print("\n=== Runtime Analysis ===")
    print(f"CLIP text feature computation time: {np.mean(runtimes[0]):.4f} seconds")
    print(f"CLIP prompt ensemble prediction time: {np.mean(runtimes[1]):.4f} seconds")



if __name__ == "__main__":

    t = time.time()
    with open(f"_temp_{t}.txt", "w") as f:
        contextlib.redirect_stdout(f)
        contextlib.redirect_stderr(f)  

        # _prepare_data_cifar100()    
        # _prepare_data_openimages()
        main()
        #zero_shot_runtime_analysis()






    