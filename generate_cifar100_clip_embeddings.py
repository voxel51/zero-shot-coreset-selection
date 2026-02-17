import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np


dataset = foz.load_zoo_dataset("cifar100", split="train")

model = foz.load_zoo_model("clip-vit-base32-torch")
embeddings = dataset.compute_embeddings(model, batch_size=16, num_workers=8, progress=True)
embeddings = np.array(embeddings)
np.save(f"./clip_embeddings_cifar100_train_full.npy", embeddings)