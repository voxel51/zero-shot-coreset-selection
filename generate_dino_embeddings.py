import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np


#model = foz.load_zoo_model("dinov2-vits14-reg-torch")
model = foz.load_zoo_model("resnet18-imagenet-torch")

dataset = foz.load_zoo_dataset("cifar100", split="train")

embeddings = dataset.compute_embeddings(model, batch_size=64, num_workers=4)

np.save("./data/resnet_embeddings_cifar100_train_full.npy", embeddings)