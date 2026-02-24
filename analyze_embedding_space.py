import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. BASIC STATS
# ---------------------------------------------------------


def basic_stats(X, name="embedding"):
    print(f"\n=== {name}: Basic Stats ===")
    norms = np.linalg.norm(X, axis=1)
    print("Mean norm:", norms.mean())
    print("Std norm:", norms.std())
    print("Min/Max norm:", norms.min(), norms.max())

    # Pairwise cosine similarity
    cos = 1 - pairwise_distances(X, metric="cosine")
    print("Cosine similarity: mean/std:", cos.mean(), cos.std())

    # Pairwise L2 distance
    l2 = pairwise_distances(X, metric="euclidean")
    print("L2 distance: mean/std:", l2.mean(), l2.std())

    # Pairwise L1 distance
    l1 = pairwise_distances(X, metric="l1")
    print("L1 distance: mean/std:", l1.mean(), l1.std())

    # Pairwise fractional metric with p=0.5
    frac = pairwise_distances(X, metric="minkowski", p=0.5)
    print("Fractional (p=0.5) distance: mean/std:", frac.mean(), frac.std())


# ---------------------------------------------------------
# 2. PCA + INTRINSIC DIMENSION
# ---------------------------------------------------------


def pca_diagnostics(X, name="embedding"):
    print(f"\n=== {name}: PCA Diagnostics ===")
    pca = PCA()
    pca.fit(X)

    eigvals = pca.explained_variance_

    # Participation ratio = intrinsic dimension estimate
    pr = (eigvals.sum() ** 2) / (np.sum(eigvals**2))
    print("Intrinsic dimension (participation ratio):", pr)

    # Variance explained by top components
    top_n = np.round(pr).astype(int) + 1
    print(f"Top {top_n} eigenvalues:", eigvals[:top_n])
    print(
        f"Variance explained by top {top_n} components:",
        pca.explained_variance_ratio_[:top_n].sum(),
    )

    return pca, eigvals


# ---------------------------------------------------------
# 3. NEIGHBORHOOD STRUCTURE
# ---------------------------------------------------------


def neighbor_structure(X, metric="cosine", name="embedding"):
    print(f"\n=== {name}: Nearest Neighbor Structure ({metric}) ===")
    dists = pairwise_distances(X, metric=metric)

    dists_nn = dists.copy()
    np.fill_diagonal(dists_nn, np.inf)
    nn = np.argmin(dists_nn, axis=1)
    nn_dist = np.min(dists_nn, axis=1)

    dists_far = dists.copy()
    np.fill_diagonal(dists_far, -np.inf)
    far = np.argmax(dists_far, axis=1)
    far_dist = np.max(dists_far, axis=1)

    print(
        "Nearest-neighbor distance min/mean/max/std:",
        nn_dist.min(),
        nn_dist.mean(),
        nn_dist.max(),
        nn_dist.std(),
    )
    print(
        "Farthest-neighbor distance min/mean/max/std:",
        far_dist.min(),
        far_dist.mean(),
        far_dist.max(),
        far_dist.std(),
    )

    return nn, far


# ---------------------------------------------------------
# 4. PCA PROJECTION + WHITENING
# ---------------------------------------------------------


def pca_project(X, pca, k):
    # Project to top-k components
    Z = pca.transform(X)[:, :k]
    return Z


def whiten(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# ---------------------------------------------------------
# 5. FULL PIPELINE WRAPPER
# ---------------------------------------------------------


def full_diagnostic(X, name="embedding"):
    basic_stats(X, name)
    pca, eigvals = pca_diagnostics(X, name)

    # Try cosine and L2 neighbors
    neighbor_structure(X, metric="cosine", name=name)
    neighbor_structure(X, metric="euclidean", name=name)

    # # Try PCA projection
    # for k in [16, 32, 64, 128]:
    #     Z = pca_project(X, pca, k)
    #     print(f"\n--- PCA-{k} diagnostics for {name} ---")
    #     basic_stats(Z, f"{name}-PCA{k}")
    #     neighbor_structure(Z, metric="cosine", name=f"{name}-PCA{k}")

    # # Try whitening
    # Xw = whiten(X)
    # print(f"\n--- Whitening diagnostics for {name} ---")
    # basic_stats(Xw, f"{name}-whitened")
    # neighbor_structure(Xw, metric="cosine", name=f"{name}-whitened")

    print(f"\n================================================================")


if __name__ == "__main__":
    # Example usage
    # X = np.random.randn(1000, 512)  # Simulated embedding matrix
    # full_diagnostic(X, name="Simulated Embeddings")

    num_embeddings = 10_000

    clip_embeddings = np.load("./data/clip_embeddings_cifar100_train_full.npy")[
        :num_embeddings
    ]
    clip_embeddings_normed = clip_embeddings / np.linalg.norm(
        clip_embeddings, axis=1, keepdims=True
    )
    full_diagnostic(clip_embeddings, name="CLIP CIFAR-100 Train")
    full_diagnostic(clip_embeddings_normed, name="CLIP CIFAR-100 Train (Normed)")
    # dino_embeddings = np.load("./data/dino_embeddings_cifar100_train_full.npy")[:num_embeddings]
    # full_diagnostic(dino_embeddings, name="DINO CIFAR-100 Train")
    # resnet_embeddings = np.load("./data/resnet_embeddings_cifar100_train_full.npy")[:num_embeddings]
    # full_diagnostic(resnet_embeddings, name="ResNet CIFAR-100 Train")
