import numpy as np
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# 1. PCA on UNNORMALIZED CLIP embeddings
# ---------------------------------------------------------


def fit_pca(unnorm_embeddings, k=None):
    """
    unnorm_embeddings: (N, D) raw CLIP embeddings BEFORE L2-normalization
    k: number of PCA components to keep
    """

    pca = PCA(n_components=k)
    Z = pca.fit_transform(unnorm_embeddings)
    return pca, Z


# ---------------------------------------------------------
# 2. Sampling in PCA space
# ---------------------------------------------------------


def sample_pca_gaussian(pca, num_samples=1):
    """
    Gaussian sampling in PCA space:
    z ~ N(0, diag(eigenvalues))
    """
    k = pca.n_components_
    std = np.sqrt(pca.explained_variance_)
    z = np.random.randn(num_samples, k) * std
    return z


def sample_pca_triangular(pca, Z, num_samples=1):
    """
    Triangular sampling in PCA space:
    For each PC dimension, sample from triangular(min, mean, max)
    """
    mins = Z.min(axis=0)
    maxs = Z.max(axis=0)
    means = Z.mean(axis=0)

    samples = []
    for _ in range(num_samples):
        s = np.array(
            [
                np.random.triangular(mins[i], means[i], maxs[i])
                for i in range(pca.n_components_)
            ]
        )
        samples.append(s)
    return np.vstack(samples)


# ---------------------------------------------------------
# 3. Back-project to original 512-D space
# ---------------------------------------------------------


def back_project(pca, z_samples):
    """
    z_samples: (num_samples, k)
    Returns: (num_samples, D) unnormalized embeddings
    """
    return pca.inverse_transform(z_samples)


# ---------------------------------------------------------
# 4. L2-normalize to get valid CLIP embeddings
# ---------------------------------------------------------


def l2_normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ---------------------------------------------------------
# 5. Full pipeline wrapper
# ---------------------------------------------------------


def generate_synthetic_clip_embeddings(
    unnorm_embeddings, k=50, num_samples=10, method="gaussian"
):
    """
    unnorm_embeddings: (N, 512) raw CLIP embeddings BEFORE normalization
    method: "gaussian" or "triangular"
    """
    # Step 1: PCA
    pca, Z = fit_pca(unnorm_embeddings, k=k)

    # Step 2: Sampling
    if method == "gaussian":
        z_samples = sample_pca_gaussian(pca, num_samples)
    elif method == "triangular":
        z_samples = sample_pca_triangular(pca, Z, num_samples)
    else:
        raise ValueError("method must be 'gaussian' or 'triangular'")

    # Step 3: Back-project
    X_raw = back_project(pca, z_samples)

    # Step 4: Normalize to CLIP geometry
    X_norm = l2_normalize(X_raw)

    return X_norm
