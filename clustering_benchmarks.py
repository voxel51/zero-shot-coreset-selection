from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import scipy.sparse as sp

import pandas as pd


# ...existing code...

def _temp_imbalanced_train_set_sweep(
    labels: Union[np.ndarray, list[int]],
    target_total: int = 9500,
    sweep_strength: float = 1.0, 
    order: str = "label",          
    min_per_class: int = 50,
    include_max_available: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Smooth exponential-like sweep from ~min_per_class up to val=caps.max().

    Strategy:
      - Ensure presence via `forced = min(min_per_class, caps)`
      - Let val = caps.max(); ensure at least one class with caps==val hits val (if requested & feasible)
      - Put an exponential (geometric) ramp over the last T classes in the chosen ordering:
            ramp[i] = min * (val/min)^(i/(T-1)), i=0..T-1
        Others stay near min.
      - Choose T (2..C) to make total count closest to target_total (approximate match).
      - Finally clamp to caps and forced.
    """
    labels = np.asarray(labels)
    n = int(labels.size)
    rng = np.random.default_rng(seed)

    classes, y_inv = np.unique(labels, return_inverse=True)
    C = int(classes.size)
    if C == 0 or n == 0:
        return np.array([], dtype=int)

    by_class = [np.flatnonzero(y_inv == c) for c in range(C)]
    for c in range(C):
        rng.shuffle(by_class[c])

    caps = np.array([len(ix) for ix in by_class], dtype=int)
    val = int(caps.max())

    # total budget cannot exceed availability
    k = int(target_total)
    k = max(1, min(k, int(caps.sum())))

    min_pc = max(1, int(min_per_class))
    forced = np.minimum(min_pc, caps).astype(int)

    # Choose ordering for the sweep
    if order == "label":
        rank = np.arange(C)
    elif order == "frequency":
        rank = np.argsort(caps, kind="stable")
    else:
        raise ValueError("order must be 'label' or 'frequency'.")

    # If requested, ensure a global-max class is at the high end of the sweep ordering
    c_max = None
    if include_max_available and val > 0:
        max_candidates = np.flatnonzero(caps == val)
        if max_candidates.size > 0:
            pos = np.empty(C, dtype=int)
            pos[rank] = np.arange(C)
            c_max = int(max_candidates[np.argmax(pos[max_candidates])])
            if int(rank[-1]) != c_max:
                i = int(np.where(rank == c_max)[0][0])
                rank[i], rank[-1] = rank[-1], rank[i]

    # Degenerate: can't sweep if val <= 0
    if val <= 0:
        counts = forced.copy()
    else:
        # Build geometric ramp values for a given T (T>=2)
        def ramp_values(T: int) -> np.ndarray:
            if T <= 1:
                return np.array([float(min_pc)], dtype=float)
            if val <= min_pc:
                return np.full(T, float(min_pc), dtype=float)
            r = float(val) / float(min_pc)
            i = np.arange(T, dtype=float)
            return float(min_pc) * np.power(r, i / float(T - 1))

        # Choose T that best matches k (approx). Always allow at least 2 if include_max is requested.
        T_min = 2 if (include_max_available and c_max is not None and C >= 2) else 1
        best_T = T_min
        best_err = float("inf")

        # Evaluate candidate T values; O(C^2) worst-case is fine for typical class counts
        for T in range(T_min, C + 1):
            rv = ramp_values(T)
            total = float(min_pc) * float(C - T) + float(rv.sum())
            err = abs(total - float(k))
            if err < best_err:
                best_err = err
                best_T = T

        T = best_T
        rv = ramp_values(T)

        # Assign: first C-T in rank -> ~min_pc; last T -> ramp
        counts_f = np.full(C, float(min_pc), dtype=float)
        tail = rank[C - T :]
        counts_f[tail] = rv

        # Force exact max on at least one global-max class (best effort)
        if include_max_available and c_max is not None:
            counts_f[c_max] = float(val)

        # Clamp by caps and forced presence (best effort)
        counts = np.rint(counts_f).astype(int)
        counts = np.minimum(counts, caps)
        counts = np.maximum(counts, forced)

        # If forcing max overshoots per-class cap (shouldn't), re-clamp
        counts = np.minimum(counts, caps)

    # Materialize indices from per-class counts
    chosen = []
    for c in range(C):
        take = int(counts[c])
        if take > 0:
            chosen.append(by_class[c][:take])

    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    rng.shuffle(chosen)
    return chosen

# ...existing code...


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


# -----------------------------
# kNN (FAISS) -> sparse graphs
# -----------------------------
def faiss_knn_inner_product(
    X: np.ndarray,
    *,
    k: int,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate kNN with FAISS HNSW using inner product (best with L2-normalized vectors).
    Returns (nbrs, sims) with shapes (n, k). Self-neighbor is removed.
    """
    import faiss  # type: ignore

    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape

    index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(ef_construction)
    index.hnsw.efSearch = int(ef_search)

    index.add(X)

    sims, nbrs = index.search(X, k + 1)  # includes self
    nbrs = nbrs[:, 1:]
    sims = sims[:, 1:]
    assert nbrs.shape == (n, k)
    assert sims.shape == (n, k)
    return nbrs.astype(np.int32), sims.astype(np.float32)


def csr_from_knn(
    nbrs: np.ndarray,
    weights: np.ndarray,
    *,
    n: int,
    symmetrize: Literal["union", "mutual"] = "union",
    drop_self_loops: bool = True,
) -> sp.csr_matrix:
    """
    Build weighted sparse adjacency from neighbor lists.
    nbrs: (n, k) int
    weights: (n, k) float
    """
    nbrs = np.asarray(nbrs)
    weights = np.asarray(weights)
    k = nbrs.shape[1]

    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = nbrs.reshape(-1).astype(np.int32)
    data = weights.reshape(-1).astype(np.float32)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    if drop_self_loops:
        A.setdiag(0)
    A.eliminate_zeros()

    if symmetrize == "union":
        A = A.maximum(A.T)
    elif symmetrize == "mutual":
        A = A.minimum(A.T)
    else:
        raise ValueError("symmetrize must be 'union' or 'mutual'")

    if drop_self_loops:
        A.setdiag(0)
    A.eliminate_zeros()
    return A


# -----------------------------
# SNN weights (Jaccard overlap)
# -----------------------------
def snn_jaccard_from_knn(
    nbrs: np.ndarray,
    *,
    symmetrize: Literal["union", "mutual"] = "union",
) -> sp.csr_matrix:
    """
    SNN graph with edge weight = Jaccard(N(i), N(j)).
    Computes weights only for edges present in the (symmetrized) kNN graph.

    Note: This is OK for ~10k–50k. For ~1M, pure-Python overlap loops will be too slow.
    """
    nbrs = np.asarray(nbrs, dtype=np.int32)
    n, k = nbrs.shape
    nbrs_sorted = np.sort(nbrs, axis=1)

    # Build unweighted kNN adjacency
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = nbrs.reshape(-1)
    A = sp.csr_matrix((np.ones(rows.shape[0], dtype=np.uint8), (rows, cols)), shape=(n, n))
    A.setdiag(0)
    A.eliminate_zeros()

    if symmetrize == "union":
        M = A.maximum(A.T)
    elif symmetrize == "mutual":
        M = A.minimum(A.T)
    else:
        raise ValueError("symmetrize must be 'union' or 'mutual'")

    coo = M.tocoo()
    src = coo.row.astype(np.int32)
    dst = coo.col.astype(np.int32)

    def inter_size(a: np.ndarray, b: np.ndarray) -> int:
        i = j = c = 0
        # both sorted length k
        while i < a.size and j < b.size:
            av = a[i]
            bv = b[j]
            if av == bv:
                c += 1
                i += 1
                j += 1
            elif av < bv:
                i += 1
            else:
                j += 1
        return c

    inter = np.empty(src.shape[0], dtype=np.int16)
    for t in range(src.shape[0]):
        inter[t] = inter_size(nbrs_sorted[src[t]], nbrs_sorted[dst[t]])

    union = (2 * k - inter).astype(np.float32)
    w = (inter.astype(np.float32) / np.maximum(union, 1e-6)).astype(np.float32)

    W = sp.csr_matrix((w, (src, dst)), shape=(n, n))
    W = W.maximum(W.T)
    W.setdiag(0)
    W.eliminate_zeros()
    return W


# -----------------------------
# UMAP graph / embedding
# -----------------------------
def umap_graph(
    X: np.ndarray,
    *,
    n_neighbors: int,
    metric: str = "cosine",
    random_state: int = 0,
) -> sp.csr_matrix:
    import umap.umap_ as umap  # type: ignore

    reducer = umap.UMAP(
        n_components=2,  # coords are irrelevant here; we just want reducer.graph_
        n_neighbors=int(n_neighbors),
        metric=metric,
        random_state=int(random_state),
    )
    _ = reducer.fit_transform(X)
    G = reducer.graph_.tocsr()
    G = G.maximum(G.T)
    G.setdiag(0)
    G.eliminate_zeros()
    return G


def umap_embed(
    X: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int,
    metric: str = "cosine",
    random_state: int = 0,
) -> np.ndarray:
    import umap.umap_ as umap  # type: ignore

    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=int(n_neighbors),
        metric=metric,
        random_state=int(random_state),
    )
    return reducer.fit_transform(X)


# -----------------------------
# Leiden / DBSCAN
# -----------------------------
def leiden_labels(
    A: sp.csr_matrix,
    *,
    resolution: float,
    seed: int = 0,
) -> np.ndarray:
    A = A.tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()

    # Prefer scikit-network (works directly with CSR)
    try:
        from sknetwork.clustering import Leiden  # type: ignore

        algo = Leiden(resolution=float(resolution), random_state=int(seed))
        return algo.fit_predict(A).astype(np.int32)
    except Exception:
        pass

    # Fallback: igraph/leidenalg
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore

        coo = A.tocoo()
        edges = list(zip(coo.row.tolist(), coo.col.tolist()))
        g = ig.Graph(n=A.shape[0], edges=edges, directed=False)
        g.es["weight"] = coo.data.tolist()

        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=int(seed),
        )
        return np.array(part.membership, dtype=np.int32)
    except Exception as e:
        raise RuntimeError(
            "No Leiden backend available. Install one of:\n"
            "  pip install scikit-network\n"
            "or\n"
            "  pip install python-igraph leidenalg\n"
        ) from e


def dbscan_on_features(
    X: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    metric: str = "euclidean",
) -> np.ndarray:
    from sklearn.cluster import DBSCAN  # type: ignore

    return DBSCAN(eps=float(eps), min_samples=int(min_samples), metric=metric).fit_predict(X).astype(np.int32)


def dbscan_precomputed_sparse(
    D: sp.csr_matrix,
    *,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """
    Approximate DBSCAN using a sparse precomputed distance matrix:
    - store only distances for (say) kNN edges; missing entries are treated as "far".
    """
    from sklearn.cluster import DBSCAN  # type: ignore

    D = D.tocsr()
    D.setdiag(0)
    D.eliminate_zeros()
    return DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="precomputed").fit_predict(D).astype(np.int32)


# -----------------------------
# Scoring
# -----------------------------
def clustering_scores(y_true: np.ndarray, y_pred: np.ndarray, *, drop_noise: bool = False) -> Dict[str, float]:
    from sklearn.metrics import (  # type: ignore
        adjusted_rand_score,
        normalized_mutual_info_score,
        v_measure_score,
        homogeneity_score,
        completeness_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if drop_noise:
        mask = y_pred != -1
        if mask.sum() == 0:
            return {"ari": np.nan, "nmi": np.nan, "v": np.nan, "hom": np.nan, "comp": np.nan}
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "v": float(v_measure_score(y_true, y_pred)),
        "hom": float(homogeneity_score(y_true, y_pred)),
        "comp": float(completeness_score(y_true, y_pred)),
    }


def clustering_summary(y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.asarray(y_pred)
    n = y_pred.size
    noise = int((y_pred == -1).sum())
    clusters = [c for c in np.unique(y_pred) if c != -1]
    return {
        "n_clusters": float(len(clusters)),
        "noise_frac": float(noise / max(n, 1)),
    }


@dataclass(frozen=True)
class RunResult:
    method: str
    params: Dict[str, Any]
    summary: Dict[str, float]
    scores: Dict[str, float]


def to_records(results: List[RunResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results:
        row: Dict[str, Any] = {"method": r.method, **r.params, **r.summary, **r.scores}
        rows.append(row)
    return rows



def main():
    

    # embeddings = np.load("./data/clip_embeddings_cifar100_train_full.npy")
    # labels = np.load("./data/labels_cifar100_train_full.npy")

    # idxs = _temp_imbalanced_train_set_sweep(labels, target_total=9500, sweep_strength=5, min_per_class=10, order="frequency", seed=42)
    # print(f"Selected {idxs.size} samples with class distribution:")
    # print(np.sort(np.bincount(labels[idxs])))

    # np.save("./data/clip_embeddings_cifar100_train_sweepImbalanced.npy", embeddings[idxs])
    # np.save("./data/labels_cifar100_train_sweepImbalanced.npy", labels[idxs])
    # np.save("./data/idxs_cifar100_train_sweepImbalanced.npy", idxs)

    # import sys 
    # sys.exit()

    embeddings = np.load("./data/clip_embeddings_cifar100_train_sweepImbalanced.npy")
    labels = np.load("./data/labels_cifar100_train_sweepImbalanced.npy")


    X = l2_normalize(embeddings)
    y = labels

    results = []

    # ---------- UMAP + Leiden (cluster on UMAP neighbor graph) ----------
    for nn in [10, 20, 40]:
        G = umap_graph(X, n_neighbors=nn, metric="cosine", random_state=0)
        for res in [0.005, 0.01, 0.02]:
            pred = leiden_labels(G, resolution=res, seed=0)
            results.append(
                RunResult(
                    method="umap_graph+leiden",
                    params={"umap_nn": nn, "leiden_res": res},
                    summary=clustering_summary(pred),
                    scores=clustering_scores(y, pred),
                )
            )

    # ---------- UMAP + DBSCAN (DBSCAN on low-D UMAP coordinates) ----------
    # Tip: use 10D for clustering, 2D for plotting.
    for nn in [10, 20, 40]:
        Z = umap_embed(X, n_components=10, n_neighbors=nn, metric="cosine", random_state=0)
        for eps in [0.3, 0.5, 0.8]:
            for ms in [5, 10, 20]:
                pred = dbscan_on_features(Z, eps=eps, min_samples=ms, metric="euclidean")
                results.append(
                    RunResult(
                        method="umap10+dbscan",
                        params={"umap_nn": nn, "db_eps": eps, "db_min_samples": ms},
                        summary=clustering_summary(pred),
                        scores=clustering_scores(y, pred),
                    )
                )

    # ---------- kNN + Leiden ----------
    for k in [15, 30, 50]:
        nbrs, sims = faiss_knn_inner_product(X, k=k, ef_search=64)
        A = csr_from_knn(nbrs, sims, n=X.shape[0], symmetrize="union")  # weights = similarity
        for res in [0.2, 0.5, 1.0]:
            pred = leiden_labels(A, resolution=res, seed=0)
            results.append(
                RunResult(
                    method="knn_ip+leiden",
                    params={"k": k, "leiden_res": res},
                    summary=clustering_summary(pred),
                    scores=clustering_scores(y, pred),
                )
            )

    # ---------- kNN + DBSCAN (approx via sparse precomputed distances on kNN edges) ----------
    # Use distance = 1 - inner_product_similarity (valid if vectors are L2-normalized).
    for k in [15, 30, 50]:
        nbrs, sims = faiss_knn_inner_product(X, k=k, ef_search=64)
        D = csr_from_knn(nbrs, 1.0 - sims, n=X.shape[0], symmetrize="union")  # weights = distance
        for eps in [0.2, 0.3, 0.4]:
            for ms in [5, 10, 20]:
                pred = dbscan_precomputed_sparse(D, eps=eps, min_samples=ms)
                results.append(
                    RunResult(
                        method="knn_ip_sparse+dbscan",
                        params={"k": k, "db_eps": eps, "db_min_samples": ms},
                        summary=clustering_summary(pred),
                        scores=clustering_scores(y, pred),
                    )
                )

    # ---------- SNN + Leiden ----------
    for k in [15, 30]:
        nbrs, _sims = faiss_knn_inner_product(X, k=k, ef_search=64)
        W = snn_jaccard_from_knn(nbrs, symmetrize="union")  # weights in [0,1]
        for res in [0.2, 0.5, 1.0]:
            pred = leiden_labels(W, resolution=res, seed=0)
            results.append(
                RunResult(
                    method="snn_jaccard+leiden",
                    params={"k": k, "leiden_res": res},
                    summary=clustering_summary(pred),
                    scores=clustering_scores(y, pred),
                )
            )

    # ---------- SNN + DBSCAN (approx via sparse precomputed distances on SNN edges) ----------
    for k in [15, 30]:
        nbrs, _sims = faiss_knn_inner_product(X, k=k, ef_search=64)
        W = snn_jaccard_from_knn(nbrs, symmetrize="union")
        D = W.copy()
        D.data = (1.0 - D.data).astype(np.float32)  # distance = 1 - Jaccard
        for eps in [0.6, 0.7, 0.8]:
            for ms in [5, 10, 20]:
                pred = dbscan_precomputed_sparse(D, eps=eps, min_samples=ms)
                results.append(
                    RunResult(
                        method="snn_jaccard_sparse+dbscan",
                        params={"k": k, "db_eps": eps, "db_min_samples": ms},
                        summary=clustering_summary(pred),
                        scores=clustering_scores(y, pred),
                    )
                )

    df = pd.DataFrame(to_records(results)).sort_values(["ari", "nmi"], ascending=False)
    print(df.head(30))



if __name__ == "__main__":    
    main()