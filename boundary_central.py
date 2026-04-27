import numpy as np
from typing import Union
import igraph as ig
import leidenalg as la
import umap.umap_ as umap


"""
For unbalanced across 0.1, 0.3, 0.5, 0.7:
UMAP neighbors: 20
Resolution: 0.02
Fraction boundary: 0.7
"""


def umap_clustering(embeddings, resolution_parameter=0.04, n_neighbors_umap=15):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors_umap,
        metric="cosine",
    )
    _ = reducer.fit_transform(embeddings)

    graph_sparse = reducer.graph_.tocsr()

    # Treat neighborhood as undirected to avoid "outgoing only" bias
    graph_sparse = graph_sparse.maximum(graph_sparse.T).tocsr()

    sources, targets = graph_sparse.nonzero()
    weights = graph_sparse.data
    nodes = list(range(graph_sparse.shape[0]))
    G = ig.Graph(n=len(nodes), edges=list(zip(sources, targets)), edge_attrs={"weight": weights})

    partition = la.CPMVertexPartition(G, resolution_parameter=resolution_parameter)
    optimiser = la.Optimiser()
    _ = optimiser.optimise_partition(partition)

    cluster_labels = np.array(partition.membership)

    return cluster_labels, graph_sparse


def zero_shot_clustering(
    embeddings,
    labels: np.ndarray | None = None,
    labels_path: str = "./data/clip_pred_labels_cifar100_train_sweepImbalanced.npy",
    n_neighbors_umap: int = 15,
):
    """Build a UMAP neighbor graph over `embeddings` and return cluster labels.

    This is a lightweight wrapper used by `boundaryness_centrality_scores()`.
    The intent is that `labels` are precomputed (e.g. CLIP zero-shot predicted
    labels) and must align 1:1 with rows in `embeddings`.
    """

    if labels is None:
        labels = np.load(labels_path)

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("zero-shot labels must be a 1D array")
    if len(labels) != len(embeddings):
        raise ValueError(
            "zero-shot labels length does not match embeddings: "
            f"len(labels)={len(labels)} vs len(embeddings)={len(embeddings)}. "
            "Pass the correct `labels=` for the current embeddings (or set `labels_path=`)."
        )

    reducer = umap.UMAP(
        n_components=2,          # irrelevant for the graph; OK to keep
        n_neighbors=n_neighbors_umap,
        metric="cosine",
    )
    reducer.fit(embeddings)
    graph_sparse = reducer.graph_.tocsr()
    graph_sparse = graph_sparse.maximum(graph_sparse.T).tocsr()  # undirected

    return labels, graph_sparse


def boundaryness_centrality_scores(embeddings, cluster_func=zero_shot_clustering, **cluster_kwargs):

    cluster_labels, graph_sparse = cluster_func(embeddings, **cluster_kwargs)
    # print(f"Detected {len(partition)} clusters in the UMAP graph.")

    n = len(embeddings)

    boundary_scores = np.zeros(n, dtype=float)
    centrality_scores = np.zeros(n, dtype=float)

    for i in range(n):

        row = graph_sparse.getrow(i)
        neighbors = row.indices
        neighbor_weights = row.data

        if neighbors.size == 0:
            continue

        neighbor_clusters = cluster_labels[neighbors]

        # --- weighted fraction of connectivity that stays inside i's own cluster ---
        same = (neighbor_clusters == cluster_labels[i])
        w_total = float(neighbor_weights.sum())
        w_same = float(neighbor_weights[same].sum()) if w_total > 0 else 0.0
        frac_same = (w_same / w_total) if w_total > 0 else 0.0

        # Boundaryness: high if most weight goes to other clusters (even if it's just one other cluster)
        boundary_scores[i] = 1.0 - frac_same

        # Centrality: high if neighborhood is "pure" in i's own cluster
        # (optionally scaled by mean weight to prefer strongly-connected points)
        centrality_scores[i] = frac_same * float(neighbor_weights.mean())

    return boundary_scores, centrality_scores, cluster_labels

def select_balanced_boundary_central(
    cluster_labels: np.ndarray,
    boundary_scores: np.ndarray,
    centrality_scores: np.ndarray,
    subset_size: float,
    embeddings: np.ndarray,
    val_labels: np.ndarray = None,
    bc_over_zcore: float = 1.0,
    frac_boundary: float = 0.3,
    ignore_scores: bool = False,
) -> np.ndarray:

    assert 0.0 <= frac_boundary <= 1.0

    n = len(embeddings)
    if len(cluster_labels) != n or len(boundary_scores) != n or len(centrality_scores) != n:
        raise ValueError(
            "Input length mismatch in select_balanced_boundary_central(): "
            f"len(embeddings)={n}, len(cluster_labels)={len(cluster_labels)}, "
            f"len(boundary_scores)={len(boundary_scores)}, len(centrality_scores)={len(centrality_scores)}"
        )

    clusters = np.unique(cluster_labels)
    n_clusters = len(clusters)
    n_to_select = int(subset_size * n)

    idxs_by_cluster = {c: np.flatnonzero(cluster_labels == c) for c in clusters}
    cluster_sizes_baseset = {c: len(idxs_by_cluster[c]) for c in clusters}

    # --- maximally balanced quotas (subject to cluster sizes) ---
    cluster_sizes_coresets = n_to_select // n_clusters
    if val_labels is None:
        quota_coreset = {c: min(cluster_sizes_coresets, cluster_sizes_baseset[c]) for c in clusters}
    else:
       
        # Target distribution q over the *same label space* as `clusters`
        # (labels in val_labels that aren't in `clusters` get ignored)
        q_counts = np.array([np.sum(val_labels == c) for c in clusters], dtype=float)
        q = q_counts / q_counts.sum()

        # Caps = how many items are available per cluster in the base pool
        caps = np.array([cluster_sizes_baseset[c] for c in clusters], dtype=int)

        rng = np.random.default_rng()
        counts = _counts_from_probs_with_caps(q, n_to_select, caps, rng)

        # Per-cluster quotas that sum (as close as possible) to n_to_select
        quota_coreset = {c: int(counts[i]) for i, c in enumerate(clusters)}
        
    remaining = n_to_select - sum(quota_coreset.values())


    # distribute the remainder to clusters that still have capacity
    while remaining > 0:
        eligible = [c for c in clusters if quota_coreset[c] < cluster_sizes_baseset[c]]
        if not eligible:
            break
        # give +1 to clusters with most remaining capacity
        eligible.sort(key=lambda c: (cluster_sizes_baseset[c] - quota_coreset[c]), reverse=True)
        for c in eligible:
            if remaining == 0:
                break
            if quota_coreset[c] < cluster_sizes_baseset[c]:
                quota_coreset[c] += 1
                remaining -= 1

    selected = []
    
    if ignore_scores:
        for c in clusters:
            idxs = idxs_by_cluster[c]
            q = quota_coreset[c]
            if q > 0:
                pick = np.random.choice(idxs, size=q, replace=False)
                selected.extend(pick.tolist())
        return np.array(selected, dtype=int)

    else:
        #zcore_scores = _get_zcore_like_scores(embeddings)
        zcore_scores = np.zeros(len(embeddings))
        # --- per-cluster pick: frac_boundary boundary + (1-frac_boundary) central ---
        for c in clusters:
            idxs = idxs_by_cluster[c]
            q = quota_coreset[c]
            if q == 0:
                continue

            # zcore_scores = _get_zcore_like_scores(embeddings[idxs])

            if bc_over_zcore > 0.0:
                zcore_boundary = boundary_scores[idxs]* bc_over_zcore + zcore_scores[idxs] * (1.0 - bc_over_zcore)
                # zcore_boundary = boundary_scores[idxs]* bc_over_zcore + zcore_scores * (1.0 - bc_over_zcore)

                q_boundary = int(round(frac_boundary * q))
                q_boundary = min(q_boundary, q)
                q_central = q - q_boundary

                # boundary: highest boundary_scores
                b_order = idxs[np.argsort(-zcore_boundary)]
                b_pick = b_order[:q_boundary]

                # central: highest centrality_scores among remaining
                remaining_idxs = np.setdiff1d(idxs, b_pick, assume_unique=False)
                zcore_central = centrality_scores[remaining_idxs]* bc_over_zcore + zcore_scores[remaining_idxs] * (1.0 - bc_over_zcore)

                c_order = remaining_idxs[np.argsort(-zcore_central)]
                c_pick = c_order[:q_central]

                selected.extend(np.concatenate([b_pick, c_pick]).tolist())
                
            else:
                order_local = np.argsort(-zcore_scores)
                pick_global = idxs[order_local[:q]]
                selected.extend(pick_global.tolist())


    # --- if short (many small clusters and a few large ones), 
    # fill globally by "best of both" score ---
    if len(selected) < n_to_select:
        print(f"Warning: only selected {len(selected)} samples out of {n_to_select} due to cluster size limits. Filling the rest globally.")

        all_idxs = np.arange(len(cluster_labels))
        leftover = np.setdiff1d(all_idxs, np.array(selected, dtype=int), assume_unique=False)
        combined = np.maximum(boundary_scores[leftover], centrality_scores[leftover])
        fill_order = leftover[np.argsort(-combined)]
        need = n_to_select - len(selected)
        selected.extend([int(i) for i in fill_order[:need]])

    return np.array(selected, dtype=int)


def _get_zcore_like_scores(embeddings, sample_dim=2, redund_nn=1000, redund_exp=4.0):
    
    rng = np.random.default_rng()

    n, dim = embeddings.shape
    n_samples = min(n*10, 1000_000)

    redund_nn = min(redund_nn, n - 2)

    mins = np.min(embeddings, axis=0)
    maxs = np.max(embeddings, axis=0)
    meds = np.median(embeddings, axis=0)

    scores = np.zeros(n)

    eps = 1e-12

    for i in range(n_samples):
        # Random embedding dimension.
        dims = rng.choice(dim, sample_dim, replace=False)

        # # Coverage score.
        # sample = rng.triangular(
        #     mins[dims], meds[dims], maxs[dims]
        # )

        left = mins[dims]
        right = maxs[dims]
        mode = np.clip(meds[dims], left, right)

        # triangular() cannot handle left == right; treat those dims as constant
        sample = np.empty_like(left, dtype=float)
        deg = np.abs(right - left) <= eps
        if np.any(deg):
            sample[deg] = left[deg]
        if np.any(~deg):
            sample[~deg] = rng.triangular(left[~deg], mode[~deg], right[~deg])

        embed_dist = np.sum(abs(embeddings[:, dims] - sample), axis=1)
        # diff = np.abs(embeddings[:, dims] - sample)
        # embed_dist = np.sum(diff ** 0.25, axis=1) ** 4

        idx = np.argmin(embed_dist)
        scores[idx] += 1

        # Redundancy score.
        cover_sample = embeddings[
            idx, dims
        ]  # sample closest to the current randomly drawn one
        nn_dist = np.sum(abs(embeddings[:, dims] - cover_sample), axis=1)

        k = 1 + redund_nn
        nn_k = np.argpartition(nn_dist, k)[:k]
        nn = nn_k[nn_k != idx]
        order = np.argsort(nn_dist[nn], kind="stable")
        nn = nn[order][:redund_nn]

        eps = 1e-12
        dist_penalty = 1 / ((nn_dist[nn]+eps) ** redund_exp)
        dist_penalty /= np.sum(dist_penalty)
        scores[nn] -= dist_penalty

    
    # Normalize scores to [0, 1]
    score_min = np.min(scores)
    scores = (scores - score_min) / (np.max(scores) - score_min)

    return scores


def score_based_selection(scores, subset_size):
    n = len(scores)
    k = int(subset_size * n)
    order = np.argsort(-scores)
    selected = order[:k]
    return selected

def _balancedness(labels: Union[np.ndarray, list[int]]) -> float:
    """
    Return the normalized entropy of the label distribution in the selected indices. 
    Higher is more balanced.
    """

    _, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()

    entropy = -np.sum(probs * np.log2(probs))

    num_classes = len(counts)

    return entropy / np.log2(num_classes) 



def _balancedness_from_counts(counts: np.ndarray) -> float:
    """Same metric as _balancedness(), but computed directly from integer class counts."""
    counts = np.asarray(counts, dtype=int)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    probs = counts / counts.sum()
    ent = -np.sum(probs * np.log2(probs))
    return float(ent / np.log2(len(counts)))


def _counts_from_probs_with_caps(
    probs: np.ndarray,
    k: int,
    caps: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Convert target probabilities into integer counts that sum to k, with per-class caps.
    Uses largest-remainder, then redistributes any shortfall to classes with remaining capacity.
    """
    probs = np.asarray(probs, dtype=float)
    caps = np.asarray(caps, dtype=int)

    # Only keep eligible classes (cap > 0)
    eligible = caps > 0
    probs = np.where(eligible, probs, 0.0)

    s = probs.sum()
    if s <= 0:
        return np.zeros_like(caps)

    probs = probs / s

    raw = probs * k
    base = np.floor(raw).astype(int)
    base = np.minimum(base, caps)

    # Distribute remainder by fractional parts, respecting caps
    remaining = k - int(base.sum())
    frac = raw - np.floor(raw)

    # random tie-break to avoid deterministic bias
    jitter = rng.random(frac.shape) * 1e-12
    order = np.argsort(-(frac + jitter))

    i = 0
    while remaining > 0 and i < len(order):
        c = order[i]
        if base[c] < caps[c]:
            base[c] += 1
            remaining -= 1
        i += 1

    # If still short (because of caps), fill from any class with capacity, preferring higher probs
    if remaining > 0:
        order2 = np.argsort(-(probs + rng.random(probs.shape) * 1e-12))
        j = 0
        while remaining > 0 and j < len(order2):
            c = order2[j]
            if base[c] < caps[c]:
                base[c] += 1
                remaining -= 1
            else:
                j += 1

    return base


def train_set_with_target_balancedness(
    labels: Union[np.ndarray, list[int]],
    subset_size: float = 0.1,
    target_balancedness: float = 0.9,
    tol: float = 0.01,
    max_iter: int = 100,
    gamma_max: float = 64.0,
    min_per_class: int = 1,
) -> np.ndarray:
    """
    Construct a subset whose _balancedness(labels_subset) is approximately target_balancedness.

    Notes:
    - This targets your current _balancedness metric (normalized entropy over *present* classes).
    """
    
    
    n = len(labels)
    
    rng = np.random.default_rng()

    classes, y_inv = np.unique(labels, return_inverse=True)
    C = len(classes)

    # Group indices by class and shuffle
    by_class = [np.flatnonzero(y_inv == c) for c in range(C)]
    for c in range(C):
        rng.shuffle(by_class[c])

    caps = np.array([len(ix) for ix in by_class], dtype=int)

    k = int(round(subset_size * n))
    k = max(1, min(k, int(caps.sum())))

    forced = np.zeros(C, dtype=int)
    if min_per_class > 0:
        forced = np.minimum(min_per_class, caps)
        k_free = k - forced.sum()
        caps_free = caps - forced
    else:
        k_free = k
        caps_free = caps

    # Base "availability" distribution q over eligible classes
    eps = 1e-12
    q = caps_free.astype(float)
    q = np.where(q > 0, q, 0.0)
    if q.sum() <= 0:
        # only forced part possible (or nothing)
        counts = forced
    else:
        q = q / q.sum()

        def counts_for_gamma(gamma: float) -> np.ndarray:
            # p ∝ q^gamma; gamma=0 => uniform over eligible classes
            p = np.where(q > 0, (q + eps) ** gamma, 0.0)
            if p.sum() > 0:
                p = p / p.sum()
            counts_free = _counts_from_probs_with_caps(p, k_free, caps_free, rng)
            return forced + counts_free

        def bal_for_gamma(gamma: float) -> float:
            return _balancedness_from_counts(counts_for_gamma(gamma))

        # Check achievable range within [0, gamma_max]
        b_lo = bal_for_gamma(0.0)         # typically highest
        b_hi = bal_for_gamma(gamma_max)   # typically lowest

        # If already within tolerance at endpoints, pick endpoint
        if abs(b_lo - target_balancedness) <= tol:
            counts = counts_for_gamma(0.0)
        elif abs(b_hi - target_balancedness) <= tol:
            counts = counts_for_gamma(gamma_max)
        else:
            # Binary search gamma (balancedness generally decreases as gamma increases)
            lo, hi = 0.0, gamma_max
            best_counts = counts_for_gamma(lo)
            best_err = abs(_balancedness_from_counts(best_counts) - target_balancedness)

            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                mid_counts = counts_for_gamma(mid)
                mid_b = _balancedness_from_counts(mid_counts)
                err = abs(mid_b - target_balancedness)

                if err < best_err:
                    best_err = err
                    best_counts = mid_counts

                if err <= tol:
                    best_counts = mid_counts
                    break

                # If too balanced (metric too high), increase gamma to skew more
                if mid_b > target_balancedness:
                    lo = mid
                else:
                    hi = mid

            counts = best_counts

    # Materialize indices from per-class counts
    chosen = []
    for c in range(C):
        take = int(counts[c])
        if take > 0:
            chosen.append(by_class[c][:take])
    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    rng.shuffle(chosen)
    return chosen


def ideal_train_set(labels, subset_size=0.1):
    """
    Returns indices for a maximally balanced subset (as equal per-class as possible),
    with safe redistribution if some classes don't have enough samples.

    labels: 1D array-like of class labels
    subset_size: float in (0, 1]
    """

    labels = np.asarray(labels)
    n = len(labels)
    if not (0 < subset_size <= 1):
        raise ValueError("subset_size must be in (0, 1].")

    rng = np.random.default_rng()

    classes, y_inv = np.unique(labels, return_inverse=True)
    C = len(classes)

    # Group indices by class
    by_class = [np.flatnonzero(y_inv == c) for c in range(C)]
    for c in range(C):
        rng.shuffle(by_class[c])

    target_total = int(round(subset_size * n))
    target_total = max(1, min(target_total, n))

    base = target_total // C
    rem = target_total % C

    # Distribute the remainder across random classes to keep things even
    order = rng.permutation(C)
    desired = np.full(C, base, dtype=int)
    desired[order[:rem]] += 1

    chosen = []
    chosen_counts = np.zeros(C, dtype=int)

    # First pass: take up to desired from each class
    for c in range(C):
        take = min(desired[c], len(by_class[c]))
        if take:
            chosen.append(by_class[c][:take])
            chosen_counts[c] += take
            by_class[c] = by_class[c][take:]  # remove taken

    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)

    # Redistribute any shortfall while keeping counts as balanced as possible
    short = target_total - len(chosen)
    if short > 0:
        # Iteratively allocate one sample at a time to the currently-most-underfilled class
        # among those with remaining capacity.
        while short > 0:
            candidates = [c for c in range(C) if len(by_class[c]) > 0]
            if not candidates:
                break  # no more samples anywhere

            # Prefer classes with smallest chosen_counts (most underrepresented)
            min_count = min(chosen_counts[c] for c in candidates)
            under = [c for c in candidates if chosen_counts[c] == min_count]
            c = rng.choice(under)

            chosen = np.append(chosen, by_class[c][0])
            by_class[c] = by_class[c][1:]
            chosen_counts[c] += 1
            short -= 1

    rng.shuffle(chosen)
    return chosen


def random_train_set(embeddings, subset_size=0.3):

    num_samples = int(len(embeddings) * subset_size)
    return np.random.choice(len(embeddings), size=num_samples, replace=False)



def _temp_create_and_save_imbalanced_dataset(percentage_full=0.3,
                                            full_embeddings_path="./data/clip_embeddings_cifar100_train_full.npy",
                                            full_labels_path="./data/labels_cifar100_train_full.npy"):
    """
    Create and save an imbalanced version of the CIFAR-100 training set, 
    by including randomly selected percentage_full classes in full and the rest at only 10% of their original size.
    """
    
    full_embeddings = np.load(full_embeddings_path)
    full_labels = np.load(full_labels_path)
    classes = np.unique(full_labels)
    rng = np.random.default_rng()
    n_classes = len(classes)
    n_full = int(round(percentage_full * n_classes))
    full_classes = rng.choice(classes, size=n_full, replace=False)
    idxs_full = np.isin(full_labels, full_classes)
    idxs_partial = ~idxs_full
    partial_classes = classes[~np.isin(classes, full_classes)]
    idxs_partial_selected = np.isin(full_labels, partial_classes) & (rng.random(len(full_labels)) < 0.1)
    idxs_final = idxs_full | idxs_partial_selected
    np.save(f"./data/cifar100_imbalanced{int(percentage_full*100)}_clip_embeddings.npy", full_embeddings[idxs_final])
    np.save(f"./data/cifar100_imbalanced{int(percentage_full*100)}_labels.npy", full_labels[idxs_final])
    



if __name__ == "__main__":

    # _temp_create_and_save_imbalanced_dataset(percentage_full=0.3)
    # import sys 
    # sys.exit(0)

    # labels_base = np.load("./data/cifar100_imbalanced_labels.npy")
    # embeddings_base = np.load("./data/cifar100_imbalanced_clip_embeddings.npy")

    # print(_balancedness(labels_base))

    labels_base = np.load("./data/cifar100_imbalanced30_labels.npy")
    embeddings_base = np.load("./data/cifar100_imbalanced30_clip_embeddings.npy")

    print(_balancedness(labels_base))


    # idxs = train_set_with_target_balancedness(labels_base, subset_size=0.1, target_balancedness=0.98)

    # print(_balancedness(labels_base[idxs]))

    # import sys 
    # sys.exit(0)

    # print(_balancedness(labels_base))

    # print()
    for subset_fraction in [0.1, 0.3, 0.5, 0.7]:
    # for subset_fraction in [0.7]:

        for frac_boundary in [0.5, 0.7, 0.9]:
            for n_neighbors_umap in [10, 20, 30]:
                for resolution_parameter in [0.01,0.02, 0.03, 0.04]:

                    # subset_fraction = 0.1
                    balancednesses = []
                    num_clusters_list = []
                    for _ in range(5):
                        #scores = _get_zcore_like_scores(embeddings_base)
                        # idxs = ideal_train_set(labels_base, subset_size=subset_fraction)
                        boundary_scores, centrality_scores, cluster_labels = boundaryness_centrality_scores(embeddings_base, resolution_parameter=resolution_parameter, n_neighbors_umap=n_neighbors_umap)
                        idxs = select_balanced_boundary_central(
                            cluster_labels, boundary_scores, centrality_scores, subset_fraction, embeddings_base, frac_boundary=frac_boundary
                        )

                        balancednesses.append(_balancedness(labels_base[idxs]))
                        num_clusters_list.append(len(np.unique(cluster_labels)))

                    print(f"Subset fraction: {subset_fraction}, Fraction boundary: {frac_boundary}, UMAP neighbors: {n_neighbors_umap}, Resolution parameter: {resolution_parameter}, Average balancedness: {np.mean(balancednesses)}, Average number of clusters: {np.mean(num_clusters_list)}")