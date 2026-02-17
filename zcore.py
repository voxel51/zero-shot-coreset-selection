import multiprocessing
from multiprocessing import shared_memory

import numpy as np


def zcore_scores(
    raw_embeddings_,
    num_workers=4,
    n_samples=1000000,
    rand_init=True,
    use_multiprocessing=True,
):

    embeddings_, valid_idx = _filter_embeddings_nonempty(raw_embeddings_)
    embeddings_ = np.array(embeddings_, dtype=np.float32)
    embed_info = _embedding_preprocess(embeddings_)

    # 1e6 seems enough, even for large datasets.
    n_samples = min(n_samples, embed_info["n"] * 10)

    # prepare shared memory for embeddings when using multiprocessing
    shm = None
    if use_multiprocessing and num_workers > 1:
        embeddings_arr = np.ascontiguousarray(embeddings_)
        shm = shared_memory.SharedMemory(create=True, size=embeddings_arr.nbytes)
        shm_array = np.ndarray(
            embeddings_arr.shape, dtype=embeddings_arr.dtype, buffer=shm.buf
        )
        shm_array[:] = embeddings_arr[:]  # single copy into shared memory
        shared_name = shm.name
        shape = embeddings_arr.shape
        dtype_str = str(embeddings_arr.dtype)

        # Parallel sample and score.
        n_parallel_sample = int(n_samples / num_workers)
        parallel_input = [(embed_info, n_parallel_sample)] * num_workers

        pool = multiprocessing.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(shared_name, shape, dtype_str),
        )
        parallel_scores = pool.starmap(_zcore_scores, parallel_input)

        pool.close()
        pool.join()

        # cleanup shared memory in parent after workers have exited
        if shm is not None:
            shm.close()
            shm.unlink()

        # Aggregate scores.
        if rand_init:
            scores = np.random.uniform(0, 1, embed_info["n"])
            for s in parallel_scores:
                # scores += s
                scores += s[0]
        else:
            scores = np.sum(parallel_scores, axis=0)

        covs = np.sum([s[1] for s in parallel_scores], axis=0)
        redunds = np.sum([s[2] for s in parallel_scores], axis=0)

    else:
        # non-multiprocess path: expose embeddings to local globals
        _init_worker_local(embeddings_)
        scores, covs, redunds = _zcore_scores(embed_info, n_samples)


    np.save(f"./data/coverages_dims=8.npy", covs)
    np.save(f"./data/redundancies_dims=8.npy", redunds)


    # Normalize scores.
    score_min = np.min(scores)
    scores = (scores - score_min) / (np.max(scores) - score_min)

    if len(valid_idx) < len(raw_embeddings_):
        # Map back to original indices
        full_scores = np.full(len(raw_embeddings_), None, dtype=np.float32)
        full_scores[valid_idx] = scores
        return full_scores
    

    return scores.astype(np.float32)


def _zcore_scores(embed_info, n_samples, sample_dim=8, redund_nn=1000, redund_exp=4, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    redund_nn = min(redund_nn, embed_info["n"] - 2)

    scores = np.zeros(embed_info["n"])
    coverages = np.zeros(embed_info["n"])
    redundancies = np.zeros(embed_info["n"])

    for i in range(n_samples):
        # Random embedding dimension.
        dim = rng.choice(embed_info["n_dim"], sample_dim, replace=False)

        # Coverage score.
        sample = rng.triangular(
            embed_info["min"][dim], embed_info["med"][dim], embed_info["max"][dim]
        )

        embed_dist = np.sum(abs(embeddings[:, dim] - sample), axis=1)
        # diff = np.abs(embeddings[:, dim] - sample)
        # embed_dist = np.sum(diff ** 0.25, axis=1) ** 4

        idx = np.argmin(embed_dist)
        scores[idx] += 1
        coverages[idx] += 1

        # Redundancy score.
        cover_sample = embeddings[
            idx, dim
        ]  # sample cloest to the current randomly drawn one
        nn_dist = np.sum(abs(embeddings[:, dim] - cover_sample), axis=1)
        # diff_nn = np.abs(embeddings[:, dim] - cover_sample)
        # nn_dist = np.sum(diff_nn ** 0.25, axis=1) ** 4

        k = 1 + redund_nn
        nn_k = np.argpartition(nn_dist, k)[:k]
        nn = nn_k[nn_k != idx]
        order = np.argsort(nn_dist[nn], kind="stable")
        nn = nn[order][:redund_nn]

        # if i % 100 == 0:
        #     print("cover_sample", cover_sample)
        #     print("nn_dist[nn][:1]", nn_dist[nn][:1])
        #     print("nn_dist[nn][-1:]", nn_dist[nn][-1:])
        #     print()

        # if nn_dist[nn[0]] == 0:
        #     scores[nn[0]] -= 1
        # else:
        dist_penalty = 1 / (nn_dist[nn] ** redund_exp)
        dist_penalty /= sum(dist_penalty)
        scores[nn] -= dist_penalty
        redundancies[nn] += dist_penalty

    return scores, coverages, redundancies


def _zcore_scores_vectorized(
    embed_info,
    n_samples,
    sample_dim=2,
    redund_nn=1000,
    redund_exp=4,
    batch_size=None,
    rng=None,
):
    """Vectorized implementation of Zcore score computation."""

    n = embed_info["n"]
    n_dim = embed_info["n_dim"]
    redund_nn = min(redund_nn, n - 2)

    if rng is None:
        rng = np.random.default_rng()

    # Heuristic to avoid allocating gigantic (n, n_samples, sample_dim) arrays
    if batch_size is None:
        approx_elems = n * n_samples * sample_dim
        # Target ~100M float32 elements max (~400MB) per batch
        if approx_elems > 100_000_000:
            batch_size = max(1, 100_000_000 // (n * sample_dim))
        else:
            batch_size = n_samples

    scores = np.zeros(n, dtype=np.float64)

    for start in range(0, n_samples, batch_size):
        T = min(batch_size, n_samples - start)

        # Row-wise "choice without replacement": take first sample_dim columns of a random argpartition
        # This is equivalent to choosing sample_dim unique dims per trial.
        rand_keys = rng.random((T, n_dim), dtype=np.float32)
        dims = np.argpartition(rand_keys, sample_dim, axis=1)[:, :sample_dim]

        mins = embed_info["min"][dims]
        meds = embed_info["med"][dims]
        maxs = embed_info["max"][dims]
        samples = rng.triangular(mins, meds, maxs).astype(np.float32)

        #import ipdb; ipdb.set_trace()

        # Coverage distances: L1 over selected dims, for all embeddings and trials
        embs_sel = embeddings[:, dims]                      # (n, T, sample_dim)

        dists = np.sum(np.abs(embs_sel - samples[None, :, :]), axis=2)  # (n, T)
        idx = np.argmin(dists, axis=0)                      # (T,)

        np.add.at(scores, idx, 1.0)
        #scores += np.bincount(idx, minlength=n)

        # Redundancy distances to the chosen cover sample in each trial
        cover = np.take_along_axis(embeddings[idx], dims, axis=1)                       # (T, sample_dim)
        nn_dists = np.sum(np.abs(embs_sel - cover[None, :, :]), axis=2)  # (n, T)

        # Exclude self from neighbors
        nn_dists[idx, np.arange(T)] = np.inf

        # Take redund_nn smallest per trial and sort them stably
        nn_idx = np.argpartition(nn_dists, redund_nn, axis=0)[:redund_nn, :]  # (redund_nn, T)
        nn_vals = np.take_along_axis(nn_dists, nn_idx, axis=0)                # (redund_nn, T)
        order = np.argsort(nn_vals, axis=0, kind="stable")
        nn_idx = np.take_along_axis(nn_idx, order, axis=0)
        nn_vals = np.take_along_axis(nn_vals, order, axis=0)

        first = nn_vals[0, :]
        zero_cols = np.where(first == 0)[0]
        if zero_cols.size:
            # Subtract 1 from the exact-duplicate nearest neighbor
            counts = np.bincount(nn_idx[0, zero_cols], minlength=n)
            scores -= counts

        nonzero_cols = np.where(first != 0)[0]
        if nonzero_cols.size:
            vals = nn_vals[:, nonzero_cols]
            weights = 1.0 / (vals ** redund_exp)
            weights /= np.sum(weights, axis=0, keepdims=True)
            targets = nn_idx[:, nonzero_cols].ravel()
            w = weights.ravel()
            np.subtract.at(scores, targets, w)

    return scores


def _embedding_preprocess(embeddings):
    embed_info = {
        "n": len(embeddings),
        "n_dim": len(embeddings[0]),
        "min": np.min(embeddings, axis=0),
        "max": np.max(embeddings, axis=0),
        "med": np.median(embeddings, axis=0),
    }
    return embed_info


def _init_worker(shared_name, shape, dtype_str):
    # Worker initializer: attach to the parent's shared memory block
    # exposed via `shared_name`. Store numpy view in the global `embeddings`.
    global embeddings, _worker_shm
    _worker_shm = shared_memory.SharedMemory(name=shared_name)
    embeddings = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_worker_shm.buf)


def _init_worker_local(embeddings_):
    # Non-multiprocessing path: keep embeddings in global variable
    global embeddings
    embeddings = embeddings_


def select_coreset(sample_collection, scores, coreset_size):
    # Select top-k samples based on zcore scores.
    all_ids = list(sample_collection.values("id"))
    idxs = np.argsort(-scores)[:coreset_size]
    sample_ids = [all_ids[i] for i in idxs]
    coreset = sample_collection.select(sample_ids, ordered=True)
    return coreset


def _filter_embeddings_nonempty(embeddings_list):
    """Returns (valid_embeddings_list, valid_indices)."""

    valid_idx, valid_embs = [], []
    dim = None
    for i, v in enumerate(embeddings_list):
        if v is None:
            continue
        if dim is None:
            dim = len(v)
        if len(v) != dim:
            raise ValueError(
                f"All embeddings must have the same dimension, "
                f"but found {len(v)} and {dim}. "
                f"This is likely due to using different embedding models."
            )
        valid_idx.append(i)
        valid_embs.append(v)

    return valid_embs, valid_idx


def compare_coresets(scores_1, scores_2, coreset_size=0.3):
    """Computes the overlap between two coresets selected using the given scores."""

    size = len(scores_1)
    assert size == len(scores_2), "Scores must have the same length"
    n = int(size * coreset_size)

    idxs_1 = np.argsort(-scores_1)[:n]
    idxs_2 = np.argsort(-scores_2)[:n]

    set_1 = set(idxs_1)
    set_2 = set(idxs_2)

    intersection = set_1.intersection(set_2)
    overlap = len(intersection) / n

    return overlap


def _expected_overlap(coreset_size=0.3, dataset_size=10000):
    """Computes the expected overlap between two random coresets of the given size."""
    overlaps = []
    for scores_1, scores_2 in zip(np.random.rand(100,dataset_size), np.random.rand(100,dataset_size)):
        overlaps.append(compare_coresets(scores_1, scores_2, coreset_size=coreset_size))

    print("Average overlap:", np.mean(overlaps))


def pca_reduction(embeddings, n_components=64):
    """Reduces the dimensionality of the embeddings using PCA."""

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings


def _number_n_samples():

    _embeddings = np.load("/tmp/embeddings_cifar100_train_10000.npy")

    for n in [100, 1000, 10_000]:
        print("Number of embeddings:", n)
        print()
        embeddings = _embeddings[:n]
        _init_worker_local(embeddings)
        embed_info = _embedding_preprocess(embeddings)
        
        for n_samples in [1000, 10_000, 100_000, 1000_000]:
            print("Number of samples:", n_samples)
            scores_1 = _zcore_scores(embed_info, n_samples=n_samples)
            scores_2 = _zcore_scores(embed_info, n_samples=n_samples)
            print("Overlap:", compare_coresets(scores_1, scores_2, coreset_size=0.3))

        print()
        print("-"*40)
        print()
        

def _try_pca():

    embeddings = np.load("/tmp/embeddings_cifar100_train_10000.npy")

    for n_components in [8, 16, 32, 64, 128]:

        print("PCA components:", n_components)

        reduced = pca_reduction(embeddings, n_components=n_components)
        _do_two_runs(reduced)


def _do_two_runs(embeddings=None):
     # import fiftyone.zoo as foz

    # # dataset = foz.load_zoo_dataset(
    # #     "quickstart"
    # # )

    # dataset = foz.load_zoo_dataset("cifar100", max_samples=10000, split="train")

    # model = foz.load_zoo_model("clip-vit-base32-torch")
    # embeddings = dataset.compute_embeddings(model, batch_size=16)
    # embs_arr = np.array(embeddings)
    # np.save("/tmp/embeddings_cifar100_train_10000.npy", embs_arr)
    # import sys  
    # sys.exit(0)

    if embeddings is None:
        embeddings = np.load("/tmp/embeddings_cifar100_train_10000.npy")
    _init_worker_local(embeddings)
    

    # scores = zcore_scores(embeddings, use_multiprocessing=True)

    # coreset = select_coreset(dataset, scores, coreset_size=10)

    #embeddings = np.random.randn(10, 512).astype(np.float32)
    embed_info = _embedding_preprocess(embeddings)
    n_samples = 1000_000
    n_samples = n_samples if n_samples < embed_info["n"] * 10 else embed_info["n"] * 10
    #import time
    #t1 = time.time()
    # seed = 42
    scores_1 = _zcore_scores(embed_info, sample_dim=2, n_samples=n_samples)
    # print(scores_1)
    # print(np.argsort(scores_1))

    scores_2 = _zcore_scores(embed_info, sample_dim=2, n_samples=n_samples)
    # print(scores_2)
    # print(np.argsort(scores_2))

    print("Overlap:", compare_coresets(scores_1, scores_2, coreset_size=0.3))


def _compare_vectorized_vs_loop():
    embeddings = np.ascontiguousarray(np.random.randn(30000, 512).astype(np.float32))


    _init_worker_local(embeddings)
    embed_info = _embedding_preprocess(embeddings)

    # n_dim = embed_info["n_dim"]
    n_samples = 1000_000
    n_samples = n_samples if n_samples < embed_info["n"] * 10 else embed_info["n"] * 10
    # sample_dim = 2
    

    # rng = np.random.default_rng(42)

    # # Row-wise "choice without replacement": take first sample_dim columns of a random argpartition
    # # This is equivalent to choosing sample_dim unique dims per trial.
    # rand_keys = rng.random((n_samples, n_dim), dtype=np.float64)
    # dims = np.argpartition(rand_keys, sample_dim, axis=1)[:, :sample_dim]

    # mins = embed_info["min"][dims]
    # meds = embed_info["med"][dims]
    # maxs = embed_info["max"][dims]
    # samples = rng.triangular(mins, meds, maxs)

    import time

    t1 = time.time()
    #scores_loop = _zcore_scores(embed_info, sample_dim=sample_dim, n_samples=n_samples, rng=np.random.default_rng(42), samples=samples, dims=dims)
    scores_loop = _zcore_scores(embed_info, n_samples=n_samples)

    #for batch_size in [2,4,8,16,32,64,128]:
    # for batch_size in [4,8,16]:
    #     print("Batch size:", batch_size)
    #     t2 = time.time()
    #     scores = _zcore_scores_vectorized(embed_info, n_samples=n_samples, batch_size=batch_size)
    #     t3 = time.time()
    #     print("Vectorized time:", t3 - t2)
    t2 = time.time()
    # #scores_vec = _zcore_scores_vectorized(embed_info, sample_dim=sample_dim, n_samples=n_samples, rng=np.random.default_rng(42), samples=samples, dims=dims)
    # scores_vec = _zcore_scores_vectorized(embed_info, n_samples=n_samples, batch_size=64)
    # t3 = time.time()
    print("Loop time:", t2 - t1)
    # print("Vectorized time:", t3 - t2)
    # #diff = np.abs(scores_loop - scores_vec)

    # #print(scores_loop[:10])
    # print(scores_vec[:10])

    #print("Max difference between loop and vectorized:", np.max(diff))


# Usage example
if __name__ == "__main__":


    #_do_two_runs()
    #_compare_vectorized_vs_loop()
    #_try_pca()
    _number_n_samples()


