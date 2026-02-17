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
    n_samples = n_samples if n_samples < embed_info["n"] * 10 else embed_info["n"] * 10

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
                scores += s
        else:
            scores = np.sum(parallel_scores, axis=0)

    else:
        # non-multiprocess path: expose embeddings to local globals
        _init_worker_local(embeddings_)
        scores = _zcore_scores(embed_info, n_samples)

    # Normalize scores.
    score_min = np.min(scores)
    scores = (scores - score_min) / (np.max(scores) - score_min)

    if len(valid_idx) < len(raw_embeddings_):
        # Map back to original indices
        full_scores = np.full(len(raw_embeddings_), None, dtype=np.float32)
        full_scores[valid_idx] = scores
        return full_scores

    return scores.astype(np.float32)


def _zcore_scores(embed_info, n_samples, sample_dim=2, redund_nn=1000, redund_exp=4):

    redund_nn = min(redund_nn, embed_info["n"] - 2)

    scores = np.zeros(embed_info["n"])

    for _ in range(n_samples):
        # Random embedding dimension.
        dim = np.random.choice(embed_info["n_dim"], sample_dim, replace=False)

        # Coverage score.
        sample = np.random.triangular(
            embed_info["min"][dim], embed_info["med"][dim], embed_info["max"][dim]
        )
        embed_dist = np.sum(abs(embeddings[:, dim] - sample), axis=1)
        idx = np.argmin(embed_dist)
        scores[idx] += 1

        # Redundancy score.
        cover_sample = embeddings[
            idx, dim
        ]  # sample cloest to the current randomly drawn one
        nn_dist = np.sum(abs(embeddings[:, dim] - cover_sample), axis=1)

        k = 1 + redund_nn
        nn_k = np.argpartition(nn_dist, k)[:k]
        nn = nn_k[nn_k != idx]
        order = np.argsort(nn_dist[nn], kind="stable")
        nn = nn[order][:redund_nn]

        if nn_dist[nn[0]] == 0:
            scores[nn[0]] -= 1
        else:
            dist_penalty = 1 / (nn_dist[nn] ** redund_exp)
            dist_penalty /= sum(dist_penalty)
            scores[nn] -= dist_penalty

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


# Usage example
if __name__ == "__main__":
    import fiftyone.zoo as foz

    dataset = foz.load_zoo_dataset(
        "quickstart", drop_existing_dataset=True, persistent=True
    )
    model = foz.load_zoo_model("clip-vit-base32-torch")
    embeddings = dataset.compute_embeddings(model, batch_size=2)

    scores = zcore_scores(embeddings, use_multiprocessing=True)

    coreset = select_coreset(dataset, scores, coreset_size=10)
