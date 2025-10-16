import multiprocessing
import numpy as np
import torch


def zcore_score(embeddings_, num_workers, n_samples=1e6, rand_init=True):

    embed_info = embedding_preprocess(embeddings_)

    # embeddings = torch.tensor(embeddings_).share_memory_()

    n_samples = n_samples if n_samples < embed_info["n"] * 10 else embed_info["n"] * 10
    print(n_samples)
    # Parallel sample and score.
    n_parallel_sample = int(n_samples / num_workers)
    parallel_input = [
        (embeddings_, embed_info, n_parallel_sample) for _ in range(num_workers)
    ]

    # multiprocessing.set_start_method("spawn", force=True)

    # pool = multiprocessing.Pool(num_workers, initializer=init_worker,
    #                             initargs=(embeddings_,))

    # pool = multiprocessing.Pool(num_workers)
    # parallel_scores = pool.starmap(sample_score, parallel_input)
    # pool.close()

    # init_worker(embeddings_)
    parallel_scores = [sample_score(*args) for args in parallel_input]

    print(len(parallel_scores))

    # Postprocess.
    if rand_init:
        scores = np.random.uniform(0, 1, embed_info["n"])
        for s in parallel_scores:
            scores += s
    else:
        scores = np.sum(parallel_scores, axis=0)
    score_min = np.min(scores)
    scores = (scores - score_min) / (np.max(scores) - score_min)

    return scores.astype(np.float32)


def donothing():
    return


def sample_score(
    embeddings, embed_info, n_sample, sample_dim=2, redund_nn=5, redund_exp=2
):

    scores = np.zeros(embed_info["n"])

    for _ in range(n_sample):

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
        cover_sample = embeddings[idx, dim]
        nn_dist = np.sum(abs(embeddings[:, dim] - cover_sample), axis=1)
        nn = np.argsort(nn_dist)[1:]
        if nn_dist[nn[0]] == 0:
            scores[nn[0]] -= 1
        else:
            nn = nn[:redund_nn]
            dist_penalty = 1 / (nn_dist[nn] ** redund_exp)
            dist_penalty /= sum(dist_penalty)
            scores[nn] -= dist_penalty

    return scores


def embedding_preprocess(embeddings):
    embed_info = {
        "n": len(embeddings),
        "n_dim": len(embeddings[0]),
        "min": np.min(embeddings, axis=0),
        "max": np.max(embeddings, axis=0),
        "med": np.median(embeddings, axis=0),
    }
    return embed_info


def init_worker(embeddings_):
    # Parallelize embeddings across pool workers to reduce memory footprint.
    global embeddings
    embeddings = embeddings_
