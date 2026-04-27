import multiprocessing
import numpy as np

def zcore_scores(embeddings_, num_workers=4, n_samples=1e6, rand_init=True
):

    embed_info = embedding_preprocess(embeddings_)

    # Parallel sample and score.
    n_parallel_sample = int(n_samples / num_workers)
    parallel_input = [(embed_info, n_parallel_sample) 
                      for n in range(num_workers)]
    pool = multiprocessing.Pool(num_workers, initializer=init_worker,
                                initargs=(embeddings_,))
    parallel_scores = pool.starmap(sample_score, parallel_input)
    pool.close()
    pool.join()

    # Postprocess. 
    if rand_init:
        scores = np.random.uniform(0, 1, embed_info["n"])
        for s in parallel_scores: scores += s
    else: scores = np.sum(parallel_scores, axis=0)
    score_min = np.min(scores)
    scores = (scores - score_min) / (np.max(scores) - score_min)
    
    return scores.astype(np.float32)

def sample_score(embed_info, n_sample, sample_dim=2, redund_nn=5, redund_exp=2):

    scores = np.zeros(embed_info["n"]) 

    for _ in range(n_sample):

        # Random embedding dimension.
        dim = np.random.choice(embed_info["n_dim"], sample_dim, 
                               replace=False)

        # Coverage score.
        sample = np.random.triangular(
            embed_info["min"][dim],
            embed_info["med"][dim],
            embed_info["max"][dim]
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
        "med": np.median(embeddings, axis=0)
    }
    return embed_info

def init_worker(embeddings_):
    # Parallelize embeddings across pool workers to reduce memory footprint.
    global embeddings
    embeddings = embeddings_
