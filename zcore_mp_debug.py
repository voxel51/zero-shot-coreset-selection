import atexit
import logging
import multiprocessing as mp
import os
import platform
import sys
import time
from multiprocessing import shared_memory

import numpy as np

LOGPATH = "/tmp/my_debug_log.log"


def setup_file_logger():
    logger = logging.getLogger("my_logger")
    if not logger.handlers:
        fh = logging.FileHandler(LOGPATH)
        fmt = logging.Formatter(
            "%(asctime)s %(process)d %(threadName)s %(levelname)s %(message)s"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
    return logger


logger = setup_file_logger()

# Optional process-introspection (psutil)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def _attach_mp_logger():
    mp_log = logging.getLogger("multiprocessing")
    # Ensure mp logger writes to same file once
    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == LOGPATH
        for h in mp_log.handlers
    ):
        fh = logging.FileHandler(LOGPATH)
        fmt = logging.Formatter(
            "%(asctime)s %(process)d %(threadName)s %(levelname)s mp:%(message)s"
        )
        fh.setFormatter(fmt)
        mp_log.addHandler(fh)
        mp_log.setLevel(logging.INFO)


def _log_runtime_context(phase: str):
    proc = mp.current_process()
    info = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "name": proc.name,
        "daemon": proc.daemon,
        "start_method": mp.get_start_method(allow_none=True),
        "cpu_count": mp.cpu_count(),
        "python": platform.python_version(),
        "exe": sys.executable,
    }
    if psutil:
        try:
            p = psutil.Process()
            info.update(
                {
                    "num_threads": p.num_threads(),
                    "rss_mb": round(p.memory_info().rss / (1024**2), 1),
                    "open_fds": getattr(p, "num_fds", lambda: None)(),
                    "cpu_affinity": getattr(p, "cpu_affinity", lambda: None)(),
                }
            )
        except Exception:
            pass
    logger.info(f"[{phase}] runtime_ctx={info}")


def _log_process_tree(phase: str, max_ancestors: int = 3):
    if not psutil:
        return
    try:
        p = psutil.Process()
        ancestors = []
        cur = p
        for _ in range(max_ancestors):
            parent = cur.parent()
            if not parent:
                break
            ancestors.append(f"{parent.pid}:{parent.name()}")
            cur = parent
        children = [f"{c.pid}:{c.name()}" for c in p.children(recursive=False)]
        logger.info(f"[{phase}] ancestors={ancestors} children={children}")
    except Exception:
        logger.exception(f"[{phase}] failed to log process tree")


# Attach mp logger and log module import context once
_attach_mp_logger()
_log_runtime_context("module_import")
_log_process_tree("module_import")


def _choose_ctx():
    methods = mp.get_all_start_methods()
    if sys.platform.startswith("win"):
        return mp.get_context("spawn")
    if sys.platform == "darwin":
        return mp.get_context("spawn")
    # Linux
    if "forkserver" in methods:
        return mp.get_context("forkserver")
    return mp.get_context("fork")


def zcore_scores(
    embeddings_, num_workers: int = 4, n_samples: int | float = 1e6,
    rand_init: bool = True, use_multiprocessing: bool = True
):
    embed_info = _embedding_preprocess(embeddings_)

    # Cap n_samples to avoid extreme runtimes
    max_allowed = embed_info["n"] * 10
    n_samples = int(n_samples if n_samples < max_allowed else max_allowed)

    shm = None
    if use_multiprocessing and num_workers > 1:
        # Shared memory for zero-copy embeddings in workers
        embeddings_arr = np.ascontiguousarray(embeddings_)
        shm = shared_memory.SharedMemory(create=True, size=embeddings_arr.nbytes)
        shm_array = np.ndarray(
            embeddings_arr.shape, dtype=embeddings_arr.dtype, buffer=shm.buf
        )
        shm_array[:] = embeddings_arr[:]

        shared_name = shm.name
        shape = embeddings_arr.shape
        dtype_str = str(embeddings_arr.dtype)

        logger.info(
            f"[parent pid={os.getpid()}] created SharedMemory name={shared_name} "
            f"shape={shape} dtype={dtype_str} bytes={embeddings_arr.nbytes}"
        )

        # Parallel sample and score
        n_parallel_sample = int(n_samples / num_workers)
        parallel_input = [(embed_info, n_parallel_sample)] * num_workers

        _log_runtime_context("operator_parent_before_pool")
        _log_process_tree("operator_parent_before_pool")

        t0 = time.time()
        pool = None
        try:
            #ctx = _choose_ctx()
            #ctx = mp.get_context("spawn")
            pool = mp.Pool(
            #pool = ctx.Pool(
                num_workers,
                initializer=_init_worker,
                initargs=(shared_name, shape, dtype_str),
            )
            logger.info(
                f"[parent pid={os.getpid()}] Pool started "
                f"(workers={num_workers}, per_worker_samples={n_parallel_sample})"
            )
            _log_process_tree("operator_parent_after_pool_start")

            parallel_scores = pool.starmap(_zcore_scores, parallel_input)

            logger.info(
                f"[parent pid={os.getpid()}] Pool starmap completed in "
                f"{time.time()-t0:.3f}s"
            )
        except Exception:
            logger.exception(f"[parent pid={os.getpid()}] Pool task failed")
            raise
        finally:
            if pool is not None:
                try:
                    pool.close()
                    pool.join()
                    logger.info(f"[parent pid={os.getpid()}] Pool closed and joined")
                except Exception:
                    logger.exception(
                        f"[parent pid={os.getpid()}] Pool close/join failed"
                    )

        # Cleanup shared memory in parent
        if shm is not None:
            try:
                shm.close()
                shm.unlink()
                logger.info(
                    f"[parent pid={os.getpid()}] SharedMemory {shared_name} "
                    "closed+unlinked"
                )
            except Exception:
                logger.exception(
                    f"[parent pid={os.getpid()}] error cleaning SharedMemory "
                    f"{shared_name}"
                )

        # Aggregate scores
        if rand_init:
            scores = np.random.uniform(0, 1, embed_info["n"])
            for s in parallel_scores:
                scores += s
        else:
            scores = np.sum(parallel_scores, axis=0)

    else:
        # Single-process path
        _init_worker_local(embeddings_)
        logger.info(
            f"[single-process pid={os.getpid()}] computing scores "
            f"n_samples={n_samples}"
        )
        scores = _zcore_scores(embed_info, n_samples)

    # Normalize scores
    score_min = np.min(scores)
    score_max = np.max(scores)
    scores = (scores - score_min) / (score_max - score_min)

    return scores.astype(np.float32)





def _zcore_scores(
    embed_info, n_sample, sample_dim: int = 2, redund_nn: int = 5, redund_exp: int = 2
):
    t0 = time.time()
    pid = os.getpid()
    logger.debug(
        f"[worker pid={pid}] _zcore_scores start n={embed_info['n']} "
        f"n_sample={n_sample} sample_dim={sample_dim} "
        f"redund_nn={redund_nn} redund_exp={redund_exp}"
    )

    scores = np.zeros(embed_info["n"])

    for _ in range(n_sample):
        # Random embedding dimension
        dim = np.random.choice(embed_info["n_dim"], sample_dim, replace=False)

        # Coverage score
        sample = np.random.triangular(
            embed_info["min"][dim], embed_info["med"][dim], embed_info["max"][dim]
        )
        embed_dist = np.sum(abs(embeddings[:, dim] - sample), axis=1)
        idx = np.argmin(embed_dist)
        scores[idx] += 1

        # Redundancy score
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

    logger.debug(f"[worker pid={pid}] _zcore_scores done in {time.time()-t0:.3f}s")
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
    # Worker initializer: attach to parent's shared memory, set globals
    setup_file_logger()  # ensure handler exists in child
    pid = os.getpid()
    global embeddings, _worker_shm
    _worker_shm = shared_memory.SharedMemory(name=shared_name)
    embeddings = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_worker_shm.buf)
    logger.info(
        f"[worker pid={pid}] initialized, attached SharedMemory name={shared_name} "
        f"shape={shape} dtype={dtype_str}"
    )

    _log_runtime_context("worker_init")
    _log_process_tree("worker_init")

    def _on_exit():
        try:
            _worker_shm.close()
            logger.info(f"[worker pid={pid}] exiting, SharedMemory closed")
        except Exception:
            logger.exception(
                f"[worker pid={pid}] error while closing SharedMemory on exit"
            )

    atexit.register(_on_exit)


def _init_worker_local(embeddings_):
    # Non-multiprocessing path: keep embeddings in global variable
    global embeddings
    embeddings = embeddings_
    logger.debug(f"[single-process pid={os.getpid()}] local embeddings initialized")


def select_coreset(sample_collection, scores, coreset_size):
    # Select top-k samples based on zcore scores
    all_ids = list(sample_collection.values("id"))
    idxs = np.argsort(-scores)[:coreset_size]
    sample_ids = [all_ids[i] for i in idxs]
    coreset = sample_collection.select(sample_ids, ordered=True)
    return coreset