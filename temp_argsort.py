import numpy as np
import time

# 1. Setup (1M samples, 16 dimensions)
n_samples = 1_000_000
n_dims = 16
k_neighbors = 1000
X = np.random.random((n_samples, n_dims)).astype(np.float32)
query_point = X[0]  # Single query point

print(f"{'Method':<15} | {'Query (s)':<12}")
print("-" * 45)


# Measure Query Time (average over 5 runs)
query_times_argsort = []
query_times_partition = []

for _ in range(5):
    # Method A: Full Argsort
    start = time.time()
    distances = np.linalg.norm(X - query_point, axis=1)
    indices_argsort = np.argsort(distances)[:k_neighbors]
    query_times_argsort.append(time.time() - start)
    
    # Method B: Argpartition (Optimized Brute Force)
    start = time.time()
    distances = np.linalg.norm(X - query_point, axis=1)
    indices_partition = np.argpartition(distances, k_neighbors)[:k_neighbors]
    query_times_partition.append(time.time() - start)

print(f"{'np.argsort':<15} | {np.mean(query_times_argsort):<12.6f}")
print(f"{'np.argpartition':<15} | {np.mean(query_times_partition):<12.6f}")