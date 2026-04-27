import faiss
import numpy as np
import time

# 1. Setup
n_samples = 1_000_000
n_dims = 512
k = 1000
X = np.random.random((n_samples, n_dims)).astype('float32')
query_point = X[:1]

# 2. Exact Manhattan Search (Flat Index)
# We explicitly pass faiss.METRIC_L1
index_l1 = faiss.IndexFlat(n_dims, faiss.METRIC_L1)
index_l1.add(X)

start = time.time()
distances, indices = index_l1.search(query_point, k)
print(f"FAISS L1 Query Time: {time.time() - start:.6f}s")

# 3. Approximate Manhattan Search (IVF)
nlist = 1000
quantizer = faiss.IndexFlat(n_dims, faiss.METRIC_L1) 
index_ivf = faiss.IndexIVFFlat(quantizer, n_dims, nlist, faiss.METRIC_L1)

index_ivf.train(X)
index_ivf.add(X)
index_ivf.nprobe = 10

start = time.time()
distances, indices = index_ivf.search(query_point, k)
print(f"FAISS IVF L1 Query Time: {time.time() - start:.6f}s")
