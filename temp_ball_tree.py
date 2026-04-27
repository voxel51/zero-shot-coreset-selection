import numpy as np
import time
from sklearn.neighbors import BallTree

# 1. Generate synthetic high-dimensional data
n_samples = 1_000_000
n_dims = 16
X = np.random.random((n_samples, n_dims)).astype(np.float32)
query_point = X[:1]  # Query using the first point for speed

"""
For 1000000x512:

Leaf Size    | Build (s)    | Query (s)   
------------------------------------------
10           | 102.4476     | 0.564438    
20           | 110.1540     | 0.592767    
40           | 98.4158      | 0.508193    
80           | 105.9838     | 0.583005    
160          | 100.4730     | 0.515496    
320          | 103.7201     | 0.606235    
640          | 99.1289      | 0.608323  
"""

# 2. Define leaf sizes to test
leaf_sizes = [10, 20, 40, 80, 160, 320, 640]

print(f"{'Leaf Size':<12} | {'Build (s)':<12} | {'Query (s)':<12}")
print("-" * 42)

for ls in leaf_sizes:
    # Measure Build Time
    start_build = time.time()
    tree = BallTree(X, leaf_size=ls)
    build_time = time.time() - start_build
    
    # Measure Query Time (average over 5 runs for stability)
    query_times = []
    for _ in range(5):
        start_query = time.time()
        tree.query(query_point, k=5)
        query_times.append(time.time() - start_query)
    
    avg_query = np.mean(query_times)
    print(f"{ls:<12} | {build_time:<12.4f} | {avg_query:<12.6f}")