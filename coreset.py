"""
Core algorithms for zero-shot coreset selection.

This module contains the mathematical functions for computing redundancy,
coverage, and z-scores, independent of FiftyOne.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def compute_redundancy_score(embeddings):
    """
    Compute redundancy score for each sample.
    
    Redundancy is measured as the average similarity to other samples.
    Higher redundancy means the sample is more similar to others.
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
    
    Returns:
        redundancy_scores: numpy array of shape (n_samples,)
    """
    # Compute pairwise cosine distances
    distances = cosine_distances(embeddings)
    
    # Convert distances to similarities (1 - distance)
    similarities = 1 - distances
    
    # Redundancy is the average similarity to all other samples
    # Exclude self-similarity (diagonal)
    n_samples = len(embeddings)
    redundancy_scores = (similarities.sum(axis=1) - 1) / (n_samples - 1)
    
    return redundancy_scores


def compute_coverage_score(embeddings, k=10):
    """
    Compute coverage score for each sample.
    
    Coverage is measured as how well a sample represents its local neighborhood.
    Higher coverage means the sample is more representative.
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        k: number of nearest neighbors to consider
    
    Returns:
        coverage_scores: numpy array of shape (n_samples,)
    """
    # Compute pairwise cosine distances
    distances = cosine_distances(embeddings)
    
    # For each sample, find the average distance to k nearest neighbors
    # (excluding itself)
    n_samples = len(embeddings)
    k = min(k, n_samples - 1)
    
    coverage_scores = np.zeros(n_samples)
    for i in range(n_samples):
        # Get distances to all other samples
        sample_distances = distances[i]
        # Sort and get k nearest (excluding self at index 0)
        nearest_distances = np.sort(sample_distances)[1:k+1]
        # Coverage is inversely related to average distance
        coverage_scores[i] = 1.0 / (1.0 + np.mean(nearest_distances))
    
    return coverage_scores


def compute_zscores(redundancy_scores, coverage_scores):
    """
    Compute z-scores from redundancy and coverage metrics.
    
    Z-score combines redundancy (lower is better) and coverage (higher is better).
    
    Args:
        redundancy_scores: numpy array of redundancy scores
        coverage_scores: numpy array of coverage scores
    
    Returns:
        zscores: numpy array of z-scores
    """
    # Normalize scores to have mean=0, std=1
    redundancy_z = (redundancy_scores - np.mean(redundancy_scores)) / np.std(redundancy_scores)
    coverage_z = (coverage_scores - np.mean(coverage_scores)) / np.std(coverage_scores)
    
    # Z-score: high coverage, low redundancy is best
    # So we want: high coverage_z, low redundancy_z
    zscores = coverage_z - redundancy_z
    
    return zscores
