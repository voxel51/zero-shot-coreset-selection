#!/usr/bin/env python
"""
Tests for zero-shot coreset selection plugin.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions from coreset.py
from coreset import (
    compute_redundancy_score,
    compute_coverage_score,
    compute_zscores
)


def test_redundancy_score():
    """Test redundancy score computation."""
    print("Testing redundancy score computation...")
    
    # Create simple test embeddings
    # Sample 1: [1, 0], Sample 2: [0, 1], Sample 3: [1, 0]
    # Sample 1 and 3 are identical, so they should have high redundancy
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    
    redundancy = compute_redundancy_score(embeddings)
    
    assert len(redundancy) == 3, "Should have 3 redundancy scores"
    assert redundancy[0] > redundancy[1], "Sample 1 should be more redundant than sample 2"
    assert np.isclose(redundancy[0], redundancy[2]), "Samples 1 and 3 should have similar redundancy"
    
    print(f"  Redundancy scores: {redundancy}")
    print("  ✓ Redundancy score test passed!")


def test_coverage_score():
    """Test coverage score computation."""
    print("\nTesting coverage score computation...")
    
    # Create embeddings where one sample is central
    embeddings = np.array([
        [0.0, 0.0],   # Central point
        [1.0, 0.0],   # Points around it
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0]
    ])
    
    coverage = compute_coverage_score(embeddings, k=4)
    
    assert len(coverage) == 5, "Should have 5 coverage scores"
    # Central point should have higher coverage (lower avg distance to neighbors)
    assert coverage[0] > min(coverage[1:]), "Central point should have good coverage"
    
    print(f"  Coverage scores: {coverage}")
    print("  ✓ Coverage score test passed!")


def test_zscores():
    """Test z-score computation."""
    print("\nTesting z-score computation...")
    
    # Create sample scores
    redundancy = np.array([0.8, 0.5, 0.3, 0.6])
    coverage = np.array([0.4, 0.7, 0.9, 0.5])
    
    zscores = compute_zscores(redundancy, coverage)
    
    assert len(zscores) == 4, "Should have 4 z-scores"
    # Sample with high coverage and low redundancy should have high z-score
    # Sample 3 has lowest redundancy (0.3) and highest coverage (0.9)
    assert zscores[2] == max(zscores), "Sample with high coverage and low redundancy should have highest z-score"
    
    print(f"  Z-scores: {zscores}")
    print("  ✓ Z-score test passed!")


def test_normalization():
    """Test that z-scores are properly normalized."""
    print("\nTesting z-score normalization...")
    
    # Create random scores
    np.random.seed(42)
    redundancy = np.random.rand(100)
    coverage = np.random.rand(100)
    
    zscores = compute_zscores(redundancy, coverage)
    
    # Z-scores should be roughly centered around 0
    mean = np.mean(zscores)
    std = np.std(zscores)
    
    assert abs(mean) < 0.5, f"Mean should be close to 0, got {mean}"
    assert 0.5 < std < 2.0, f"Std should be reasonable, got {std}"
    
    print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
    print("  ✓ Normalization test passed!")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    # Test with minimal samples (3 samples, k=2)
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    
    redundancy = compute_redundancy_score(embeddings)
    coverage = compute_coverage_score(embeddings, k=2)
    zscores = compute_zscores(redundancy, coverage)
    
    assert len(redundancy) == 3, "Should handle minimal samples"
    assert len(coverage) == 3, "Should handle minimal samples"
    assert len(zscores) == 3, "Should handle minimal samples"
    assert not np.any(np.isnan(zscores)), "Should not produce NaN values"
    
    print("  ✓ Edge cases test passed!")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running Zero-Shot Coreset Selection Tests")
    print("="*60)
    
    try:
        test_redundancy_score()
        test_coverage_score()
        test_zscores()
        test_normalization()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
