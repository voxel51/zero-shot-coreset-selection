# Plugin Architecture

This document describes the architecture and components of the zero-shot coreset selection plugin.

## File Structure

```
zero-shot-corest-selection/
├── __init__.py           # Main plugin file with FiftyOne operators
├── coreset.py            # Core algorithms (redundancy, coverage, z-scores)
├── fiftyone.yml          # Plugin configuration
├── requirements.txt      # Python dependencies
├── setup.py              # Package installation script
├── examples.py           # Example usage script
├── tests.py              # Unit tests for core algorithms
├── README.md             # User documentation
├── LICENSE               # Apache 2.0 license
└── .gitignore            # Git ignore rules
```

## Components

### 1. Core Algorithms (`coreset.py`)

The core mathematical functions implement the zero-shot coreset selection algorithm:

- **`compute_redundancy_score(embeddings)`**
  - Computes average cosine similarity to all other samples
  - Higher score = more redundant (similar to others)
  - Used to identify unique samples

- **`compute_coverage_score(embeddings, k=10)`**
  - Computes inverse of average distance to k-nearest neighbors
  - Higher score = more representative of local neighborhood
  - Used to identify samples that cover the data distribution

- **`compute_zscores(redundancy_scores, coverage_scores)`**
  - Normalizes both scores to zero mean, unit variance
  - Combines as: `zscore = coverage_z - redundancy_z`
  - Higher zscore = better coreset candidate (unique + representative)

### 2. FiftyOne Operators (`__init__.py`)

#### ComputeZScores Operator
- **Purpose**: Compute z-scores for all samples in a dataset
- **Inputs**:
  - Brain key or embedding field name
  - Number of neighbors (k) for coverage computation
  - Output field name for storing z-scores
- **Outputs**:
  - Stores three fields per sample:
    - `zscore`: Combined z-score
    - `zscore_redundancy`: Redundancy component
    - `zscore_coverage`: Coverage component

#### ZScoresPanel Operator
- **Purpose**: Display z-score statistics in a panel
- **Outputs**:
  - Number of samples with z-scores
  - Mean z-score
  - Standard deviation
  - Min/max values

### 3. Plugin Configuration (`fiftyone.yml`)

Registers the plugin with FiftyOne:
- Plugin name: `@voxel51/zero_shot_coreset_selection`
- Version: `1.0.0`
- Operators: `compute_zscores`, `zscores_panel`
- Panels: `zscores_panel`

## Workflow

```
┌─────────────────────┐
│   Load Dataset      │
│  (with embeddings)  │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  Compute Z-Scores   │
│  (ComputeZScores    │
│    operator)        │
└──────────┬──────────┘
           │
           ├─> compute_redundancy_score()
           ├─> compute_coverage_score()
           └─> compute_zscores()
           │
           v
┌─────────────────────┐
│ Z-scores stored in  │
│   sample fields     │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  View Z-Scores      │
│  (ZScoresPanel)     │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Select Top N        │
│  Samples for        │
│    Coreset          │
└─────────────────────┘
```

## Algorithm Details

### Redundancy Score

For each sample `i`:
```
similarity(i, j) = 1 - cosine_distance(embedding[i], embedding[j])
redundancy[i] = mean(similarity(i, j) for all j ≠ i)
```

### Coverage Score

For each sample `i`:
```
distances[i] = sorted([cosine_distance(embedding[i], embedding[j]) for all j ≠ i])
nearest_k_distances[i] = distances[i][0:k]
coverage[i] = 1 / (1 + mean(nearest_k_distances[i]))
```

### Z-Score

```
redundancy_z = (redundancy - mean(redundancy)) / std(redundancy)
coverage_z = (coverage - mean(coverage)) / std(coverage)
zscore = coverage_z - redundancy_z
```

## Mathematical Intuition

The z-score identifies samples that are:
1. **Unique** (low redundancy): Not similar to many other samples
2. **Representative** (high coverage): Close to their neighbors, covering a region well

Combining these properties ensures the coreset:
- Contains diverse samples (via low redundancy)
- Represents the data distribution (via high coverage)
- Is optimal for annotation budgets

## Testing

Run tests with:
```bash
python tests.py
```

Tests validate:
- Redundancy computation (identical samples have high redundancy)
- Coverage computation (central samples have high coverage)
- Z-score computation (high coverage + low redundancy = high z-score)
- Normalization (z-scores centered at 0)
- Edge cases (minimal samples, k neighbors)
