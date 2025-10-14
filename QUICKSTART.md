# Quick Start Guide

Get started with zero-shot coreset selection in 5 minutes!

## Prerequisites

- Python 3.7+
- FiftyOne 0.23.0+
- A dataset with embeddings

## Installation

```bash
# Install the plugin
fiftyone plugins download https://github.com/voxel51/zero-shot-corest-selection

# Or install locally
git clone https://github.com/voxel51/zero-shot-corest-selection.git
cd zero-shot-corest-selection
fiftyone plugins install .
```

## Quick Example

### Step 1: Prepare Your Dataset

```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

# Load a sample dataset (or use your own)
dataset = foz.load_zoo_dataset("quickstart")

# Generate embeddings using a foundation model
fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",
    brain_key="clip_embeddings"
)

print(f"Dataset has {len(dataset)} samples")
```

### Step 2: Compute Z-Scores

Launch the FiftyOne App:

```python
session = fo.launch_app(dataset)
```

In the App:
1. Click the operators menu (⊞ grid icon in top-right)
2. Select **"Compute Z-Scores"**
3. Configure:
   - **Embedding brain key**: Select "clip_embeddings"
   - **Number of neighbors**: 10 (default is fine)
   - **Output field name**: "zscore" (default)
4. Click **"Execute"**

Wait for completion (you'll see a success notification).

### Step 3: View Z-Scores

1. Open the panels menu (in the App sidebar)
2. Select **"Z-Scores Panel"**
3. View statistics:
   - Number of samples
   - Mean z-score
   - Standard deviation
   - Min/max values

### Step 4: Select Your Coreset

#### In the App UI:

1. Sort samples by z-score:
   - Click the samples grid menu
   - Select "Sort by"
   - Choose "zscore" field
   - Select descending order (highest first)

2. Select top N samples:
   - Use the lasso tool or checkbox to select
   - Tag selected samples: "coreset"

#### Programmatically:

```python
# Select top 100 samples by z-score
coreset_size = 100
coreset_view = dataset.sort_by("zscore", reverse=True).limit(coreset_size)

# Tag them
for sample in coreset_view:
    sample.tags.append("coreset")
    sample.save()

# View only coreset samples
session.view = dataset.match_tags("coreset")

print(f"Selected {len(coreset_view)} samples for the coreset")
```

### Step 5: Use Your Coreset

```python
# Create a new dataset with just the coreset
coreset_dataset = fo.Dataset("my_coreset")
coreset_dataset.add_samples(coreset_view)

# Or export for annotation
coreset_view.export(
    export_dir="/path/to/export",
    dataset_type=fo.types.ImageDirectory,
)

print(f"Coreset ready for annotation!")
```

## Understanding the Results

### Z-Score Interpretation

- **High z-score** (> 0.5): Best coreset candidates
  - Unique samples (low redundancy)
  - Representative of local region (high coverage)
  - Prioritize these for annotation

- **Medium z-score** (-0.5 to 0.5): Average samples
  - Balanced redundancy and coverage
  - Consider if budget allows

- **Low z-score** (< -0.5): Less valuable
  - Either very redundant or not representative
  - Skip these to save annotation budget

### Field Breakdown

After computing z-scores, each sample has three fields:

```python
sample = dataset.first()

# Combined score (use this for selection)
print(f"Z-score: {sample.zscore}")

# Individual components (for analysis)
print(f"Redundancy: {sample.zscore_redundancy}")  # Lower is better
print(f"Coverage: {sample.zscore_coverage}")      # Higher is better
```

## Tips

1. **Adjust k-neighbors**:
   - Small k (5-10): Focus on local neighborhoods
   - Large k (20-50): Consider broader distribution
   - Default (10) works well for most cases

2. **Coreset size**:
   - Start with 10% of your dataset
   - Adjust based on annotation budget
   - Monitor diversity with the panel

3. **Multiple rounds**:
   - Annotate first coreset
   - Train model
   - Compute z-scores on remaining data
   - Select next coreset batch

## Next Steps

- Read the full [README](README.md) for details
- Check [ARCHITECTURE](ARCHITECTURE.md) to understand the algorithm
- See [examples.py](examples.py) for more code samples
- Read the [paper](https://arxiv.org/pdf/2411.15349) for theory

## Troubleshooting

**No embeddings found**: Make sure you've computed embeddings first using `fiftyone.brain.compute_similarity()`

**Plugin not showing**: Restart FiftyOne App or run `fiftyone plugins list` to verify installation

**Slow computation**: For large datasets (>10k samples), consider working with a subset first

## Support

- GitHub Issues: https://github.com/voxel51/zero-shot-corest-selection/issues
- FiftyOne Slack: https://slack.voxel51.com
- Documentation: https://docs.voxel51.com
