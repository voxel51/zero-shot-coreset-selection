# Zero-Shot Coreset Selection Plugin

A [FiftyOne](https://docs.voxel51.com/) plugin that implements zero-shot coreset selection based on the paper: https://arxiv.org/pdf/2411.15349

This plugin enables selection of representative subsets from unlabeled image datasets using redundancy and coverage metrics computed in a foundation-model-generated embedding space. It provides:

- **Z-Score Computation**: Computes z-scores that combine redundancy and coverage metrics
- **Interactive Panel**: Visualizes z-score statistics in the FiftyOne App
- **Coreset Selection**: Identifies the most representative samples for annotation or analysis

## Installation

```bash
fiftyone plugins download https://github.com/voxel51/zero-shot-corest-selection
```

Or clone this repository and install locally:

```bash
git clone https://github.com/voxel51/zero-shot-corest-selection
cd zero-shot-corest-selection
pip install -r requirements.txt
fiftyone plugins install .
```

## Usage

### Prerequisites

Your dataset must have embeddings computed. You can generate embeddings using FiftyOne's brain methods:

```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

# Load a dataset
dataset = foz.load_zoo_dataset("quickstart")

# Compute embeddings using a model
fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",
    brain_key="clip_embeddings"
)
```

### Computing Z-Scores

1. Open your dataset in the FiftyOne App
2. Click the operators menu (grid icon) and select "Compute Z-Scores"
3. Configure the parameters:
   - **Embedding brain key**: Select the brain key with your embeddings
   - **Number of neighbors (k)**: Number of nearest neighbors for coverage (default: 10)
   - **Output field name**: Field to store z-scores (default: "zscore")
4. Click "Execute"

The operator will compute three fields for each sample:
- `zscore`: Combined z-score (higher is better for coreset selection)
- `zscore_redundancy`: Redundancy score (similarity to other samples)
- `zscore_coverage`: Coverage score (representativeness)

### Viewing Z-Scores Panel

1. Open the panels menu in the FiftyOne App
2. Select "Z-Scores Panel"
3. View statistics including:
   - Number of samples
   - Mean z-score
   - Standard deviation
   - Min and max values

### Selecting the Coreset

After computing z-scores, you can select the top samples for your coreset:

```python
# In Python
import fiftyone as fo

dataset = fo.load_dataset("your-dataset")

# Sort by z-score (descending) and select top N samples
coreset_size = 100
coreset_view = dataset.sort_by("zscore", reverse=True).limit(coreset_size)

# Create a tag for these samples
for sample in coreset_view:
    sample.tags.append("coreset")
    sample.save()

# Or create a new dataset with just the coreset
coreset_dataset = dataset.clone("coreset")
coreset_dataset.match_samples(coreset_view)
```

## How It Works

The plugin implements a zero-shot coreset selection method that:

1. **Redundancy Score**: Measures how similar each sample is to others in the embedding space
   - Computed as average cosine similarity to all other samples
   - Lower redundancy = more unique sample

2. **Coverage Score**: Measures how well each sample represents its local neighborhood
   - Computed based on average distance to k-nearest neighbors
   - Higher coverage = more representative sample

3. **Z-Score**: Combines redundancy and coverage into a single metric
   - Normalized to zero mean and unit variance
   - Formula: `z-score = coverage_z - redundancy_z`
   - Higher z-score = better candidate for coreset (unique + representative)

Samples with high z-scores are good coreset candidates because they are:
- **Unique**: Low redundancy with other samples
- **Representative**: High coverage of their local neighborhood

## Development

To modify the plugin:

```bash
# Install in development mode
pip install -e .

# Test changes
fiftyone app launch
```

## Citation

If you use this plugin, please cite the original paper:

```
@article{zeroshot2024,
  title={Zero-Shot Coreset Selection},
  year={2024},
  url={https://arxiv.org/pdf/2411.15349}
}
```

## License

Apache 2.0
