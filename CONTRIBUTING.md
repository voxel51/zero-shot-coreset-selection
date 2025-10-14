# Contributing Guide

Thank you for your interest in contributing to the zero-shot coreset selection plugin!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/voxel51/zero-shot-corest-selection.git
cd zero-shot-corest-selection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FiftyOne (if not already installed):
```bash
pip install fiftyone>=0.23.0
```

## Running Tests

Run the test suite to ensure everything works:

```bash
python tests.py
```

All tests should pass before submitting changes.

## Testing the Plugin

### Local Installation

Install the plugin locally for development:

```bash
# From the plugin directory
fiftyone plugins install .
```

### Testing with FiftyOne

1. Create a test dataset with embeddings:
```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

# Load a sample dataset
dataset = foz.load_zoo_dataset("quickstart")

# Generate embeddings
fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",
    brain_key="clip_embeddings"
)

# Launch the app
session = fo.launch_app(dataset)
```

2. In the FiftyOne App:
   - Open the operators menu (⊞ icon)
   - Select "Compute Z-Scores"
   - Configure and execute
   - View results in "Z-Scores Panel"

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add docstrings to all functions
- Keep functions focused and small
- Comment complex logic

## Making Changes

1. **Core Algorithms** (`coreset.py`):
   - Keep independent of FiftyOne
   - Add tests for new functionality
   - Maintain backward compatibility

2. **Operators** (`__init__.py`):
   - Follow FiftyOne operator patterns
   - Provide clear input descriptions
   - Handle errors gracefully
   - Show informative notifications

3. **Documentation**:
   - Update README.md for user-facing changes
   - Update ARCHITECTURE.md for structural changes
   - Include examples where helpful

## Submitting Changes

1. Run tests: `python tests.py`
2. Check syntax: `python -m py_compile *.py`
3. Update documentation if needed
4. Submit a pull request with:
   - Clear description of changes
   - Test results
   - Use case or motivation

## Ideas for Contributions

- **Performance improvements**: Optimize for large datasets
- **Additional metrics**: Implement alternative coreset selection criteria
- **Visualization**: Enhanced panel with charts/plots
- **Batch processing**: Support for processing very large datasets
- **Export functionality**: Export coreset to new dataset
- **Parameter tuning**: Auto-suggest optimal k value
- **Documentation**: More examples and use cases

## Questions?

Open an issue on GitHub or reach out to the maintainers.
