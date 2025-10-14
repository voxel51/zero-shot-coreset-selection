#!/usr/bin/env python
"""
Example script demonstrating the zero-shot coreset selection plugin.

This script shows how to:
1. Load or create a dataset
2. Generate embeddings
3. Compute z-scores
4. Select a coreset
"""

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob


def example_usage():
    """
    Example workflow for using the zero-shot coreset selection plugin.
    """
    print("Loading dataset...")
    # Load a sample dataset (you can replace this with your own)
    dataset = foz.load_zoo_dataset("quickstart")
    
    print("Computing embeddings...")
    # Compute embeddings using a foundation model
    # This creates the embedding space needed for coreset selection
    fob.compute_similarity(
        dataset,
        model="clip-vit-base32-torch",
        brain_key="clip_embeddings"
    )
    
    print("Dataset ready!")
    print(f"Total samples: {len(dataset)}")
    print("\nNext steps:")
    print("1. Launch FiftyOne App: fiftyone app launch")
    print("2. Open the dataset")
    print("3. Run 'Compute Z-Scores' operator from the operators menu")
    print("4. View 'Z-Scores Panel' to see statistics")
    print("5. Sort by 'zscore' field to identify top coreset candidates")
    
    # Launch the app
    session = fo.launch_app(dataset)
    
    print("\n" + "="*60)
    print("After computing z-scores in the App, run the code below")
    print("to programmatically select the coreset:")
    print("="*60)
    print("""
# Select top 100 samples by z-score
coreset_size = 100
coreset_view = dataset.sort_by("zscore", reverse=True).limit(coreset_size)

# Tag coreset samples
for sample in coreset_view:
    sample.tags.append("coreset")
    sample.save()

# View only coreset samples
session.view = dataset.match_tags("coreset")
""")
    
    # Wait for user input
    input("\nPress Enter to close the session...")


if __name__ == "__main__":
    example_usage()
