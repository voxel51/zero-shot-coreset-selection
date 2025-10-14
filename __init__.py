"""
Zero-Shot Coreset Selection Plugin for FiftyOne

This plugin implements zero-shot coreset selection based on the paper:
https://arxiv.org/pdf/2411.15349

It computes z-scores for redundancy and coverage metrics in an embedding space
and provides a panel to visualize these scores.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone as fo
import numpy as np

from .coreset import (
    compute_redundancy_score,
    compute_coverage_score,
    compute_zscores
)


class ComputeZScores(foo.Operator):
    """Operator to compute z-scores for coreset selection."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_zscores",
            label="Compute Z-Scores",
            description="Compute z-scores for zero-shot coreset selection",
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Get available embedding fields
        view = ctx.dataset.view()
        brain_keys = ctx.dataset.list_brain_runs()
        
        if brain_keys:
            embedding_choices = types.DropdownView()
            for key in brain_keys:
                embedding_choices.add_choice(key, label=key)
            
            inputs.str(
                "brain_key",
                label="Embedding brain key",
                description="Select the brain key with embeddings to use",
                view=embedding_choices,
                required=True,
            )
        else:
            inputs.str(
                "embedding_field",
                label="Embedding field",
                description="Field containing embeddings (e.g., 'embeddings')",
                required=True,
                default="embeddings",
            )
        
        inputs.int(
            "k_neighbors",
            label="Number of neighbors (k)",
            description="Number of nearest neighbors for coverage computation",
            default=10,
            required=True,
        )
        
        inputs.str(
            "output_field",
            label="Output field name",
            description="Field name to store z-scores",
            default="zscore",
            required=True,
        )
        
        return types.Property(inputs)
    
    def execute(self, ctx):
        view = ctx.dataset.view()
        brain_key = ctx.params.get("brain_key", None)
        embedding_field = ctx.params.get("embedding_field", "embeddings")
        k_neighbors = ctx.params.get("k_neighbors", 10)
        output_field = ctx.params.get("output_field", "zscore")
        
        # Get embeddings
        if brain_key:
            # Try to get embeddings from brain run
            brain_info = ctx.dataset.get_brain_info(brain_key)
            if "embeddings_field" in brain_info.config:
                embedding_field = brain_info.config.embeddings_field
        
        # Extract embeddings from samples
        embeddings_list = []
        sample_ids = []
        
        for sample in view:
            embedding = sample.get_field(embedding_field)
            if embedding is not None:
                embeddings_list.append(embedding)
                sample_ids.append(sample.id)
        
        if not embeddings_list:
            ctx.ops.notify(
                f"No embeddings found in field '{embedding_field}'",
                variant="error"
            )
            return
        
        embeddings = np.array(embeddings_list)
        
        # Compute scores
        redundancy_scores = compute_redundancy_score(embeddings)
        coverage_scores = compute_coverage_score(embeddings, k=k_neighbors)
        zscores = compute_zscores(redundancy_scores, coverage_scores)
        
        # Store scores in dataset
        for sample_id, zscore, redundancy, coverage in zip(
            sample_ids, zscores, redundancy_scores, coverage_scores
        ):
            sample = ctx.dataset[sample_id]
            sample[output_field] = float(zscore)
            sample[f"{output_field}_redundancy"] = float(redundancy)
            sample[f"{output_field}_coverage"] = float(coverage)
            sample.save()
        
        ctx.ops.notify(
            f"Computed z-scores for {len(sample_ids)} samples",
            variant="success"
        )
    
    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("message", label="Result")
        return types.Property(outputs)


class ZScoresPanel(foo.Operator):
    """Panel operator to display z-scores."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="zscores_panel",
            label="Z-Scores Panel",
            description="Display z-scores for coreset selection",
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        return types.Property(inputs)
    
    def execute(self, ctx):
        pass
    
    def resolve_output(self, ctx):
        outputs = types.Object()
        
        # Get current view
        view = ctx.dataset.view()
        
        # Check if z-scores exist
        schema = ctx.dataset.get_field_schema()
        zscore_fields = [f for f in schema.keys() if "zscore" in f.lower()]
        
        if not zscore_fields:
            outputs.str(
                "message",
                label="No z-scores found",
                description="Please run 'Compute Z-Scores' operator first",
            )
            return types.Property(outputs)
        
        # Get statistics
        primary_field = "zscore" if "zscore" in zscore_fields else zscore_fields[0]
        
        stats = {
            "count": 0,
            "mean": 0,
            "std": 0,
            "min": 0,
            "max": 0,
        }
        
        zscores = []
        for sample in view:
            zscore = sample.get_field(primary_field)
            if zscore is not None:
                zscores.append(zscore)
        
        if zscores:
            zscores = np.array(zscores)
            stats["count"] = len(zscores)
            stats["mean"] = float(np.mean(zscores))
            stats["std"] = float(np.std(zscores))
            stats["min"] = float(np.min(zscores))
            stats["max"] = float(np.max(zscores))
        
        # Display statistics
        outputs.str(
            "field",
            label="Z-Score Field",
            default=primary_field,
        )
        
        outputs.int(
            "count",
            label="Number of Samples",
            default=stats["count"],
        )
        
        outputs.float(
            "mean",
            label="Mean Z-Score",
            default=stats["mean"],
        )
        
        outputs.float(
            "std",
            label="Std Dev",
            default=stats["std"],
        )
        
        outputs.float(
            "min",
            label="Min Z-Score",
            default=stats["min"],
        )
        
        outputs.float(
            "max",
            label="Max Z-Score",
            default=stats["max"],
        )
        
        return types.Property(outputs)


def register(plugin):
    """Register the plugin operators."""
    plugin.register(ComputeZScores)
    plugin.register(ZScoresPanel)
