"""Zcore plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from fiftyone import ViewField as F

import fiftyone.operators as foo
import fiftyone.zoo as foz
import fiftyone.core.fields as fof
from fiftyone.operators import types
from .utils import (
    get_embeddings,
    # _get_label_fields,
    # _get_sample_fields,
    # _get_target_view,
    # _get_zoo_models_with_embeddings,
    # get_target_view
)

from .zcore import zcore_score


class ComputeZCoreScores(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_zcore_score",
            label="Compute ZCore Scores",
            description="Compute ZCore Scores on image samples",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
            allow_delegated_execution=True,
            allow_immediate_execution=True,
            default_choice_to_delegated=True,
            dynamic=True,
        )

    def resolve_input(self, ctx):

        inputs = types.Object()

        # TODO: distentangle the get_target_view and _get_target_view functions
        # target_view = get_target_view(ctx, inputs)
        target_view = ctx.dataset
        get_embeddings(ctx, inputs, target_view)

        view = types.View(label="Compute Zcore Scores")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        # target = ctx.params.get("target", None)
        embeddings = ctx.params.get("embeddings", None) or None
        model = ctx.params.get("model")
        model = foz.load_zoo_model(model)
        batch_size = ctx.params.get("batch_size", None)
        num_workers = ctx.params.get("num_workers", None)
        skip_failures = ctx.params.get("skip_failures", True)

        # No multiprocessing allowed when running synchronously
        # TODO: why is this? original zcore implementation does multiproessing with good speedup

        # sample_collection = _get_target_view(ctx, target)
        sample_collection = ctx.dataset
        if not embeddings:
            embeddings = sample_collection.compute_embeddings(
                model,
                embeddings_field=None,
                batch_size=batch_size,
                num_workers=num_workers,
                skip_failures=skip_failures,
            )

        scores = zcore_score(embeddings, num_workers=num_workers)
        # import numpy as np
        # scores = np.random.rand(len(sample_collection))  # placeholder for testing
        print("scores")
        print(scores)

        # if not ctx.delegated:
        #     ctx.trigger("reload_dataset")

        # if sample_collection.get_field("zcore_score") is None:
        #     sample_collection.add_sample_field("zcore_score", fof.FloatField)

        scores = scores.astype(float)  # convert numpy float32 -> Python float
        sample_collection.set_values("zcore_score", scores.tolist())

        # now this reads the saved values
        print(sample_collection.values("zcore_score")[:20])

        return scores

    """
    def resolve_output(self, ctx):
        outputs = types.Object()
        view = types.View(label="Zcore scores have been added to samples")
        return types.Property(outputs, view=view)
    """


def register(plugin):
    # plugin.register(FindExactDuplicates)
    # plugin.register(DisplayExactDuplicates)
    # plugin.register(RemoveAllExactDuplicates)
    # plugin.register(DeduplicateExactDuplicates)
    # plugin.register(FindApproximateDuplicates)
    # plugin.register(DisplayApproximateDuplicates)
    # plugin.register(RemoveAllApproximateDuplicates)
    # plugin.register(DeduplicateApproximateDuplicates)
    plugin.register(ComputeZCoreScores)
    # plugin.register(SelectCoreset)
