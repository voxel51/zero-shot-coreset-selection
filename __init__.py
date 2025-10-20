"""Zcore plugin.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone.operators as foo
import fiftyone.zoo as foz
from fiftyone.operators import types

from .utils import get_embeddings
from .zcore import zcore_scores


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

        # Currently only supports dataset-level computation
        target_view = ctx.dataset
        get_embeddings(ctx, inputs, target_view)

        view = types.View(label="Compute Zcore Scores")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        embeddings = ctx.params.get("embeddings", None) or None
        model = ctx.params.get("model")
        model = foz.load_zoo_model(model)
        batch_size = ctx.params.get("batch_size", None)
        num_workers = ctx.params.get("num_workers", None)
        skip_failures = ctx.params.get("skip_failures", True)

        sample_collection = ctx.dataset
        if not embeddings:
            embeddings = sample_collection.compute_embeddings(
                model,
                embeddings_field=None,
                batch_size=batch_size,
                num_workers=num_workers,
                skip_failures=skip_failures,
            )

        scores = zcore_scores(embeddings, num_workers=num_workers)

        scores = scores.astype(float)  # convert numpy float32 -> Python float
        sample_collection.set_values("zcore_score", scores.tolist())

        if not ctx.delegated:
            ctx.trigger("reload_dataset")


def register(plugin):
    plugin.register(ComputeZCoreScores)
