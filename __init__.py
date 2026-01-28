"""Zcore plugin.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone.operators as foo
import fiftyone.zoo as foz
from fiftyone.operators import execution_cache, types

from .utils import get_embeddings
from .zcore import select_coreset, zcore_scores


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
        get_embeddings(ctx, inputs, ctx.target_view())

        inputs.str(
            "zcore_score_field",
            default="zcore_score",
            label="Zcore score field name",
            description=(
                "The name of the field in which to store the zcore scores. "
                "Defaults to 'zcore_score'."
            ),
            required=False,
        )

        size = _get_view_length(ctx)
        n = int(ctx.params.get("coreset_size") or 0)

        slider = types.SliderView(
            value_precision=0,
            step=1,
            space=12,
            label_position="top",
            value_label_display="auto",
        )

        inputs.int(
            "coreset_size",
            default=0,
            min=0,
            max=size,
            required=False,
            label=f"Coreset size ({n} / {size})",
            view=slider,
            description=(
                "Optionally, the size of the coreset to select. If this value is set, "
                "a view containing the selected coreset will be created."
            ),
        )

        view = types.View(label="Compute Zcore Scores")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        embeddings = ctx.params["embeddings"]
        zcore_score_field = ctx.params.get("zcore_score_field", "zcore_score")
        model = ctx.params.get("model")
        model = foz.load_zoo_model(model) if model else None
        batch_size = ctx.params.get("batch_size", None)
        num_workers = ctx.params.get("num_workers", None)
        skip_failures = ctx.params.get("skip_failures", True)
        coreset_size = ctx.params["coreset_size"]

        sample_collection = ctx.target_view()

        if model:
            sample_collection.compute_embeddings(
                model,
                embeddings_field=embeddings,
                batch_size=batch_size,
                num_workers=num_workers,
                skip_failures=skip_failures,
            )

        use_multiprocessing = ctx.delegated
        embeddings = sample_collection.values(embeddings)
        scores = zcore_scores(
            embeddings, num_workers=num_workers, use_multiprocessing=use_multiprocessing
        )

        scores = scores.astype(float)  # convert numpy float32 -> Python float
        sample_collection.set_values(zcore_score_field, scores.tolist())

        if coreset_size > 0:
            coreset = select_coreset(sample_collection, scores, coreset_size)
            ctx.ops.set_view(view=coreset)
        # in delegated execution mode, there is no executor present
        if not ctx.delegated:
            ctx.trigger("reload_dataset")


@execution_cache(prompt_scoped=True, residency="ephemeral")
def _get_view_length(ctx):
    return len(ctx.target_view())


def register(plugin):
    plugin.register(ComputeZCoreScores)
