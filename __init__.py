"""Zcore plugin.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import importlib
import os

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz
from fiftyone.core.utils import add_sys_path
from fiftyone.operators import execution_cache, types

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    import utils

    importlib.reload(utils)
    import zcore

    importlib.reload(zcore)


class ComputeZCoreScores(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_zcore_score",
            label="Compute ZCore Scores ",
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

        inputs.view_target(ctx, allow_selected_samples=True)
        view_target = ctx.params.get("view_target", None)

        utils.get_embeddings(ctx, inputs, view_target)

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

        size = _get_view_length(ctx, view_target)
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

        if n > 0:
            inputs.str(
                "coreset_name",
                default="zcore_coreset",
                label="Coreset view name",
                description=(
                    "The name to use for the view containing the selected coreset."
                ),
                required=True,
            )

        view = types.View(label="Compute Zcore Scores")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        embedding_fields = ctx.params["embeddings"]
        new_embedding_fields = ctx.params.get("new_embedding_fields", None)
        zcore_score_field = ctx.params.get("zcore_score_field")
        models = ctx.params.get("models")
        batch_size = ctx.params.get("batch_size", None)
        num_workers = ctx.params.get("num_workers") or 1
        skip_failures = ctx.params.get("skip_failures", True)
        coreset_size = ctx.params["coreset_size"]

        sample_collection = ctx.target_view()

        if new_embedding_fields:
            for embedding_name, model_name in zip(new_embedding_fields, models):
                model = foz.load_zoo_model(model_name)
                sample_collection.compute_embeddings(
                    model,
                    embeddings_field=embedding_name,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    skip_failures=skip_failures,
                )

        use_multiprocessing = ctx.delegated
        embeddings = [
            sample_collection.values(embedding_name)
            for embedding_name in embedding_fields
        ]

        scores = zcore.zcore_scores(
            embeddings, num_workers=num_workers, use_multiprocessing=use_multiprocessing
        )

        scores = scores.astype(float)  # convert numpy float32 -> Python float
        if coreset_size > 0:
            # Must select coreset BEFORE set_values to avoid view invalidation
            coreset = zcore.select_coreset(sample_collection, scores, coreset_size)

        sample_collection.set_values(zcore_score_field, scores.tolist())

        if coreset_size > 0:
            dataset = (
                sample_collection
                if isinstance(sample_collection, fo.Dataset)
                else sample_collection._dataset
            )
            dataset.save_view(
                name=ctx.params["coreset_name"], view=coreset, overwrite=True
            )
        # in delegated execution mode, there is no executor present
        if not ctx.delegated:
            ctx.trigger("reload_dataset")


@execution_cache(prompt_scoped=True, residency="ephemeral")
def _get_view_length(ctx, view_target):
    return len(ctx.target_view())


def register(plugin):
    plugin.register(ComputeZCoreScores)
