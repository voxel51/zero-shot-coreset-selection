"""All functions in this file are copied from fiftyone-plugins/plugins/brain/__init__.py

It would be better to make this functionality available in fiftyone and import from there."""


import fiftyone as fo
import fiftyone.zoo.models as fozm
from packaging.version import Version
import fiftyone.constants as foc

import fiftyone.operators.types as types
import fiftyone.core.patches as fop


def _get_sample_fields(sample_collection, field_types):
    schema = sample_collection.get_field_schema(flat=True)
    bad_roots = tuple(k + "." for k, v in schema.items() if isinstance(v, fo.ListField))
    return [
        path
        for path, field in schema.items()
        if isinstance(field, field_types) and not path.startswith(bad_roots)
    ]


def _get_allowed_model_licenses(ctx, inputs):
    license = ctx.secrets.get("FIFTYONE_ZOO_ALLOWED_MODEL_LICENSES", None)
    if license is None:
        return None

    licenses = license.split(",")

    inputs.view(
        "licenses",
        types.Notice(
            label=(f"Only models with licenses {licenses} will be available below")
        ),
    )

    return licenses


def _get_zoo_models_with_embeddings(ctx, inputs):
    # @todo can remove this if we require `fiftyone>=1.4.0`
    if Version(foc.VERSION) >= Version("1.4.0"):
        licenses = _get_allowed_model_licenses(ctx, inputs)
        kwargs = dict(license=licenses)
    else:
        licenses = None
        kwargs = {}

    if hasattr(fozm, "_list_zoo_models"):
        manifest = fozm._list_zoo_models(**kwargs)
    else:
        # Can remove this code path if we require fiftyone>=1.0.0
        manifest = fozm._load_zoo_models_manifest()

    # pylint: disable=no-member
    available_models = set()
    for model in manifest:
        if model.has_tag("embeddings"):
            available_models.add(model.name)

    return available_models, licenses


def get_embeddings(ctx, inputs, view):
    embeddings_fields = set(_get_sample_fields(view, fo.VectorField))

    embeddings_choices = types.AutocompleteView()
    for field_name in sorted(embeddings_fields):
        embeddings_choices.add_choice(field_name, label=field_name)

    loc = "sample field"

    inputs.str(
        "embeddings",
        default=None,
        label="Embeddings",
        description=(
            f"An optional {loc} containing pre-computed embeddings to use. "
            f"Or when a model is provided, a new {loc} in which to store the "
            "embeddings"
        ),
        view=embeddings_choices,
    )

    embeddings = ctx.params.get("embeddings", None)

    if embeddings not in embeddings_fields:
        model_names, _ = _get_zoo_models_with_embeddings(ctx, inputs)

        model_choices = types.AutocompleteView()
        for name in sorted(model_names):
            model_choices.add_choice(name, label=name)

        inputs.enum(
            "model",
            model_choices.values(),
            default=None,
            required=False,
            label="Model",
            description=(
                "An optional name of a model from the "
                "[FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/models.html) "
                "to use to generate embeddings"
            ),
            view=model_choices,
        )

        model = ctx.params.get("model", None)

        inputs.int(
            "num_workers",
            default=None,
            label="Num workers",
            description=(
                "Number of workers to use for embeddings generation (if applicable) and zcore score computation"
            ),
        )

        if model:
            inputs.int(
                "batch_size",
                default=None,
                label="Batch size",
                description=(
                    "A batch size to use when computing embeddings " "(if applicable)"
                ),
            )

            inputs.bool(
                "skip_failures",
                default=True,
                label="Skip failures",
                description=(
                    "Whether to gracefully continue without raising an error "
                    "if embeddings cannot be generated for a sample"
                ),
            )


def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "BASE_VIEW":
        return ctx.view._base_view

    if target == "DATASET":
        return ctx.dataset

    return ctx.view


def get_target_view(ctx, inputs, allow_selected=True):
    has_base_view = isinstance(ctx.view, fop.PatchesView)
    if has_base_view:
        has_view = ctx.view != ctx.view._base_view
    else:
        has_view = ctx.view != ctx.dataset.view()
    has_selected = allow_selected and bool(ctx.selected)
    default_target = None

    if has_view or has_selected:
        target_choices = types.RadioGroup(orientation="horizontal")

        if has_base_view:
            target_choices.add_choice(
                "BASE_VIEW",
                label="Base view",
                description="Process the base view",
            )
        else:
            target_choices.add_choice(
                "DATASET",
                label="Entire dataset",
                description="Process the entire dataset",
            )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Process the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Process only the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            required=True,
            label="Target view",
            view=target_choices,
        )

    target = ctx.params.get("target", default_target)

    return _get_target_view(ctx, target)
