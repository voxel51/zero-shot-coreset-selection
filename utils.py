"""All functions in this file are (modified) copies from
fiftyone-plugins/plugins/brain/__init__.py.

It would be better to make this functionality available in fiftyone
and import from there."""

import fiftyone as fo
import fiftyone.constants as foc
import fiftyone.operators.types as types
import fiftyone.zoo.models as fozm
from fiftyone.operators import execution_cache
from packaging.version import Version

EMBEDDINGS_FIELD_TYPE = fo.VectorField


@execution_cache(prompt_scoped=True, residency="ephemeral")
def _get_sample_fields(ctx, view_target):

    target_view = ctx.target_view()

    schema = target_view.get_field_schema(flat=True)
    bad_roots = tuple(k + "." for k, v in schema.items() if isinstance(v, fo.ListField))
    return [
        path
        for path, field in schema.items()
        if isinstance(field, EMBEDDINGS_FIELD_TYPE) and not path.startswith(bad_roots)
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


def get_embeddings(ctx, inputs, view_target):
    embeddings_fields = set(_get_sample_fields(ctx, view_target))

    embeddings_choices = types.AutocompleteView()
    for field_name in sorted(embeddings_fields):
        embeddings_choices.add_choice(field_name, label=field_name)

    inputs.int(
        "num_embeddings",
        label="Number of embeddings",
        default=1,
        min=1,
        max=2,
        required=True,
        description=(
            "The number of embeddings to use for zcore score computation. "
            "If more than one is selected, embeddings will be concatenated and "
            "treated as a single embedding vector."
            "Note that a maximum of 2 embeddings can be selected. "
            "This is because functionally, "
            "using more than 2 embeddings has diminishing returns for zcore "
            "score quality, "
            "while significantly increasing memory usage and computation time."
        ),
    )

    num_embeddings = ctx.params.get("num_embeddings", None)

    inputs.int(
        "num_workers",
        default=None,
        label="Num workers",
        description=(
            "Number of workers to use for embeddings generation (if applicable) "
            "and zcore score computation"
        ),
    )

    loc = "sample field"

    for i in range(1, num_embeddings + 1):

        inputs.str(
            f"embeddings_{i}",
            label=f"Embeddings {i}",
            required=True,
            description=(
                f"A {loc} containing pre-computed embeddings to use. "
                f"Or when a model is provided, a new {loc} in which to store the "
                "embeddings"
            ),
            view=embeddings_choices,
        )

        embeddings = ctx.params.get(f"embeddings_{i}", None)

        if embeddings not in embeddings_fields:
            model_names, _ = _get_zoo_models_with_embeddings(ctx, inputs)

            model_choices = types.AutocompleteView()
            for name in sorted(model_names):
                model_choices.add_choice(name, label=name)

            inputs.enum(
                f"model_{i}",
                model_choices.values(),
                default=None,
                required=False,
                label=f"Model {i}",
                description=(
                    "An optional name of a model from the "
                    "[FiftyOne Model Zoo]"
                    "(https://docs.voxel51.com/user_guide/model_zoo/models.html) "
                    "to use to generate embeddings"
                ),
                view=model_choices,
            )

            model = ctx.params.get(f"model_{i}", None)

        if model:
            inputs.int(
                "batch_size",
                default=None,
                label="Batch size",
                description=("A batch size to use when computing embeddings."),
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
