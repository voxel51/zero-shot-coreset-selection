import fiftyone.operators as foo
import fiftyone.zoo as foz
import pytest


@pytest.fixture
def quickstart_dataset():
    dataset = foz.load_zoo_dataset("quickstart")
    return dataset.take(20, seed=42)


@pytest.mark.dependency()
def test_with_embeddings_computation(quickstart_dataset):
    ctx = {
        "dataset": quickstart_dataset._dataset,
        "view": quickstart_dataset,
        "params": {
            "model": "resnet18-imagenet-torch",
            "embeddings": "embeddings_resnet18",
            "zcore_score_field": "zcore_score_resnet18",
        },
    }
    _ = foo.execute_operator(
        "@51labs/zero-shot-coreset-selection/compute_zcore_score", ctx
    )
    assert "zcore_score_resnet18" in quickstart_dataset.get_field_schema()
    assert quickstart_dataset.first()["zcore_score_resnet18"] is not None


@pytest.mark.dependency(depends=["test_with_embeddings_computation"])
def test_with_precomputed_embeddings(quickstart_dataset):
    quickstart_dataset._dataset.delete_sample_field("zcore_score_resnet18")
    ctx = {
        "dataset": quickstart_dataset._dataset,
        "view": quickstart_dataset,
        "params": {
            "embeddings": "embeddings_resnet18",
            "zcore_score_field": "zcore_score_resnet18",
        },
    }
    _ = foo.execute_operator(
        "@51labs/zero-shot-coreset-selection/compute_zcore_score", ctx
    )
    assert "zcore_score_resnet18" in quickstart_dataset.get_field_schema()
    assert quickstart_dataset.first()["zcore_score_resnet18"] is not None
