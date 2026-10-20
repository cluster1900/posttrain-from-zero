import pytest
from examples.ch06_model_structure import TARGETS, inspect_model


@pytest.fixture(scope="module")
def inspected():
    return inspect_model()


def test_targets_formula_and_freezing(inspected):
    model, report = inspected
    rows = report["target_instances"]
    assert {row["name"].split(".")[-1] for row in rows} == set(TARGETS)
    assert report["lora_trainable_parameters"] == sum(64 * (r["in"] + r["out"]) for r in rows)
    assert report["wrapped_unique_parameters"] == (
        report["base_unique_parameters"] + report["lora_trainable_parameters"]
    )
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    assert not model.get_input_embeddings().weight.requires_grad
    assert not model.get_output_embeddings().weight.requires_grad
    assert any(row["name"].endswith("in_proj_a") for row in rows)
    assert any(row["name"].endswith("in_proj_b") for row in rows)


def test_two_layer_fixture_is_not_full_model(inspected):
    _, report = inspected
    assert report["layer_types"] == ["linear_attention", "full_attention"]
    assert report["hidden_size"] == 64
    assert report["embedding_unique_parameters"] == 248320 * 64
    assert report["kind"] == "reader_experiment_not_project_metrics"
    assert report["meta_only"] is False


def test_meta_and_materialized_shapes_agree(inspected):
    _, actual = inspected
    _, meta = inspect_model(meta=True)
    for key in ("base_unique_parameters", "lora_trainable_parameters", "target_instances"):
        assert actual[key] == meta[key]
