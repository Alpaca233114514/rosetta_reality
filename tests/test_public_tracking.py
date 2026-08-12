import pytest

from rosetta_reality.tracking import sanitize_metrics, validate_public_payload


def test_public_payload_accepts_revision_identity_and_metrics() -> None:
    payload = {
        "model_id": "lerobot/smolvla_base",
        "model_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "formal_plan_sha256": "a" * 64,
        "normalization_source_split": "train",
        "workspace_dirty": True,
        "paper_url": "https://arxiv.org/abs/2506.01844",
        "loss": 0.5,
        "test_split_loaded": False,
    }

    validate_public_payload(payload)
    assert sanitize_metrics({"loss": 0.5, "message": "ignored"}, mode="train") == {
        "train/loss": 0.5
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"hf_token": "redacted"},
        {"value": "hf_" + "abcdefghijklmnopqrstuvwxyz"},
        {"path": "C:" + "\\Users\\person\\model"},
        {"path": "/" + "home/person/model"},
        {"path": "/" + "root/private/run.log"},
        {"path": "/" + "workspace/checkpoints/model"},
        {"message": "artifact at /" + "tmp/home/run.json"},
        {"url": "https://example.invalid/run?write_token=value"},
    ],
)
def test_public_payload_rejects_sensitive_fields(payload: dict) -> None:
    with pytest.raises(ValueError):
        validate_public_payload(payload)


def test_public_metrics_reject_nonfinite_values_and_sensitive_keys() -> None:
    with pytest.raises(ValueError, match="finite"):
        sanitize_metrics({"loss": float("nan")}, mode="train")
    with pytest.raises(ValueError, match="sensitive"):
        sanitize_metrics({"api_token": 1.0}, mode="train")
