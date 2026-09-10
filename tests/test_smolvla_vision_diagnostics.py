"""Synthetic probes only: no real dataset, weights, accelerator or optimizer."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow
import pyarrow.parquet as parquet
import pytest

from rosetta_reality.vla.vision_diagnostics import (
    fit_probe,
    paired_alignment,
    read_frame_zero,
    ridge_predict,
    spatial_features,
    validate_splits,
    verify_deploy_artifact,
)


def _config():
    return SimpleNamespace(
        episodes=(1, 2, 3),
        fields=SimpleNamespace(
            episode_index="episode_index",
            frame_index="frame_index",
            action="action",
            state="state",
        ),
    )


def test_hidden_rows_filtered_before_materialization(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    parquet.write_table(
        pa.table(
            {
                "episode_index": [1, 2, 3, 1],
                "frame_index": [0, 0, 0, 1],
                "action": [[1.0], [2.0], [float("nan")], [99.0]],
                "state": [[0.0], [0.0], [float("nan")], [1.0]],
            }
        ),
        tmp_path / "data/rows.parquet",
    )
    actual_dataset = arrow.dataset
    scanned = []

    class Scanner:
        def to_table(self, **kwargs):
            assert "filter" in kwargs
            table = actual_dataset(tmp_path / "data", format="parquet").to_table(**kwargs)
            # This check happens before production code can call to_pylist().
            assert table["episode_index"].to_pylist() == [1, 2]
            scanned.append(True)
            return table

    monkeypatch.setattr(arrow, "dataset", lambda *args, **kwargs: Scanner())
    rows = read_frame_zero(tmp_path, _config(), [2, 1], [3])
    assert [row["episode_index"] for row in rows] == [2, 1]
    assert scanned == [True]


@pytest.mark.parametrize("episodes", [[3], [1, 1], [], [9]])
def test_forbidden_or_invalid_request_rejected_before_scan(tmp_path, monkeypatch, episodes):
    def forbidden(*args, **kwargs):
        pytest.fail("An invalid request must never construct a dataset scanner.")

    monkeypatch.setattr(arrow, "dataset", forbidden)
    with pytest.raises(ValueError):
        read_frame_zero(tmp_path, _config(), episodes, [3])


@pytest.mark.parametrize("ids", [[1, 1, 2], [1]])
def test_missing_or_duplicate_frame_zero_fails(tmp_path, ids):
    (tmp_path / "data").mkdir()
    parquet.write_table(
        pa.table(
            {
                "episode_index": ids,
                "frame_index": [0] * len(ids),
                "action": [[1.0]] * len(ids),
                "state": [[0.0]] * len(ids),
            }
        ),
        tmp_path / "data/rows.parquet",
    )
    with pytest.raises(ValueError, match="Exactly one"):
        read_frame_zero(tmp_path, _config(), [1, 2], [3])


def test_spatial_pool_retains_layout_that_mean_pool_loses():
    first = np.asarray([[1.0], [0.0], [0.0], [0.0]])
    second = first[::-1]
    np.testing.assert_equal(first.mean(axis=0), second.mean(axis=0))
    assert not np.array_equal(spatial_features(first, (2, 2)), spatial_features(second, (2, 2)))
    with pytest.raises(ValueError, match="explicit matching grid"):
        spatial_features(first, (3, 2))


def test_validation_labels_cannot_choose_alpha():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(35, 12))
    y = x[:, :2] * 0.3
    original = fit_probe(x, y, 30)
    changed = y.copy()
    changed[30:] += 1000
    contaminated = fit_probe(x, changed, 30)
    assert original["alpha"] == contaminated["alpha"]
    assert original["train_cv_tuning_mae"] == contaminated["train_cv_tuning_mae"]
    assert contaminated["validation_mae"] > original["validation_mae"] + 900


def test_dual_ridge_matches_primal_solution():
    rng = np.random.default_rng(5)
    x, y, target = rng.normal(size=(8, 20)), rng.normal(size=(8, 2)), rng.normal(size=(3, 20))
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    weights = np.linalg.solve(z.T @ z + 0.1 * np.eye(20), z.T @ (y - y.mean(axis=0)))
    expected = ((target - x.mean(axis=0)) / x.std(axis=0)) @ weights + y.mean(axis=0)
    np.testing.assert_allclose(ridge_predict(x, y, target, 0.1), expected, atol=1e-10)


def test_image_sensitivity_is_not_alignment():
    truth = np.arange(6.0).reshape(3, 2)
    wrong = truth[::-1]
    result = paired_alignment(wrong, truth, truth)
    assert result["image_output_shift"] > 0
    assert result["paired_mae_gain"] < 0
    assert result["gating"] is False
    good = paired_alignment(truth, wrong, truth)
    assert good["paired_mae_gain"] > 0


def test_constant_dimensions_emit_null_not_nan():
    constant = np.ones((3, 2))
    result = paired_alignment(constant, constant, constant)
    assert result["output_target_correlation_per_dimension"] == [None, None]
    assert result["paired_mae_gain"] == 0
    with pytest.raises(ValueError):
        paired_alignment(constant, constant, constant * float("nan"))


def test_splits_cannot_overlap():
    validate_splits([1], [2], [3])
    with pytest.raises(ValueError, match="disjoint"):
        validate_splits([1], [2], [1, 3])


def test_context_uses_one_checked_root_for_labels_and_images(monkeypatch):
    from pathlib import Path

    import rosetta_reality.data as data
    import rosetta_reality.vla.vision_diagnostics as diagnostics

    seen = []
    sentinel = Path("/synthetic-pinned-cache")

    def resolve(config, repository_root, *, validate_checksums):
        assert validate_checksums is True
        return sentinel, SimpleNamespace(resolved_revision=config.revision)

    def rows(root, config, episodes, hidden):
        assert root == sentinel
        assert set(episodes).isdisjoint(hidden)
        seen.extend(episodes)
        return [{config.fields.action: [0.0], config.fields.state: [0.0]} for _ in episodes]

    monkeypatch.setattr(data, "resolve_prepared_cache", resolve)
    monkeypatch.setattr(diagnostics, "read_frame_zero", rows)
    context = diagnostics.load_frame_zero_context(Path(__file__).resolve().parents[1], "train")
    assert context["root"] == sentinel
    assert seen == context["train"]
    assert set(seen).isdisjoint(context["validation"])


def test_deploy_inventory_detects_corruption_before_model_load(tmp_path):
    import hashlib
    import json

    payloads = {
        "config.json": "{}",
        "normalization.json": json.dumps({"source_split": "train", "hidden_test_loaded": False}),
        "action_contract.json": "{}",
        "pretrained_model/config.json": "{}",
        "pretrained_model/model.safetensors": "synthetic checksum bytes; never loaded",
    }
    for name, value in payloads.items():
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        target.write_text(value, encoding="utf-8")
    manifest = {
        "artifact_id": "test-001",
        "experiment_id": "experiment-001",
        "status": "verified",
        "hidden_test_loaded": False,
        "reload": {"exact_tensor_equality": True},
        "files": {
            name: hashlib.sha256(value.encode()).hexdigest() for name, value in payloads.items()
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert len(verify_deploy_artifact(tmp_path, "test-001", "experiment-001")) == 64
    # Only mutate this test's own synthetic temporary artifact.
    (tmp_path / "pretrained_model/model.safetensors").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_deploy_artifact(tmp_path, "test-001", "experiment-001")
