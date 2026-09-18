"""Exercise the real doctor against the supervisor's isolated output roots."""

import pytest

from scripts.autodl_doctor import _validate_roots
from scripts.run_canonical_fullframes_posttrain import prepare_output_roots


def roots(tmp_path, monkeypatch):
    durable = tmp_path / "durable"
    durable.mkdir()
    monkeypatch.setenv("ROSETTA_AUTODL_PLATFORM_ROOT", str(tmp_path))
    monkeypatch.setenv("ROSETTA_AUTODL_ROOT", str(durable))
    for variable, name in (
        ("ROSETTA_DATA_ROOT", "data"),
        ("ROSETTA_MODELS_ROOT", "models"),
        ("ROSETTA_CHECKPOINT_ROOT", "checkpoints"),
        ("ROSETTA_ARTIFACT_ROOT", "artifacts"),
        ("ROSETTA_RUN_ROOT", "runs"),
        ("TRACKIO_DIR", "runs/trackio"),
        ("HF_HOME", "models/hf_home"),
    ):
        path = durable / name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(variable, str(path))
    job = durable / "workspace/runs/new-attempt"
    job.mkdir(parents=True)
    return job


def test_supervisor_roots_pass_actual_doctor(tmp_path, monkeypatch):
    job = roots(tmp_path, monkeypatch)
    prepare_output_roots(job)
    result = _validate_roots({"storage": {"require_data_disk": True}})
    assert result["runs"] == job / "results"
    assert result["trackio"] == job / "results/trackio"
    assert result["artifacts"].parent == job.parent


def test_old_missing_result_and_stale_trackio_are_rejected(tmp_path, monkeypatch):
    job = roots(tmp_path, monkeypatch)
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(job / "results"))
    with pytest.raises(ValueError, match="Durable runs root"):
        _validate_roots({"storage": {"require_data_disk": True}})
    (job / "results").mkdir()
    with pytest.raises(ValueError, match="TRACKIO_DIR"):
        _validate_roots({"storage": {"require_data_disk": True}})
