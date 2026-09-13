"""Iris gates require the actual completed, backed-up source artifact."""

import json
import os
import sys
import time

import pytest

from scripts import iris_gate
from scripts.iris_runtime import sha


def source(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"synthetic sealed model")
    (tmp_path / "worker-exited.json").write_text(json.dumps({"error": None}))
    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in tmp_path.iterdir()}
    manifest = tmp_path / "handoff-manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    monkeypatch.setattr(iris_gate, "MANIFEST_SHA", sha(manifest))
    (tmp_path / "transfer-receipt.json").write_text(
        json.dumps({"manifest_sha256": sha(manifest), "all_file_sha256_matched": True})
    )
    return tmp_path


def test_verified_completed_source_is_accepted(tmp_path, monkeypatch):
    iris_gate.check_source(source(tmp_path, monkeypatch))


@pytest.mark.parametrize("failure", ["weight", "receipt", "manifest"])
def test_gate_rejects_identity_or_backup_failure_before_loading_models(
    tmp_path, monkeypatch, failure
):
    root = source(tmp_path, monkeypatch)
    if failure == "weight":
        (root / "model.safetensors").write_bytes(b"changed")
    elif failure == "receipt":
        (root / "transfer-receipt.json").write_text("{}")
    else:
        (root / "handoff-manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        iris_gate.check_source(root)


@pytest.mark.parametrize("failure", ["expired", "watchdog"])
def test_gate_dispatch_requires_live_registration(tmp_path, monkeypatch, failure):
    from scripts import run_iris_furnace

    monkeypatch.setattr(iris_gate, "ROOT", tmp_path)
    job = tmp_path / "runs" / iris_gate.NAME
    job.mkdir(parents=True)
    (job / "registration.json").write_text(
        json.dumps(
            {
                "id": iris_gate.NAME,
                "execution_authorized": True,
                "started": time.time() - 1,
                "deadline": time.time() + (60 if failure == "watchdog" else -1),
            }
        )
    )
    (job / "watchdog.json").write_text(json.dumps({"pid": 123, "ticks": "123"}))
    monkeypatch.setattr(run_iris_furnace, "alive", lambda *args: failure != "watchdog")
    monkeypatch.setattr(sys, "argv", ["iris_gate.py", "render", "--job", str(job)])
    with pytest.raises(ValueError, match="expired or independent watchdog absent"):
        iris_gate.main()


@pytest.mark.parametrize("raises", [False, True])
def test_native_loader_uses_data_root_and_restores_gate_output_root(tmp_path, monkeypatch, raises):
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("ROSETTA_AUTODL_ROOT", str(tmp_path))
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(tmp_path / "gate-results"))

    def native(*args, **kwargs):
        assert os.environ["ROSETTA_RUN_ROOT"] == str(tmp_path / "runs")
        if raises:
            raise ValueError("native failure")
        return "context"

    monkeypatch.setattr(iris_gate, "load_native", native)
    if raises:
        with pytest.raises(ValueError, match="native failure"):
            iris_gate.load_gate_context()
    else:
        assert iris_gate.load_gate_context() == "context"
    assert os.environ["ROSETTA_RUN_ROOT"] == str(tmp_path / "gate-results")
