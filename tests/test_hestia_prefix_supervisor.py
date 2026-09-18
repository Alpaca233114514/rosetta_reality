import json
from types import SimpleNamespace

import pytest

from scripts import run_hestia_prefix_path as control


def test_missing_authorization_precedes_any_job_creation(tmp_path):
    job = tmp_path / "never-created"
    args = SimpleNamespace(execute_authorized=False, shutdown_authorized=True)
    with pytest.raises(ValueError, match="authorization"):
        control.supervise(args, {"output": str(job)})
    assert not job.exists()


def test_stale_pid_identity_never_signals_a_process(tmp_path, monkeypatch):
    (tmp_path / "active-child.json").write_text(json.dumps({"pid": 123, "ticks": "old"}))
    monkeypatch.setattr(control, "alive", lambda *_: False)

    def forbidden(*_):
        raise AssertionError("Unrelated PID was signaled")

    monkeypatch.setattr(control.os, "killpg", forbidden)
    control.stop_child(tmp_path)


def test_escaping_job_path_is_rejected_before_creation(monkeypatch):
    monkeypatch.setenv("ROSETTA_AUTODL_RUNTIME_PROFILE", "fixture")
    monkeypatch.setenv("ROSETTA_TORCH_DEVICE", "cuda")
    args = SimpleNamespace(execute_authorized=True, shutdown_authorized=True)
    with pytest.raises(ValueError, match="within the fresh workspace"):
        control.supervise(args, {"output": "../escaped"})
