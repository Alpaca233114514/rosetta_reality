"""Exercise real process-group termination and evidence-preserving writes."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "coverage_job", ROOT / "scripts/run_visual_coverage_job.py"
)
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)


def test_termination_targets_only_owned_process_group():
    owned = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        JOB.terminate_group(owned)
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        if owned.poll() is None:
            owned.kill()
        unrelated.terminate()
        owned.wait(timeout=5)
        unrelated.wait(timeout=5)


def test_existing_evidence_cannot_be_overwritten(tmp_path):
    report = tmp_path / "report.json"
    JOB.save(report, {"status": "failed"})
    before = report.read_bytes()
    with pytest.raises(FileExistsError):
        JOB.save(report, {"status": "passed"})
    assert report.read_bytes() == before


def test_shutdown_rejects_changed_platform_wrapper(monkeypatch):
    monkeypatch.setattr(JOB, "digest", lambda _path: "unrecognized")
    with pytest.raises(AssertionError, match="wrapper changed"):
        JOB.shutdown()


@pytest.mark.parametrize("stage", ["preflight-b1", "preflight-b4", "smoke2", "main256"])
def test_generated_plan_roundtrips_through_real_launcher(tmp_path, monkeypatch, stage):
    from run_smolvla_v2 import _resolve_plan

    draft = ROOT / "configs/vla/visual-coverage40-20260910-001" / f"{stage}.json"
    plan = json.loads(draft.read_text())
    plan["status"] = "preregistered"
    monkeypatch.setattr(JOB, "ROOT", tmp_path)
    target = tmp_path / f"{stage}.yaml"
    record = JOB.save_stage_plan(target, plan)
    loaded, _, _ = _resolve_plan(target)
    assert loaded == plan
    assert record["sha256"] == JOB.digest(target)
    assert type(loaded["training"]["optimizer"]["eps"]) is float
    assert type(loaded["training"]["optimizer"]["weight_decay"]) is float


def test_old_json_scientific_notation_reproduces_actual_failure(tmp_path):
    from run_smolvla_v2 import _resolve_plan

    draft = ROOT / "configs/vla/visual-coverage40-20260910-001/preflight-b1.json"
    plan = json.loads(draft.read_text())
    plan["status"] = "preregistered"
    target = tmp_path / "old-format.json"
    target.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="AdamW optimizer contract"):
        _resolve_plan(target)


def test_stage_writer_preserves_failure_for_invalid_numeric_value(tmp_path, monkeypatch):
    draft = ROOT / "configs/vla/visual-coverage40-20260910-001/preflight-b1.json"
    plan = json.loads(draft.read_text())
    plan["status"] = "preregistered"
    plan["training"]["optimizer"]["eps"] = "1e-08"
    monkeypatch.setattr(JOB, "ROOT", tmp_path)
    target = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError, match="AdamW optimizer contract"):
        JOB.save_stage_plan(target, plan)
    assert target.is_file()


@pytest.mark.parametrize("state_shape", [(1, 14), (1, 2, 14), (1, 1, 6)])
def test_state_contract_rejects_missing_history_or_wrong_dimensions(state_shape):
    from visual_coverage_job_checks import validate_state_action_shapes

    batch = {
        "observation.state": SimpleNamespace(shape=state_shape),
        "action": SimpleNamespace(shape=(1, 50, 14)),
    }
    with pytest.raises(ValueError, match="Native observation.state shape differs"):
        validate_state_action_shapes(batch, n_obs_steps=1, chunk_size=50, action_dimension=14)


def test_state_contract_accepts_registered_single_frame_history():
    from visual_coverage_job_checks import validate_state_action_shapes

    batch = {
        "observation.state": SimpleNamespace(shape=(1, 1, 14)),
        "action": SimpleNamespace(shape=(1, 50, 14)),
    }
    validate_state_action_shapes(batch, n_obs_steps=1, chunk_size=50, action_dimension=14)
