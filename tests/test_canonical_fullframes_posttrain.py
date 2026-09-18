"""Static contract tests for the canonical full-frame post-training adapter."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "canonical_fullframes_posttrain",
    ROOT / "scripts/canonical_fullframes_posttrain.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def current_plan(tmp_path):
    """Synthetic current-source plan; never a historical execution permit."""
    plan = json.loads(MODULE.DEFAULT_PLAN.read_text())
    plan["implementation"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in plan["implementation"]
    }
    plan["synthetic_unit_test"] = True
    path = tmp_path / "synthetic-current-plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_current_plan_binds_cuda_runtime_and_fixed_gate_protocol(current_plan):
    plan = MODULE.validate_plan(current_plan, verify_runtime_inputs=False)
    assert plan["fixed_endpoint_step"] == 5000
    assert plan["posttrain_runtime"]["accelerator"] == "cuda"
    assert plan["posttrain_runtime"]["requires_separate_cuda_window"] is True
    assert plan["gate4"]["seeds"] == [1000, 1001, 1002, 1003, 1004]
    assert plan["gate4"]["minimum_task_success_rate"] == 0.2


def test_training_schedule_is_exactly_20000_unique_pairs():
    schedule = MODULE.expected_training_schedule()
    assert len(schedule) == 20000
    assert len({tuple(pair) for pair in schedule}) == 20000
    assert {episode for episode, _ in schedule} == set(MODULE.TRAIN_EPISODES)
    assert {frame for _, frame in schedule} == set(range(500))


def test_plan_rejects_gate_threshold_drift(tmp_path, current_plan):
    source = json.loads(current_plan.read_text())
    source["gate4"]["minimum_task_success_rate"] = 0.1
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Gate 4 success threshold drift"):
        MODULE.validate_plan(path, verify_runtime_inputs=False)


def test_default_admission_still_requires_actual_runtime_inputs(
    tmp_path, current_plan, monkeypatch
):
    name = "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Source identity source changed: runs/"):
        MODULE.validate_plan(current_plan)
