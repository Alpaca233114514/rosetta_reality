"""Static contract tests for the canonical full-frame post-training adapter."""

import importlib.util
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


def test_registered_plan_binds_cuda_runtime_and_fixed_gate_protocol():
    plan = MODULE.validate_plan()
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


def test_plan_rejects_gate_threshold_drift(tmp_path):
    import json

    source = json.loads(MODULE.DEFAULT_PLAN.read_text())
    source["gate4"]["minimum_task_success_rate"] = 0.1
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Gate 4 success threshold drift"):
        MODULE.validate_plan(path)
