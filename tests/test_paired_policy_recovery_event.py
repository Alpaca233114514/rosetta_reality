"""Registered recovery geometry semantics, using no model or simulator."""
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/evaluate_paired_policy_v2.py"
_spec = importlib.util.spec_from_file_location("_paired_policy_event_test", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
Tracker = _module.RecoveryEventTracker


def test_registered_bank_cannot_be_silently_omitted():
    plan = {"bank_manifest_sha256": "a" * 64}
    with pytest.raises(ValueError, match="must agree"):
        _module.validate_bank_selection(plan, None)
    with pytest.raises(ValueError, match="must agree"):
        _module.validate_bank_selection({}, Path("unregistered-bank"))
    _module.validate_bank_selection(plan, Path("registered-bank"))
    _module.validate_bank_selection({}, None)


def contacts(**changed):
    return {"socket_grasp_contact": True, "peg_grasp_contact": True, "peg_socket_contact": True} | changed


def test_recovery_requires_five_steps_and_success_strictly_later():
    tracker = Tracker("approach")
    for step in range(1, 5):
        assert not tracker.observe(step, contacts(), safe=True, task_success=False)["geometric_recovery"]
    fifth = tracker.observe(5, contacts(), safe=True, task_success=False)
    assert fifth["geometric_recovery"] and not fifth["verified_recovery_chain"]
    assert tracker.observe(6, contacts(), safe=True, task_success=True)["verified_recovery_chain"]


def test_success_on_fifth_step_is_not_a_prior_recovery_chain():
    tracker = Tracker("approach")
    for step in range(1, 6):
        row = tracker.observe(step, contacts(), safe=True, task_success=step == 5)
    assert row["geometric_recovery"] and not row["verified_recovery_chain"]


def test_align_requires_peg_socket_contact_and_contiguous_window():
    tracker = Tracker("align")
    for step in range(1, 5):
        tracker.observe(step, contacts(), safe=True, task_success=False)
    tracker.observe(5, contacts(peg_socket_contact=False), safe=True, task_success=False)
    for step in range(6, 10):
        assert not tracker.observe(step, contacts(), safe=True, task_success=False)["geometric_recovery"]
    assert tracker.observe(10, contacts(), safe=True, task_success=False)["recovery_step"] == 10


@pytest.mark.parametrize("trigger", ["approach", "grasp"])
def test_grasp_family_does_not_require_insertion_contact(trigger):
    tracker = Tracker(trigger)
    for step in range(1, 6):
        row = tracker.observe(step, contacts(peg_socket_contact=False), safe=True, task_success=False)
    assert row["geometric_recovery"]


def test_unsafe_step_cannot_qualify_successful_chain():
    tracker = Tracker("approach")
    for step in range(1, 7):
        row = tracker.observe(step, contacts(), safe=step != 6, task_success=step == 6)
    assert row["geometric_recovery"] and not row["verified_recovery_chain"]


def test_missing_observations_remain_unmeasured_and_step_gaps_rejected():
    tracker = Tracker("approach")
    assert tracker.result()["correction_steps_observed"] == 0
    with pytest.raises(ValueError, match="consecutive"):
        tracker.observe(2, contacts(), safe=True, task_success=False)
    with pytest.raises(ValueError, match="booleans"):
        Tracker("align").observe(1, {"socket_grasp_contact":True,"peg_grasp_contact":True}, safe=True, task_success=False)
