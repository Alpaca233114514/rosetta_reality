"""Synthetic numeric regressions; no scientific success or simulator claims."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("paired_numerics", Path(__file__).resolve().parents[1]/"scripts/verify_paired_numerics_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    dims = [{"name": "left_waist", "minimum": -3, "maximum": 3},
            {"name": "other", "minimum": -3, "maximum": 3}]
    registered = {"joint_name": "left_waist", "trigger": "grasp", "joint_delta_rad": .04}
    plan = {"deviation_min_joint_change_rad": .002, "correction_max_steps": 150, "recovery_sustain_steps": 5}
    rows = []
    for step in range(7):
        geometry = {"robot_state": [.01, .2], "socket_grasp_contact": True,
                    "peg_grasp_contact": True, "peg_socket_contact": True}
        rows.append({"step": step, "state": [0., .2], "action": [.04, .2] if step == 0 else [0., .2],
                     "action_source": "registered_deliberate_deviation" if step == 0 else "state_conditioned_teacher",
                     "images": str(step), "images_sha256": str(step), "pts_seconds": step/50,
                     "before": {"robot_state": [0., .2]}, "after": {"geometry": geometry,
                     "joint_limits": 0, "unexpected_collisions": 0,
                     "info": {"clipped_fields": [], "is_success": step == 6}}, "safe": True, "phase": "grasp"})
    commands = [{k: r[k] for k in ("step", "action", "state", "action_source", "images")} for r in rows]
    attempt = {"rows": 7, "kind": "R", "deviation": {"step": 0, "trigger": "grasp", "joint": "left_waist",
               "duration_steps": 1, "delta_rad": .04, "pre_state": [0., .2], "observed_joint_change_rad": .01,
               "verified": True}, "correction_start": 1, "recovery": {"step": 5, "sustain_steps": 5},
               "safety_stop": False, "task_success": True, "verified_recovery_chain": True}
    return rows, commands, copy.deepcopy(rows), attempt, registered, dims, plan


def test_full_row_independent_numeric_fixture():
    result = module.verify_attempt(*fixture())
    assert result["rows_verified"] == 7 and result["recovery_window_verified"]


@pytest.mark.parametrize("mutation", ["movement", "delta", "extra_joint", "command", "phase", "clip", "correction", "empty_window", "contact"])
def test_corrupted_numeric_or_recovery_is_rejected(mutation):
    args = fixture()
    rows, commands, trace, attempt, registered, dims, plan = args
    if mutation == "movement":
        attempt["deviation"]["observed_joint_change_rad"] = .02
    elif mutation in ("delta", "extra_joint"):
        index = 0 if mutation == "delta" else 1
        for record in (rows[0], commands[0], trace[0]):
            record["action"][index] += .01
    elif mutation == "command":
        commands[1]["state"] = [.1, .2]
    elif mutation == "phase":
        rows[0]["phase"] = trace[0]["phase"] = "align"
    elif mutation == "clip":
        rows[2]["after"]["info"]["clipped_fields"] = ["other"]
        trace[2]["after"]["info"]["clipped_fields"] = ["other"]
    elif mutation == "correction":
        attempt["correction_start"] = 2
    elif mutation == "empty_window":
        attempt["recovery"]["step"] = 1
    elif mutation == "contact":
        rows[3]["after"]["geometry"]["peg_grasp_contact"] = False
        trace[3]["after"]["geometry"]["peg_grasp_contact"] = False
    with pytest.raises(ValueError):
        module.verify_attempt(*args)


def test_failed_uncompleted_command_preserves_denominator():
    args = fixture()
    args[1].append({"step": 7})
    assert module.verify_attempt(*args)["uncompleted_commands"] == 1
