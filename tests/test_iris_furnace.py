"""Local Iris protocol, admission, negative-outcome and delivery counterexamples."""

from __future__ import annotations

import copy
import io
import json
import tarfile

import numpy as np
import pytest

from rosetta_reality.vla.training.features import FEATURE_FACTORIES
from rosetta_reality.vla.training.plan import validate_plan_structure
from scripts.iris_delivery import extract, package, safe_name
from scripts.iris_protocol import (
    RUN,
    STAGES,
    baseline,
    build_plan,
    estimate_main_seconds,
    verify_stage,
)
from scripts.iris_runtime import save, sha
from scripts.run_iris_furnace import build_template, validate_template


def test_four_plan_names_are_fresh_and_preserve_the_learning_contract():
    original = baseline()
    untouched = copy.deepcopy(original)
    names = set()
    for arm, coefficient in (("control", 0), ("treatment", 0.01)):
        for kind in ("smoke", "main"):
            plan = build_plan(
                original,
                arm + "-" + kind,
                coefficient=coefficient,
                calibration={"path": "runs/iris/calibration.json", "sha256": "1" * 64},
                smoke=kind == "smoke",
            )
            validate_plan_structure(plan, known_features=FEATURE_FACTORIES)
            assert plan["training"] == original["training"]
            assert plan["normalization"] == original["normalization"]
            assert plan["initialization"] == original["initialization"]
            assert plan["optimizer_smoke"]["steps"] == (2 if kind == "smoke" else 1280)
            assert plan["run_name"].startswith(RUN)
            names.add(plan["run_name"])
    assert len(names) == 4 and original == untouched


@pytest.mark.parametrize("coefficient", [True, "0.01", float("nan"), 0.1])
def test_unregistered_lambda_rejected_by_native_schema(coefficient):
    plan = build_plan(
        baseline(),
        "bad",
        coefficient=coefficient,
        calibration={"path": "runs/calibration.json", "sha256": "1" * 64},
    )
    with pytest.raises(ValueError):
        validate_plan_structure(plan, known_features=FEATURE_FACTORIES)


def test_missing_calibration_cannot_generate_training_plan():
    with pytest.raises(ValueError):
        build_plan(baseline(), "treatment-main", coefficient=0.01)


def test_persisted_plan_resolves_with_native_launcher(tmp_path):
    from scripts.iris_protocol import write_plan
    from scripts.run_smolvla_v2 import _resolve_plan

    (tmp_path / "plans").mkdir()
    path = write_plan(
        tmp_path,
        "treatment-main",
        coefficient=0.01,
        calibration={"path": "runs/iris/calibration.json", "sha256": "1" * 64},
    )
    plan, _, _ = _resolve_plan(path)
    assert plan["run_name"] == RUN + "-treatment-main"
    assert plan["image_key_scene_contract"]["coefficient"] == 0.01
    with pytest.raises(FileExistsError):
        write_plan(tmp_path, "treatment-main")


def test_main_timing_requires_both_observed_smokes_and_does_not_expand_budget():
    records = [{"steps": 2, "update_seconds": [1.5, 0.5]}] * 2
    assert estimate_main_seconds(records) == 2520
    assert estimate_main_seconds([{"steps": 2, "update_seconds": [0.1, 2.0]}] * 2) > 3600
    with pytest.raises(ValueError):
        estimate_main_seconds(records[:1])
    with pytest.raises(ValueError):
        estimate_main_seconds([{"steps": 2, "update_seconds": [1.0, float("nan")]}] * 2)


def test_failed_or_missing_stage_prevents_later_admission(tmp_path):
    save(
        tmp_path / "registration.json",
        {
            "id": RUN,
            "execution_authorized": True,
            "shutdown_authorized": True,
            "started": 100,
            "deadline": 3700,
            "sources": {},
        },
    )
    for prior in STAGES[: STAGES.index("control-main")]:
        save(tmp_path / (prior + ".done.json"), {"stage": prior, "exit_code": 0})
    verify_stage(tmp_path, "control-main", 200)
    path = tmp_path / "treatment-smoke-check.done.json"
    path.write_text(json.dumps({"stage": "treatment-smoke-check", "exit_code": 2}))
    with pytest.raises(ValueError):
        verify_stage(tmp_path, "control-main", 200)
    with pytest.raises(ValueError):
        verify_stage(tmp_path, "control-main", 4000)


def test_template_rejects_extra_optimizer_budget():
    template = build_template()
    validate_template(template)
    template["maximum_optimizer_steps"] += 1
    with pytest.raises(ValueError):
        validate_template(template)


@pytest.mark.parametrize("name", ["../x", "/x", "a/../b", "a\\b", "C:x", "a//b", "."])
def test_delivery_paths(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_delivery_round_trip_and_second_extract_refusal(tmp_path):
    job = tmp_path / RUN
    job.mkdir()
    (job / "result.json").write_text('{"status":"negative_result"}')
    (job / "weights").mkdir()
    (job / "weights/model.safetensors").write_bytes(b"synthetic weights")
    manifest_sha = package(job)
    archive = tmp_path / (RUN + ".tar")
    out = tmp_path / "verified"
    result = extract(archive, out, sha(archive), manifest_sha)
    assert result["all_file_sha256_matched"] and result["files"] == 2
    assert (out / "weights/model.safetensors").read_bytes() == b"synthetic weights"
    with pytest.raises(ValueError):
        extract(archive, out, sha(archive), manifest_sha)


@pytest.mark.parametrize("kind", ["symlink", "traversal", "duplicate"])
def test_archive_rejects_unsafe_members_before_creating_destination(tmp_path, kind):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as tar:
        member = tarfile.TarInfo("../escape" if kind == "traversal" else "item")
        if kind == "symlink":
            member.type, member.linkname = tarfile.SYMTYPE, "/outside"
            tar.addfile(member)
        else:
            member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
            if kind == "duplicate":
                tar.addfile(member, io.BytesIO(b"x"))
    out = tmp_path / "output"
    with pytest.raises(ValueError):
        extract(path, out, sha(path), "0" * 64)
    assert not out.exists()


def arrays():
    generator = np.random.default_rng(100)
    target = generator.normal(size=(45, 50, 14))
    dims = []
    for side in ("left", "right"):
        for i in range(7):
            grip = i == 6
            dims.append(
                {
                    "name": side + ("_gripper" if grip else f"_joint_{i}"),
                    "unit": "normalized" if grip else "radian",
                    "encoding": "0_closed_1_open" if grip else "absolute",
                    "minimum": -10.0,
                    "maximum": 10.0,
                }
            )
    values = {
        "standard_targets": target,
        "normalized_targets": target.copy(),
        "noise": np.zeros((4, 1, 50, 32)),
        "valid_mask": np.ones(target.shape, dtype=bool),
        "internal_grippers": np.zeros((4, 45, 50, 2)),
        "standard_predictions": np.repeat(target[None], 4, axis=0),
        "normalized_predictions": np.repeat(target[None], 4, axis=0),
    }
    control, treatment = copy.deepcopy(values), copy.deepcopy(values)
    control["standard_predictions"][:, 40:] += 0.2
    treatment["standard_predictions"][:, 40:] += 0.1
    return control, treatment, dims


def test_scientific_acceptance_rejects_gripper_damage_despite_left_gain():
    from scripts.analyze_iris import compare

    control, treatment, dimensions = arrays()
    good = compare(control, treatment, dimensions)
    assert good["passed"] and not good["m2_complete"]
    treatment["standard_predictions"][:, 40:, :, 6] += 1.0
    bad = compare(control, treatment, dimensions)
    assert not bad["passed"] and bad["status"] == "negative_result"
    assert bad["metrics"]["full/left_joint/mae"]["strict_left_improvement"]


def test_scientific_acceptance_rejects_target_identity_change():
    from scripts.analyze_iris import compare

    control, treatment, dimensions = arrays()
    treatment["standard_targets"][40, 0, 0] += 1
    with pytest.raises(ValueError):
        compare(control, treatment, dimensions)


@pytest.mark.parametrize("finished", [False, True])
def test_watchdog_preserves_completed_worker_transfer(tmp_path, monkeypatch, finished):
    from scripts import run_iris_furnace as furnace

    save(
        tmp_path / "registration.json",
        {
            "deadline": 1,
            "shutdown_deadline": 10,
            "supervisor_pid": 123,
            "supervisor_ticks": "old",
        },
    )
    save(tmp_path / "active-child.json", {"pid": 456, "ticks": "child"})
    if finished:
        save(tmp_path / "worker-exited.json", {"completed_stages": []})
    now = [2.0]
    actions = []
    monkeypatch.setattr(furnace.time, "time", lambda: now[0])
    monkeypatch.setattr(furnace.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(furnace, "alive", lambda *args: True)
    monkeypatch.setattr(furnace, "terminate", lambda *args: actions.append(("terminate", args)))
    monkeypatch.setattr(furnace.os, "kill", lambda *args: actions.append(("signal", args)))
    monkeypatch.setattr(furnace, "shutdown", lambda *args: actions.append(("shutdown", ())))
    furnace.watch(tmp_path)
    assert [action[0] for action in actions] == (
        ["shutdown"] if finished else ["terminate", "signal", "shutdown"]
    )


def test_terminate_does_not_signal_reused_pid(monkeypatch):
    from scripts import run_iris_furnace as furnace

    monkeypatch.setattr(furnace, "ticks", lambda pid: "new-process")

    def unexpected_signal(*args):
        pytest.fail("A reused PID must not be signalled")

    monkeypatch.setattr(furnace.os, "killpg", unexpected_signal)
    furnace.terminate(123, "original-process")
