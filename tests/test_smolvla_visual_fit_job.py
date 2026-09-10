"""Main-training gates and complete historical B reproduction counterexamples."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_smolvla_visual_fit import evidence

from rosetta_reality.vla import visual_coverage as legacy
from rosetta_reality.vla import visual_fit as fit
from rosetta_reality.vla import visual_fit_job as job


def test_historical_control_requires_entire_array_not_only_first_action():
    arrays, meta = evidence("B")
    old_meta = copy.deepcopy(meta)
    old_meta["protocol"] = legacy.PROTOCOL
    old_meta.pop("formula_protocol")
    old_meta["arm_identity"].pop("training_recipe")
    old_meta["common_identity"]["execution_contract_sha256"] = "9" * 64
    result = job.compare_historical_control(arrays, old_meta, arrays, meta)
    assert result["status"] == "passed" and all(result["exact_arrays"].values())
    changed = copy.deepcopy(arrays)
    changed["normalized_predictions"][2, 41, 49, 0] += 0.001
    result = job.compare_historical_control(arrays, old_meta, changed, meta)
    assert result["status"] == "failed"
    assert not result["exact_arrays"]["normalized_predictions"]


def test_historical_control_rejects_changed_input_identity():
    arrays, meta = evidence("B")
    old = copy.deepcopy(meta)
    old["protocol"] = legacy.PROTOCOL
    old.pop("formula_protocol")
    old["arm_identity"].pop("training_recipe")
    old["image_hashes"][0] = "e" * 64
    with pytest.raises(ValueError, match="metadata"):
        job.compare_historical_control(arrays, old, arrays, meta)


def live_gate(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    names = [
        "scripts/run_hestia_fit.py", "scripts/check_hestia_collector_cpu.py",
        "src/rosetta_reality/vla/visual_fit_job.py", job.PLAN_DOC,
    ]
    sources = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("registered synthetic source")
        sources[name] = fit.file_hash(path)

    def save(name, data):
        path = run / name
        path.write_text(json.dumps(data))
        return {"path": path.relative_to(tmp_path).as_posix(),
                "sha256": fit.file_hash(path), "accepted": True}

    registration = {"status": "authorized_worker", "run_name": fit.RUN_NAMES["C"],
        "model_execution_authorized": True, "shutdown_authorized": True,
        "compute_seconds": 1800, "resource_limits": job.LIMITS, "source_files": sources}
    save("registration.json", registration)
    watch = {"pid": 1234, "process_start_ticks": 77, "started_unix": 1000,
        "deadline_unix": 2800, "external_watchdog_verified": True,
        "registration_sha256": fit.file_hash(run / "registration.json")}
    save("watchdog.json", watch)
    monkeypatch.setattr(job.time, "time", lambda: 1100)
    monkeypatch.setattr(job.checks, "process_start_ticks", lambda pid: 77)
    monkeypatch.setattr(job.os, "kill", lambda pid, sig: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    samples = [[ep, 0] for _ in range(128) for ep in fit.TRAIN40]
    schedule = {"status": "passed", "native_order_verified": True, "sample_identities": samples,
        "expected_schedule_sha256": hashlib.sha256(
            json.dumps(samples, separators=(",", ":")).encode()).hexdigest()}
    permit = {"status": "authorized_sealed",
        "registration_sha256": fit.file_hash(run / "registration.json"),
        "watchdog_sha256": fit.file_hash(run / "watchdog.json"),
        "plan": save("plan.yaml", {"synthetic": True}),
        "schedule": save("schedule.json", schedule),
        "prerequisite_evidence": {name: save(name + ".json", {"status": "passed"})
                                  for name in job.REQUIRED_BEFORE_TRAIN}}
    save("training-permit.json", permit)
    return run, save, registration, watch, permit


def test_live_complete_training_gate(tmp_path, monkeypatch):
    run, _, _, _, _ = live_gate(tmp_path, monkeypatch)
    plan, samples = job.authorize_training(tmp_path, run)
    assert plan.name == "plan.yaml" and len(samples) == 5120


@pytest.mark.parametrize("failure", [
    "no_authorization", "reused_pid", "too_late", "expired", "source_changed",
    "missing_prerequisite", "failed_prerequisite", "changed_plan", "hidden_sample",
])
def test_pretraining_gate_rejects_incomplete_or_changed_evidence(tmp_path, monkeypatch, failure):
    run, save, registration, _watch, permit = live_gate(tmp_path, monkeypatch)
    if failure == "no_authorization":
        registration["model_execution_authorized"] = False
        save("registration.json", registration)
    elif failure == "reused_pid":
        monkeypatch.setattr(job.checks, "process_start_ticks", lambda pid: 99)
    elif failure in {"too_late", "expired"}:
        monkeypatch.setattr(job.time, "time", lambda: 1700 if failure == "too_late" else 2801)
    elif failure == "source_changed":
        (tmp_path / "scripts/run_hestia_fit.py").write_text("changed")
    elif failure == "missing_prerequisite":
        permit["prerequisite_evidence"].pop("sample_contract")
        save("training-permit.json", permit)
    elif failure == "failed_prerequisite":
        ref = save("environment.json", {"status": "failed"})
        permit["prerequisite_evidence"]["environment"] = ref
        save("training-permit.json", permit)
    elif failure == "changed_plan":
        (run / "plan.yaml").write_text("changed")
    else:
        schedule = job.load(run / "schedule.json")
        schedule["sample_identities"][0] = [fit.HIDDEN5[0], 0]
        permit["schedule"] = save("schedule.json", schedule)
        save("training-permit.json", permit)
    with pytest.raises((ValueError, TimeoutError)):
        job.authorize_training(tmp_path, run)


def test_prior_smoke_cannot_cover_different_native_training_source():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    report = job.load(root / job.SMOKE_REPORT)
    sources = job.load(root / job.SMOKE_REGISTRATION)["job_sources"]
    job.verify_smoke_reuse(report, sources, sources)
    changed = dict(sources)
    changed["src/rosetta_reality/vla/training/observed_launch.py"] = "0" * 64
    with pytest.raises(ValueError, match="training source"):
        job.verify_smoke_reuse(report, sources, changed)
