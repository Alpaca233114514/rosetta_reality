"""Hestia job gates and historical-control comparison without model imports."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from rosetta_reality.vla import visual_coverage as legacy
from rosetta_reality.vla import visual_fit as fit
from rosetta_reality.vla import visual_fit_contract as checks

JOB_REL = Path("runs/hestia-fit40-001")
PLAN_DOC = "reports/training/m2-smolvla-hestia-main-plan-2026-09-10.md"
SMOKE_REPORT = "reports/training/m2-smolvla-hestia-gpu-preflight-result-2026-09-10.json"
SMOKE_REGISTRATION = "reports/training/hestia-preflight-20260910/002-registration.json"
SMOKE_CPU = "reports/training/hestia-preflight-20260910/002-cpu-result.json"
HISTORICAL_B = (
    "dev/visual-hermes-post-20260910-02cbf63/"
    "runs/visual-hermes-coverage40-eval-001/B-first/manifest.json"
)
REQUIRED_BEFORE_TRAIN = {
    "environment", "sample_contract", "resource_preflight",
    "two_step_smoke_reload", "B_training_integrity",
}
LIMITS = {
    "cuda_allocated_bytes": 8 * 1024**3,
    "cuda_reserved_bytes": 10 * 1024**3,
    "host_rss_bytes": 10 * 1024**3,
}


def load(path):
    return json.loads(Path(path).read_text())


def verify_live_job(root: Path, job: Path) -> tuple[dict, dict]:
    registration = load(job / "registration.json")
    if (
        registration.get("status") != "authorized_worker"
        or registration.get("run_name") != fit.RUN_NAMES["C"]
        or registration.get("model_execution_authorized") is not True
        or registration.get("shutdown_authorized") is not True
        or registration.get("compute_seconds") != 1800
        or registration.get("resource_limits") != LIMITS
    ):
        raise ValueError("A separately authorized Hestia main job is required")
    sources = registration.get("source_files", {})
    required = {
        "scripts/run_hestia_fit.py", "scripts/check_hestia_collector_cpu.py",
        "src/rosetta_reality/vla/visual_fit_job.py", PLAN_DOC,
    }
    if not required <= set(sources):
        raise ValueError("Main supervisor source identity is incomplete")
    for name, sha in sources.items():
        if fit.file_hash(checks.relative_file(root, name)) != sha:
            raise ValueError("Main job source changed: " + name)
    watch = load(job / "watchdog.json")
    if (
        watch.get("registration_sha256") != fit.file_hash(job / "registration.json")
        or watch.get("external_watchdog_verified") is not True
        or type(watch.get("started_unix")) not in (int, float)
        or type(watch.get("deadline_unix")) not in (int, float)
        or watch["deadline_unix"] - watch["started_unix"] != 1800
        or not watch["started_unix"] <= time.time() < watch["deadline_unix"]
        or checks.process_start_ticks(watch["pid"]) != watch.get("process_start_ticks")
    ):
        raise ValueError("The registered main watchdog is absent, changed or expired")
    os.kill(watch["pid"], 0)
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Main execution must remain offline")
    return registration, watch


def authorize_training(root: Path, job: Path) -> tuple[Path, list]:
    """No torch import is permitted until every training prerequisite passes."""
    registration, watch = verify_live_job(root, job)
    if watch["deadline_unix"] - time.time() < 1200:
        raise TimeoutError("Insufficient shared budget for training and full comparison")
    permit = load(job / "training-permit.json")
    if (
        permit.get("status") != "authorized_sealed"
        or permit.get("registration_sha256") != fit.file_hash(job / "registration.json")
        or permit.get("watchdog_sha256") != fit.file_hash(job / "watchdog.json")
        or set(permit.get("prerequisite_evidence", {})) != REQUIRED_BEFORE_TRAIN
    ):
        raise ValueError("Main training permit is incomplete")
    for ref in permit["prerequisite_evidence"].values():
        if (
            ref.get("accepted") is not True
            or checks.sealed_json(root, ref).get("status") != "passed"
        ):
            raise ValueError("A main training prerequisite failed")
    plan = permit["plan"]
    path = checks.relative_file(root, plan["path"])
    if fit.file_hash(path) != plan["sha256"]:
        raise ValueError("Candidate plan changed after preflight")
    if registration["source_files"].get(PLAN_DOC) != fit.file_hash(root / PLAN_DOC):
        raise ValueError("Main registration changed")
    schedule = checks.sealed_json(root, permit["schedule"])
    if (
        schedule.get("status") != "passed"
        or schedule.get("native_order_verified") is not True
        or len(schedule.get("sample_identities", [])) != 5120
    ):
        raise ValueError("Native main schedule is not sealed")
    samples = schedule["sample_identities"]
    counts = {ep: 0 for ep in fit.TRAIN40}
    for pair in samples:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(v) is not int for v in pair) or pair[0] not in counts or pair[1] != 0):
            raise ValueError("Native main schedule contains an unregistered identity")
        counts[pair[0]] += 1
    digest = hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest()
    if set(counts.values()) != {128} or digest != schedule.get("expected_schedule_sha256"):
        raise ValueError("Native main schedule allocation or digest changed")
    return path, schedule["sample_identities"]


def verify_smoke_reuse(report: dict, prior_sources: dict, current_sources: dict) -> None:
    """Reuse the completed two-step run only when the executed training path matches."""
    if (
        report.get("status") != "gpu_preflight_passed"
        or report.get("optimizer_updates") != 2
        or report.get("completed_samples") != 8
        or report.get("recovery_state_complete") is not True
        or report.get("reload", {}).get("normalized_and_standard_arrays_exact") is not True
        or len(set(report["reload"].get("independent_processes", []))) != 2
        or report.get("parameter_counts", {}).get("vlm") != {"changed": 0, "total": 345}
    ):
        raise ValueError("Prior two-step smoke/reload was not accepted")
    required = {
        "scripts/run_smolvla_v2.py", "scripts/train_smolvla_v2.py",
        "src/rosetta_reality/vla/processor.py",
        "src/rosetta_reality/vla/training/observed_launch.py",
        "src/rosetta_reality/vla/training/observation.py",
        "src/rosetta_reality/vla/training/features.py",
        "src/rosetta_reality/vla/training/launch.py",
        "src/rosetta_reality/vla/training/plan.py",
    }
    for name in required:
        if (
            not fit._sha(prior_sources.get(name))
            or current_sources.get(name) != prior_sources[name]
        ):
            raise ValueError("Prior smoke does not cover the current training source: " + name)


def compare_historical_control(old_arrays, old_metadata, new_arrays, new_metadata) -> dict:
    """The same B model must reproduce every old value under unchanged inputs."""
    legacy.validate_bundle(old_arrays, old_metadata)
    fit.validate_bundle(new_arrays, new_metadata)
    if old_metadata["arm"] != "B" or new_metadata["arm"] != "B":
        raise ValueError("Historical reproduction requires B in both bundles")
    ignored = {"protocol", "formula_protocol", "common_identity", "arm_identity", "process"}
    for key in (set(old_metadata) | set(new_metadata)) - ignored:
        if old_metadata.get(key) != new_metadata.get(key):
            raise ValueError("Historical B input/metadata changed: " + key)
    for key, value in old_metadata["common_identity"].items():
        if key != "execution_contract_sha256" and new_metadata["common_identity"].get(key) != value:
            raise ValueError("Historical B common identity changed: " + key)
    identity = copy.deepcopy(new_metadata["arm_identity"])
    identity.pop("training_recipe", None)
    if identity != old_metadata["arm_identity"]:
        raise ValueError("Historical B checkpoint or saved plan changed")
    exact = {
        name: old_arrays[name].dtype == new_arrays[name].dtype
        and np.array_equal(old_arrays[name], new_arrays[name])
        for name in legacy.ARRAY_NAMES
    }
    return {"status": "passed" if all(exact.values()) else "failed", "exact_arrays": exact,
            "entire_chunk_compared": True, "historical_evidence_modified": False,
            "additional_optimizer_steps": 0, "m2_complete": False}
