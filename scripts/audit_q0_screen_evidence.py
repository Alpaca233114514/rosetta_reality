"""Independent saved-file audit of the closed 20260927 Q0 screening attempt.

This checks identities and report arithmetic, never loads a model or simulator,
and cannot establish that a saved measurement reflects the physical world.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

MANIFEST_SHA = "b9ef3a845877d68dc5c054a4edfc6052930140d2e72c11c04a984b208f5fad87"
PLAN_SHA = "57a2e1b9f345a200a4ed3e825a63b4f015c5cd06017dbd2fa2bf5e10eb236b45"
CANDIDATE = "mink-coupled-translation-teacher-20260927-001"
EXPERIMENT = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(root: Path) -> dict:
    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    require(sha(manifest_path) == MANIFEST_SHA, "Original manifest identity mismatch")
    manifest = read(manifest_path)
    seen, total = set(), 0
    for entry in manifest["files"]:
        name = entry["path"]
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and "\\" not in name and name not in seen, "Unsafe or duplicate manifest entry")
        path = root.joinpath(*relative.parts)
        require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(root),
                "Manifest member escaped evidence root")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                "Manifest member size or SHA mismatch")
        total += entry["bytes"]
        seen.add(name)
    require(len(seen) == 127 and total == manifest["bytes"] == 4663991,
            "Original evidence count or size mismatch")
    ended = read(root / "worker-exited.json")
    require(ended["manifest_sha256"] == MANIFEST_SHA and ended["status"] == "failed",
            "Exit record does not bind failed original attempt")
    registration = read(root / "registration.json")
    require(registration["plan_sha256"] == PLAN_SHA, "Execution plan binding mismatch")
    gate_root = root / "gate-runs" / EXPERIMENT / "teacher_gate"
    candidate_registration_path = gate_root / "registrations" / (CANDIDATE + ".json")
    candidate_registration = read(candidate_registration_path)
    g0 = read(gate_root / CANDIDATE / "g0-001.json")
    g1 = read(gate_root / CANDIDATE / "g1-001.json")
    require(g0["passed"] is True and all(g0["checks"].values()), "G0 report inconsistent")
    for evidence in (g0, g1):
        require(evidence["candidate_id"] == CANDIDATE
                and evidence["registration_sha256"] == sha(candidate_registration_path)
                and evidence["preregistration_sha256"] == candidate_registration["preregistration"]["sha256"]
                and evidence["code_identity"] == candidate_registration["code_identity"],
                "Candidate, protocol or code binding mismatch")
    require(g1["seeds"] == [10] and g1["budget_steps"] == 500
            and len(g1["episodes"]) == 1, "G1 seed or budget changed")
    episode = g1["episodes"][0]
    expected_criteria = {
        "calibration_reward_reached": episode["maximum_reward"] >= 4.0,
        "terminated_within_budget": 0 < episode["rollout_length"] <= 500,
        "zero_joint_limit_violations": episode["joint_limit_violations"] == 0,
        "zero_unexpected_collisions": episode["unexpected_collisions"] == 0,
        "zero_refusals": not episode["refusals"],
    }
    require(g1["criteria"] == expected_criteria, "G1 criteria arithmetic mismatch")
    passed = episode["success"] is True and all(expected_criteria.values())
    require(g1["passed"] == passed is False, "G1 outcome arithmetic mismatch")
    require(episode["rollout_length"] == 500 and episode["maximum_reward"] == 3.0
            and episode["action_clip_entry_count"] == 0, "Closed result changed")
    require(not any((gate_root / CANDIDATE).glob("g2-*.json")), "Unexpected later G2 execution")
    failed = read(root / "failure.json")
    require(failed["phase"] == "g1" and failed["automatic_retry"] is False,
            "Failure boundary changed")
    phases = failed["phases"]
    require([p["stage"] for p in phases] == ["check_env", "synthetic", "benchmark", "register", "g0", "g1"]
            and [p["exit_code"] for p in phases] == [0, 0, 0, 0, 0, 1], "Phase chain inconsistent")
    benchmark = read(root / "benchmark.json")
    estimate = 6 * benchmark["reset_and_observation_seconds"] + (
        benchmark["ik_loop_54_seconds"] + benchmark["maximum_step_render_observer_seconds"]
    ) * 3000 * 1.5
    require(math.isfinite(estimate) and math.isclose(estimate, benchmark["estimated_screening_seconds"], rel_tol=1e-12),
            "Benchmark estimate arithmetic mismatch")
    require(benchmark["passed"] is True and estimate <= 1800
            and benchmark["environment_steps"] == 3 and benchmark["solve_count"] == 54,
            "Benchmark budget inconsistent")
    return {"schema": "rosetta.q0_saved_evidence_audit.v1", "passed": True,
            "verification_scope": "saved_file_integrity_bindings_and_report_arithmetic",
            "original_manifest_sha256": MANIFEST_SHA, "files_verified": len(seen), "bytes_verified": total,
            "g0_passed": True, "g1_passed": False, "g1_steps": 500, "g1_maximum_reward": 3.0,
            "g1_success": False, "g2_executed": False, "benchmark_steps": 3,
            "total_reported_environment_steps": 503, "benchmark_estimated_seconds": estimate,
            "provider_qualified": False, "new_model_calls": 0, "new_environment_steps": 0,
            "physical_measurements_independently_replayed": False,
            "shutdown_or_platform_status_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output is create-only")
    try:
        result = audit(args.input)
    except (ValueError, KeyError, OSError, TypeError) as error:
        result = {"schema": "rosetta.q0_saved_evidence_audit.v1", "passed": False,
                  "error_type": type(error).__name__, "provider_qualified": False}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
