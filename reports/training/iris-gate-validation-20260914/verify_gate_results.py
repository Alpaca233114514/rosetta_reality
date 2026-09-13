"""Independently verify recovered Iris Gate bindings, episode counts and criteria."""
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

job = Path(sys.argv[1])
source = Path(sys.argv[2])


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source_sha = "8f1f80d9676c2a8f25c587d9bf7462fe41778c6fa992681096f7284dbcb9812b"
assert digest(source / "handoff-manifest.json") == source_sha
source_manifest = load(source / "handoff-manifest.json")
worker = load(job / "worker-exited.json")
assert worker["error"] is None and worker["optimizer_steps"] == 0
gates = job / "results/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/gates"
summary = {}
actions = 0
for arm, suffix in (("control", "451"), ("treatment", "452")):
    plan_path = job / (arm + "-sim.yaml")
    plan = yaml.safe_load(plan_path.read_text())
    artifact = job / "artifact-metadata" / arm / "manifest.json"
    manifest = load(artifact)
    expected_model = source_manifest["files"][f"exports/{arm}/model.safetensors"]["sha256"]
    assert manifest["selected_checkpoint_model_sha256"] == expected_model
    assert manifest["source_manifest_sha256"] == source_sha
    assert manifest["selected_checkpoint_step"] == 1280
    assert plan["gate3"]["maximum_steps"] == 20 and plan["gate3"]["seed"] == 20260809
    assert plan["gate4"]["seeds"] == list(range(1000, 1005))
    assert plan["gate4"]["maximum_steps"] == 500
    assert plan["gate4"]["minimum_task_success_rate"] == 0.2
    assert plan["inference"]["chunk_execution_steps"] == 1
    reports = {}
    for gate in ("gate3", "gate4"):
        path = gates / f"{gate}-smolvla-sim-{suffix}.json"
        report = load(path)
        assert report["artifact_manifest_sha256"] == digest(artifact)
        assert report["simulation_plan_sha256"] == digest(plan_path)
        assert report["hidden_test_loaded"] is False
        bridge = load(job / (arm + "-" + gate + "-bridge.json"))
        assert bridge["status"] == "passed" and bridge["exact_raw_and_projected_chunks"]
        assert load(job / (arm + "-" + gate + "-parameter-check.json")) == {
            "unchanged": True, "optimizer_steps": 0}
        reports[gate] = report
    first, final = reports["gate3"], reports["gate4"]
    assert first["status"] == "passed" and all(first["acceptance_criteria"].values())
    assert first["metrics"]["rollout_length"] == 20
    assert final["code_identity"] == first["code_identity"]
    first_sha = digest(gates / f"gate3-smolvla-sim-{suffix}.json")
    assert final["gate3_report_sha256"] == first_sha
    episodes = []
    for index, seed in enumerate(range(1000, 1005)):
        p = gates / f"gate4-smolvla-sim-{suffix}-episodes/episode-{index:02d}-seed-{seed}.json"
        row = load(p)
        assert row["status"] == "complete" and row["seed"] == row["policy_noise_seed"] == seed
        assert row["code_identity"] == first["code_identity"]
        assert row["gate3_report_sha256"] == first_sha
        assert row["artifact_manifest_sha256"] == digest(artifact)
        assert row["simulation_plan_sha256"] == digest(plan_path)
        assert row["metrics"] == final["episodes"][index]
        assert row["metrics"]["rollout_length"] == row["metrics"]["policy_inference_calls"] == 500
        episodes.append(row["metrics"])
    aggregate = final["aggregate"]
    for name in ("invalid_action_rate", "executed_limit_violation_rate", "raw_limit_violation_rate",
                 "policy_output_limit_violation_rate", "unprojected_limit_violation_rate"):
        assert math.isclose(sum(e[name] for e in episodes) / 5, aggregate[name], abs_tol=1e-12)
    for name in ("joint_limit_violations", "unexpected_collisions"):
        assert sum(e[name] for e in episodes) == aggregate[name]
    success_rate = sum(e["success"] for e in episodes) / 5
    assert success_rate == aggregate["task_success_rate"]
    assert final["acceptance_criteria"]["minimum_task_success_rate"] == (success_rate >= 0.2)
    assert final["status"] == ("passed" if all(final["acceptance_criteria"].values()) else "failed")
    actions += first["metrics"]["rollout_length"] + sum(e["rollout_length"] for e in episodes)
    summary[arm] = {"gate3": first["status"], "gate4": final["status"],
                    "successes": sum(e["success"] for e in episodes),
                    "maximum_rewards": [e["maximum_reward"] for e in episodes],
                    "failed_criteria": [k for k, v in final["acceptance_criteria"].items() if not v]}
print(json.dumps({"status": "passed", "arms": summary, "gate_reports_verified": 4,
                  "episode_reports_verified": 10, "closed_loop_actions": actions,
                  "artifact_plan_gate3_code_bindings_verified": True, "optimizer_steps": 0}))
