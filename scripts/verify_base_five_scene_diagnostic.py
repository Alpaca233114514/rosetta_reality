"""Read-only independent trace arithmetic for actual base diagnostic closeout."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    return value


def verify(directory, plan_path, contract_path):
    directory = Path(directory)
    plan, contract = read(plan_path), read(contract_path)
    assert sha(contract_path) == "ef0c8ce81dec7203bfecf37233282c36fc9ae0e5abd9a93b2c0125a4fe420b13"
    summary, completion = read(directory / "summary.json"), read(directory / "completion.json")
    integrity = read(directory / "parameter-integrity.json")
    assert completion["status"] == "completed", "Closeout did not complete"
    assert completion["summary_sha256"] == sha(directory / "summary.json")
    assert summary["status"] == "measurements_complete_pending_closeout"
    assert summary["provenance"]["plan_sha256"] == sha(plan_path)
    assert summary["provenance"]["model_sha256"] == plan["model_sha256"]
    assert summary["provenance"]["artifact_manifest_sha256"] == plan["artifact_manifest_sha256"]
    assert summary["registered_seed_count"] == summary["measured_seed_count"] == 5
    assert summary["unmeasured_seeds"] == summary["incomplete_seeds"] == []
    assert integrity["unchanged"] is True and integrity["before"] == integrity["after"]
    assert integrity["optimizer_steps"] == summary["optimizer_steps"] == 0
    assert summary["final_checks"]["parameter_integrity"] == integrity
    assert summary["provenance"]["parameter_before_sha256"] == integrity["before"]
    resources, caps = summary["final_checks"]["resources"], plan["resources"]
    assert resources["elapsed_seconds"] < caps["maximum_seconds"]
    for name in ("rss_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes", "evidence_bytes"):
        assert resources[name] <= caps[name]
    assert resources["cuda_peak_allocated_bytes"] <= caps["cuda_allocated_bytes"]
    assert resources["cuda_peak_reserved_bytes"] <= caps["cuda_reserved_bytes"]
    for item in (summary, completion):
        assert item["shutdown_requested"] is False
    assert summary["formal_gate_result"] is None
    assert summary["original_gate4_protocol_equivalent"] is False
    assert summary["historical_base_gate3_status"] == "failed"
    assert summary["historical_base_gate4_status"] == "not_measured"
    dimensions = contract["dimensions"]
    f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
    lower, upper = [f32(d["minimum"]) for d in dimensions], [f32(d["maximum"]) for d in dimensions]
    def finite_vector(value):
        assert len(value) == len(dimensions)
        assert all(type(x) in (int, float) and math.isfinite(x) for x in value)
    independent = []
    for seed in plan["seeds"]:
        episode = read(directory / f"seed-{seed}.json")
        trace_path = directory / f"seed-{seed}.jsonl"
        assert episode["trace_sha256"] == sha(trace_path)
        assert episode["seed"] == episode["policy_noise_seed"] == seed
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert events and events[0]["event"] == "reset" and events[0]["seed"] == seed
        reset, steps = events[0], events[1:]
        assert 0 <= len(steps) <= 500
        limit_count, collision_count = reset["joint_limit_violations"], len(reset["unexpected_pairs"])
        if limit_count or collision_count:
            assert not steps, "Unsafe reset executed commands"
        successes, rewards, clips = [], [], 0
        for index, row in enumerate(steps):
            assert row["event"] == "step" and row["seed"] == seed and row["step"] == index + 1
            for name in ("raw_action", "processed_action", "executed_action", "state_before", "state_after"):
                finite_vector(row[name])
            expected = [min(max(v, low), high) for v, low, high in zip(row["raw_action"], lower, upper)]
            assert row["executed_action"] == expected, "Executed projection differs from raw slot0"
            counted = sum(a != b for a, b in zip(row["raw_action"], expected))
            assert row["raw_clipped_fields"] == counted
            clips += counted
            limit_count += row["joint_limit_violations"]
            collision_count += len(row["unexpected_pairs"])
            if row["joint_limit_violations"] or row["unexpected_pairs"] or row["done"]:
                assert index == len(steps) - 1, "Execution continued after episode stop"
            assert math.isfinite(row["reward"])
            rewards.append(row["reward"])
            successes.append(row["success"])
        actual = {"seed": seed, "rollout_length": len(steps), "success": any(successes),
                  "maximum_reward": max(rewards, default=0.), "joint_limit_violations": limit_count,
                  "unexpected_collisions": collision_count, "raw_clipped_fields": clips}
        assert all(episode[key] == value for key, value in actual.items())
        expected_outcome = ("physical_safety_stop" if limit_count or collision_count
                            else "task_success" if actual["success"] else "task_failed")
        assert episode["outcome"] == expected_outcome
        if expected_outcome != "physical_safety_stop":
            assert len(steps) == 500 or steps[-1]["done"]
        assert episode == next(e for e in summary["episodes"] if e["seed"] == seed)
        independent.append(actual)
    assert summary["success_count"] == sum(row["success"] for row in independent)
    return {"status": "passed", "plan_sha256": sha(plan_path),
            "completion_sha256": sha(directory / "completion.json"), "independent_episodes": independent,
            "registered_seed_count": 5, "actual_success_count": sum(row["success"] for row in independent),
            "formal_gate_result": None, "shutdown_requested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "plan", "contract"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.plan, args.contract), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
