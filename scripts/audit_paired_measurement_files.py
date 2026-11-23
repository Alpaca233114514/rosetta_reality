"""Read-only paired measurement audit. Standard library; no model/physics imports.

Integrity/count agreement is separate from evidence sufficiency. Missing physical
state, noise or geometry is never reconstructed from seeds or logical actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def safe(root, name):
    relative = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or relative.is_absolute()
            or ".." in relative.parts):
        raise ValueError("Unsafe evidence path")
    target = (Path(root) / relative).resolve(strict=True)
    if not target.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError("Evidence escaped root")
    return target


def verify_inventory(root, manifest):
    checked = set()
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        canonical = PurePosixPath(name).as_posix()
        if canonical in checked or len(expected) != 64:
            raise ValueError("Duplicate or invalid inventory entry")
        path = safe(root, name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Inventory hash mismatch: " + canonical)
        checked.add(canonical)
    actual = {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()}
    if not checked or checked != actual:
        raise ValueError("Inventory is not the exact received regular-file set")
    return {"files_verified": len(checked), "manifest_sha256": sha(manifest)}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def vector(value, size):
    if (not isinstance(value, list) or len(value) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError("Missing/nonfinite/wrong-width action or observation")
    return value


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid nonnegative count")
    return value


def coverage(rows, fields):
    return {key: {"present_nonnull": sum(key in r and r[key] is not None for r in rows),
                  "expected": len(rows)} for key in fields}


def audit_trace(rows, contract, *, bank=False):
    """Recompute saved observations, not physical safety from 14-D policy state."""
    dimensions = contract["dimensions"]
    size = len(dimensions)
    limits = collisions = clips = 0
    success = False
    maximum_reward = None
    first_limit = first_collision = None
    clip_events, steps = [], []
    previous = None
    stopped = False
    reset = None
    streak = 0
    recovery_step = terminal_step = None
    recovery_event = None
    recovery_criterion = None
    for index, row in enumerate(rows):
        if not bank and index == 0:
            if row.get("event") != "reset":
                raise ValueError("Five-scene trace lacks reset event")
            reset = row
        else:
            if not bank and row.get("event") != "step":
                raise ValueError("Unexpected trace event")
            if stopped or row.get("step") != len(steps) + 1 or type(row.get("step")) is not int:
                raise ValueError("Step gap, duplicate or continued execution after stop")
            vector(row["executed_action"], size)
            vector(row["state_after"], size)
            if not bank:
                vector(row["state_before"], size)
                if previous is not None and row["state_before"] != previous:
                    raise ValueError("Adjacent state observations disagree")
                raw = vector(row["raw_action"], size)
                processed = vector(row["processed_action"], size)
                expected = [max(f32(d["minimum"]), min(f32(d["maximum"]), v))
                            for d, v in zip(dimensions, raw)]
                if processed != expected or row["executed_action"] != expected:
                    raise ValueError("Raw/projected/logical command chain mismatch")
                changed = [i for i, (a, b) in enumerate(zip(raw, expected)) if a != b]
                if integer(row["raw_clipped_fields"]) != len(changed):
                    raise ValueError("Clip count disagrees with full logical action")
                if changed:
                    clip_events.append({"step": row["step"], "dimensions": [
                        {"name": dimensions[i]["name"], "raw": raw[i],
                         "projected": processed[i], "executed": row["executed_action"][i],
                         "state_after": row["state_after"][i],
                         "minimum": f32(dimensions[i]["minimum"]),
                         "maximum": f32(dimensions[i]["maximum"])} for i in changed]})
            clip_key = "raw_clips" if bank else "raw_clipped_fields"
            clips += integer(row[clip_key])
            flag = row["task_success" if bank else "success"]
            if type(flag) is not bool or type(row["reward"]) not in (float, int) or not math.isfinite(row["reward"]):
                raise ValueError("Invalid task observation")
            success |= flag
            maximum_reward = row["reward"] if maximum_reward is None else max(maximum_reward, row["reward"])
            previous = row["state_after"]
            steps.append(row)
            if len(steps) > (150 if bank else 500):
                raise ValueError("Registered step budget exceeded")
        count = integer(row["joint_limit_violations"])
        pairs = row["unexpected_contacts" if bank else "unexpected_pairs"]
        if not isinstance(pairs, list) or any(not isinstance(p, list) or len(p) != 2 or any(not isinstance(n, str) for n in p) for p in pairs):
            raise ValueError("Invalid contact pair observation")
        limits += count
        collisions += len(pairs)
        if count and first_limit is None:
            first_limit = row
        if pairs and first_collision is None:
            first_collision = row
        stopped = bool(count or pairs or row.get("done", False))
        if bank:
            event = row["recovery_event"]
            criterion = event["criterion"]
            if recovery_criterion is not None and criterion != recovery_criterion:
                raise ValueError("Recovery criterion changed within episode")
            recovery_criterion = criterion
            if criterion not in ("both_grasps", "both_grasps_and_peg_socket_contact"):
                raise ValueError("Unknown recovery criterion")
            names = ["socket_grasp_contact", "peg_grasp_contact"]
            if criterion == "both_grasps_and_peg_socket_contact":
                names.append("peg_socket_contact")
            if any(type(row["geometry"].get(name)) is not bool for name in names):
                raise ValueError("Missing contact evidence")
            streak = streak + 1 if not (count or pairs) and all(row["geometry"][n] for n in names) else 0
            if streak >= 5 and recovery_step is None:
                recovery_step = row["step"]
            if row["task_success"] and terminal_step is None:
                terminal_step = row["step"]
            recovery_event = {"criterion": criterion, "required_consecutive_steps": 5,
                "correction_steps_observed": row["step"], "recovery_step": recovery_step,
                "geometric_recovery": recovery_step is not None, "terminal_success_step": terminal_step,
                "verified_recovery_chain": bool(not (limits or collisions) and recovery_step is not None
                    and terminal_step is not None and recovery_step < terminal_step)}
            if event != recovery_event:
                raise ValueError("Recovery event disagrees with actual saved contacts")
    if not rows or (not steps and not stopped):
        raise ValueError("Empty or unexplained zero-step trace")
    return {
        "steps": len(steps), "success": success, "maximum_reward": maximum_reward,
        "joint_limit_violations": limits, "unexpected_contacts": collisions,
        "raw_clipped_fields": clips, "first_limit": first_limit,
        "first_collision": first_collision, "clip_events": clip_events,
        "last_step": steps[-1] if steps else None,
        "initial_logical_state": steps[0].get("state_before") if steps else None,
        "logical_action_chain_exact": True if steps and not bank else None,
        "physical_joint_attribution": "not_captured",
        "recovery_event": recovery_event,
        "coverage": coverage(steps, ["raw_action", "processed_action", "executed_action",
            "state_before", "state_after", "geometry", "physics_before", "physics_after",
            "noise", "processor_identity", "full_predicted_chunk"]),
        "reset_coverage": coverage([reset] if reset is not None else [{}],
            ["integration_state", "qpos", "qvel", "object_poses", "observation_hashes"]),
    }


def load_trace(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank trace record")
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(row, dict):
                raise ValueError("Trace record must be an object")
            rows.append(row)
    return rows


def audit_arm(root, arm):
    directory = root / ("measurement-" + arm)
    plan_path = directory / "registration.json"
    plan = read(plan_path)
    summary = read(directory / "summary.json")
    if (summary["plan_sha256"] != sha(plan_path) or plan != read(root / f"evaluation-plan-{arm}.json")
            or plan["arm"] != arm or plan["seeds"] != list(range(1000, 1005))
            or summary["optimizer_updates"] != 0 or summary["formal_gate_result"] is not None
            or summary["original_gate4_protocol_equivalent"] is not False
            or summary["status"] != "completed" or summary["error_type"] is not None):
        raise ValueError("Plan/summary identity or interpretation disagreement")
    artifact = root / "artifacts" / arm
    manifest = read(artifact / "manifest.json")
    if sha(artifact / "manifest.json") != plan["artifact_manifest_sha256"]:
        raise ValueError("Artifact manifest identity mismatch")
    for name in ("config.json", "normalization.json", "action_contract.json"):
        if sha(artifact / name) != manifest["files"][name]:
            raise ValueError("Returned artifact payload mismatch")
    if sha(artifact / "normalization.json") != plan["D_normalization_sha256"]:
        raise ValueError("Normalization mismatch")
    contract = read(artifact / "action_contract.json")
    episodes = []
    for saved in summary["episodes"]:
        seed = saved["seed"]
        path = directory / f"seed-{seed}.jsonl"
        if sha(path) != saved["trace_sha256"] or read(directory / f"seed-{seed}.json") != saved:
            raise ValueError("Episode trace/summary binding mismatch")
        rows = load_trace(path)
        if any(r.get("seed") != seed for r in rows):
            raise ValueError("Scene identity differs inside trace")
        result = audit_trace(rows, contract)
        mapping = {"rollout_length": "steps", "maximum_reward": "maximum_reward", "success": "success",
                   "joint_limit_violations": "joint_limit_violations", "unexpected_collisions": "unexpected_contacts",
                   "raw_clipped_fields": "raw_clipped_fields"}
        if any(saved[key] != result[value] for key, value in mapping.items()):
            raise ValueError("Episode arithmetic mismatch")
        unsafe = bool(result["joint_limit_violations"] or result["unexpected_contacts"])
        expected_outcome = "physical_safety_stop" if unsafe else "task_success" if result["success"] else "task_failed"
        if saved["outcome"] != expected_outcome or (not unsafe and not result["last_step"]["done"] and result["steps"] != plan["maximum_steps"]):
            raise ValueError("Episode ending disagrees with trace")
        episodes.append({"seed": seed, "outcome": saved["outcome"], "trace_sha256": sha(path), **result})
    if ([e["seed"] for e in episodes] != plan["seeds"] or summary["registered_scenes"] != 5
            or summary["measured_scenes"] != len(episodes) or summary["unmeasured_seeds"] != []
            or summary["success_count"] != sum(e["success"] for e in episodes)):
        raise ValueError("Five-scene denominator mismatch")
    bank = read(directory / "recovery-bank" / "bank-summary.json")
    bank_results = []
    ids = set()
    for saved in bank["rows"]:
        identity = saved["pair_id"]
        if identity in ids or identity not in {f"bank-003b-scene-{s}" for s in range(2440, 2450)}:
            raise ValueError("Duplicate/unregistered bank state")
        ids.add(identity)
        path = directory / "recovery-bank" / (identity + ".jsonl")
        if saved["status"] == "unmeasured_branch_not_created":
            if path.exists() or saved["measured_steps"] != 0 or saved["branch_task_success"] is not None:
                raise ValueError("Uncreated branch mislabeled")
            bank_results.append({"pair_id": identity, "measured": False, "status": saved["status"]})
            continue
        result = audit_trace(load_trace(path), contract, bank=True)
        if (result["steps"] != saved["measured_steps"] or result["success"] is not saved["branch_task_success"]
                or result["recovery_event"] != saved["recovery_event"]):
            raise ValueError("Bank arithmetic mismatch")
        unsafe = bool(result["joint_limit_violations"] or result["unexpected_contacts"])
        if unsafe != (saved["status"] == "physical_safety_stop"):
            raise ValueError("Bank safety status mismatch")
        bank_results.append({"pair_id": identity, "measured": True, "teacher_feasible": saved["teacher_feasible"],
                             "status": saved["status"], "trace_sha256": sha(path), **result})
    if len(ids) != 10 or bank["registered_states"] != 10 or bank["measured_states"] != sum(r["measured"] for r in bank_results):
        raise ValueError("Bank denominator mismatch")
    if bank["branch_task_success_count"] != sum(r.get("success", False) for r in bank_results):
        raise ValueError("Bank success aggregate mismatch")
    for key in ("geometric_recovery", "verified_recovery_chain"):
        if bank[key + "_count"] != sum(bool((r.get("recovery_event") or {}).get(key)) for r in bank_results):
            raise ValueError("Bank recovery aggregate mismatch")
    probes = [read(root / f"probe-{arm}-{n}.json") for n in (1, 2)]
    proof = read(root / f"reload-proof-{arm}.json")
    keys = ("arm", "model_sha256", "parameter_before", "parameter_after", "raw", "projected", "optimizer_updates", "noise", "scene_seed")
    if (any(probes[0][k] != probes[1][k] for k in keys) or probes[0]["pid"] == probes[1]["pid"]
            or proof["status"] != "passed" or proof["full_action_array_exact"] is not True
            or probes[0]["parameter_before"] != probes[0]["parameter_after"]
            or probes[0]["model_sha256"] != plan["model_sha256"]
            or manifest["selected_checkpoint_model_sha256"] != plan["model_sha256"]
            or manifest["selected_checkpoint_step"] != 5000
            or manifest["reload"]["proof_sha256"] != sha(root / f"reload-proof-{arm}.json")
            or any(proof[f"probe_{n}_sha256"] != sha(root / f"probe-{arm}-{n}.json") for n in (1, 2))):
        raise ValueError("Independent reload proof mismatch")
    integrity = read(directory / "parameter-integrity.json")
    if integrity["before"] != integrity["after"] or integrity["unchanged"] is not True:
        raise ValueError("Parameter digest changed")
    return {"arm": arm, "episodes": episodes, "bank": bank_results,
            "source_sha256": plan["source_sha256"],
            "processor_file_identities": {k: v for k, v in manifest["files"].items() if "processor" in k},
            "normalization_sha256": plan["D_normalization_sha256"],
            "reload_scope": {"scene_seed": probes[0]["scene_seed"], "noise": probes[0]["noise"],
                             "full_array_equal": True, "rollout_noise_parity_proven": False}}


def audit(root, manifest):
    root = Path(root)
    inventory = verify_inventory(root, manifest)
    arms = [audit_arm(root, arm) for arm in "ABC"]
    return {"schema": "rosetta.paired_measurement_file_audit.v1", "integrity_and_counts_passed": True,
            "evidence_sufficient_for_causal_attribution": False, "inventory": inventory, "arms": arms,
            "model_or_simulator_executed": False, "formal_gate_result": None,
            "limitations": ["Logical state is not full physical qpos", "No actual full noise tensors",
                            "No per-step effective processor identity", "No five-scene grasp/object geometry",
                            "No bank reset/restore event", "No physical joint name/value/range at limit event",
                            "No actuator command/ctrl snapshots", "First arm difference is not first error"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("Audit output must be outside the immutable received-file root")
    result = audit(args.root, args.manifest)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"integrity_and_counts_passed": True, "files": result["inventory"]["files_verified"],
                      "evidence_sufficient_for_causal_attribution": False}))


if __name__ == "__main__":
    main()
