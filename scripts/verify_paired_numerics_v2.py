"""Independent full-row command/deviation/recovery arithmetic, no ML imports.

This supplements the collection's saved-file audit. It never changes labels or
claims teacher qualification, legacy-scene isolation, or policy success.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(value, dimensions, name):
    require(isinstance(value, list) and len(value) == dimensions, name + " shape")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in value), name + " finite")
    return value


def verify_attempt(rows, commands, trace, attempt, registered, dimensions, plan):
    """All rows, including failures; only observed recovery claims get admission checks."""
    names = [d["name"] for d in dimensions]
    count = len(names)
    require(registered["joint_name"] in names, "unregistered joint")
    joint = names.index(registered["joint_name"])
    require(len(rows) == len(trace) == attempt["rows"], "trace/label row count")
    # A failed env.step may leave exactly one pre-written command without a row.
    require(len(rows) <= len(commands) <= len(rows) + 1, "command denominator")
    deviations = []
    clipped_steps = []
    for step, row in enumerate(rows):
        require(row["step"] == trace[step]["step"] == commands[step]["step"] == step,
                "nonconsecutive step")
        for key in ("action", "state", "action_source", "images"):
            require(row[key] == trace[step][key] == commands[step][key], "command/trace/label " + key)
        for key in ("before", "after", "phase", "safe", "images_sha256", "pts_seconds"):
            require(row[key] == trace[step][key], "trace/label " + key)
        before = vector(row["before"]["robot_state"], count, "before state")
        after = vector(row["after"]["geometry"]["robot_state"], count, "after state")
        action = vector(row["action"], count, "command")
        require(row["state"] == before, "observation state alignment")
        require(row["pts_seconds"] == step / 50, "timestamp alignment")
        for dim, value in zip(dimensions, action):
            require(dim["minimum"] <= value <= dim["maximum"], "command outside unclipped contract")
        info = row["after"]["info"]
        require("clipped_fields" in info and isinstance(info["clipped_fields"], list), "missing actual clip receipt")
        if info["clipped_fields"]:
            clipped_steps.append(step)
        safe = not (row["after"]["joint_limits"] or row["after"]["unexpected_collisions"] or info["clipped_fields"])
        require(row["safe"] is safe, "physical safety receipt disagreement")
        if row["action_source"] == "registered_deliberate_deviation":
            deviations.append(step)
            dev = attempt["deviation"]
            require(attempt["kind"] == "R" and isinstance(dev, dict), "deviation outside R")
            require(step == dev["step"] and row["phase"] == dev["trigger"] == registered["trigger"], "deviation phase/index")
            require(dev["joint"] == registered["joint_name"] and dev["duration_steps"] == 1, "deviation joint/duration")
            require(dev["delta_rad"] == registered["joint_delta_rad"] and dev["pre_state"] == before, "deviation registration/state")
            # Float32 addition has rounding; bound by a single rounded operation.
            tolerance = 2e-7 * max(1.0, abs(before[joint]))
            require(abs((action[joint] - before[joint]) - registered["joint_delta_rad"]) <= tolerance,
                    "registered delta differs from actual command")
            require(all(action[i] == before[i] for i in range(count) if i != joint), "extra deliberate joint change")
            movement = abs(after[joint] - before[joint])
            if safe:
                require(dev["observed_joint_change_rad"] == movement, "summary joint movement not actual")
                require(dev["verified"] is (movement >= plan["deviation_min_joint_change_rad"]), "movement threshold disagreement")
                require(attempt["correction_start"] == step + 1, "correction start")
        else:
            require(row["action_source"] == "state_conditioned_teacher", "unknown action provenance")
    require(len(deviations) <= 1, "more than one deviation action")
    require(not clipped_steps or attempt["safety_stop"], "clipping absent from failure summary")
    if attempt["kind"] == "S":
        require(not deviations and attempt["deviation"] is None, "S deliberate action")
    if attempt["correction_start"] is not None:
        require(len(rows) - attempt["correction_start"] <= plan["correction_max_steps"], "correction budget expanded")
    recovery = attempt["recovery"]
    if recovery is not None:
        end = recovery["step"]
        start = end - plan["recovery_sustain_steps"] + 1
        require(plan["recovery_sustain_steps"] == 5 and recovery["sustain_steps"] == 5, "registered five-step sustain")
        require(type(start) is int and type(end) is int and attempt["correction_start"] is not None
                and 0 <= attempt["correction_start"] <= start <= end < len(rows), "recovery window bounds")
        window = rows[start:end + 1]
        require(len(window) == 5 and [r["step"] for r in window] == list(range(start, end + 1)), "recovery window length")
        for row in window:
            g = row["after"]["geometry"]
            require(row["safe"] is True and row["action_source"] == "state_conditioned_teacher", "unsafe/nonexpert recovery")
            require(g["socket_grasp_contact"] is True and g["peg_grasp_contact"] is True, "recovery contact lost")
            if registered["trigger"] == "align":
                require(g["peg_socket_contact"] is True, "align contact lost")
    successes = [r["step"] for r in rows if r["after"]["info"].get("is_success") is True]
    require(attempt["task_success"] is bool(successes), "task success disagreement")
    if attempt["verified_recovery_chain"]:
        require(len(deviations) == 1 and recovery is not None and successes
                and recovery["step"] < successes[-1] and attempt["deviation"]["verified"] is True
                and not attempt["safety_stop"], "unverified recovery chain")
    return {"rows_verified": len(rows), "commands_recorded": len(commands),
            "uncompleted_commands": len(commands)-len(rows), "deviation_steps": deviations,
            "clip_steps": clipped_steps, "recovery_window_verified": recovery is not None}


def verify(root, contract_path):
    import yaml  # Official registered worker environment; no model/data package imports.
    root = root.resolve()
    manifest_bytes = (root/"manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    inventory = {entry["path"]: entry for entry in manifest["files"]}
    require(len(inventory) == len(manifest["files"]), "duplicate manifest member")

    def payload(relative):
        part = PurePosixPath(relative)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in relative and ":" not in relative,
                "unsafe evidence path")
        require(relative in inventory, "unbound evidence payload")
        target = root.joinpath(*part.parts).resolve()
        require(target.is_relative_to(root), "evidence escape")
        raw = target.read_bytes()
        entry = inventory[relative]
        require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"], "payload identity changed")
        return raw

    plan = json.loads(payload("registration.json"))
    require(hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            == manifest["plan_sha256"], "plan identity")
    summary_raw = payload("summary.json")
    summary = json.loads(summary_raw)
    contract_raw = contract_path.read_bytes()
    contract_sha = hashlib.sha256(contract_raw).hexdigest()
    contract = yaml.safe_load(contract_raw)
    require(contract["action"]["frequency_hz"] == 50 and contract["action"]["semantics"] == "absolute_joint_position_targets",
            "contract semantics/frequency")
    registered = {p["pair_id"]: p for p in plan["pairs"]}
    expected = {p+"-"+role for p in registered for role in ("S", "R")}
    require(len(summary["attempts"]) == len(expected) and {a["attempt_id"] for a in summary["attempts"]} == expected,
            "registered attempt denominator")
    results = []
    for attempt in summary["attempts"]:
        require(attempt["contract_sha256"] == contract_sha, "contract bytes identity")
        prefix = attempt["pair_id"]+"/"+attempt["kind"]+"/"
        if prefix+"labels.json" not in inventory:
            require(attempt["rows"] == 0, "missing completed rows")
            results.append({"attempt_id": attempt["attempt_id"], "rows_verified": 0, "unstarted": True})
            continue
        rows = json.loads(payload(prefix+"labels.json"))["rows"]
        streams = []
        for name in ("commands.jsonl", "trace.jsonl"):
            streams.append([json.loads(line) for line in payload(prefix+name).splitlines()] if prefix+name in inventory else [])
        result = verify_attempt(rows, *streams, attempt, registered[attempt["pair_id"]], contract["action"]["dimensions"], plan)
        results.append({"attempt_id": attempt["attempt_id"], **result})
    return {"schema": "rosetta.paired_independent_numerics.v1", "passed": True,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_raw).hexdigest(), "contract_sha256": contract_sha,
            "registered_attempts": len(expected), "attempts": results,
            "rows_verified": sum(r["rows_verified"] for r in results),
            "legacy_object_scene_nonoverlap": "not_established", "teacher_qualified": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.root, args.contract)
    except Exception as error:
        result = {"schema": "rosetta.paired_independent_numerics.v1", "passed": False,
                  "error_type": type(error).__name__, "reason": str(error)}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
