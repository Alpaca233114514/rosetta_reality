"""Data adapters. Never import or execute code from the source repository."""

import math
from pathlib import Path

from .io import child, loads, outside, read_bytes, sha


def _artifact(root, name):
    return read_bytes(child(root, name))


def _bundle(root, names):
    manifest_bytes = _artifact(root, "manifest.json")
    manifest = loads(manifest_bytes)
    if set(manifest["files"]) != set(names):
        raise ValueError("Unexpected source manifest file set")
    artifacts = {"manifest.json": manifest_bytes}
    for name in names:
        data = _artifact(root, name)
        spec = manifest["files"][name]
        digest = spec["sha256"] if isinstance(spec, dict) else spec
        if sha(data) != digest:
            raise ValueError(f"Source digest mismatch: {name}")
        if isinstance(spec, dict) and spec.get("bytes", len(data)) != len(data):
            raise ValueError(f"Source byte count mismatch: {name}")
        artifacts[name] = data
    return manifest, artifacts


def _vector(value, dimension):
    if not isinstance(value, list) or len(value) != dimension or not all(
        type(v) in (int, float) and math.isfinite(v) for v in value
    ):
        raise ValueError("Action/state vector has invalid dimension or non-finite values")


def import_trace(store, repo, relative, run_id, *, gate="diagnostic"):
    """Rosetta rollout_trace schema v1; hash + execution sequence verification."""
    outside(store.root, repo)
    root = child(repo, relative)
    manifest, artifacts = _bundle(root, ("identity.json", "summary.json", "trace.jsonl"))
    identity, summary = (loads(artifacts[n]) for n in ("identity.json", "summary.json"))
    if any(item.get("schema_version") != 1 for item in (manifest, identity, summary)):
        raise ValueError("Unsupported Rosetta trace schema")
    status = summary["status"]
    if status not in ("complete", "incomplete") or manifest["status"] != status:
        raise ValueError("Invalid trace completion status")
    dimensions = identity["dimension_names"]
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("Missing or duplicate action dimension names")
    seed = identity["options"]["seed"]
    events, phase, predictions, started = [], "prediction", 0, 0
    prediction = before = last = None
    for line_number, line in enumerate(artifacts["trace.jsonl"].splitlines(), 1):
        row = loads(line)
        if (row["schema_version"] != 1 or row["seed"] != seed
                or row["episode_index"] != identity["episode_index"] or row["event"] != phase):
            raise ValueError(f"Trace identity/order mismatch at line {line_number}")
        if phase == "prediction":
            if row["prediction"] != predictions or (last and last["done"]):
                raise ValueError("Non-contiguous predictions or execution after done")
            _vector(row["postprocessed_first_action"], len(dimensions))
            _vector(row["decoded_first_action"], len(dimensions))
            if len(row["internal_grippers"]) != identity["chunk_length"]:
                raise ValueError("Chunk length mismatch")
            for pair in row["internal_grippers"]:
                _vector(pair, len(identity["gripper_indices"]))
            prediction = row
            predictions += 1
            phase = "step_started"
        elif phase == "step_started":
            if row["step"] != started or row["prediction"] != predictions - 1:
                raise ValueError("Unpaired step start")
            _vector(row["state_before"], len(dimensions))
            _vector(row["executed_action"], len(dimensions))
            if last and row["state_before"] != last["state_after"]:
                raise ValueError("State continuity mismatch")
            before = row
            started += 1
            phase = "step"
        else:
            if any(row.get(k) != v for k, v in before.items() if k != "event"):
                raise ValueError("Step changed after execution start")
            _vector(row["state_after"], len(dimensions))
            if type(row["reward"]) not in (int, float) or not math.isfinite(row["reward"]):
                raise ValueError("Invalid reward")
            if any(type(row[k]) is not bool for k in ("success", "done", "terminated", "truncated")):
                raise ValueError("Invalid step outcome")
            events.append({"step": row["step"], "stage": "execution", "values": {
                "action": row["executed_action"], "state_before": row["state_before"],
                "state_after": row["state_after"], "reward": row["reward"],
                "success": row["success"], "input_identity": prediction["input_identity"],
                "predicted_action": prediction["postprocessed_first_action"],
                "physics_after": row.get("physics_after")},
                "evidence": {"artifact": "trace.jsonl", "line": line_number,
                             "prediction_line": line_number - 2}})
            last, phase = row, "prediction"
    if (summary["predictions"] != predictions or summary["steps_started"] != started
            or summary["steps_recorded"] != len(events)
            or not len(events) <= summary["steps_completed"] <= started):
        raise ValueError("Trace counters mismatch")
    if status == "complete":
        metrics = summary["metrics"]
        if (not events or phase != "prediction" or summary["error_type"] is not None
                or summary["steps_completed"] != len(events)
                or metrics["rollout_length"] != len(events)
                or metrics["maximum_reward"] != max(e["values"]["reward"] for e in events)
                or metrics["success"] != any(e["values"]["success"] for e in events)):
            raise ValueError("Complete outcome does not match recorded steps")
    elif not summary.get("error_type"):
        raise ValueError("Incomplete trace lacks failure reason")
    source = identity.get("identity", {})
    record = {"schema_version": 1, "id": run_id, "adapter": "rosetta.rollout_trace.v1",
              "source_run": source.get("run_id"), "source_path": Path(relative).as_posix(),
              "evidence_kind": "historical_observation", "status": status,
              "gate": {"label": gate, "acceptance": "not_inferred_from_episode"},
              "parameters": identity, "dimensions": dimensions,
              "verification": {"source_hashes": "verified", "sequence": "verified",
                               "support_arithmetic": "not_recomputed",
                               "checkpoint_bytes": "not_verified"},
              "limitations": ["Historical capture only; no new model or simulator execution.",
                              "Missing tensors cannot be recovered from this trace."]}
    return store.add(record, events, artifacts)


def import_gate(store, repo, relative, run_id):
    outside(store.root, repo)
    data = _artifact(repo, relative)
    report = loads(data)
    if report.get("schema_version") != 1 or not all(k in report for k in ("run", "gate3", "gate4", "episodes")):
        raise ValueError("Unsupported Rosetta Gate report")
    record = {"schema_version": 1, "id": run_id, "adapter": "rosetta.gate_report.v1",
              "source_run": report["run"], "source_path": Path(relative).as_posix(),
              "evidence_kind": "reported_result", "status": report.get("status", "unknown"),
              "gate": {"gate3": report["gate3"], "gate4": report["gate4"],
                       "m2_complete": report.get("m2_complete")},
              "parameters": report, "dimensions": [],
              "verification": {"source_hashes": "computed_at_import_not_upstream_verified"},
              "limitations": ["Preserves reported Gate outcomes; does not independently certify acceptance."]}
    return store.add(record, [], {"gate-report.json": data})


def import_native(store, source, run_id):
    """Backend-neutral JSON envelope for hooks/NNsight/other collectors."""
    source = Path(source).resolve()
    # For a single file only, prevent writing over that file through the store.
    if source.is_relative_to(store.root):
        raise ValueError("Native input must be outside the history store")
    data = read_bytes(source)
    value = loads(data)
    if value.get("schema_version") != 1 or value.get("evidence_kind") not in (
        "synthetic", "historical_observation", "live_observation"
    ):
        raise ValueError("Unsupported Basin native envelope")
    events, keys = value["events"], set()
    for index, event in enumerate(events):
        key = (event["step"], event["stage"])
        if type(key[0]) is not int or key[0] < 0 or not isinstance(key[1], str) or key in keys:
            raise ValueError("Duplicate or invalid (step, stage)")
        if not isinstance(event["values"], dict):
            raise ValueError("Event values must be an object")
        keys.add(key)
        event["evidence"] = {"artifact": "native.json", "pointer": f"/events/{index}"}
    record = {"schema_version": 1, "id": run_id, "adapter": "basin.native.v1",
              "source_run": value.get("source_run", run_id), "status": value.get("status", "unknown"),
              "evidence_kind": value["evidence_kind"], "gate": value.get("gate", {}),
              "parameters": value.get("parameters", {}), "dimensions": value.get("dimensions", []),
              "verification": {"source_hashes": "computed_at_import_not_upstream_verified"},
              "limitations": ["Collector claims are retained as data, not independently certified."]}
    return store.add(record, events, {"native.json": data})
