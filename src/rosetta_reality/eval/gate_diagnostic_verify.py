"""Independent NumPy verifier. Does not import recorder, model, or metric producer."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

_DTYPES = {
    "float16": "<f2",
    "float32": "<f4",
    "float64": "<f8",
    "uint8": "u1",
    "int8": "i1",
    "int16": "<i2",
    "int32": "<i4",
    "int64": "<i8",
    "bool": "?",
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root, name):
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
        raise ValueError("Unsafe evidence member")
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Unsafe evidence link")
    return target


def arrays(directory):
    """Independent typed-tree decoder; BF16 is converted exactly to float32."""
    used = set()
    with np.load(Path(directory) / "arrays.npz", allow_pickle=False) as archive:

        def read(node):
            kind = node["type"]
            if kind == "tensor":
                key = node["array"]
                if key in used:
                    raise ValueError("Tensor member reused")
                used.add(key)
                raw = archive[key]
                if raw.dtype != np.uint8 or raw.ndim != 1:
                    raise ValueError("Tensor bytes invalid")
                dtype = node["dtype"].removeprefix("torch.")
                if dtype == "bfloat16":
                    values = (raw.view("<u2").astype("<u4") << 16).view("<f4")
                elif dtype in _DTYPES:
                    values = raw.view(_DTYPES[dtype])
                else:
                    raise ValueError("Unsupported tensor dtype")
                shape = node["shape"]
                if any(type(i) is not int or i < 0 for i in shape):
                    raise ValueError("Invalid tensor shape")
                values = values.reshape(shape)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite tensor")
                return values.copy()
            if kind == "dict":
                return {k: read(v) for k, v in node["items"].items()}
            if kind in ("tuple", "list"):
                value = [read(v) for v in node["items"]]
                return tuple(value) if kind == "tuple" else value
            if kind != "scalar":
                raise ValueError("Invalid tree node")
            value = node["value"]
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite scalar")
            return value

        tree = _json(Path(directory) / "tree.json")
        result = read(tree)
        if set(archive.files) != used:
            raise ValueError("Unreferenced tensor bytes")
        return result


def _equal(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _type_tree(node):
    if node["type"] == "tensor":
        return {k: node[k] for k in ("type", "shape", "dtype")}
    if node["type"] == "dict":
        return {k: _type_tree(v) for k, v in node["items"].items()}
    if node["type"] in ("list", "tuple"):
        return (node["type"], [_type_tree(v) for v in node["items"]])
    return node


def _expected_probe_steps(source):
    rows = [json.loads(line) for line in (source / "trace/trace.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["event"] == "step"]
    chosen, seen = {0}, set()
    prior, reward = None, 0
    for i, row in enumerate(rows):
        snap = row["physics_after"]
        raw = (snap.get("value") or {}).get("contacts")
        current = None
        if snap["available"] and raw is not None:
            current = {
                tuple(sorted(pair))
                for pair in raw
                if any(x.startswith("vx300s_") for x in pair)
                and any(x == "red_peg" or x.startswith("socket-") for x in pair)
            }
        flags = [
            bool(current),
            prior is not None and current is not None and bool(prior - current),
            row["reward"] != reward,
        ]
        for event, occurs in enumerate(flags):
            if occurs and event not in seen:
                chosen.update(x for x in (i - 1, i, i + 1) if 0 <= x < len(rows))
                seen.add(event)
        prior, reward = current, row["reward"]
    selected = sorted(chosen)[:12]
    grid = [round(i * (len(rows) - 1) / 11) for i in range(12)]
    for i in grid + list(range(len(rows))):
        if len(selected) == 12:
            break
        if i not in selected:
            selected.append(i)
    return sorted(selected)


def verify_bundle(directory, *, source=None):
    directory = Path(directory)
    manifest = _json(directory / "manifest.json")
    members = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Evidence links are forbidden")
    if members != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Evidence file set differs")
    for name, spec in manifest["files"].items():
        path = _resolve(directory, name)
        if path.stat().st_size != spec["bytes"] or _sha(path) != spec["sha256"]:
            raise ValueError("Evidence checksum mismatch: " + name)
    result = _json(directory / "result.json")
    identity = _json(directory / "identity.json")
    if result["stage"] != identity["stage"] or result["status"] != manifest["status"]:
        raise ValueError("Stage or status mismatch")
    if result.get("diagnostic_only") is not True or result.get("optimizer_steps") != 0:
        raise ValueError("Diagnostic scope changed")
    if result["status"] not in ("complete", "incomplete", "control_failed"):
        raise ValueError("Unknown completion state")
    if result["stage"] == "collect":
        _collection(directory, identity, result)
    elif result["stage"] in ("replay", "probe"):
        if source is None:
            raise ValueError("Replay/probe verification requires --source collection")
        _replay(directory, Path(source), identity, result)
    elif result["stage"] not in ("audit-training", "analyze"):
        raise ValueError("Unknown stage")
    return {
        "stage": result["stage"],
        "status": "verified_" + result["status"],
        "diagnostic_only": True,
    }


def _collection(directory, identity, result):
    from .rollout_trace_verify import verify_trace

    if identity["seed"] != 3 or identity["policy_noise_seed"] != 3:
        raise ValueError("Seed 3 identity changed")
    total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    if total > identity["maximum_bytes"]:
        raise ValueError("Evidence budget exceeded")
    if not (directory / "trace/manifest.json").exists():
        if result["status"] == "complete":
            raise ValueError("Missing complete trace")
        return
    trace = verify_trace(directory / "trace")
    trace_identity = _json(directory / "trace/identity.json")
    if trace_identity["identity"] != identity["identity"]:
        raise ValueError("Trace endpoint binding differs")
    if trace_identity["dimension_names"] != identity["dimension_names"]:
        raise ValueError("Action dimension order differs")
    if trace_identity["options"] != {
        "seed": 3,
        "policy_noise_seed": 3,
        "maximum_steps": identity["maximum_steps"],
        "noise_mode": "seeded_standard_normal",
        "project_policy_output": True,
    }:
        raise ValueError("Rollout protocol differs")
    events = [
        _json_line(line) for line in (directory / "trace/trace.jsonl").read_text().splitlines()
    ]
    predictions = [r for r in events if r["event"] == "prediction"]
    steps = [r for r in events if r["event"] == "step"]
    count = result["predictions_saved"]
    if result["status"] == "complete":
        if (
            trace["status"] != "verified_complete"
            or count != len(steps)
            or count != len(predictions)
            or result.get("parameters_unchanged") is not True
            or count > identity["maximum_steps"]
            or count < 1
        ):
            raise ValueError("Incomplete collection marked complete")
        summary = _json(directory / "trace/summary.json")
        if result["metrics"] != summary["metrics"]:
            raise ValueError("Rollout metrics differ")
        if result["task_success"] != any(bool(r["success"]) for r in steps):
            raise ValueError("Task success differs")
        saved = {p.name for p in (directory / "predictions").iterdir()}
        if saved != {f"{i:04d}" for i in range(count)}:
            raise ValueError("Prediction file set differs from completed count")
    elif result["task_success"] is not None:
        raise ValueError("Incomplete collection claims task outcome")
    for index in range(count):
        row = arrays(directory / "predictions" / f"{index:04d}")
        shape = (identity["chunk_length"], len(identity["dimension_names"]))
        if (
            row["decoded"].shape != shape
            or row["projected"].shape != shape
            or row["internal"].shape != (1, *shape)
            or row["noise"].ndim != 3
            or row["noise"].shape[:2] != (1, shape[0])
        ):
            raise ValueError("Prediction shape differs")
        if index >= len(predictions):
            if result["status"] == "complete":
                raise ValueError("Unpaired prediction")
            continue
        event = predictions[index]
        original = {
            "robot_state": row["observation"]["robot_state"],
            **row["observation"]["images"],
        }
        for name, tensor in original.items():
            spec = event["input_identity"][name]
            if (
                list(tensor.shape) != spec["shape"]
                or hashlib.sha256(tensor.tobytes()).hexdigest() != spec["sha256"]
            ):
                raise ValueError("Original input differs from trace")
        if (
            hashlib.sha256(row["instruction"].encode()).hexdigest()
            != event["input_identity"]["instruction_sha256"]
        ):
            raise ValueError("Instruction differs")
        if not np.array_equal(row["decoded"][0], event["decoded_first_action"]):
            raise ValueError("Decoder boundary differs")
        if not np.array_equal(row["projected"][0], event["postprocessed_first_action"]):
            raise ValueError("Projection boundary differs")
        if index < len(steps) and not np.array_equal(
            row["projected"][0], steps[index]["executed_action"]
        ):
            raise ValueError("Execution boundary differs")


def _json_line(line):
    return json.loads(line)


def _replay(directory, source, identity, result):
    verified = verify_bundle(source)
    if verified != {"stage": "collect", "status": "verified_complete", "diagnostic_only": True}:
        raise ValueError("Source is not a complete collection")
    if identity["source_manifest_sha256"] != _sha(source / "manifest.json"):
        raise ValueError("Replay source binding differs")
    source_identity = _json(source / "identity.json")
    if identity["identity"] != source_identity["identity"]:
        raise ValueError("Replay endpoint differs")
    if identity["dimension_names"] != source_identity["dimension_names"]:
        raise ValueError("Replay action order differs")
    if (
        identity["identity"]["kind"] == "registered_model"
        and identity["pid"] == source_identity["pid"]
    ):
        raise ValueError("Replay process is not independent")
    if result["stage"] == "probe" and (
        result["forwards"] > 60 or len(identity["selected_steps"]) > 12
    ):
        raise ValueError("Probe budget exceeded")
    total_steps = _json(source / "result.json")["predictions_saved"]
    selected = (
        _expected_probe_steps(source) if result["stage"] == "probe" else list(range(total_steps))
    )
    if identity["selected_steps"] != selected:
        raise ValueError("Registered probe/replay selection differs")
    kinds = (
        ["replay", "self_copy", "image", "state", "noise"]
        if result["stage"] == "probe"
        else ["replay"]
    )
    expected_order = [(i, kind) for i in selected for kind in kinds]
    actual_order = [(r["step"], r["kind"]) for r in result["records"]]
    if actual_order != expected_order[: len(actual_order)] or (
        result["status"] == "complete" and actual_order != expected_order
    ):
        raise ValueError("Missing, duplicated, or reordered diagnostic records")
    controls, directories = {}, set()
    measured = 0
    for record in result["records"]:
        index, kind = record["step"], record["kind"]
        reference = arrays(source / "predictions" / f"{index:04d}")
        if record["status"] == "skipped_no_distinct_donor":
            if kind not in ("image", "state", "noise"):
                raise ValueError("Control cannot be skipped")
            donor_index = index - 1 if index else 1
            if donor_index < total_steps:
                donor = arrays(source / "predictions" / f"{donor_index:04d}")
                key = {"image": "images", "state": "robot_state", "noise": "noise"}[kind]
                first = reference if kind == "noise" else reference["observation"]
                second = donor if kind == "noise" else donor["observation"]
                if not _equal(first[key], second[key]):
                    raise ValueError("Distinct donor was incorrectly skipped")
            continue
        measured += 1
        directories.add(record["directory"])
        actual = arrays(_resolve(directory, record["directory"]))
        if kind in ("replay", "self_copy"):
            order = (
                "observation",
                "instruction",
                "batch",
                "noise",
                "normalized",
                "internal",
                "decoded",
                "projected",
            )
            ref_types = _type_tree(_json(source / "predictions" / f"{index:04d}" / "tree.json"))
            got_types = _type_tree(_json(directory / record["directory"] / "tree.json"))
            difference = next(
                (
                    key
                    for key in order
                    if not _equal(reference[key], actual[key]) or ref_types[key] != got_types[key]
                ),
                None,
            )
            if record["first_difference"] != difference:
                raise ValueError("Control mismatch boundary differs")
            if (record["status"] == "exact") != (difference is None):
                raise ValueError("Control status differs")
            if difference is not None and result["status"] == "complete":
                raise ValueError("Failed control marked complete")
            if difference is None:
                controls.setdefault(index, set()).add(kind)
        else:
            if controls.get(index) != {"replay", "self_copy"}:
                raise ValueError("Intervention without both exact controls")
            donor_index = index - 1 if index else 1
            if record["donor_step"] != donor_index:
                raise ValueError("Intervention donor differs")
            donor = arrays(source / "predictions" / f"{donor_index:04d}")
            expected_observation = dict(reference["observation"])
            expected_noise = reference["noise"]
            if kind == "noise":
                expected_noise = donor["noise"]
            elif kind in ("image", "state"):
                key = "images" if kind == "image" else "robot_state"
                expected_observation[key] = donor["observation"][key]
            else:
                raise ValueError("Unknown intervention")
            if (
                not _equal(actual["observation"], expected_observation)
                or not _equal(actual["noise"], expected_noise)
                or actual["instruction"] != reference["instruction"]
            ):
                raise ValueError("Intervention changed more than its single axis")
            expected_groups = {
                f"{side}_{action}/{window}"
                for side in ("left", "right")
                for action in ("joints", "gripper")
                for window in ("first_action", "full_chunk")
            }
            if set(record["deltas"]) != expected_groups or record["status"] != "measured":
                raise ValueError("Missing intervention groups")
            for group, metric in record["deltas"].items():
                side_kind, window = group.split("/")
                side, action_kind = side_kind.split("_")
                indices = [
                    i
                    for i, name in enumerate(identity["dimension_names"])
                    if name.startswith(side + "_")
                    and name.endswith("gripper") == (action_kind == "gripper")
                ]
                difference = np.abs(
                    actual["projected"][:, indices].astype(np.float64)
                    - reference["projected"][:, indices].astype(np.float64)
                )
                if window == "first_action":
                    difference = difference[:1]
                if not math.isclose(
                    metric["mean_absolute_delta"],
                    float(difference.mean()),
                    rel_tol=1e-12,
                    abs_tol=1e-14,
                ) or metric["maximum_absolute_delta"] != float(difference.max()):
                    raise ValueError("Independent probe metric differs")
                expected_unit = "normalized_command" if action_kind == "gripper" else "radian"
                if metric["unit"] != expected_unit:
                    raise ValueError("Probe unit differs")
    if result["status"] in ("complete", "control_failed") and measured != result["forwards"]:
        raise ValueError("Forward accounting differs")
    if result["status"] == "complete":
        required = {"replay", "self_copy"} if result["stage"] == "probe" else {"replay"}
        if (
            set(controls) != set(identity["selected_steps"])
            or any(v != required for v in controls.values())
            or result.get("parameters_unchanged") is not True
        ):
            raise ValueError("Missing completed replay controls")
    if result["status"] == "control_failed":
        if not result["records"] or result["records"][-1]["status"] != "mismatch":
            raise ValueError("Missing failed control")
    if result["status"] in ("complete", "control_failed"):
        persisted = {
            p.parent.relative_to(directory).as_posix()
            for p in (directory / "predictions").rglob("tree.json")
        }
        if persisted != directories:
            raise ValueError("Unreferenced prediction evidence")
