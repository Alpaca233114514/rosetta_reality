"""Exact replay and bounded, non-executed one-variable input interventions."""

from __future__ import annotations

import os
from pathlib import Path

from .gate_diagnostic_capture import capture_prediction, clone_tree
from .gate_diagnostic_io import load_json, read_tree, seal, sha, write_json, write_tree

BOUNDARIES = (
    "observation",
    "instruction",
    "batch",
    "noise",
    "normalized",
    "internal",
    "decoded",
    "projected",
)


def exact(first, second):
    import torch

    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and first.shape == second.shape
            and torch.equal(first.cpu(), second.cpu())
        )
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(exact(v, second[k]) for k, v in first.items())
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(exact(a, b) for a, b in zip(first, second))
    return first == second


def first_difference(reference, actual):
    return next((name for name in BOUNDARIES if not exact(reference[name], actual[name])), None)


def object_contacts(row):
    snapshot = row.get("physics_after", {})
    if not snapshot.get("available") or "contacts" in snapshot.get("missing_fields", []):
        return None
    contacts = snapshot.get("value", {}).get("contacts")
    if contacts is None:
        return None
    return sorted(
        {
            tuple(sorted(pair))
            for pair in contacts
            if len(pair) == 2
            and any(x.startswith("vx300s_") for x in pair)
            and any(x == "red_peg" or x.startswith("socket-") for x in pair)
        }
    )


def select_steps(rows, maximum=12):
    if not rows:
        return []
    events = {0}
    found = set()
    previous_contacts = None
    previous_reward = 0.0
    for index, row in enumerate(rows):
        contacts = object_contacts(row)
        flags = {
            "contact": bool(contacts),
            "loss": (
                previous_contacts is not None
                and contacts is not None
                and bool(set(previous_contacts) - set(contacts))
            ),
            "reward": row["reward"] != previous_reward,
        }
        for name, flag in flags.items():
            if flag and name not in found:
                found.add(name)
                events.update(i for i in (index - 1, index, index + 1) if 0 <= i < len(rows))
        previous_contacts, previous_reward = contacts, row["reward"]
    selected = sorted(events)[:maximum]
    # Uniform grid followed by all indices provides deterministic fill without duplicates.
    grid = [round(i * (len(rows) - 1) / (maximum - 1)) for i in range(maximum)]
    for index in grid + list(range(len(rows))):
        if len(selected) >= maximum:
            break
        if index not in selected:
            selected.append(index)
    return sorted(selected)


def action_deltas(reference, changed, dimension_names):
    result = {}
    for side in ("left", "right"):
        for kind in ("joints", "gripper"):
            indices = [
                i
                for i, name in enumerate(dimension_names)
                if name.startswith(side + "_") and name.endswith("gripper") == (kind == "gripper")
            ]
            if not indices:
                raise ValueError("Missing named action group")
            delta = (changed[:, indices].double() - reference[:, indices].double()).abs()
            for window, values in (("first_action", delta[:1]), ("full_chunk", delta)):
                result[f"{side}_{kind}/{window}"] = {
                    "mean_absolute_delta": float(values.mean()),
                    "maximum_absolute_delta": float(values.max()),
                    "unit": "normalized_command" if kind == "gripper" else "radian",
                }
    return result


def replay_bundle(
    online,
    source,
    output,
    *,
    probe=False,
    guard=lambda: None,
    check_unchanged=lambda: True,
    maximum_bytes=4 * 1024**3,
):
    """Called in a new CLI process after verifying source and runtime registrations."""
    from .gate_diagnostic_verify import verify_bundle

    source, output = Path(source), Path(output)
    verified = verify_bundle(source)
    if verified["stage"] != "collect" or verified["status"] != "verified_complete":
        raise ValueError("Replay requires a verified complete collection")
    identity = load_json(source / "identity.json")
    if identity["identity"]["kind"] == "registered_model" and identity["pid"] == os.getpid():
        raise ValueError("Real replay requires an independent process")
    rows = list(_step_rows(source))
    indices = select_steps(rows) if probe else list(range(len(rows)))
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "identity.json",
        {
            "schema_version": 1,
            "stage": "probe" if probe else "replay",
            "pid": os.getpid(),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "dimension_names": identity["dimension_names"],
            "selected_steps": indices,
            "identity": identity["identity"],
            "diagnostic_only": True,
        },
    )
    result = {
        "stage": "probe" if probe else "replay",
        "status": "incomplete",
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "forwards": 0,
        "records": [],
        "error_type": None,
        "interventions_executed": False,
    }
    online.configure_noise("seeded_standard_normal", 3)
    try:
        for index in indices:
            reference = read_tree(source / "predictions" / f"{index:04d}")
            donor_index = index - 1 if index else 1
            donor = (
                read_tree(source / "predictions" / f"{donor_index:04d}")
                if donor_index < len(rows)
                else None
            )
            control = None
            kinds = ("replay", "self_copy", "image", "state", "noise") if probe else ("replay",)
            for kind in kinds:
                guard()
                observation = clone_tree(reference["observation"])
                noise = reference["noise"].clone()
                if kind in ("image", "state", "noise"):
                    key = {"image": "images", "state": "robot_state", "noise": "noise"}[kind]
                    first = reference["noise"] if kind == "noise" else observation[key]
                    second = (
                        None
                        if donor is None
                        else (donor["noise"] if kind == "noise" else donor["observation"][key])
                    )
                    if donor is None or exact(first, second):
                        result["records"].append(
                            {"step": index, "kind": kind, "status": "skipped_no_distinct_donor"}
                        )
                        continue
                    if kind == "noise":
                        noise = second.clone()
                    else:
                        observation[key] = clone_tree(second)
                if probe and result["forwards"] >= 60:
                    raise RuntimeError("Probe forward budget exhausted")
                # Count attempted forwards, including a failed model call.
                result["forwards"] += 1
                _, actual = capture_prediction(
                    online, observation, reference["instruction"], forced_noise=noise
                )
                name = f"{index:04d}-{kind}"
                write_tree(
                    output / "predictions" / name,
                    actual,
                    budget_root=output,
                    maximum_bytes=maximum_bytes,
                )
                difference = first_difference(reference, actual)
                record = {
                    "step": index,
                    "kind": kind,
                    "directory": "predictions/" + name,
                    "donor_step": donor_index if kind in ("image", "state", "noise") else None,
                }
                if kind in ("replay", "self_copy"):
                    record.update(
                        status="exact" if difference is None else "mismatch",
                        first_difference=difference,
                    )
                    result["records"].append(record)
                    if difference is not None:
                        result["status"] = "control_failed"
                        return result
                    control = actual
                else:
                    record.update(
                        status="measured",
                        deltas=action_deltas(
                            control["projected"], actual["projected"], identity["dimension_names"]
                        ),
                    )
                    result["records"].append(record)
                guard()
        if not check_unchanged():
            raise ValueError("Policy parameters changed")
        result.update(status="complete", parameters_unchanged=True)
        return result
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        raise
    finally:
        seal(output, result)


def _step_rows(source):
    import json

    with (Path(source) / "trace/trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "step":
                yield row
