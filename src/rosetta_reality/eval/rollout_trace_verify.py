"""Independent, standard-library replay of trace structure and support metrics."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path


def _limit(dtype):
    value = math.pi / 2
    if dtype == "torch.float64":
        return value
    if dtype in ("torch.float32", "torch.float16"):
        code = "f" if dtype.endswith("32") else "e"
        return struct.unpack(code, struct.pack(code, value))[0]
    if dtype == "torch.bfloat16":
        bits = struct.unpack("I", struct.pack("f", value))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return struct.unpack("f", struct.pack("I", bits))[0]
    raise ValueError("Unsupported internal action dtype")


def _stats(predictions, first_only):
    output = {}
    for column, side in enumerate(("left", "right")):
        samples, violations = [], []
        for item in predictions:
            limit = _limit(item["internal_dtype"])
            if item["support_limit"] != limit:
                raise ValueError("Incorrect dtype support boundary")
            values = item["internal_grippers"][:1] if first_only else item["internal_grippers"]
            for slot, pair in enumerate(values):
                if len(pair) != 2 or not all(math.isfinite(v) for v in pair):
                    raise ValueError("Invalid dual gripper values")
                value = pair[column]
                samples.append(value)
                if value < -limit or value > limit:
                    violations.append((abs(value) - limit, item["prediction"], slot))
        output[side] = {
            "count": len(samples), "outside_count": len(violations),
            "outside_fraction": len(violations) / len(samples) if samples else None,
            "maximum_excess": max((v[0] for v in violations), default=0.0) if samples else None,
            "first_outside": {"prediction": violations[0][1], "slot": violations[0][2]}
            if violations else None,
        }
    return output


def verify_trace(directory):
    """Recompute without importing the recorder, torch, models or datasets."""
    directory = Path(directory)
    def load(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))
    manifest, summary, identity = (load(name) for name in (
        "manifest.json", "summary.json", "identity.json"))
    if set(manifest["files"]) != {"identity.json", "summary.json", "trace.jsonl"}:
        raise ValueError("Trace manifest file set differs")
    for name, expected in manifest["files"].items():
        digest = hashlib.sha256()
        with (directory / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Trace file digest differs: " + name)
    if summary["status"] not in ("complete", "incomplete"):
        raise ValueError("Unknown completion status")
    if manifest["status"] != summary["status"]:
        raise ValueError("Manifest completion status differs")
    predictions, started, steps = [], [], []
    phase = "prediction"
    with (directory / "trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row["schema_version"] != 1 or row["seed"] != identity["options"]["seed"]
                    or row["episode_index"] != identity["episode_index"]):
                raise ValueError("Trace episode identity differs")
            if row["event"] != phase:
                raise ValueError("Trace event order differs")
            if phase == "prediction":
                if row["prediction"] != len(predictions):
                    raise ValueError("Prediction sequence differs")
                if len(row["internal_grippers"]) != identity["chunk_length"]:
                    raise ValueError("Internal chunk length differs")
                predictions.append(row)
                phase = "step_started"
            elif phase == "step_started":
                if row["step"] != len(started) or row["prediction"] != len(predictions) - 1:
                    raise ValueError("Execution is not paired with prediction")
                if steps and row["state_before"] != steps[-1]["state_after"]:
                    raise ValueError("State continuity differs")
                if steps and steps[-1]["done"]:
                    raise ValueError("Execution continued after done")
                started.append(row)
                phase = "step"
            else:
                if any(row.get(key) != value for key, value in started[-1].items()
                       if key != "event"):
                    raise ValueError("Execution pre-state or action changed")
                for label in ("before", "after"):
                    expected = [row["state_" + label][i] for i in identity["gripper_indices"]]
                    if row["observed_grippers_" + label] != expected:
                        raise ValueError("Observed gripper state differs")
                steps.append(row)
                phase = "prediction"
    if (summary["predictions"] != len(predictions) or summary["steps_started"] != len(started)
            or summary["steps_recorded"] != len(steps)
            or not len(steps) <= summary["steps_completed"] <= len(started)):
        raise ValueError("Trace counters differ")
    support = {"first_action": _stats(predictions, True), "full_chunk": _stats(predictions, False)}
    executed = _stats([predictions[step["prediction"]] for step in steps], True)
    if summary["support"] != support or summary["executed_support"] != executed:
        raise ValueError("Independent support arithmetic differs")
    if summary["status"] == "complete":
        metrics = summary["metrics"]
        if (phase != "prediction" or not steps or summary["error_type"] is not None
                or summary["steps_completed"] != len(steps)
                or metrics["rollout_length"] != len(steps)
                or metrics["policy_inference_calls"] != len(predictions)
                or metrics["maximum_reward"] != max(row["reward"] for row in steps)
                or metrics["success"] != any(bool(row["success"]) for row in steps)
                or metrics["terminated"] != any(bool(row["terminated"]) for row in steps)
                or metrics["truncated"] != any(bool(row["truncated"]) for row in steps)):
            raise ValueError("Complete metrics do not match recorded execution")
    elif summary["error_type"] is None:
        raise ValueError("Incomplete trace lacks a failure reason")
    return {"status": "verified_" + summary["status"], "steps": len(steps),
            "predictions": len(predictions), "support": support, "executed_support": executed}
