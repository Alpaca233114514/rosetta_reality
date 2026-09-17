"""Create-only sidecar recording around the unchanged SmolVLA Gate rollout.

One traced rollout owns the engine binding for its entire lifetime. The caller
must not run untraced rollouts concurrently in that process. No model is loaded
by this module; the caller supplies a policy with its saved postprocessor.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path

_BINDING_LOCK = threading.Lock()
_BODIES = ("peg", "socket", "vx300s_left/gripper_link", "vx300s_right/gripper_link")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


def _tensor(value):
    import torch

    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError("Trace requires finite tensors")
    return value.detach().cpu().clone()


def _input_identity(observation, instruction):
    tensors = {"robot_state": observation["robot_state"], **observation.get("images", {})}
    result = {}
    for name, value in sorted(tensors.items()):
        tensor = _tensor(value).contiguous()
        import torch

        result[name] = {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    result["instruction_sha256"] = hashlib.sha256(instruction.encode()).hexdigest()
    return result


def _snapshot(environment):
    reader = getattr(environment, "diagnostic_snapshot", None)
    snapshot = reader() if callable(reader) else None
    if snapshot is None or not any(snapshot.get(key) for key in (
        "bodies", "contacts", "joint_limit_violations",
    )):
        return {"available": False, "value": None, "missing_bodies": list(_BODIES),
                "reason": "snapshot_missing_or_empty"}
    return {
        "available": True, "value": snapshot,
        "missing_bodies": [name for name in _BODIES if name not in snapshot.get("bodies", {})],
        "missing_fields": [name for name in ("contacts", "joint_limit_violations", "bodies")
                           if name not in snapshot],
    }


def _support(records):
    result = {}
    for window in ("first_action", "full_chunk"):
        result[window] = {}
        for side, column in (("left", 0), ("right", 1)):
            count = outside = 0
            largest = 0.0
            first = None
            for record in records:
                values = record["internal_grippers"]
                if window == "first_action":
                    values = values[:1]
                limit = record["support_limit"]
                for slot, pair in enumerate(values):
                    value = pair[column]
                    count += 1
                    excess = max(abs(value) - limit, 0.0)
                    if excess > 0:
                        outside += 1
                        largest = max(largest, excess)
                        if first is None:
                            first = {"prediction": record["prediction"], "slot": slot}
            result[window][side] = {
                "count": count, "outside_count": outside,
                "outside_fraction": outside / count if count else None,
                "maximum_excess": largest if count else None, "first_outside": first,
            }
    return result


def run_traced_rollout(engine, policy, contract, instruction, *, output, identity,
                       episode_index, **options):
    """Observe exactly one original rollout; return its unmodified metric dict.

    ``identity`` contains already-verified source/checkpoint/processor/config
    inventories. CLI execution verifies them before constructing a real policy.
    Synthetic callers must label their provenance ``synthetic``.
    """
    if contract.chunk_execution_steps != 1:
        raise ValueError("This trace version requires receding-horizon first-action execution")
    indices = [contract.dimension_names.index(name) for name in ("left_gripper", "right_gripper")]
    if identity.get("kind") not in ("synthetic", "registered_model"):
        raise ValueError("Explicit trace provenance required")
    output = Path(output)
    original_environment = engine.GymAlohaEnvironment
    if not _BINDING_LOCK.acquire(blocking=False):
        raise RuntimeError("A traced rollout already owns the engine")
    stream = None
    created = False
    predictions, steps = [], []
    pending = None
    started_steps = completed_steps = 0
    failed = None
    metrics = None
    wall_start = time.perf_counter()
    recording_seconds = 0.0

    def emit(event):
        nonlocal recording_seconds
        begin = time.perf_counter()
        row = {"schema_version": 1, "episode_index": episode_index,
               "seed": options["seed"], **event}
        text = json.dumps(row, allow_nan=False, sort_keys=True)
        stream.write(text + "\n")
        stream.flush()
        recording_seconds += time.perf_counter() - begin

    class Policy:
        def configure_noise(self, *args, **kwargs):
            return policy.configure_noise(*args, **kwargs)

        def predict(self, observation, task):
            nonlocal pending
            if pending is not None:
                raise RuntimeError("Previous prediction was not executed")
            input_identity = _input_identity(observation, task)
            prediction = policy.predict(observation, task)
            raw, projected = prediction
            raw_copy, projected_copy = _tensor(raw), _tensor(projected)
            decoders = [step for step in policy.postprocessor.steps
                        if getattr(type(step), "_registry_name", None)
                        == "rosetta_pi_aloha_postprocessor"]
            if len(decoders) != 1:
                raise ValueError("Trace requires exactly one saved action decoder")
            internal = _tensor(decoders[0].last_model_action)
            shape = (contract.chunk_length, contract.dimension)
            if (tuple(raw_copy.shape) != shape or tuple(projected_copy.shape) != shape
                    or tuple(internal.shape) != (1, *shape)):
                raise ValueError("Trace action shapes differ from the contract")
            if not internal.is_floating_point():
                raise ValueError("Internal actions must be floating point")
            pending = {
                "event": "prediction", "prediction": len(predictions),
                "input_identity": input_identity,
                "internal_dtype": str(internal.dtype),
                "support_limit": float(internal.new_tensor(math.pi / 2)),
                "internal_grippers": internal[0, :, indices].double().tolist(),
                "decoded_first_action": raw_copy[0].double().tolist(),
                "postprocessed_first_action": projected_copy[0].double().tolist(),
            }
            emit(pending)
            predictions.append(pending)
            return prediction

    class Environment:
        def __init__(self, *args, **kwargs):
            self.wrapped = original_environment(*args, **kwargs)
            self.observation = None

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def reset(self, **kwargs):
            observation = self.wrapped.reset(**kwargs)
            self.observation = observation
            return observation

        def step(self, action):
            nonlocal pending, started_steps, completed_steps
            if pending is None:
                raise RuntimeError("No prediction for executed action")
            record = {
                "step": started_steps, "prediction": pending["prediction"],
                "state_before": _tensor(self.observation["robot_state"]).double().tolist(),
                "physics_before": _snapshot(self.wrapped),
                "executed_action": _tensor(action).double().tolist(),
            }
            record["observed_grippers_before"] = [record["state_before"][i] for i in indices]
            emit({"event": "step_started", **record})
            started_steps += 1
            result = self.wrapped.step(action)
            completed_steps += 1
            observation, reward, done, info = result
            record.update(
                event="step", state_after=_tensor(observation["robot_state"]).double().tolist(),
                physics_after=_snapshot(self.wrapped), reward=float(reward), done=bool(done),
                success=info.get("is_success"), terminated=info.get("terminated"),
                truncated=info.get("truncated"),
            )
            record["observed_grippers_after"] = [record["state_after"][i] for i in indices]
            emit(record)
            steps.append(record)
            pending = None
            self.observation = observation
            return result

    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(output / "identity.json", {
            "schema_version": 1, "identity": identity, "episode_index": episode_index,
            "completion_rule": "incomplete unless summary and manifest verify as complete",
            "options": options, "instruction": instruction,
            "dimension_names": list(contract.dimension_names),
            "chunk_length": contract.chunk_length, "gripper_indices": indices,
            "support_definition": "abs(internal) > pi/2 rounded to internal tensor dtype",
            "observation_gripper_units": "normalized measured joint opening, not action command",
            "full_chunk_is_executed": False,
        })
        stream = (output / "trace.jsonl").open("x", encoding="utf-8", newline="\n")
        engine.GymAlohaEnvironment = Environment
        metrics = engine._rollout(Policy(), contract, instruction, **options)
        if pending is not None or len(steps) != metrics["rollout_length"]:
            raise RuntimeError("Incomplete prediction/execution trace")
        return metrics
    except BaseException as error:
        failed = type(error).__name__
        raise
    finally:
        engine.GymAlohaEnvironment = original_environment
        try:
            if stream is not None:
                stream.close()
            if created:
                summary = {
                    "schema_version": 1, "status": "incomplete" if failed else "complete",
                    "error_type": failed, "predictions": len(predictions),
                    "steps_started": started_steps, "steps_completed": completed_steps,
                    "steps_recorded": len(steps), "metrics": metrics,
                    "support": _support(predictions),
                    "executed_support": _support([predictions[s["prediction"]] for s in steps])[
                        "first_action"],
                    "wall_seconds": time.perf_counter() - wall_start,
                    "serialization_seconds": recording_seconds,
                    "latency_includes_diagnostic_overhead": True,
                    "expert_deviation_measured": False,
                }
                write_json(output / "summary.json", summary)
                write_json(output / "manifest.json", {
                    "schema_version": 1, "status": summary["status"],
                    "files": {p.name: file_sha(p) for p in sorted(output.iterdir())
                              if p.is_file()},
                })
        finally:
            _BINDING_LOCK.release()
