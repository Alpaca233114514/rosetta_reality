"""Seed-10 engineering restore check; no policy, dataset, teacher or Gate claims."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import enum
import hashlib
import inspect
import json
import math
import random
import sys
import time
import types
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class UnsupportedRestore(RuntimeError):
    """A state owner cannot be captured without guessing its semantics."""


class SafetyStop(RuntimeError):
    """Original adapter safety checks require stopping the entire batch."""


def safe_string(value):
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return {"absolute_path_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "length": len(value)}
    return value


def frozen_contract_fields(value):
    """Only the two source-defined frozen Action Contract dataclasses qualify."""
    cls = type(value)
    if cls.__module__ != "rosetta_reality.sim.action_contract" or cls.__name__ not in (
            "ActionContract", "ActionDimension"):
        return None
    if not dataclasses.is_dataclass(value) or not cls.__dataclass_params__.frozen:
        raise UnsupportedRestore("action_contract_not_frozen_dataclass")
    fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}

    def verify(item):
        if item is None or isinstance(item, (str, bytes, bool, int, float, enum.Enum)):
            return
        if isinstance(item, tuple):
            for child in item:
                verify(child)
            return
        if frozen_contract_fields(item) is not None:
            return
        raise UnsupportedRestore("mutable_action_contract_field")

    for item in fields.values():
        verify(item)
    return fields


def numeric_tree(value):
    """JSON-safe exact array bytes, dtype/shape and human-readable numeric values."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise UnsupportedRestore("object_array")
        if not np.isfinite(value).all():
            raise SafetyStop("nonfinite_array")
        raw = np.ascontiguousarray(value).tobytes()
        return {"dtype": value.dtype.str, "shape": list(value.shape),
                "bytes_hex": raw.hex(), "sha256": hashlib.sha256(raw).hexdigest(),
                "values": value.tolist()}
    if isinstance(value, np.generic):
        return numeric_tree(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise UnsupportedRestore("non_string_mapping_key")
        return {key: numeric_tree(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [numeric_tree(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise SafetyStop("nonfinite_scalar")
    if isinstance(value, str):
        return safe_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise UnsupportedRestore("unsupported_numeric_tree_type")


def tree_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class PythonState:
    """Capture Python owners, rejecting opaque state and unknown native extensions.

    Physics is separately covered by mjSTATE_INTEGRATION. No assumption that
    qpos/qvel alone suffice is allowed. Source hashes identify the audited owners.
    """

    def __init__(self, roots, physics):
        import numpy as np
        import torch

        self.nodes = []
        self.rngs = []
        self.sources = {}
        self.immutable = []
        self.physics = physics
        seen = {id(physics)}

        def visit(obj):
            if obj is None or isinstance(obj, (str, bytes, bool, int, float, enum.Enum,
                                               np.generic, np.dtype, type)):
                return
            if id(obj) in seen:
                return
            seen.add(id(obj))
            frozen = frozen_contract_fields(obj)
            if frozen is not None:
                source = inspect.getsourcefile(type(obj))
                if not source or not Path(source).is_file():
                    raise UnsupportedRestore("action_contract_source_unavailable")
                self.sources[type(obj).__module__] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
                self.immutable.append(obj)
                for item in frozen.values():
                    visit(item)
                return
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return
            if isinstance(obj, np.random.Generator):
                self.rngs.append((obj, "generator", copy.deepcopy(obj.bit_generator.state)))
                return
            if isinstance(obj, np.random.RandomState):
                self.rngs.append((obj, "randomstate", copy.deepcopy(obj.get_state())))
                return
            if isinstance(obj, random.Random):
                self.rngs.append((obj, "python", obj.getstate()))
                return
            if isinstance(obj, types.MethodType):
                visit(obj.__self__)
                return
            if isinstance(obj, types.FunctionType):
                if obj.__closure__:
                    raise UnsupportedRestore("stateful_callback_closure")
                return
            if isinstance(obj, dict):
                for key, item in obj.items():
                    visit(key)
                    visit(item)
                return
            if isinstance(obj, (list, tuple, set, frozenset)):
                for item in obj:
                    visit(item)
                return
            module = type(obj).__module__
            if not module.startswith(("gymnasium.", "gym_aloha.", "dm_control.",
                                      "dm_env.", "rosetta_reality.sim.gym_aloha")):
                raise UnsupportedRestore("unknown_state_owner:" + module + "." + type(obj).__name__)
            if not hasattr(obj, "__dict__"):
                raise UnsupportedRestore("opaque_state_owner:" + module + "." + type(obj).__name__)
            source = inspect.getsourcefile(type(obj))
            if not source or not Path(source).is_file():
                raise UnsupportedRestore("state_owner_source_unavailable:" + module)
            self.sources[module] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            self.nodes.append(obj)
            if len(self.nodes) > 128:
                raise UnsupportedRestore("state_owner_budget_exceeded")
            for item in vars(obj).values():
                visit(item)

        for root in roots:
            visit(root)
        # Preserve identities of owners/RNG/physics while copying their fields.
        self.memo = {id(owner): owner for owner in [physics, *self.nodes, *self.immutable,
                                                  *(item[0] for item in self.rngs)]}
        self.states = copy.deepcopy([vars(owner) for owner in self.nodes], self.memo.copy())
        self.global_python = random.getstate()
        self.global_numpy = np.random.get_state()
        self.global_torch = torch.get_rng_state().clone()
        self.immutable_hash = tree_hash(self.receipt()["immutable_values"])

    def restore(self):
        import numpy as np
        import torch

        if tree_hash(self.receipt()["immutable_values"]) != self.immutable_hash:
            raise UnsupportedRestore("frozen_contract_modified")
        restored = copy.deepcopy(self.states, self.memo.copy())
        for owner, state in zip(self.nodes, restored):
            vars(owner).clear()
            vars(owner).update(state)
        for owner, kind, state in self.rngs:
            if kind == "generator":
                owner.bit_generator.state = copy.deepcopy(state)
            else:
                getattr(owner, "set_state" if kind == "randomstate" else "setstate")(copy.deepcopy(state))
        random.setstate(self.global_python)
        np.random.set_state(self.global_numpy)
        torch.set_rng_state(self.global_torch)

    def receipt(self):
        """Serialize every captured field and RNG without host source filenames."""
        import numpy as np
        import torch

        refs = {id(owner): index for index, owner in enumerate(self.nodes)}
        rng_refs = {id(owner): index for index, (owner, _, _) in enumerate(self.rngs)}
        immutable_refs = {id(owner): index for index, owner in enumerate(self.immutable)}

        def encode(value):
            if id(value) in refs:
                return {"owner_ref": refs[id(value)]}
            if id(value) in rng_refs:
                return {"rng_ref": rng_refs[id(value)]}
            if id(value) in immutable_refs:
                return {"immutable_ref": immutable_refs[id(value)]}
            if value is self.physics:
                return {"native_physics_ref": "mjSTATE_INTEGRATION"}
            if isinstance(value, enum.Enum):
                return {"enum_type": type(value).__module__ + "." + type(value).__qualname__,
                        "member": value.name, "value": encode(value.value)}
            if isinstance(value, np.dtype):
                return {"numpy_dtype": value.str, "descr": encode(value.descr)}
            if isinstance(value, type):
                return {"class": value.__module__ + "." + value.__qualname__}
            if isinstance(value, types.MethodType):
                return {"bound_method": value.__func__.__module__ + "." + value.__func__.__qualname__,
                        "owner": encode(value.__self__)}
            if isinstance(value, types.FunctionType):
                return {"function": value.__module__ + "." + value.__qualname__,
                        "code_sha256": hashlib.sha256(value.__code__.co_code).hexdigest()}
            if isinstance(value, (np.ndarray, torch.Tensor)):
                array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                if array.dtype.hasobject:
                    raise UnsupportedRestore("object_array_in_wrapper_state")
                raw = np.ascontiguousarray(array).tobytes()
                return {"dtype": array.dtype.str, "shape": list(array.shape),
                        "bytes_hex": raw.hex(), "sha256": hashlib.sha256(raw).hexdigest()}
            if isinstance(value, dict):
                return {"mapping_items": [[encode(k), encode(v)] for k, v in value.items()]}
            if isinstance(value, (list, tuple, set, frozenset)):
                return {"container_type": type(value).__name__, "items": [encode(v) for v in value]}
            if isinstance(value, np.generic):
                return encode(value.item())
            if isinstance(value, float) and not math.isfinite(value):
                return {"float_hex": value.hex()}
            if isinstance(value, str):
                return safe_string(value)
            if value is None or isinstance(value, (int, float, bool)):
                return value
            if isinstance(value, bytes):
                return {"bytes_hex": value.hex()}
            raise UnsupportedRestore("unserialized_static_value")

        return {"owners": [{"type": type(owner).__module__ + "." + type(owner).__name__,
                            "fields": encode(state)} for owner, state in zip(self.nodes, self.states)],
                "immutable_values": [{"type": type(owner).__module__ + "." + type(owner).__name__,
                                      "fields": encode(frozen_contract_fields(owner)),
                                      "fields_sha256": tree_hash(encode(frozen_contract_fields(owner)))}
                                     for owner in self.immutable],
                "rngs": [{"kind": kind, "state": encode(state)} for _, kind, state in self.rngs],
                "global_python": encode(self.global_python),
                "global_numpy": encode(self.global_numpy), "global_torch": encode(self.global_torch)}


def run(result):
    import mujoco
    import numpy as np
    import torch
    from importlib.metadata import version
    from rosetta_reality.eval.reproducibility_capture import PhysicsReader
    from rosetta_reality.sim.action_contract import load_action_contract
    from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment

    start = time.monotonic()
    env = None

    def guard():
        if time.monotonic() - start > 120:
            raise TimeoutError("work_cap_exceeded")

    try:
        result["packages"] = {name: version(name) for name in
                              ("mujoco", "dm-control", "gym-aloha", "gymnasium", "numpy", "torch")}
        contract_path = ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
        result["contract_sha256"] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        contract = load_action_contract(contract_path)
        if contract.semantics != "absolute_joint_position_targets" or contract.control_mode != "position":
            raise UnsupportedRestore("hold_requires_absolute_contract")
        env = GymAlohaEnvironment(contract, maximum_episode_steps=500)
        obs = env.reset(seed=10)
        guard()
        result["reset_safety"] = {"joint_limit_violations": env.state_limit_violation_count(),
                                  "unexpected_collisions": env.unexpected_collision_count()}
        if any(result["reset_safety"].values()):
            raise SafetyStop("unsafe_reset_state")
        reader = PhysicsReader(env)
        if not all(hasattr(mujoco, name) for name in ("mj_getState", "mj_setState", "mj_forward")):
            raise UnsupportedRestore("integration_api_unavailable")
        getters = [name for name in dir(mujoco) if name.startswith("get_mjcb_")]
        if not {"get_mjcb_control", "get_mjcb_passive", "get_mjcb_sensor"}.issubset(getters):
            raise UnsupportedRestore("native_callback_inspection_unavailable")
        if any(getattr(mujoco, name)() is not None for name in getters):
            raise UnsupportedRestore("native_callback_state_unverified")
        if int(getattr(reader.model, "nplugin", 0)):
            raise UnsupportedRestore("native_plugin_external_state_unverified")
        snapshot = reader()["integration"].numpy().copy()
        result["reset_integration"] = numeric_tree(snapshot)
        result["integration_spec"] = reader.spec
        result["model_sha256"] = reader.model_sha
        python = PythonState([env], reader.physics)
        result["state_owner_source_sha256"] = python.sources
        result["python_state_owner_count"] = len(python.nodes)
        result["rng_owner_count"] = len(python.rngs)
        result["wrapper_restore_snapshot"] = python.receipt()
        actions = []
        traces = []
        result["replays"] = traces
        for replay in range(2):
            if replay:
                python.restore()
                mujoco.mj_setState(reader.model, reader.data, snapshot, reader.spec)
                # Recompute derived contacts/poses, then restore warmstart/control
                # again: forward is not allowed to change the saved integration.
                mujoco.mj_forward(reader.model, reader.data)
                mujoco.mj_setState(reader.model, reader.data, snapshot, reader.spec)
                if not np.array_equal(reader()["integration"].numpy(), snapshot):
                    raise UnsupportedRestore("integration_restore_roundtrip_mismatch")
            sequence = []
            traces.append(sequence)
            current = obs
            for index in range(3):
                guard()
                action = current["robot_state"].clone() if replay == 0 else actions[index].clone()
                if not bool(torch.isfinite(action).all()):
                    raise SafetyStop("nonfinite_hold_action")
                contract.validate_tensor(action, allow_chunk=False)
                if bool(contract.clip(action)[1].any()):
                    raise SafetyStop("hold_outside_contract")
                if replay == 0:
                    actions.append(action.clone())
                result["environment_steps_attempted"] += 1
                current, reward, done, info = env.step(action)
                result["environment_steps"] += 1
                record = numeric_tree({"integration": reader()["integration"],
                                       "wrapper_state": PythonState([env], reader.physics).receipt(),
                                       "observation": current, "action": action,
                                       "reward": reward, "done": done, "info": info,
                                       "contacts": env.contact_pairs(),
                                       "joint_limit_violations": env.state_limit_violation_count(),
                                       "unexpected_collisions": env.unexpected_collision_count(),
                                       "clip_mask": env.last_clip_mask})
                record["full_numeric_sha256"] = tree_hash(record)
                sequence.append(record)
                if done or record["joint_limit_violations"] or record["unexpected_collisions"] or bool(env.last_clip_mask.any()):
                    raise SafetyStop("unsafe_or_terminated_hold")
                guard()
        result["step_exact_matches"] = [a == b for a, b in zip(*traces)]
        result["physics_model_unchanged"] = reader.model_hash() == reader.model_sha
        result["passed"] = all(result["step_exact_matches"]) and result["physics_model_unchanged"]
        result["status"] = "passed" if result["passed"] else "mismatch"
    finally:
        if env is not None:
            env.close()
        result["elapsed_seconds"] = time.monotonic() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    # Reserve before environment work; retain the incomplete receipt on failure.
    with args.output.open("x", encoding="utf-8") as stream:
        result = {"schema": "rosetta.q0_state_restore.v1", "seed": 10,
                  "status": "incomplete", "passed": False, "diagnostic_only": True,
                  "provider_qualified": False, "infrastructure_failure": False, "safety_stop": False,
                  "task_success": None, "gate3_or_gate4": False, "model_forward_calls": 0,
                  "optimizer_steps": 0, "environment_step_cap": 6,
                  "environment_steps_attempted": 0, "environment_steps": 0,
                  "work_cap_seconds": 120, "hard_deadline_requires_supervisor": True,
                  "limitations": ["single_seed_reset_geometry_three_hold_steps",
                                  "not_loaded_contact_or_teacher_validation",
                                  "unknown_wrapper_or_callback_state_fails_closed"]}
        stream.write(json.dumps(result) + "\n")
        stream.flush()
        try:
            run(result)
        except Exception as error:
            result["status"] = ("unsupported" if isinstance(error, UnsupportedRestore) else
                                "safety_stop" if isinstance(error, SafetyStop) else "failed")
            result["infrastructure_failure"] = not isinstance(error, (UnsupportedRestore, SafetyStop))
            result["safety_stop"] = isinstance(error, SafetyStop)
            result["error_type"] = type(error).__name__
            # Do not serialize arbitrary exception messages containing host paths.
            result["reason"] = str(error) if isinstance(error, UnsupportedRestore) else "check_failed_see_error_type"
        result["full_numeric_sha256"] = tree_hash(result)
        stream.seek(0)
        stream.truncate()
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
