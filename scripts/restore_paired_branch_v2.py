"""Restore a saved bank branch against exact audited owner/binary identities.

Receipt decoding never imports a class named by data. Static classes/callbacks
must already exist in the freshly initialized environment, with matching code.
"""
from __future__ import annotations
import hashlib
import gzip
import json


def restore_branch(environment, reader, directory):
    import enum
    import random
    import types
    import mujoco
    import numpy as np
    import torch
    from check_q0_state_restore import PythonState, UnsupportedRestore, tree_hash

    compressed_path = directory / "branch-python.json.gz"
    if compressed_path.is_file():
        payload = json.loads(gzip.decompress(compressed_path.read_bytes()))
    else:
        payload = json.loads((directory / "branch-python.json").read_text())
    receipt = payload["python_state"]
    current = PythonState([environment], reader.physics)
    if payload["source_sha256"] != current.sources or payload["model_sha256"] != reader.model_sha:
        raise UnsupportedRestore("bank_source_or_model_mismatch")
    if payload["integration_spec"] != reader.spec:
        raise UnsupportedRestore("bank_integration_spec_mismatch")
    if len(receipt["owners"]) != len(current.nodes) or len(receipt["rngs"]) != len(current.rngs):
        raise UnsupportedRestore("bank_owner_inventory_mismatch")

    def decode(value, baseline=None):
        if not isinstance(value, (dict, list)):
            return value
        if isinstance(value, list):
            return [decode(v, baseline[i] if isinstance(baseline, (list, tuple)) and i < len(baseline) else None)
                    for i, v in enumerate(value)]
        if "owner_ref" in value:
            return current.nodes[value["owner_ref"]]
        if "rng_ref" in value:
            return current.rngs[value["rng_ref"]][0]
        if "immutable_ref" in value:
            return current.immutable[value["immutable_ref"]]
        if "native_physics_ref" in value:
            return reader.physics
        if "dtype" in value and "bytes_hex" in value:
            raw = bytes.fromhex(value["bytes_hex"])
            if hashlib.sha256(raw).hexdigest() != value["sha256"]:
                raise UnsupportedRestore("bank_array_hash")
            array = np.frombuffer(raw, dtype=np.dtype(value["dtype"])).copy().reshape(value["shape"])
            return torch.from_numpy(array) if isinstance(baseline, torch.Tensor) else array
        if "mapping_items" in value:
            answer = {}
            for k, v in value["mapping_items"]:
                key = decode(k)
                answer[key] = decode(v, baseline.get(key) if isinstance(baseline, dict) else None)
            return answer
        if "container_type" in value:
            items = decode(value["items"], baseline)
            constructors = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}
            return constructors[value["container_type"]](items)
        if "numpy_dtype" in value:
            return np.dtype(value["numpy_dtype"])
        if "bytes_hex" in value:
            return bytes.fromhex(value["bytes_hex"])
        if "float_hex" in value:
            return float.fromhex(value["float_hex"])
        if "enum_type" in value:
            if not isinstance(baseline, enum.Enum) or value["enum_type"] != type(baseline).__module__ + "." + type(baseline).__qualname__:
                raise UnsupportedRestore("bank_enum_type")
            return type(baseline)[value["member"]]
        if "class" in value:
            if not isinstance(baseline, type) or value["class"] != baseline.__module__ + "." + baseline.__qualname__:
                raise UnsupportedRestore("bank_static_class")
            return baseline
        if "function" in value or "bound_method" in value:
            key = "function" if "function" in value else "bound_method"
            if not isinstance(baseline, (types.FunctionType, types.MethodType)):
                raise UnsupportedRestore("bank_callback_type")
            func = baseline.__func__ if isinstance(baseline, types.MethodType) else baseline
            if value[key] != func.__module__ + "." + func.__qualname__:
                raise UnsupportedRestore("bank_callback_identity")
            if key == "function" and hashlib.sha256(func.__code__.co_code).hexdigest() != value["code_sha256"]:
                raise UnsupportedRestore("bank_callback_code")
            return baseline
        if "absolute_path_sha256" in value:
            if not isinstance(baseline, str) or hashlib.sha256(baseline.encode()).hexdigest() != value["absolute_path_sha256"]:
                raise UnsupportedRestore("bank_path_identity")
            return baseline
        raise UnsupportedRestore("bank_receipt_unknown_field")

    states = []
    for owner, saved in zip(current.nodes, receipt["owners"]):
        if saved["type"] != type(owner).__module__ + "." + type(owner).__name__:
            raise UnsupportedRestore("bank_owner_order")
        states.append(decode(saved["fields"], vars(owner)))
    for owner, state in zip(current.nodes, states):
        vars(owner).clear()
        vars(owner).update(state)
    for (owner, kind, baseline), saved in zip(current.rngs, receipt["rngs"]):
        if kind != saved["kind"]:
            raise UnsupportedRestore("bank_rng_kind")
        state = decode(saved["state"], baseline)
        if kind == "generator":
            owner.bit_generator.state = state
        else:
            getattr(owner, "set_state" if kind == "randomstate" else "setstate")(state)
    random.setstate(decode(receipt["global_python"], random.getstate()))
    np.random.set_state(decode(receipt["global_numpy"], np.random.get_state()))
    torch.set_rng_state(decode(receipt["global_torch"], torch.get_rng_state()))
    with np.load(directory / "branch-integration.npz", allow_pickle=False) as arrays:
        integration = arrays["state"].copy()
    mujoco.mj_setState(reader.model, reader.data, integration, reader.spec)
    mujoco.mj_forward(reader.model, reader.data)
    mujoco.mj_setState(reader.model, reader.data, integration, reader.spec)
    if not np.array_equal(reader()["integration"].numpy(), integration):
        raise UnsupportedRestore("bank_physics_roundtrip")
    restored = PythonState([environment], reader.physics).receipt()
    if tree_hash(restored) != tree_hash(receipt):
        raise UnsupportedRestore("bank_python_roundtrip")
    with np.load(directory / "branch-observation.npz", allow_pickle=False) as arrays:
        observation = {"robot_state": torch.from_numpy(arrays["robot_state"].copy()),
                       "images": {key[len("image::"):]: torch.from_numpy(arrays[key].copy())
                                  for key in arrays.files if key.startswith("image::")}}
    return {"complete_integration_and_python_restored": True,
            "model_sha256": reader.model_sha, "integration_spec": reader.spec,
            "observation": observation}
