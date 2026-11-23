"""Opt-in JSON sidecar support for a future registered collector.

Accepts only already-observed Python values. Never calls policy, physics, random,
or a safety decision. Not installed into historical runners. A future integration
must capture actual boundaries and establish observer parity before live use.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


def finite(value):
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("Nonfinite evidence")


def required(value, names):
    if not isinstance(value, dict) or any(name not in value or value[name] is None for name in names):
        raise ValueError("Required observed evidence missing")


def array(value):
    if not isinstance(value, list) or not value or any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError("Actual finite numeric array required")


def identity(value):
    if not isinstance(value, dict) or not value or any(not isinstance(k, str) or not isinstance(v, str)
            or re.fullmatch(r"[0-9a-f]{64}", v) is None for k, v in value.items()):
        raise ValueError("Nonempty SHA256 identity inventory required")


def tensor(value):
    required(value, ("shape", "dtype", "values"))
    if not value["shape"] or any(type(n) is not int or n <= 0 for n in value["shape"]):
        raise ValueError("Actual tensor shape required")
    array(value["values"])
    if math.prod(value["shape"]) != len(value["values"]):
        raise ValueError("Full tensor payload differs from shape")
    if value["dtype"] not in ("float32", "float64", "bfloat16", "float16"):
        raise ValueError("Explicit supported tensor dtype required")


def physics(value):
    finite(value)
    required(value, ("qpos", "qvel", "ctrl", "joint_limits", "contacts", "object_poses",
                     "integration_state", "compiled_model_sha256"))
    for key in ("qpos", "qvel", "ctrl", "integration_state"):
        array(value[key])
    if not isinstance(value["joint_limits"], list) or not value["joint_limits"]:
        raise ValueError("Actual limited-joint inventory required")
    for joint in value["joint_limits"]:
        required(joint, ("name", "qpos_address", "lower", "upper", "tolerance", "value", "violated"))
        address = joint["qpos_address"]
        if type(address) is not int or not 0 <= address < len(value["qpos"]):
            raise ValueError("Invalid qpos address")
        if joint["tolerance"] != 1e-5 or joint["lower"] > joint["upper"] or joint["value"] != value["qpos"][address]:
            raise ValueError("Physical joint evidence or unchanged threshold mismatch")
        actual = joint["value"] < joint["lower"] - 1e-5 or joint["value"] > joint["upper"] + 1e-5
        if type(joint["violated"]) is not bool or joint["violated"] != actual:
            raise ValueError("Physical limit flag mismatch")
    if not isinstance(value["contacts"], list):
        raise ValueError("Explicit contact inventory required")
    identity({"compiled_model": value["compiled_model_sha256"]})
    if not isinstance(value["object_poses"], dict) or not value["object_poses"]:
        raise ValueError("Observed object poses required")
    for contact in value["contacts"]:
        required(contact, ("pair", "unexpected"))
        if (not isinstance(contact["pair"], list) or len(contact["pair"]) != 2
                or any(not isinstance(v, str) for v in contact["pair"])
                or type(contact["unexpected"]) is not bool):
            raise ValueError("Invalid contact evidence")


class EvidenceSidecar:
    """Create-only, byte-capped episode evidence; incomplete files remain visible."""

    def __init__(self, path, *, maximum_bytes):
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("Explicit positive evidence byte cap required")
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.maximum_bytes = maximum_bytes
        self.bytes_written = 0
        self.step = 0
        self.initialized = False
        self.stopped = False

    def write(self, record):
        finite(record)
        text = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        size = len(text.encode("utf-8"))
        if self.bytes_written + size > self.maximum_bytes:
            raise ValueError("Evidence byte cap reached")
        self.stream.write(text)
        self.stream.flush()
        self.bytes_written += size

    def start(self, record):
        if self.initialized:
            raise ValueError("Duplicate initial state")
        required(record, ("episode_id", "source_identity", "processor_identity", "normalization_sha256",
                          "action_contract_sha256", "physics", "observation_hashes", "noise_contract"))
        physics(record["physics"])
        for key in ("processor_identity", "observation_hashes", "source_identity"):
            identity(record[key])
        identity({k: record[k] for k in ("normalization_sha256", "action_contract_sha256")})
        self.write({**record, "schema": "rosetta.paired_evidence_sidecar.v1", "event": "initial"})
        self.processor_identity = dict(record["processor_identity"])
        self.initialized = True
        self.stopped = any(j["violated"] for j in record["physics"]["joint_limits"]) or any(c["unexpected"] for c in record["physics"]["contacts"])

    def observe(self, record):
        if not self.initialized or self.stopped or record.get("step") != self.step + 1:
            raise ValueError("Invalid episode lifecycle")
        required(record, ("step", "noise", "normalized_prediction", "raw_chunk", "projected_chunk",
                          "executed_slot", "logical_command", "physics_before", "physics_after",
                          "processor_identity", "observation_hashes", "reward", "task_success", "done"))
        for key in ("noise", "normalized_prediction", "raw_chunk", "projected_chunk"):
            tensor(record[key])
        for key in ("physics_before", "physics_after"):
            physics(record[key])
        identity(record["processor_identity"])
        identity(record["observation_hashes"])
        if record["processor_identity"] != self.processor_identity:
            raise ValueError("Effective processor identity drift")
        array(record["logical_command"])
        shape = record["raw_chunk"]["shape"]
        if (len(shape) != 2 or record["projected_chunk"]["shape"] != shape
                or record["executed_slot"] != 0 or len(record["logical_command"]) != shape[1]
                or record["logical_command"] != record["projected_chunk"]["values"][:shape[1]]):
            raise ValueError("Registered first-action command chain mismatch")
        if (any(type(record[k]) is not bool for k in ("task_success", "done"))
                or type(record["reward"]) not in (int, float) or not math.isfinite(record["reward"])):
            raise ValueError("Explicit terminal flags required")
        self.write({**record, "schema": "rosetta.paired_evidence_sidecar.v1", "event": "step"})
        self.step += 1
        after = record["physics_after"]
        self.stopped = record["done"] or any(j["violated"] for j in after["joint_limits"]) or any(c["unexpected"] for c in after["contacts"])

    def close(self):
        self.stream.close()
        return {"steps": self.step, "bytes": self.bytes_written,
                "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "live_integration_verified": False}
