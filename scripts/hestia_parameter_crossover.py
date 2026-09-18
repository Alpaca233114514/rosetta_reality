"""Exact, explicitly bounded cross-attention K/V checkpoint substitutions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CONDITIONS = ("base1280", "base640", "base1280_kv640", "base640_kv1280")


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered condition")
    return {
        "base1280": (1280, None),
        "base640": (640, None),
        "base1280_kv640": (1280, 640),
        "base640_kv1280": (640, 1280),
    }[name]


def kv_keys():
    return tuple(
        f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{kind}_proj.weight"
        for layer in range(1, 16, 2)
        for kind in ("k", "v")
    )


def read_header(path):
    with Path(path).open("rb") as f:
        length = int.from_bytes(f.read(8), "little")
        if not 0 < length < 1024**2:
            raise ValueError("Unexpected checkpoint header")
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1, "BOOL": 1}
    end = 0
    for key, value in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        low, high = value["data_offsets"]
        expected = math.prod(value["shape"]) * widths[value["dtype"]]
        if low != end or high - low != expected:
            raise ValueError("Noncontiguous or malformed tensor ranges")
        end = high
    if 8 + length + end != Path(path).stat().st_size:
        raise ValueError("Checkpoint length differs")
    return header, 8 + length


def tensor_file_digests(path):
    """Read <=1 MiB at once; no policy construction or full-file materialization."""
    header, offset = read_header(path)
    result = {}
    with Path(path).open("rb") as f:
        for name, item in header.items():
            low, high = item["data_offsets"]
            f.seek(offset + low)
            h = hashlib.sha256()
            remaining = high - low
            while remaining:
                chunk = f.read(min(remaining, 1024**2))
                if not chunk:
                    raise ValueError("Truncated tensor")
                h.update(chunk)
                remaining -= len(chunk)
            result[name] = {"dtype": item["dtype"], "shape": item["shape"], "sha256": h.hexdigest()}
    return result


def compare_checkpoints(first, second):
    if set(first) != set(second) or not set(kv_keys()) <= set(first):
        raise ValueError("Checkpoint key set differs")
    changed = []
    for name in first:
        if any(first[name][k] != second[name][k] for k in ("dtype", "shape")):
            raise ValueError("Checkpoint tensor schema differs")
        if first[name]["sha256"] != second[name]["sha256"]:
            changed.append(name)
    frozen = [k for k in first if k.startswith("model.vlm_with_expert.vlm.")]
    if not frozen or set(frozen) & set(changed):
        raise ValueError("Frozen VLM differs between endpoints")
    for k in kv_keys():
        if first[k]["shape"] != [320, 320]:
            raise ValueError("Cross-attention projection shape differs")
    return {
        "tensor_count": len(first),
        "frozen_vlm_tensors": len(frozen),
        "frozen_vlm_exact": True,
        "changed_keys": sorted(changed),
        "changed_cross_kv": sorted(set(changed) & set(kv_keys())),
        "changed_other": sorted(set(changed) - set(kv_keys())),
    }


def substitute(parameters, donor, *, keys=None):
    """Validate the entire mutation first, then replace exactly registered tensors."""
    import torch

    allowed = kv_keys() if keys is None else tuple(keys)
    if (
        tuple(allowed) != kv_keys()
        or set(donor) != set(allowed)
        or not set(allowed) <= set(parameters)
    ):
        raise ValueError("Exact registered sixteen-key substitution required")
    for name in allowed:
        p, v = parameters[name], donor[name]
        if (
            p.shape != v.shape
            or tuple(p.shape) != (320, 320)
            or p.dtype != v.dtype
            or not torch.isfinite(v).all()
        ):
            raise ValueError("Invalid donor schema or values")
    with torch.no_grad():
        for name in allowed:
            parameters[name].copy_(donor[name])


def patch_policy(policy, donor_file):
    from safetensors import safe_open

    from scripts.diagnose_zen_noise_transfer import parameter_digests

    before = parameter_digests(policy)
    parameters = dict(policy.named_parameters())
    with safe_open(str(donor_file), framework="pt", device="cpu") as f:
        donor = {k: f.get_tensor(k) for k in kv_keys()}
    substitute(parameters, donor)
    after = parameter_digests(policy)
    changed = sorted(k for k in before if before[k] != after[k])
    if not set(changed) <= set(kv_keys()) or not changed:
        raise ValueError("Substitution affected unexpected tensors or was empty")
    for name, value in donor.items():
        if not parameters[name].detach().cpu().equal(value):
            raise ValueError("Donor copy was not exact")
    return {
        "allowed_keys": list(kv_keys()),
        "changed_keys": changed,
        "other_parameters_exact": True,
        "donor_copy_exact": True,
        "before": before,
        "after": after,
    }
