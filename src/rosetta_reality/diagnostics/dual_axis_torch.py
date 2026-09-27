"""Optional Torch capture primitives. No Torch import or model access at import time."""

from __future__ import annotations

import copy
import hashlib
import random
from contextlib import contextmanager

from .dual_axis import digest


def tensor_bytes(value):
    import torch

    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_identity(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
    }


def tree_identity(value):
    import torch

    if isinstance(value, torch.Tensor):
        return tensor_identity(value)
    if isinstance(value, dict):
        return {k: tree_identity(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_identity(v) for v in value]
    return value


def detached_tree(value):
    """Freeze loader ingress before in-place processor changes without retaining a graph."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: detached_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detached_tree(v) for v in value)
    return copy.deepcopy(value)


@contextmanager
def native_methods(policy, writer, *, full=False):
    """Observe native noise/time/prefix/denoiser calls without extra model execution."""
    model = getattr(policy, "model", None)
    rows, installed = [], []
    if model is None:
        yield rows
        return
    try:
        for name in ("sample_noise", "sample_time", "embed_prefix", "denoise_step"):
            original = getattr(model, name, None)
            if not callable(original):
                continue
            had = name in vars(model)
            local = vars(model).get(name)

            def wrapped(*args, _original=original, _name=name, **kwargs):
                result = _original(*args, **kwargs)
                if len(rows) >= 4096:
                    raise ValueError("Native method capture budget exceeded")
                payload = {"inputs": args, "kwargs": kwargs, "output": result}
                # Device objects occur in sample_noise/sample_time signatures.
                if _name in ("sample_noise", "sample_time"):
                    payload = {"output": result}
                rows.append(
                    {
                        "method": _name,
                        "call": sum(r["method"] == _name for r in rows),
                        "payload": tensor_tree(payload, writer)
                        if full or _name in ("sample_noise", "sample_time")
                        else tree_identity(payload),
                    }
                )
                return result

            setattr(model, name, wrapped)
            installed.append((name, had, local, wrapped))
        yield rows
    finally:
        for name, had, local, wrapper in reversed(installed):
            if getattr(model, name) is not wrapper:
                raise RuntimeError("Native method observer changed")
            if had:
                setattr(model, name, local)
            else:
                delattr(model, name)


def tensor_summary(value, *, sample_elements=256):
    import torch

    t = value.detach().reshape(-1)
    if not bool(torch.isfinite(t).all()):
        raise FloatingPointError("Nonfinite diagnostic tensor")
    stride = max(1, (t.numel() + sample_elements - 1) // sample_elements)
    sample = t[::stride][:sample_elements].float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": t.numel(),
        "l2": float(t.float().norm()),
        "sample_stride": stride,
        "sample_values": sample.tolist(),
        "sampled_not_full": t.numel() > sample.numel(),
    }


def tensor_tree(value, writer, *, prefix="tensors", chunk_bytes=8 * 1024**2):
    """Content-addressed raw bytes preserve BF16 exactly; no pickle/model objects."""
    import torch

    if isinstance(value, torch.Tensor):
        raw = tensor_bytes(value)
        identity = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        refs = []
        for offset in range(0, len(raw), chunk_bytes):
            chunk = raw[offset : offset + chunk_bytes]
            name = f"{prefix}/{hashlib.sha256(chunk).hexdigest()}.bin"
            if name not in writer.files:
                writer.write(name, chunk, raw=True)
            refs.append({"offset": offset, "path": name, **writer.files[name]})
        return {"tensor": identity, "chunks": refs, "encoding": "native_little_endian_raw"}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Tensor tree keys must be strings")
        return {
            k: tensor_tree(v, writer, prefix=prefix, chunk_bytes=chunk_bytes)
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [tensor_tree(v, writer, prefix=prefix, chunk_bytes=chunk_bytes) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported evidence type {type(value).__name__}")


def model_identity(policy):
    return {
        "parameters": {n: tensor_identity(p) for n, p in policy.named_parameters()},
        "buffers": {n: tensor_identity(b) for n, b in policy.named_buffers()},
    }


def optimizer_inventory(policy, optimizer, writer=None):
    names = {id(p): n for n, p in policy.named_parameters()}
    groups, seen, state = [], set(), {}
    base = getattr(optimizer, "optimizer", optimizer)
    for group in optimizer.param_groups:
        members = []
        for p in group["params"]:
            name = names.get(id(p))
            if name is None or name in seen:
                raise ValueError("Ambiguous optimizer parameter membership")
            seen.add(name)
            members.append(name)
            moment = base.state.get(p, {})
            state[name] = (
                tensor_tree(moment, writer)
                if writer
                else {
                    k: tensor_summary(v) if hasattr(v, "detach") else v for k, v in moment.items()
                }
            )
        groups.append({**{k: v for k, v in group.items() if k != "params"}, "parameters": members})
    return {
        "groups": groups,
        "state": state,
        "trainable_missing_from_optimizer": [
            n for n, p in policy.named_parameters() if p.requires_grad and n not in seen
        ],
        "parameters": {
            n: {
                "requires_grad": p.requires_grad,
                "dtype": str(p.dtype),
                "shape": list(p.shape),
                "in_optimizer": n in seen,
            }
            for n, p in policy.named_parameters()
        },
    }


def snapshot(policy, optimizer, writer):
    return {
        "model": tensor_tree(dict(policy.state_dict()), writer),
        "optimizer": optimizer_inventory(policy, optimizer, writer),
        "identity": model_identity(policy),
    }


def gradient_summary(policy):
    return {
        name: {"status": "none"}
        if p.grad is None
        else {
            "status": "zero" if not bool(p.grad.count_nonzero()) else "nonzero",
            **tensor_summary(
                p.grad, sample_elements=p.numel() if "norm" in name and p.numel() <= 4096 else 256
            ),
        }
        for name, p in policy.named_parameters()
        if p.requires_grad
    }


def parameter_samples(policy):
    return {
        name: tensor_summary(
            p, sample_elements=p.numel() if "norm" in name and p.numel() <= 4096 else 256
        )
        for name, p in policy.named_parameters()
        if p.requires_grad
    }


def sampled_update(before, after):
    import math

    if before.keys() != after.keys():
        raise ValueError("Trainable parameter scope changed")
    result = {}
    for name, old in before.items():
        new = after[name]
        if (old["shape"], old["dtype"], old["sample_stride"]) != (
            new["shape"],
            new["dtype"],
            new["sample_stride"],
        ):
            raise ValueError("Parameter identity changed during update")
        delta = math.sqrt(
            sum(
                (b - a) ** 2
                for a, b in zip(old["sample_values"], new["sample_values"], strict=True)
            )
        )
        norm = math.sqrt(sum(a * a for a in old["sample_values"]))
        result[name] = {
            "sample_delta_l2": delta,
            "sample_parameter_l2": norm,
            "sample_update_ratio": delta / norm if norm else None,
            "sample_stride": old["sample_stride"],
            "sampled_not_full": old["sampled_not_full"],
        }
    return result


@contextmanager
def isolated_probe(policy, *, generators=(), state_objects=()):
    """Restore stochastic/mode/call state; reject tensor mutation instead of hiding it."""
    import numpy as np
    import torch

    py, np_rng, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    own = [g.get_state().clone() for g in generators]
    modes = [(m, m.training) for m in policy.modules()]
    before = model_identity(policy)
    grads = [
        (p, p.grad, None if p.grad is None else p.grad.detach().clone())
        for p in policy.parameters()
    ]
    automatic = [(policy, "_queues")] if hasattr(policy, "_queues") else []
    model = getattr(policy, "model", None)
    if model is not None and hasattr(model, "prefix_length"):
        automatic.append((model, "prefix_length"))
    state_objects = list(state_objects) + automatic
    states = [(obj, name, copy.deepcopy(getattr(obj, name))) for obj, name in state_objects]
    try:
        policy.eval()
        yield
    finally:
        changed = model_identity(policy) != before
        for p, original, value in grads:
            if original is None:
                p.grad = None
            else:
                original.copy_(value)
                p.grad = original
        for obj, name, value in states:
            setattr(obj, name, value)
        for m, mode in modes:
            m.training = mode
        random.setstate(py)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for g, state in zip(generators, own, strict=True):
            g.set_state(state)
        if changed:
            raise RuntimeError("Probe mutated parameters/buffers; live integration rejected")


@contextmanager
def module_capture(policy, module_names, *, maximum_calls=4096, maximum_elements=4096):
    """Repeated invocations have distinct call indices. No graph retained by hooks."""
    import torch

    modules = dict(policy.named_modules())
    if (
        not module_names
        or len(module_names) != len(set(module_names))
        or not set(module_names) <= modules.keys()
    ):
        raise ValueError("Explicit existing module names required")
    rows, handles, counts = [], [], {}

    def hook(name):
        def record(module, inputs, output):
            if len(rows) >= maximum_calls:
                raise ValueError("Module capture call budget exceeded")

            def collect(value):
                if isinstance(value, torch.Tensor):
                    result = tensor_summary(value)
                    if value.numel() <= maximum_elements:
                        result["full_values"] = value.detach().float().cpu().tolist()
                    return result
                if isinstance(value, (list, tuple)):
                    return [collect(v) for v in value]
                if isinstance(value, dict):
                    return {k: collect(v) for k, v in value.items()}
                return {"unobserved_type": type(value).__name__}

            call = counts.get(name, 0)
            counts[name] = call + 1
            rows.append(
                {"module": name, "call": call, "input": collect(inputs), "output": collect(output)}
            )

        return record

    try:
        for name in module_names:
            handles.append(modules[name].register_forward_hook(hook(name)))
        yield rows
    finally:
        for handle in handles:
            handle.remove()


def endpoint_gradient_probe(policy, samples, forward, writer, *, update, state_objects=()):
    """Separate diagnostic stage: fixed train-only inputs, zero optimizer/scheduler calls."""
    import torch

    if any(sample["split"] != "train" for sample in samples):
        raise ValueError("Gradient probes must use train-only samples")
    for sample in samples:
        with isolated_probe(policy, state_objects=state_objects):
            policy.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss, payload = forward(policy, sample)
                if loss.numel() != 1 or not bool(torch.isfinite(loss)):
                    raise ValueError("Invalid endpoint diagnostic loss")
                loss.backward()
            writer.event(
                {
                    "axis": "probe",
                    "update": update,
                    "sample_id": sample["sample_id"],
                    "episode": sample["episode"],
                    "frame": sample["frame"],
                    "boundary": "model_internal",
                },
                {
                    "gradient": gradient_summary(policy),
                    "loss": float(loss.detach()),
                    "inputs": tensor_tree(payload, writer),
                    "optimizer_updates": 0,
                },
                reasons=["endpoint_gradient"],
            )


def admission_result(control, observed, *, maximum_overhead=0.15):
    """Consumed by registration, never grants model execution authorization."""
    fields = (
        "outputs",
        "gradients",
        "parameters",
        "buffers",
        "rng",
        "next_batch",
        "optimizer",
        "scheduler",
    )
    missing = [key for key in fields if key not in control or key not in observed]
    mismatches = [key for key in fields if key not in missing and control[key] != observed[key]]
    base_time, new_time = control.get("routine_seconds"), observed.get("routine_seconds")
    import math

    overhead = (
        new_time / base_time - 1
        if type(base_time) in (float, int)
        and math.isfinite(base_time)
        and base_time > 0
        and type(new_time) in (float, int)
        and math.isfinite(new_time)
        and new_time > 0
        else None
    )
    return {
        "status": "passed"
        if not missing and not mismatches and overhead is not None and overhead <= maximum_overhead
        else "failed",
        "missing": missing,
        "mismatches": mismatches,
        "routine_overhead": overhead,
        "maximum_overhead": maximum_overhead,
        "model_execution_authorized": False,
        "control_identity": digest(control),
        "observed_identity": digest(observed),
    }
