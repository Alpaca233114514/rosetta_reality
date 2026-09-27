"""Optional TorchLens backend. Core Basin import/query remains standard-library only.

Call only inside a disposable, resource-bounded worker. This first backend accepts
synthetic captures only; no model loading, downloading, training or intervention.
"""

import importlib.metadata
import warnings
from pathlib import Path

from .io import dumps, sha, write_new

VERSION = "2.23.0"
MAX_BYTES = 16 * 1024 * 1024


def capture_synthetic(model, inputs, *, run_id, selected_modules, forward_seed, output):
    """Capture selected module outputs as a native v1 envelope, without certifying parity.

The operator constructs the model/inputs in the worker. No untrusted module names,
pickles, weights or executable specifications are loaded from evidence files.
Explicit seed is required because TorchLens trace reseeds global RNG by design.
Python RNG transparency is known to fail: output is not a live-integration approval.
"""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Nonempty source run required")
    if type(forward_seed) is not int or not 0 <= forward_seed < 2**32:
        raise ValueError("Explicit 32-bit forward seed required")
    if not isinstance(selected_modules, (list, tuple)) or not selected_modules:
        raise ValueError("Explicit nonempty selected modules required")
    if not all(isinstance(name, str) and name for name in selected_modules):
        raise ValueError("Invalid module selection")
    if len(set(selected_modules)) != len(selected_modules):
        raise ValueError("Duplicate module selection")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    envelope = {"schema_version": 1, "source_run": run_id, "evidence_kind": "synthetic",
                "status": "incomplete", "dimensions": [], "gate": {}, "events": [],
                "parameters": {"collector": "basin.torchlens.v1", "torchlens": VERSION,
                               "forward_seed": forward_seed,
                               "selected_modules": list(selected_modules),
                               "backend_source_sha256": sha(Path(__file__).read_bytes()),
                               "consistency": "not_assessed",
                               "live_integration_accepted": False,
                               "graph_completeness": "not_independently_verified"}}
    try:
        if importlib.metadata.version("torchlens") != VERSION:
            raise ValueError("TorchLens version differs from audited pin")
        import torch
        import torchlens as tl

        if model.training:
            raise ValueError("This backend requires an eval-mode synthetic model")
        modules = dict(model.named_modules())
        if any(name not in modules for name in selected_modules):
            raise ValueError("Selected module missing")
        selector = tl.module(selected_modules[0])
        for name in selected_modules[1:]:
            selector = selector | tl.module(name)
        envelope["parameters"]["torch"] = torch.__version__
        with torch.no_grad(), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            trace = tl.trace(model, inputs, save=selector, capture=tl.options.CaptureOptions(
                random_seed=forward_seed, inference_only=True, save_raw_input=False,
                save_raw_output=False, capture_tensor_grad_hooks=False, save_code_context=False,
                detach_saved_activations=True, unwrap_when_done=True, cache=False))
        envelope["parameters"]["warnings"] = [str(item.message) for item in captured]
        for index, op in enumerate(trace):
            if index >= 4096:
                raise ValueError("Operation budget exceeded")
            label = str(op.module or "<root>")
            path, separator, call = label.rpartition(":")
            module = path if separator and call.isdigit() else label
            call_index = int(call) if separator and call.isdigit() else int(op.fx_call_index)
            value = {"module": module, "call_index": call_index, "label": str(op.label),
                     "parents": [str(parent) for parent in op.parents],
                     "shape": list(op.shape), "dtype": str(op.dtype),
                     "saved": bool(op.has_saved_activation), "finite": None}
            if op.has_saved_activation:
                tensor = op.out.detach().cpu().contiguous()
                if tensor.numel() > 4096:
                    raise ValueError("Activation budget exceeded")
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Non-finite activation")
                value.update({"finite": True, "activation": tensor.tolist(),
                              "sha256": sha(
                                  tensor.reshape(-1).view(torch.uint8).numpy().tobytes())})
            envelope["events"].append({"step": 0,
                "stage": f"module:{module}/call:{call_index}/op:{index}", "values": value})
        envelope["status"] = "complete"  # capture completed, consistency is still not_assessed
        data = dumps(envelope).encode()
        if len(data) > MAX_BYTES:
            raise ValueError("Envelope byte budget exceeded")
        write_new(output / "native.json", data)
        return envelope
    except Exception as exc:
        failure = {**envelope, "status": "incomplete", "events": []}
        failure["parameters"] = {**envelope["parameters"], "error_type": type(exc).__name__,
                                 "unpublished_event_count": len(envelope["events"])}
        write_new(output / "failure.json", dumps(failure))
        raise
