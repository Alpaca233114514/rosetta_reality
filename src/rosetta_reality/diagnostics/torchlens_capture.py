"""Synthetic-only TorchLens experiment and backend-neutral Basin envelope.

No live policy integration: each arm must run in its own disposable process.
TorchLens may wrap global torch functions; it must never enter a training process.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import warnings
from pathlib import Path

MAX_BYTES = 16 * 1024 * 1024
MAX_ELEMENTS = 4096
TORCHLENS_VERSION = "2.23.0"
SELECTED = ("image", "text", "state", "shared", "head")
FORWARD_SEED = 2718
CASES = ("normal", "python_rng", "numpy_rng", "exception", "nonfinite")


def resource_envelope(cgroup=Path("/sys/fs/cgroup")):
    memory = (cgroup / "memory.max").read_text().strip()
    quota, period = (cgroup / "cpu.max").read_text().split()
    if memory == "max" or quota == "max":
        raise ValueError("Bounded cgroup required")
    if not 0 < int(memory) <= 4 * 1024**3 or not 0 < int(quota) / int(period) <= 2:
        raise ValueError("Container resource budget exceeded")
    return {"memory_bytes": int(memory), "cpus": int(quota) / int(period)}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, value: dict, *, limit: int = MAX_BYTES) -> None:
    """Validate/size-check before publishing; never truncate or overwrite."""
    if not 0 < limit <= MAX_BYTES:
        raise ValueError("Invalid JSON byte limit")
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(data) > limit:
        raise ValueError("JSON byte budget exceeded")
    with path.open("xb") as stream:
        stream.write(data)


def envelope(run_id: str) -> dict:
    return {
        "schema_version": 1, "evidence_kind": "synthetic", "source_run": run_id,
        "status": "incomplete", "parameters": {"collector": "rosetta.torchlens.v1"},
        "dimensions": [], "gate": {}, "events": [],
    }


def tensor_record(tensor, *, include_values=True):
    import torch

    value = tensor.detach().cpu().contiguous()
    finite = bool(torch.isfinite(value).all())
    result = {"shape": list(value.shape), "dtype": str(value.dtype), "finite": finite,
              "sha256": digest(value.reshape(-1).view(torch.uint8).numpy().tobytes())}
    if not finite:
        raise ValueError("Non-finite tensor observation")
    if include_values:
        if value.numel() > MAX_ELEMENTS:
            raise ValueError("Selected tensor element budget exceeded")
        result["activation"] = value.tolist()
    return result


def tree_record(value):
    import torch

    if isinstance(value, torch.Tensor):
        return {"tensor": tensor_record(value)}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Only string dictionary keys are supported")
        return {"dict": {key: tree_record(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [tree_record(item) for item in value]}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("Unsupported or non-finite output leaf")


def rng_record():
    import numpy as np
    import torch

    state = np.random.get_state()
    return {"python": digest(repr(random.getstate()).encode()),
            "numpy": digest(state[1].tobytes() + repr((state[0], *state[2:])).encode()),
            "torch_cpu": digest(torch.get_rng_state().numpy().tobytes())}


def synthetic_model(case="normal"):
    import numpy as np
    import torch
    from torch import nn

    class TinyMultimodal(nn.Module):
        def __init__(self):
            super().__init__()
            self.image = nn.Linear(12, 4)
            self.text = nn.Embedding(8, 4)
            self.state = nn.Linear(3, 4)
            self.shared = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)
            self.register_buffer("calls", torch.zeros((), dtype=torch.int64))
            self.last_output = None

        def forward(self, image, text, state):
            self.calls.add_(1)
            x = self.image(image.flatten(1)) + self.text(text).mean(1) + self.state(state)
            x = self.shared(x).tanh()
            x = self.shared(x) + torch.rand_like(x) * 0.01
            if case == "python_rng":
                x = x + random.random()
            if case == "numpy_rng":
                x = x + float(np.random.random())
            result = self.head(x)
            if case == "exception":
                raise RuntimeError("Synthetic forward failure")
            if case == "nonfinite":
                result = result / 0
            self.last_output = {"action": result, "aux": (x, [state, None])}
            return self.last_output

    return TinyMultimodal().eval()


def model_state(model):
    return {name: tensor_record(value) for name, value in model.state_dict().items()}


def execute_arm(backend: str, case: str, run_id: str, *, result=None, capture_backend=None) -> dict:
    """One arm, called only by the isolated worker CLI, never a live model."""
    resources = resource_envelope()
    import numpy as np
    import torch

    if backend not in {"plain", "torchlens", "reference", "basin-torchlens"} or case not in CASES:
        raise ValueError("Unsupported synthetic capture")
    if backend == "basin-torchlens" and capture_backend is None:
        raise ValueError("Basin capture requires the explicit isolated integration worker")
    if backend == "torchlens":
        if importlib.metadata.version("torchlens") != TORCHLENS_VERSION:
            raise ValueError("TorchLens version differs from audited pin")
        import torchlens as tl

    random.seed(1729)
    np.random.seed(1729)
    torch.manual_seed(1729)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = synthetic_model(case)
    args = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 24,
            torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 3))
    # TorchLens trace reseeds by design. Register the same forward RNG starting
    # point for all synthetic arms; never restore RNG after execution to hide drift.
    random.seed(FORWARD_SEED)
    np.random.seed(FORWARD_SEED)
    torch.manual_seed(FORWARD_SEED)
    result = envelope(run_id) if result is None else result
    parameters = result["parameters"]
    parameters.update({"backend": backend, "case": case, "selected_modules": list(SELECTED),
                       "resources": resources,
                       "model_seed": 1729, "forward_seed": FORWARD_SEED,
                       "consistency": "not_evaluated_until_supervisor_comparison",
                       "torch": torch.__version__, "numpy": np.__version__,
                       "torchlens": TORCHLENS_VERSION if "torchlens" in backend else None,
                       "source_sha256": digest(Path(__file__).read_bytes()),
                       "input_before": tree_record(args), "state_before": model_state(model),
                       "rng_before": rng_record(), "device": "cpu", "backward": False})
    with torch.no_grad():
        if backend == "reference":
            # Independent, test-only module-output oracle. The uninstrumented
            # plain arm remains the reference for output/state/RNG parity.
            handles, counts = [], {}

            def hook(name):
                def observe(_module, _inputs, output):
                    counts[name] = counts.get(name, 0) + 1
                    result["events"].append({"step": 0,
                        "stage": f"module:{name}/call:{counts[name]}",
                        "values": {"module": name, "call_index": counts[name],
                                   **tensor_record(output)}})
                return observe

            try:
                for name, module in model.named_modules():
                    if name in SELECTED:
                        handles.append(module.register_forward_hook(hook(name)))
                model(*args)
            finally:
                for handle in handles:
                    handle.remove()
        elif backend == "plain":
            model(*args)
        elif backend == "basin-torchlens":
            captured = capture_backend(model, args)
            if (captured["status"] != "complete" or captured["source_run"] != run_id
                    or captured["evidence_kind"] != "synthetic" or captured["gate"] != {}):
                raise ValueError("Basin raw capture contract mismatch")
            parameters["collector"] = "basin.torchlens.v1"
            parameters["basin_capture"] = captured["parameters"]
            result["events"].extend(captured["events"])
        else:
            selector = tl.module(SELECTED[0])
            for name in SELECTED[1:]:
                selector = selector | tl.module(name)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                trace = tl.trace(model, args, save=selector, capture=tl.options.CaptureOptions(
                    inference_only=True, save_raw_input=False, save_raw_output=False,
                    random_seed=FORWARD_SEED,
                    capture_tensor_grad_hooks=False, save_code_context=False, cache=False,
                    detach_saved_activations=True, unwrap_when_done=True))
            parameters["capture_warnings"] = [str(warning.message) for warning in captured]
            parameters["graph_completeness"] = "not_independently_verified"
            # No hooks or extra forward calls are installed by our collector.
            for index, op in enumerate(trace):
                if index >= 4096:
                    raise ValueError("Graph operation budget exceeded")
                module_label = str(op.module or "<root>")
                path, separator, call = module_label.rpartition(":")
                module = path if separator and call.isdigit() else module_label
                call_index = int(call) if separator and call.isdigit() else int(op.fx_call_index)
                values = {"label": str(op.label), "module": module,
                          "call_index": call_index,
                          "parents": [str(parent) for parent in op.parents],
                          "shape": list(op.shape), "dtype": str(op.dtype),
                          "finite": None, "saved": bool(op.has_saved_activation),
                          "module_outputs": [str(item) for item in op.output_of_module_calls]}
                if op.has_saved_activation:
                    values.update(tensor_record(op.out))
                result["events"].append({"step": 0,
                    "stage": f"module:{module}/call:{call_index}/op:{index}",
                    "values": values})
    parameters.update({"input_after": tree_record(args), "state_after": model_state(model),
                       "rng_after": rng_record(), "output": tree_record(model.last_output)})
    if int(model.calls) != 1:
        raise ValueError("Collector executed more than one forward")
    result["events"].append({"step": 0, "stage": "model:output/call:1",
                             "values": {"output": parameters["output"]}})
    result["status"] = "complete"
    return result


def compare_arms(control: dict, repeat: dict, observed: dict) -> dict:
    fields = ("input_before", "input_after", "state_before", "state_after",
              "rng_before", "rng_after", "output")
    if any(arm["status"] != "complete" for arm in (control, repeat, observed)):
        raise ValueError("Incomplete arm cannot establish parity")
    aa = {key: control["parameters"][key] == repeat["parameters"][key] for key in fields}
    ab = {key: control["parameters"][key] == observed["parameters"][key] for key in fields}
    return {"plain_repeat": aa, "torchlens_vs_plain": ab,
            "passed": all(aa.values()) and all(ab.values()),
            "scope": "synthetic CPU forward only; no SmolVLA or Gate claim"}


def assess_acceptance(control, repeat, observed, reference):
    """Separate transparency from measured offline usefulness; never waive strict parity."""
    strict = compare_arms(control, repeat, observed)
    reference_check = compare_arms(control, repeat, reference)
    expected = {(e["values"]["module"], e["values"]["call_index"]): e["values"]
                for e in reference["events"] if "module" in e["values"]}
    actual = {(e["values"]["module"], e["values"]["call_index"]): e["values"]
              for e in observed["events"] if e["values"].get("saved")
              and e["values"].get("module") in SELECTED}
    coverage = expected.keys() == actual.keys() and bool(expected)
    activation_equal = coverage and all(
        all(expected[key][field] == actual[key][field]
            for field in ("shape", "dtype", "sha256", "activation")) for key in expected)
    output_state = all(value for field, value in strict["torchlens_vs_plain"].items()
                       if field != "rng_after") and all(strict["plain_repeat"].values())
    return {"strict_transparency": strict,
            "reference_observer_transparent": reference_check["passed"],
            "selected_activation_coverage": coverage,
            "selected_activation_exact": activation_equal,
            "selected_activation_count": len(expected),
            "rng_after_equal": {key: control["parameters"]["rng_after"][key]
                                == observed["parameters"]["rng_after"][key]
                                for key in control["parameters"]["rng_after"]},
            "output_and_state_exact": output_state,
            "isolated_case_evidence_usable": bool(
                output_state and activation_equal and reference_check["passed"]),
            "live_integration_accepted": False,
            "graph_completeness": "not_independently_verified",
            "scope": "Only this fixed-seed synthetic case; does not accept arbitrary RNG state"}
