"""Train-only decomposition of actual native CUDA value updates; no import side effects."""

import hashlib
import json

LAYERS = tuple(range(1, 16, 2))
CONDITIONS = ("base1280", "base640") + tuple(
    f"base{base}_v{mode}{1920 - base}"
    for mode in ("full", "global", "token", "scene")
    for base in (1280, 640)
)


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered value component condition")
    base = 1280 if name.startswith("base1280") else 640
    return base, 1920 - base


def component_mode(name):
    base, donor = condition_spec(name)
    return "native" if name == f"base{base}" else name.split("_v")[1].removesuffix(str(donor))


def digest(tensor):
    a = tensor.detach().cpu().float().numpy()
    a = a.copy()
    a[a == 0] = 0  # Reciprocal subtraction may differ only in the sign bit of zero.
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def split_update(delta_train):
    """Only caller-provided training rows; no labels are accepted."""
    if delta_train.ndim != 4 or delta_train.shape[0] != 40:
        raise ValueError("Exactly forty training rows required")
    token_mean = delta_train.double().mean(0).float()
    global_mean = delta_train.double().mean((0, 2), keepdim=False).unsqueeze(1).float()
    return global_mean, token_mean - global_mean


def intervention_output(native, donor, global_mean, token_pattern, mode, sign):
    if mode == "native":
        return native
    if mode == "full":
        return donor
    delta = donor.float() - native.float()
    if mode == "global":
        correction = sign * global_mean
    elif mode == "token":
        correction = sign * token_pattern
    elif mode == "scene":
        correction = delta - sign * (global_mean + token_pattern)
    else:
        raise ValueError("Unregistered component")
    return (native.float() + correction).to(native.dtype)


class ValueComponents:
    """Observe native V outputs and replace one preregistered component at all eight layers."""

    def __init__(self, policy, donor_file, condition, calibration_dir):
        import numpy as np
        import torch
        from safetensors import safe_open

        self.condition = condition
        self.base, _ = condition_spec(condition)
        self.mode = component_mode(condition)
        self.sign = 1 if self.base == 1280 else -1
        self.row = None
        self.handles = []
        self.cache = {}
        self.calls = {}
        self.means = {}
        self.reference = None
        self.calibration_dir = calibration_dir
        if condition != "base1280":
            path = calibration_dir / "calibration.json"
            native_result = json.loads((calibration_dir / "result.json").read_text())
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != native_result["value_component"]["calibration_sha256"]
            ):
                raise ValueError("Calibration seal differs from verified native result")
            self.reference = json.loads(path.read_text())
            npz = calibration_dir / "calibration.npz"
            if hashlib.sha256(npz.read_bytes()).hexdigest() != self.reference["arrays_sha256"]:
                raise ValueError("Calibration content changed")
            if self.reference["training_rows"] != list(range(40)):
                raise ValueError("Calibration must exclude development rows")
            with np.load(npz, allow_pickle=False) as data:
                for layer in LAYERS:
                    self.means[layer] = tuple(
                        torch.from_numpy(data[f"{layer}_{part}"].copy()).to("cuda")
                        for part in ("global", "token")
                    )
        modules = dict(policy.named_modules())
        self.donors = {}
        with safe_open(str(donor_file), framework="pt", device="cpu") as saved:
            for layer in LAYERS:
                name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.v_proj"
                module = modules[name]
                donor = saved.get_tensor(name + ".weight")
                if (
                    tuple(donor.shape) != (320, 320)
                    or donor.dtype != module.weight.dtype
                    or module.bias is not None
                    or not torch.isfinite(donor).all()
                ):
                    raise ValueError("V projection donor schema differs")
                self.donors[layer] = donor.to(module.weight.device)
        try:
            for layer in LAYERS:
                name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.v_proj"
                self.handles.append(modules[name].register_forward_hook(self.hook(layer)))
        except Exception:
            self.close()
            raise

    def hook(self, layer):
        def apply(module, inputs, output):
            import torch
            import torch.nn.functional as functional

            if self.row not in range(45) or len(inputs) != 1:
                raise ValueError("Missing registered row context")
            x = inputs[0]
            if tuple(x.shape) != (1, 64, 320) or output.shape != x.shape:
                raise ValueError("Native V prefix shape changed")
            if output.dtype != torch.bfloat16 or not torch.is_autocast_enabled("cuda"):
                raise ValueError("Native CUDA BF16 autocast required")
            key = (self.row, layer)
            self.calls[key] = self.calls.get(key, 0) + 1
            if key not in self.cache:
                donor = functional.linear(x, self.donors[layer])
                if donor.dtype != output.dtype or not torch.isfinite(donor).all():
                    raise ValueError("Donor projection invalid")
                delta = self.sign * (donor.float() - output.float())
                identities = {"input": digest(x), "delta": digest(delta)}
                if self.reference is not None:
                    expected = self.reference["rows"][f"{self.row}:{layer}"]
                    if identities != expected:
                        raise ValueError("Frozen prefix or reciprocal actual value update differs")
                self.cache[key] = (x.detach().clone(), output.detach().clone(), donor, identities)
            saved_x, saved_y, donor, _ = self.cache[key]
            if not torch.equal(x, saved_x) or not torch.equal(output, saved_y):
                raise ValueError("Prefix depends on noise, denoise iteration, or intervention")
            if self.mode in ("native", "full"):
                return output if self.mode == "native" else donor
            global_mean, token_pattern = self.means[layer]
            return intervention_output(
                output, donor, global_mean, token_pattern, self.mode, self.sign
            )

        return apply

    def finish(self, output):
        import numpy as np
        import torch

        expected = {(row, layer) for row in range(45) for layer in LAYERS}
        if set(self.cache) != expected or any(self.calls.get(k) != 40 for k in expected):
            raise ValueError(
                "Exactly 45 rows, eight layers, four noises, ten denoise calls required"
            )
        calibration = None
        if self.condition == "base1280":
            arrays = {}
            for layer in LAYERS:
                # The only rows passed to decomposition are the preregistered train40.
                delta = torch.stack(
                    [
                        self.cache[(row, layer)][2].float() - self.cache[(row, layer)][1].float()
                        for row in range(40)
                    ]
                )
                global_mean, token_pattern = split_update(delta)
                arrays[f"{layer}_global"] = global_mean.cpu().numpy()
                arrays[f"{layer}_token"] = token_pattern.cpu().numpy()
            with (output / "calibration.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            calibration = {
                "training_rows": list(range(40)),
                "labels_used": False,
                "arrays_sha256": hashlib.sha256(
                    (output / "calibration.npz").read_bytes()
                ).hexdigest(),
                "rows": {
                    f"{row}:{layer}": self.cache[(row, layer)][3] for row, layer in sorted(expected)
                },
            }
            with (output / "calibration.json").open("x") as stream:
                json.dump(calibration, stream, indent=2)
        report = {
            "condition": self.condition,
            "mode": self.mode,
            "parameter_intervention": False,
            "activation_intervention": self.mode != "native",
            "projected_donor_evaluations": len(self.cache),
            "hook_calls": sum(self.calls.values()),
            "native_inputs_and_unmodified_outputs_repeat_exactly": True,
            "reciprocal_delta_matches_calibration": self.reference is not None,
            "calibration_sha256": hashlib.sha256(
                (
                    (output if calibration else self.calibration_dir) / "calibration.json"
                ).read_bytes()
            ).hexdigest(),
            "training_rows_for_component_means": list(range(40)),
            "labels_used_for_components": False,
            "rounding": (
                "native CUDA BF16 donor; partial addition in float32 then BF16; "
                "full returns exact donor"
            ),
            "attention_weights_claimed_fixed": False,
        }
        with (output / "intervention.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        return report

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
