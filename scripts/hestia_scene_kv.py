"""Same-checkpoint real-image K/V scene ablations; imports do not load models."""

from __future__ import annotations

import hashlib
import json

LAYERS = tuple(range(1, 16, 2))
MODES = ("native", "self", "kmean", "vmean", "kvmean")
CONDITIONS = tuple(f"base{base}_{mode}" for mode in MODES for base in (640, 1280))


def validate_protocol(plan):
    expected = {
        "id": "hestia-scene-kv-ablation-20260913-001",
        "launchable": True,
        "maximum_policy_forwards": 1800,
        "optimizer_steps": 0,
        "full_prefix_shape": [1, 241, 320],
        "replace_token_range": [0, 64],
        "dev_calibration": "train40_only",
        "train_calibration": "train39_excluding_self",
        "conditions": list(CONDITIONS),
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen scene K/V protocol differs or is non-launchable")


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered scene K/V condition")
    base, mode = name.split("_")
    return int(base.removeprefix("base")), mode


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(tensor):
    value = tensor.detach().cpu().contiguous()
    # Preserve dtype and signed-zero bits in the actual native tensors.
    import torch

    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(str((tuple(value.shape), value.dtype)).encode() + raw).hexdigest()


def calibration_means(training):
    """Only forty label-free training rows enter this function, never development."""
    import numpy as np

    a = np.asarray(training)
    if a.shape != (40, 64, 320) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError("Forty finite float32 training rows required")
    values = a.astype(np.float64)
    # Direct means avoid subtractive cancellation in leave-one-out sums.
    dev = values.mean(0).astype(np.float32)
    train = np.stack([np.delete(values, i, axis=0).mean(0) for i in range(40)]).astype(np.float32)
    return train, dev


def replace_image(native, replacement):
    import torch

    if tuple(native.shape) != (1, 241, 320) or tuple(replacement.shape) != (64, 320):
        raise ValueError("Full prefix and image replacement shape required")
    if native.dtype != torch.bfloat16 or replacement.dtype != native.dtype:
        raise ValueError("Native BF16 dtype required")
    if replacement.device != native.device or not torch.isfinite(replacement).all():
        raise ValueError("Replacement device or finite contract differs")
    output = native.clone()
    output[:, :64] = replacement
    if not torch.equal(output[:, 64:], native[:, 64:]):
        raise ValueError("Unregistered prefix positions changed")
    return output


class SceneKV:
    def __init__(self, policy, condition, job):
        import numpy as np
        import torch

        self.condition = condition
        self.base, self.mode = condition_spec(condition)
        self.row = None
        self.handles, self.cache, self.calls = [], {}, {}
        self.applied_outputs = {}
        self.layout = None
        self.reference, self.reference_sha = None, None
        self.banks, self.means = {}, {}
        if self.mode != "native":
            root = job / f"base{self.base}_native"
            path = root / "intervention.json"
            self.reference = json.loads(path.read_text())
            result = json.loads((root / "result.json").read_text())
            if (
                result["scene_kv"] != self.reference
                or self.reference["condition"] != f"base{self.base}_native"
            ):
                raise ValueError("Native calibration seal differs")
            self.reference_sha = file_hash(path)
            for split, rows in (("train", list(range(40))), ("dev", list(range(40, 45)))):
                entry = self.reference["calibration_files"][split]
                if entry["rows"] != rows:
                    raise ValueError("Calibration split row identity differs")
                bank_path = root / f"calibration-{split}.npz"
                if file_hash(bank_path) != entry["sha256"]:
                    raise ValueError("Calibration bytes differ")
                with np.load(bank_path, allow_pickle=False) as archive:
                    expected = {f"{layer}_{kind}" for layer in LAYERS for kind in ("k", "v")}
                    if set(archive.files) != expected:
                        raise ValueError("Calibration layer/key set differs")
                    self.banks[split] = {key: archive[key] for key in archive.files}
                    for value in self.banks[split].values():
                        if (
                            value.shape != (len(rows), 64, 320)
                            or value.dtype != np.float32
                            or not np.isfinite(value).all()
                        ):
                            raise ValueError("Calibration layout or dtype differs")
            for key, training in self.banks["train"].items():
                # There is deliberately no dev-bank argument to this operation.
                loo, mean = calibration_means(training)
                self.means[key] = (torch.from_numpy(loo), torch.from_numpy(mean))
        modules = dict(policy.named_modules())
        try:
            for layer in LAYERS:
                for kind in ("k", "v"):
                    name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{kind}_proj"
                    module = modules[name]
                    if tuple(module.weight.shape) != (320, 320) or module.bias is not None:
                        raise ValueError("Projection schema differs")
                    self.handles.append(module.register_forward_hook(self.hook(layer, kind)))
        except Exception:
            self.close()
            raise

    def set_layout(self, mask):
        import torch

        if tuple(mask.shape) != (1, 241) or mask.dtype != torch.bool:
            raise ValueError("Full native prefix mask required")
        if not mask[:, :64].all() or mask[:, 64:192].any() or not mask[:, 240:].all():
            raise ValueError("Real/empty camera and state mask differs")
        identity = {
            "shape": [1, 241, 320],
            "mask_sha256": digest(mask),
            "attended_tokens": int(mask.sum()),
        }
        if self.layout is not None and self.layout != identity:
            raise ValueError("Prefix layout varies")
        if self.reference is not None and self.reference["layout"] != identity:
            raise ValueError("Same-checkpoint mask differs")
        self.layout = identity

    def hook(self, layer, kind):
        def apply(_module, inputs, output):
            import torch

            if self.row not in range(45) or len(inputs) != 1 or self.layout is None:
                raise ValueError("Registered row/layout missing")
            x = inputs[0]
            if tuple(x.shape) != (1, 241, 320) or output.shape != x.shape:
                raise ValueError("Full 241-token projection required")
            if (
                output.device.type != "cuda"
                or output.dtype != torch.bfloat16
                or not torch.is_autocast_enabled("cuda")
            ):
                raise ValueError("Native CUDA BF16 autocast required")
            if not torch.isfinite(output).all() or not torch.isfinite(x).all():
                raise ValueError("Nonfinite native projection")
            key = (self.row, layer, kind)
            self.calls[key] = self.calls.get(key, 0) + 1
            if key not in self.cache:
                identity = {"input": digest(x), "native": digest(output)}
                if (
                    self.reference is not None
                    and identity != self.reference["rows"][f"{self.row}:{layer}:{kind}"]
                ):
                    raise ValueError("Same-base native projection drift")
                self.cache[key] = (x.detach().clone(), output.detach().clone(), identity)
            old_x, old_y, _ = self.cache[key]
            if not torch.equal(old_x, x) or not torch.equal(old_y, output):
                raise ValueError("Prefix varies with noise, denoising, or intervention")
            if self.mode == "native":
                return output
            bank_key = f"{layer}_{kind}"
            if self.mode == "self":
                split = "train" if self.row < 40 else "dev"
                row = self.row if self.row < 40 else self.row - 40
                replacement = torch.from_numpy(self.banks[split][bank_key][row].copy())
            elif self.mode == "kvmean" or self.mode == kind + "mean":
                train, dev = self.means[bank_key]
                replacement = train[self.row] if self.row < 40 else dev
            else:
                return output
            changed = replace_image(
                output, replacement.to(device=output.device, dtype=output.dtype)
            )
            if self.calls[key] == 1:
                self.applied_outputs[f"{self.row}:{layer}:{kind}"] = digest(changed)
            return changed

        return apply

    def finish(self, output):
        import numpy as np

        expected = {
            (row, layer, kind) for row in range(45) for layer in LAYERS for kind in ("k", "v")
        }
        if set(self.cache) != expected or any(self.calls.get(key) != 40 for key in expected):
            raise ValueError("Missing row/layer/noise/denoising calls")
        applied = {
            f"{row}:{layer}:{kind}"
            for row, layer, kind in expected
            if self.mode in ("self", "kvmean") or self.mode == kind + "mean"
        }
        if set(self.applied_outputs) != applied:
            raise ValueError("Applied projection set differs")
        calibration = {}
        if self.mode == "native":
            for split, rows in (("train", list(range(40))), ("dev", list(range(40, 45)))):
                arrays = {
                    f"{layer}_{kind}": np.stack(
                        [
                            self.cache[(row, layer, kind)][1][0, :64].cpu().float().numpy()
                            for row in rows
                        ]
                    )
                    for layer in LAYERS
                    for kind in ("k", "v")
                }
                path = output / f"calibration-{split}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                calibration[split] = {"rows": rows, "sha256": file_hash(path)}
        report = {
            "condition": self.condition,
            "mode": self.mode,
            "base_step": self.base,
            "parameter_intervention": False,
            "activation_intervention": self.mode != "native",
            "native_inputs_and_unmodified_outputs_repeat_exactly": True,
            "native_reference_sha256": self.reference_sha,
            "layout": self.layout,
            "hook_calls": sum(self.calls.values()),
            "calibration_files": calibration,
            "rows": {
                f"{row}:{layer}:{kind}": self.cache[(row, layer, kind)][2]
                for row, layer, kind in sorted(expected)
            },
            "dev_calibration_rows": list(range(40)),
            "train_calibration_excludes_self": True,
            "labels_used": False,
            "attention_weights_claimed_fixed": False,
            "applied_output_sha256": self.applied_outputs,
        }
        with (output / "intervention.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        return report

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
