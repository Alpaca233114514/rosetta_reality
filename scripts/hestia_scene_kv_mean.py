"""Same-checkpoint, train-only image K/V means with sealed native controls."""

from __future__ import annotations

import hashlib
import json

LAYERS = tuple(range(1, 16, 2))
CONDITIONS = (
    "base640_native",
    "base1280_native",
    "base640_krestore",
    "base640_vrestore",
    "base1280_krestore",
    "base1280_vrestore",
    "base640_kmean",
    "base640_vmean",
    "base1280_kmean",
    "base1280_vmean",
)


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered scene K/V condition")
    return int(name.split("_")[0].removeprefix("base")), None


def component_mode(name):
    condition_spec(name)
    return name.split("_")[1]


def digest(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    # Preserve BF16 bit patterns, including signed zero; no numeric conversion.
    return hashlib.sha256(
        str(value.dtype).encode()
        + str(tuple(value.shape)).encode()
        + value.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_mean(values, rows):
    """Rows are explicit indices; development rows never enter the accumulator."""
    import torch

    if values.ndim != 4 or not rows or len(set(rows)) != len(rows):
        raise ValueError("Invalid calibration shape or duplicate training rows")
    if min(rows) < 0 or max(rows) >= values.shape[0] or not torch.isfinite(values).all():
        raise ValueError("Invalid calibration rows or nonfinite value")
    return values[rows].double().mean(0).to(torch.bfloat16)


def replace_image(native, mean, mode):
    """Restoration directly copies the original bytes, avoiding BF16 cancellation."""
    if mode not in {"mean", "restore"} or mean.shape != native[:, :64].shape:
        raise ValueError("Invalid image replacement contract")
    if mean.dtype != native.dtype or mean.device != native.device:
        raise ValueError("Replacement precision or device differs")
    output = native.clone()
    output[:, :64] = mean
    if mode == "restore":
        output[:, :64] = native[:, :64]
    return output


class SceneKVMean:
    def __init__(
        self, policy, checkpoint_sha256, condition, calibration_dir, episodes, train_episodes
    ):
        import numpy as np
        import torch

        self.base, _ = condition_spec(condition)
        self.condition, self.mode = condition, component_mode(condition)
        if (
            len(episodes) != 45
            or len(set(episodes)) != 45
            or len(train_episodes) != 40
            or len(set(train_episodes)) != 40
            or not set(train_episodes) < set(episodes)
        ):
            raise ValueError("Registered train40/dev5 split required")
        self.episodes, self.train_episodes = episodes, train_episodes
        self.train_rows = [episodes.index(ep) for ep in train_episodes]
        self.checkpoint_sha256 = checkpoint_sha256
        self.calibration_dir = calibration_dir
        self.handles, self.cache, self.calls, self.means = [], {}, {}, {}
        self.row, self.layout, self.reference = None, None, None
        if self.mode != "native":
            path = calibration_dir / "calibration.json"
            result = json.loads((calibration_dir / "result.json").read_text())
            if (
                result.get("condition") != f"base{self.base}_native"
                or result.get("same_device_control", {}).get("passed") is not True
                or result["scene_kv_mean"]["calibration_sha256"] != sha(path)
            ):
                raise ValueError("Calibration not sealed by matching verified native endpoint")
            for key in ("normalized_predictions", "standard_predictions"):
                if result["same_device_control"][key]["max_abs"] != 0:
                    raise ValueError("Native calibration endpoint must be exact")
            self.reference = json.loads(path.read_text())
            expected = {
                "base_step": self.base,
                "checkpoint_sha256": checkpoint_sha256,
                "episodes": episodes,
                "training_episodes": train_episodes,
                "training_rows": self.train_rows,
                "labels_used": False,
            }
            if any(self.reference.get(k) != v for k, v in expected.items()):
                raise ValueError("Calibration checkpoint or split identity differs")
            npz = calibration_dir / "calibration.npz"
            if sha(npz) != self.reference["arrays_sha256"]:
                raise ValueError("Calibration arrays changed")
            with np.load(npz, allow_pickle=False) as arrays:
                expected_keys = {f"{layer}_{kind}" for layer in LAYERS for kind in ("k", "v")}
                if set(arrays.files) != expected_keys:
                    raise ValueError("Calibration projection set differs")
                for key in expected_keys:
                    array = arrays[key]
                    if array.shape != (1, 64, 320) or array.dtype != np.float32:
                        raise ValueError("Calibration array layout differs")
                    value = torch.from_numpy(array.copy()).to(torch.bfloat16)
                    if not torch.isfinite(value).all() or not np.array_equal(
                        value.float().numpy(), array
                    ):
                        raise ValueError("Calibration must preserve exact BF16 means")
                    self.means[key] = value
        modules = dict(policy.named_modules())
        try:
            for layer in LAYERS:
                for kind in ("k", "v"):
                    name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{kind}_proj"
                    module = modules[name]
                    if tuple(module.weight.shape) != (320, 320) or module.bias is not None:
                        raise ValueError("Native projection schema differs")
                    key = f"{layer}_{kind}"
                    if key in self.means:
                        self.means[key] = self.means[key].to(module.weight.device)
                    self.handles.append(module.register_forward_hook(self.hook(key)))
        except Exception:
            self.close()
            raise

    def set_layout(self, mask):
        import torch

        if (
            tuple(mask.shape) != (1, 241)
            or mask.dtype != torch.bool
            or not mask[:, :64].all()
            or mask[:, 64:192].any()
            or not mask[:, 240:].all()
            or int(mask.sum()) != 73
        ):
            raise ValueError("Registered 241-position prefix with 73 active tokens required")
        layout = {"shape": [1, 241, 320], "mask_sha256": digest(mask), "attended_tokens": 73}
        if self.layout is not None and self.layout != layout:
            raise ValueError("Prefix mask varies")
        if self.reference is not None and self.reference["layout"] != layout:
            raise ValueError("Prefix mask differs from calibration")
        self.layout = layout

    def hook(self, key):
        def apply(_module, inputs, native):
            import torch

            if self.row not in range(45) or self.layout is None or len(inputs) != 1:
                raise ValueError("Missing row or full-prefix context")
            x = inputs[0]
            if tuple(x.shape) != (1, 241, 320) or native.shape != x.shape:
                raise ValueError("Full native GEMM prefix shape differs")
            if native.dtype != torch.bfloat16 or not torch.is_autocast_enabled("cuda"):
                raise ValueError("Native CUDA BF16 autocast required")
            if not torch.isfinite(x).all() or not torch.isfinite(native).all():
                raise ValueError("Nonfinite prefix")
            pair = (self.row, key)
            self.calls[pair] = self.calls.get(pair, 0) + 1
            # First call binds full prefix; repeats compare device tensors cheaply.
            if pair not in self.cache:
                identity = {"input": digest(x), "native": digest(native)}
                if self.reference is not None:
                    if identity != self.reference["rows"][f"{self.row}:{key}"]:
                        raise ValueError("Prefix differs from same-checkpoint native calibration")
                self.cache[pair] = (x.detach().clone(), native.detach().clone(), identity)
            saved_x, saved_y, _ = self.cache[pair]
            if not torch.equal(x, saved_x) or not torch.equal(native, saved_y):
                raise ValueError("Prefix varies across noise, denoise or intervention")
            if self.mode == "native" or not self.mode.startswith(key[-1]):
                return native
            mode = self.mode[1:]
            result = replace_image(native, self.means[key], mode)
            if not torch.equal(result[:, 64:], native[:, 64:]):
                raise ValueError("Unselected prefix positions changed")
            if mode == "restore" and not torch.equal(result, native):
                raise ValueError("Restoration must be exact")
            return result

        return apply

    def finish(self, output):
        import numpy as np
        import torch

        expected = {
            (row, f"{layer}_{kind}") for row in range(45) for layer in LAYERS for kind in ("k", "v")
        }
        if set(self.cache) != expected or any(self.calls.get(pair) != 40 for pair in expected):
            raise ValueError("Native hook coverage incomplete")
        if self.mode == "native":
            arrays = {}
            for layer in LAYERS:
                for kind in ("k", "v"):
                    key = f"{layer}_{kind}"
                    # Gather train rows only; dev activations are identities, never mean inputs.
                    values = torch.stack([self.cache[(r, key)][1][:, :64] for r in self.train_rows])
                    mean = train_mean(values, list(range(len(self.train_rows))))
                    arrays[key] = mean.cpu().float().numpy()
            with (output / "calibration.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            calibration = {
                "base_step": self.base,
                "checkpoint_sha256": self.checkpoint_sha256,
                "episodes": self.episodes,
                "training_episodes": self.train_episodes,
                "training_rows": self.train_rows,
                "labels_used": False,
                "layout": self.layout,
                "arrays_sha256": sha(output / "calibration.npz"),
                "rows": {f"{r}:{k}": self.cache[(r, k)][2] for r, k in sorted(expected)},
            }
            with (output / "calibration.json").open("x") as stream:
                json.dump(calibration, stream, indent=2)
        report = {
            "condition": self.condition,
            "base_step": self.base,
            "mode": self.mode,
            "checkpoint_sha256": self.checkpoint_sha256,
            "layout": self.layout,
            "hook_calls": sum(self.calls.values()),
            "parameter_intervention": False,
            "unchanged_other_projection_and_positions": True,
            "native_inputs_and_unmodified_outputs_repeat_exactly": True,
            "calibration_sha256": sha(
                (output if self.mode == "native" else self.calibration_dir) / "calibration.json"
            ),
            "training_rows": self.train_rows,
            "labels_used_for_means": False,
            "attention_weights_claimed_fixed": False,
        }
        with (output / "intervention.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        self.cache.clear()
        return report

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
