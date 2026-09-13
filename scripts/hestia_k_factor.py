"""Per-head image-key direction/magnitude factorial; train40 calibration only."""

from __future__ import annotations

import json

from scripts.hestia_scene_kv_mean import LAYERS, SceneKVMean, digest, sha, train_mean

CONDITIONS = tuple(
    "base640_" + mode
    for mode in ("native", "krestore", "kmean", "kdirection", "kmagnitude", "knormavg", "kdiravg")
)


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered key-factor condition")
    return 640, None


def component_mode(name):
    condition_spec(name)
    return name.split("_")[1]


def factors(native, mean, average_norm, head_dim, mode):
    """Operate in actual expert attention coordinates; one final BF16 cast."""
    import torch

    if mode not in {component_mode(c) for c in CONDITIONS} - {"native"}:
        raise ValueError("Unknown factor")
    if native.shape != (1, 241, 320) or mean.shape != (1, 64, 320):
        raise ValueError("Invalid full-prefix contract")
    if head_dim <= 0 or 320 % head_dim or native.dtype != torch.bfloat16:
        raise ValueError("Invalid attention head contract")
    k = native[:, :64].double().reshape(1, 64, -1, head_dim)
    m = mean.double().reshape_as(k)
    r = torch.linalg.vector_norm(k, dim=-1, keepdim=True)
    s = torch.linalg.vector_norm(m, dim=-1, keepdim=True)
    if (
        not torch.isfinite(k).all()
        or not torch.isfinite(m).all()
        or (r <= 1e-12).any()
        or (s <= 1e-12).any()
    ):
        raise ValueError("Nonfinite or zero key direction")
    if (
        average_norm.shape != r.shape
        or not torch.isfinite(average_norm).all()
        or (average_norm <= 1e-12).any()
    ):
        raise ValueError("Invalid train-only average norm")
    output = native.clone()
    if mode == "krestore":
        output[:, :64] = mean
        output[:, :64] = native[:, :64]
    elif mode == "kmean":
        output[:, :64] = mean
    else:
        direction = m / s if mode in {"kdirection", "kdiravg"} else k / r
        radius = {
            "kdirection": r,
            "kmagnitude": s,
            "knormavg": average_norm,
            "kdiravg": average_norm,
        }[mode]
        output[:, :64] = (direction * radius).reshape_as(mean).to(native.dtype)
    if not torch.equal(output[:, 64:], native[:, 64:]):
        raise ValueError("Unselected positions changed")
    return output


class KeyFactor(SceneKVMean):
    def __init__(
        self, policy, checkpoint_sha256, condition, calibration_dir, episodes, train_episodes
    ):
        import numpy as np
        import torch

        condition_spec(condition)
        core = policy.model.vlm_with_expert
        self.head_dim = int(core.lm_expert.layers[1].self_attn.head_dim)
        self.heads = int(core.num_key_value_heads)
        self.query_heads = int(core.num_attention_heads)
        if self.head_dim * self.heads != 320 or self.query_heads % self.heads:
            raise ValueError("Expert grouped-query head layout differs")
        for layer in LAYERS:
            attn = core.lm_expert.layers[layer].self_attn
            if (
                attn.head_dim != self.head_dim
                or attn.q_proj.out_features != self.query_heads * self.head_dim
            ):
                raise ValueError("Expert head layout varies by layer")
        if core.config.text_config.head_dim != self.head_dim:
            raise ValueError("Native attention scale differs from expert head dimension")
        self.factor_mode = component_mode(condition)
        self.norms, self.transformed = {}, {}
        # Reuse native full-prefix observer, never its replacement/calibration loader.
        super().__init__(
            policy, checkpoint_sha256, "base640_native", calibration_dir, episodes, train_episodes
        )
        self.condition = condition
        if self.factor_mode != "native":
            result = json.loads((calibration_dir / "result.json").read_text())
            path = calibration_dir / "calibration.json"
            if result["condition"] != "base640_native" or result["k_factor"][
                "calibration_sha256"
            ] != sha(path):
                raise ValueError("Native calibration seal differs")
            if result["same_device_control"].get("passed") is not True or any(
                result["same_device_control"][key]["max_abs"] != 0
                for key in ("normalized_predictions", "standard_predictions")
            ):
                raise ValueError("Exact native endpoint required")
            self.reference = json.loads(path.read_text())
            expected = {
                "checkpoint_sha256": checkpoint_sha256,
                "episodes": episodes,
                "training_episodes": train_episodes,
                "training_rows": self.train_rows,
                "head_dim": self.head_dim,
                "kv_heads": self.heads,
                "query_heads": self.query_heads,
                "labels_used": False,
            }
            if any(self.reference.get(k) != v for k, v in expected.items()):
                raise ValueError("Calibration checkpoint, split or head identity differs")
            npz = calibration_dir / "calibration.npz"
            if sha(npz) != self.reference["arrays_sha256"]:
                raise ValueError("Calibration array SHA differs")
            with np.load(npz, allow_pickle=False) as arrays:
                if set(arrays.files) != {
                    f"{layer}_{kind}" for layer in LAYERS for kind in ("mean", "norm")
                }:
                    raise ValueError("Calibration array set differs")
                for layer in LAYERS:
                    m, n = arrays[f"{layer}_mean"], arrays[f"{layer}_norm"]
                    if (
                        m.shape != (1, 64, 320)
                        or m.dtype != np.float32
                        or n.shape != (1, 64, self.heads, 1)
                        or n.dtype != np.float64
                    ):
                        raise ValueError("Calibration array layout differs")
                    mean = torch.from_numpy(m.copy()).to(torch.bfloat16)
                    if not np.array_equal(mean.float().numpy(), m):
                        raise ValueError("Mean is not exact BF16")
                    device = next(policy.parameters()).device
                    self.means[f"{layer}_k"] = mean.to(device)
                    self.norms[f"{layer}_k"] = torch.from_numpy(n.copy()).to(device)

    def hook(self, key):
        observer = super().hook(key)

        def apply(module, inputs, native):
            import torch

            observed = observer(module, inputs, native)
            if self.factor_mode == "native" or key.endswith("v"):
                return observed
            pair = (self.row, key)
            if pair not in self.transformed:
                self.transformed[pair] = factors(
                    native, self.means[key], self.norms[key], self.head_dim, self.factor_mode
                )
            output = self.transformed[pair]
            if not torch.equal(output[:, 64:], native[:, 64:]):
                raise ValueError("Non-image prefix changed")
            return output

        return apply

    def finish(self, output):
        import numpy as np
        import torch

        expected = {(r, f"{layer}_{k}") for r in range(45) for layer in LAYERS for k in ("k", "v")}
        if set(self.cache) != expected or any(self.calls.get(p) != 40 for p in expected):
            raise ValueError("Native hook coverage incomplete")
        if self.factor_mode == "native":
            arrays, raw = {}, {}
            for layer in LAYERS:
                values = torch.stack([self.cache[(r, f"{layer}_k")][1][:, :64] for r in range(45)])
                selected = (
                    values[self.train_rows].double().reshape(40, 1, 64, self.heads, self.head_dim)
                )
                arrays[f"{layer}_mean"] = train_mean(values, self.train_rows).cpu().float().numpy()
                arrays[f"{layer}_norm"] = (
                    torch.linalg.vector_norm(selected, dim=-1, keepdim=True).mean(0).cpu().numpy()
                )
                raw[str(layer)] = values.cpu().float().numpy()
            with (output / "calibration.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            with (output / "native-keys.npz").open("xb") as stream:
                np.savez_compressed(stream, **raw)
            calibration = {
                "base_step": 640,
                "checkpoint_sha256": self.checkpoint_sha256,
                "episodes": self.episodes,
                "training_episodes": self.train_episodes,
                "training_rows": self.train_rows,
                "labels_used": False,
                "head_dim": self.head_dim,
                "kv_heads": self.heads,
                "query_heads": self.query_heads,
                "layout": self.layout,
                "arrays_sha256": sha(output / "calibration.npz"),
                "native_keys_sha256": sha(output / "native-keys.npz"),
                "rows": {f"{r}:{k}": self.cache[(r, k)][2] for r, k in sorted(expected)},
            }
            with (output / "calibration.json").open("x") as stream:
                json.dump(calibration, stream, indent=2)
        report = {
            "condition": self.condition,
            "base_step": 640,
            "mode": self.factor_mode,
            "head_dim": self.head_dim,
            "kv_heads": self.heads,
            "query_heads": self.query_heads,
            "layout": self.layout,
            "hook_calls": sum(self.calls.values()),
            "parameter_intervention": False,
            "unchanged_other_projection_and_positions": True,
            "native_inputs_and_unmodified_outputs_repeat_exactly": True,
            "labels_used_for_means": False,
            "calibration_sha256": sha(
                (output if self.factor_mode == "native" else self.calibration_dir)
                / "calibration.json"
            ),
            "replacement_sha256": {
                f"{r}:{k}": digest(value) for (r, k), value in sorted(self.transformed.items())
            },
        }
        with (output / "intervention.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        self.cache.clear()
        self.transformed.clear()
        return report
