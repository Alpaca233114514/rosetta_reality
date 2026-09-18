"""Bounded same-sample Q/probability replay on pinned Hestia cross attention."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.hestia_scene_kv import LAYERS, calibration_means, digest, file_hash, replace_image

RUN_ID = "hestia-q-replay-20260913-001"
MODES = ("native", "qself", "aself", "kmean", "kmean_ablock", "kmean_qnative")
CONDITIONS = ("base640", "base1280")


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered replay checkpoint")
    return int(name[4:]), "replay"


def validate_protocol(plan):
    expected = {
        "id": RUN_ID,
        "launchable": True,
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "maximum_policy_forwards": 2160,
        "optimizer_steps": 0,
        "full_prefix_shape": [1, 241, 320],
        "replace_token_range": [0, 64],
        "work_deadline_seconds": 4200,
        "shutdown_deadline_seconds": 4800,
        "dev_calibration": "train40_only",
        "train_calibration": "train39_excluding_self",
    }
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen Q replay protocol differs")


def attention_parts(core, mask, batch, head_dim, q, k, v, replacement=None):
    """Keep exactly the pinned eager operation order, including autocast behavior."""
    import torch

    nh, nk = core.num_attention_heads, core.num_key_value_heads
    ng = nh // nk
    length = k.shape[1]
    k = k[:, :, :, None, :].expand(batch, length, nk, ng, head_dim)
    k = k.reshape(batch, length, nk * ng, head_dim)
    v = v[:, :, :, None, :].expand(batch, length, nk, ng, head_dim)
    v = v.reshape(batch, length, nk * ng, head_dim)
    q, k = q.to(dtype=torch.float32), k.to(dtype=torch.float32)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    weights = torch.matmul(q, k.transpose(2, 3))
    weights *= head_dim**-0.5
    weights = weights.to(dtype=torch.float32)
    masked = torch.where(mask[:, None, :, :], weights, torch.finfo(weights.dtype).min)
    probs = torch.nn.functional.softmax(masked, dim=-1)
    probs = probs.to(dtype=v.dtype)
    if replacement is not None:
        if replacement.shape != probs.shape or replacement.dtype != probs.dtype:
            raise ValueError("Probability replay layout differs")
        probs = replacement
    result = torch.matmul(probs, v.permute(0, 2, 1, 3))
    result = result.permute(0, 2, 1, 3).reshape(batch, -1, nk * ng * head_dim)
    return result, probs


class Replay:
    def __init__(self, policy, base, source, output):
        import numpy as np
        import torch

        self.base, self.source, self.output = base, Path(source), Path(output)
        self.core = policy.model.vlm_with_expert
        self.original_attention = self.core.eager_attention_forward
        self.original_cross = self.core.forward_cross_attn_layer
        self.handles, self.refs = [], {}
        self.layer, self.row, self.noise_index, self.mode = None, None, None, None
        self.call = 0
        self.completed = {mode: 0 for mode in MODES}
        self.layout = None
        self.events = (self.output / "attention.jsonl").open("x")
        self.projection_calls = 0
        self.kmeans, self.banks = {}, {}
        self.projection_reference = json.loads(
            (self.source / f"base{base}_native/intervention.json").read_text()
        )
        try:
            for split in ("train", "dev"):
                path = self.source / f"base{base}_native/calibration-{split}.npz"
                entry = self.projection_reference["calibration_files"][split]
                if file_hash(path) != entry["sha256"]:
                    raise ValueError("Calibration seal changed")
                with np.load(path, allow_pickle=False) as bank:
                    self.banks[split] = {k: bank[k] for k in bank.files}
            for layer in LAYERS:
                loo, mean = calibration_means(self.banks["train"][f"{layer}_k"])
                self.kmeans[layer] = (torch.from_numpy(loo), torch.from_numpy(mean))
            modules = dict(policy.named_modules())
            for layer in LAYERS:
                for kind in ("k", "v"):
                    name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{kind}_proj"
                    module = modules[name]
                    if tuple(module.weight.shape) != (320, 320) or module.bias is not None:
                        raise ValueError("Projection layout changed")
                    self.handles.append(module.register_forward_hook(self.projection(layer, kind)))
            self.core.forward_cross_attn_layer = self.cross
            self.core.eager_attention_forward = self.attention
        except Exception:
            self.close()
            raise

    def set_layout(self, mask):
        import torch

        if tuple(mask.shape) != (1, 241) or mask.dtype != torch.bool:
            raise ValueError("Prefix mask layout changed")
        identity = {
            "shape": [1, 241, 320],
            "mask_sha256": digest(mask),
            "attended_tokens": int(mask.sum()),
        }
        if identity != self.projection_reference["layout"]:
            raise ValueError("Historical prefix mask changed")
        self.layout = identity

    def begin(self, row, noise_index, mode):
        if row not in range(45) or noise_index not in range(4) or mode not in MODES:
            raise ValueError("Unregistered sample/noise/mode")
        if mode == "native":
            self.refs = {}
            self.case = (row, noise_index)
        elif self.case != (row, noise_index) or len(self.refs) != 80:
            raise ValueError("Replay reference sample differs or is incomplete")
        if self.completed[mode] != noise_index * 45 + row:
            raise ValueError("Replay sample ordering changed")
        if (
            mode != "native"
            and self.completed[MODES[MODES.index(mode) - 1]] != noise_index * 45 + row + 1
        ):
            raise ValueError("Earlier sample control incomplete")
        self.row, self.noise_index, self.mode, self.call = row, noise_index, mode, 0

    def end(self):
        if self.call != 80:
            raise ValueError("Cross-attention count differs")
        self.completed[self.mode] += 1
        self.events.flush()

    def projection(self, layer, kind):
        def hook(_module, inputs, output):
            import torch

            if self.mode not in MODES or self.layout is None or len(inputs) != 1:
                raise ValueError("Unregistered projection context")
            x = inputs[0]
            if tuple(output.shape) != (1, 241, 320) or x.shape != output.shape:
                raise ValueError("Full prefix projection required")
            if output.dtype != torch.bfloat16 or output.device.type != "cuda":
                raise ValueError("Native CUDA BF16 required")
            identity = {"input": digest(x), "native": digest(output)}
            if identity != self.projection_reference["rows"][f"{self.row}:{layer}:{kind}"]:
                raise ValueError("Native projection drift")
            self.projection_calls += 1
            if kind == "k" and self.mode.startswith("kmean"):
                loo, mean = self.kmeans[layer]
                value = loo[self.row] if self.row < 40 else mean
                return replace_image(output, value.to(device=output.device, dtype=output.dtype))
            return output

        return hook

    def cross(self, *args, **kwargs):
        if self.layer is not None:
            raise ValueError("Nested cross attention")
        self.layer = kwargs.get("layer_idx", args[2] if len(args) > 2 else None)
        try:
            return self.original_cross(*args, **kwargs)
        finally:
            self.layer = None

    def attention(self, mask, batch, hd, q, k, v):
        import numpy as np
        import torch

        if self.layer is None:
            return self.original_attention(mask, batch, hd, q, k, v)
        if (
            self.layer not in LAYERS
            or tuple(mask.shape) != (1, 50, 241)
            or q.shape[1] != 50
            or k.shape[1] != 241
            or v.shape != k.shape
        ):
            raise ValueError("Expert cross-attention signature changed")
        expected_layer = LAYERS[self.call % 8]
        step = self.call // 8
        if expected_layer != self.layer or step >= 10:
            raise ValueError("Layer or denoising order changed")
        if not all(torch.isfinite(t).all() for t in (q, k, v)):
            raise ValueError("Nonfinite attention input")
        key = (step, self.layer)
        live_q = q
        if self.mode == "native":
            original = self.original_attention(mask, batch, hd, q, k, v)
            result, probs = attention_parts(self.core, mask, batch, hd, q, k, v)
            if not torch.equal(original, result):
                raise ValueError("Instrumented eager arithmetic differs from native")
            self.refs[key] = (
                q.detach().clone(),
                probs.detach().clone(),
                mask.detach().clone(),
                v.detach().clone(),
            )
            if self.noise_index == 0 and self.row in (0, 40):
                path = self.output / f"sentinel-{self.row}-{step}-{self.layer}.npz"
                values = {"q": q, "k": k, "v": v, "probs": probs, "output": result}
                # float32 exactly carries BF16; dtype metadata permits exact restoration.
                arrays = {name: t.detach().cpu().float().numpy() for name, t in values.items()}
                arrays["mask"] = mask.cpu().numpy()
                arrays["dtypes"] = np.asarray([str(t.dtype) for t in values.values()])
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
        else:
            q0, p0, mask0, v0 = self.refs[key]
            if not torch.equal(mask, mask0) or not torch.equal(v, v0):
                raise ValueError("Non-intervened V or mask changed")
            if self.mode in ("qself", "kmean_qnative"):
                q = q0
            replacement = p0 if self.mode in ("aself", "kmean_ablock") else None
            result, probs = attention_parts(self.core, mask, batch, hd, q, k, v, replacement)
        event = {
            "row": self.row,
            "noise": self.noise_index,
            "mode": self.mode,
            "step": step,
            "layer": self.layer,
            "q": digest(q),
            "live_q": digest(live_q),
            "k": digest(k),
            "v": digest(v),
            "mask": digest(mask),
            "probs": digest(probs),
            "output": digest(result),
            "q_dtype": str(q.dtype),
            "p_dtype": str(probs.dtype),
            "q_shape": list(q.shape),
            "p_shape": list(probs.shape),
        }
        if self.mode != "native":
            q0, p0, _, _ = self.refs[key]
            event["live_q_delta_l2"] = float((live_q.float() - q0.float()).square().sum().sqrt())
            event["prob_delta_l2"] = float((probs.float() - p0.float()).square().sum().sqrt())
        self.events.write(json.dumps(event, allow_nan=False) + "\n")
        self.call += 1
        return result

    def finish(self):
        if self.completed != {m: 180 for m in MODES} or self.projection_calls != 172800:
            raise ValueError("Incomplete replay collection")
        return {
            "completed": self.completed,
            "projection_calls": self.projection_calls,
            "cross_attention_calls": sum(self.completed.values()) * 80,
            "labels_used_for_calibration": False,
            "layout": self.layout,
        }

    def close(self):
        self.core.eager_attention_forward = self.original_attention
        self.core.forward_cross_attn_layer = self.original_cross
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if not self.events.closed:
            self.events.close()
