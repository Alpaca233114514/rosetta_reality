"""Isolate one image-key direction layer with observational native softmax capture."""

from __future__ import annotations

import json

from scripts.hestia_k_factor import KeyFactor, factors
from scripts.hestia_scene_kv_mean import LAYERS, SceneKVMean

CONDITIONS = ("base640_native", "base640_krestore", "base640_kdirection") + tuple(
    f"base640_kd{layer:02d}" for layer in LAYERS
)


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered direction-layer condition")
    return 640, None


def component_mode(name):
    condition_spec(name)
    return name.split("_")[1]


def selected_layers(name):
    mode = component_mode(name)
    if mode == "native":
        return ()
    if mode in {"krestore", "kdirection"}:
        return LAYERS
    return (int(mode[2:]),)


def attention_summary(logits, probs, value_dtype):
    """Summarize actual eager logits and the BF16 probabilities used by its V product."""
    import torch

    if tuple(probs.shape) != (1, 15, 50, 241) or logits.shape != probs.shape:
        raise ValueError("Native expert attention shape differs")
    p = probs.detach().to(value_dtype).float()[0]
    if not torch.isfinite(p).all() or (p < 0).any() or p[:, :, 64:192].any():
        raise ValueError("Invalid attention probabilities or padded camera mass")
    parts = [
        p[..., :64].sum(-1),
        p[..., 192:240].sum(-1),
        p[..., 240:].sum(-1),
        p[..., 64:192].sum(-1),
    ]
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
    conditional = p[..., :64] / parts[0][..., None].clamp_min(1e-30)
    image_entropy = -(conditional * conditional.clamp_min(1e-30).log()).sum(-1)
    image_logits = logits.detach().float()[0, :, :, :64]
    stats = torch.stack(
        [
            *parts,
            entropy,
            image_entropy,
            image_logits.mean(-1),
            image_logits.std(-1, unbiased=False),
        ],
        -1,
    ).mean(1)
    return p[..., :64].mean(1).cpu().numpy(), stats.cpu().numpy()


class DirectionLayers(KeyFactor):
    def __init__(
        self, policy, checkpoint_sha256, condition, calibration_dir, episodes, train_episodes
    ):
        self.selected = selected_layers(condition)
        self.core, self.original_cross, self.original_eager = None, None, None
        self.attention, self.current_layer = {}, None
        mode = component_mode(condition)
        surrogate = (
            condition if mode in {"native", "krestore", "kdirection"} else "base640_kdirection"
        )
        super().__init__(
            policy, checkpoint_sha256, surrogate, calibration_dir, episodes, train_episodes
        )
        self.condition = condition
        if (self.heads, self.head_dim, self.query_heads) != (5, 64, 15):
            raise ValueError("Previously measured 5/64/15 head layout required")
        self.core = policy.model.vlm_with_expert
        self.original_cross = self.core.forward_cross_attn_layer
        self.original_eager = self.core.eager_attention_forward
        self.core.forward_cross_attn_layer = self.cross
        self.core.eager_attention_forward = self.eager

    def cross(self, *args, **kwargs):
        if self.current_layer is not None:
            raise ValueError("Nested expert cross attention")
        layer = args[2] if len(args) > 2 else kwargs["layer_idx"]
        if layer not in LAYERS:
            raise ValueError("Unregistered cross-attention layer")
        self.current_layer = layer
        try:
            return self.original_cross(*args, **kwargs)
        finally:
            self.current_layer = None

    def eager(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        import torch.nn.functional as functional

        args = (attention_mask, batch_size, head_dim, query_states, key_states, value_states)
        layer = self.current_layer
        # Only the first noise / first denoise pass, all fixed train40/dev5 rows.
        if (
            layer is None
            or query_states.shape[1] != 50
            or self.calls.get((self.row, f"{layer}_k")) != 1
        ):
            return self.original_eager(*args)
        if key_states.shape != (1, 241, 5, 64) or query_states.shape != (1, 50, 15, 64):
            raise ValueError("Attention capture coordinate contract differs")
        pair = self.row, layer
        if pair in self.attention:
            raise ValueError("Duplicate attention capture")
        original_softmax = functional.softmax
        captured = []

        def observe(input, *soft_args, **soft_kwargs):
            probs = original_softmax(input, *soft_args, **soft_kwargs)
            captured.append(attention_summary(input, probs, value_states.dtype))
            return probs

        functional.softmax = observe
        try:
            output = self.original_eager(*args)
        finally:
            functional.softmax = original_softmax
        if len(captured) != 1:
            raise ValueError("Expected exactly one native eager softmax")
        self.attention[pair] = captured[0]
        return output

    def hook(self, key):
        observer = SceneKVMean.hook(self, key)

        def apply(module, inputs, native):
            observed = observer(module, inputs, native)
            if key.endswith("v") or int(key.split("_")[0]) not in self.selected:
                return observed
            pair = self.row, key
            if pair not in self.transformed:
                self.transformed[pair] = factors(
                    native, self.means[key], self.norms[key], self.head_dim, self.factor_mode
                )
            return self.transformed[pair]

        return apply

    def finish(self, output):
        import numpy as np

        expected = {(row, layer) for row in range(45) for layer in LAYERS}
        if set(self.attention) != expected:
            raise ValueError("Attention observation coverage incomplete")
        replacements = {(row, f"{layer}_k") for row in range(45) for layer in self.selected}
        if set(self.transformed) != replacements:
            raise ValueError("Selected direction coverage differs")
        image = np.stack(
            [self.attention[row, layer][0] for row in range(45) for layer in LAYERS]
        ).reshape(45, 8, 15, 64)
        summary = np.stack(
            [self.attention[row, layer][1] for row in range(45) for layer in LAYERS]
        ).reshape(45, 8, 15, 8)
        with (output / "attention.npz").open("xb") as stream:
            np.savez_compressed(stream, image_probability=image, summary=summary)
        from scripts.hestia_scene_kv_mean import sha

        observation = {
            "condition": self.condition,
            "selected_layers": list(self.selected),
            "captures": len(expected),
            "noise_index": 0,
            "denoise_index": 0,
            "query_reduction": "mean over all 50 action queries",
            "probabilities": "native softmax output cast to native value dtype before observation",
            "summary_fields": [
                "image_mass",
                "language_mass",
                "state_mass",
                "padded_camera_mass",
                "entropy",
                "within_image_entropy",
                "image_logit_mean",
                "image_logit_std",
            ],
            "attention_sha256": sha(output / "attention.npz"),
            "replacement_image_sha256": {
                f"{row}:{key}": self.image_digest(value)
                for (row, key), value in self.transformed.items()
            },
            "attention_output_returned_unchanged": True,
        }
        with (output / "attention.json").open("x") as stream:
            json.dump(observation, stream, indent=2)
        self.attention.clear()
        return super().finish(output)

    @staticmethod
    def image_digest(value):
        from scripts.hestia_scene_kv_mean import digest

        return digest(value[:, :64])

    def close(self):
        if self.core is not None and self.original_cross is not None:
            self.core.forward_cross_attn_layer = self.original_cross
            self.core.eager_attention_forward = self.original_eager
            self.original_cross = self.original_eager = None
        super().close()
