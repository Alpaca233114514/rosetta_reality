"""Factor image attention mass and conditional routing without replacing native kernels."""

from __future__ import annotations

import json

from scripts.hestia_k_direction_head import DirectionHeads as ParentHeads
from scripts.hestia_k_direction_layer import attention_summary
from scripts.hestia_k_factor import KeyFactor
from scripts.hestia_scene_kv_mean import LAYERS, sha

LAYER = 1
QUERY_HEADS = tuple(range(3, 9))
CONDITIONS = tuple("base640_" + mode for mode in ("native", "rrestore", "rfull", "rintra", "rmass"))


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("Unregistered routing condition")
    return 640, None


def component_mode(name):
    condition_spec(name)
    return name.split("_")[1]


def reference_for(plan, condition):
    mode = component_mode(condition)
    if mode in {"native", "rrestore"}:
        return plan["reference_endpoints"]["640"]
    return plan["reference_direction"] if mode == "rfull" else None


def factor_probabilities(native, changed, mode, value_dtype):
    import torch

    if mode not in {"rrestore", "rfull", "rintra", "rmass"}:
        raise ValueError("Unknown routing factor")
    if native.shape != (1, 15, 50, 241) or native.shape != changed.shape:
        raise ValueError("Registered softmax shape differs")
    if native.dtype != torch.float32 or changed.dtype != native.dtype:
        raise ValueError("Original float32 softmax required")
    if not torch.isfinite(native).all() or not torch.isfinite(changed).all():
        raise ValueError("Nonfinite probability")
    other = [h for h in range(15) if h not in QUERY_HEADS]
    if not torch.equal(native[:, other], changed[:, other]):
        raise ValueError("Unselected native probabilities changed")
    if (native < 0).any() or (changed < 0).any() or native[..., 64:192].any():
        raise ValueError("Invalid or unmasked probability")
    if changed[..., 64:192].any():
        raise ValueError("Changed padded probability")
    n = native[:, QUERY_HEADS].double()
    d = changed[:, QUERY_HEADS].double()
    nm, dm = n[..., :64].sum(-1, keepdim=True), d[..., :64].sum(-1, keepdim=True)
    if (nm <= 1e-30).any() or (dm <= 1e-30).any():
        raise ValueError("Image conditional distribution undefined")
    pre_error = 0.0
    if mode in {"rrestore", "rfull"}:
        output = native if mode == "rrestore" else changed
    else:
        chosen = n.clone()
        if mode == "rintra":
            chosen[..., :64] = nm * (d[..., :64] / dm)
            expected_mass = nm
        else:
            chosen[..., :64] = dm * (n[..., :64] / nm)
            chosen[..., 64:] = d[..., 64:]
            expected_mass = dm
        pre_error = float((chosen[..., :64].sum(-1, keepdim=True) - expected_mass).abs().max())
        if pre_error > 1e-12:
            raise ValueError("Pre-cast image mass factorization failed")
        output = native.clone()
        output[:, QUERY_HEADS] = chosen.to(native.dtype)
    if not torch.equal(output[:, other], native[:, other]):
        raise ValueError("Unselected output probabilities changed")
    post = output.to(value_dtype).double()
    npost = native.to(value_dtype).double()
    sum_error = float((post.sum(-1) - 1).abs().max())
    mass_leak, conditional_leak = 0.0, 0.0
    if mode == "rintra":
        mass_leak = float(
            (post[:, QUERY_HEADS, :, :64].sum(-1) - npost[:, QUERY_HEADS, :, :64].sum(-1))
            .abs()
            .max()
        )
        if mass_leak > 0.004:
            raise ValueError("Post-BF16 image mass leakage exceeds bound")
    if mode == "rmass":
        pi, ni = post[:, QUERY_HEADS, :, :64], npost[:, QUERY_HEADS, :, :64]
        if (pi.sum(-1) <= 0).any() or (ni.sum(-1) <= 0).any():
            raise ValueError("Post-BF16 conditional undefined")
        conditional_leak = float(
            (
                0.5 * (pi / pi.sum(-1, keepdim=True) - ni / ni.sum(-1, keepdim=True)).abs().sum(-1)
            ).max()
        )
        if conditional_leak > 0.004:
            raise ValueError("Post-BF16 conditional leakage exceeds bound")
    if sum_error > 0.004 or not torch.isfinite(post).all() or post[..., 64:192].any():
        raise ValueError("Post-BF16 probability contract failed")
    return output, dict(
        pre_mass_error=pre_error,
        post_sum_error=sum_error,
        post_intra_mass_leak=mass_leak,
        post_mass_conditional_tv_leak=conditional_leak,
    )


class DirectionHeads(ParentHeads):
    def __init__(
        self, policy, checkpoint_sha256, condition, calibration_dir, episodes, train_episodes
    ):
        self.routing_mode = component_mode(condition)
        self.routing_calls, self.routing_maxima = 0, {}
        surrogate = "base640_native" if self.routing_mode == "native" else "base640_kdirection"
        super().__init__(
            policy, checkpoint_sha256, surrogate, calibration_dir, episodes, train_episodes
        )
        self.condition = condition
        if self.routing_mode != "native":
            self.selected_heads = (1, 2)

    def eager(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        import torch
        import torch.nn.functional as functional

        if self.routing_mode == "native" or self.current_layer != LAYER:
            return super().eager(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
        if key_states.shape != (1, 241, 5, 64) or query_states.shape != (1, 50, 15, 64):
            raise ValueError("Routing attention coordinate contract differs")
        original_softmax = functional.softmax
        native_capture, changed_capture = [], []
        native_keys = self.cache[self.row, "1_k"][1].reshape_as(key_states)
        args = (attention_mask, batch_size, head_dim, query_states)

        def capture(input, *soft_args, **soft_kwargs):
            probs = original_softmax(input, *soft_args, **soft_kwargs)
            native_capture.append((input, probs))
            return probs

        def replace(input, *soft_args, **soft_kwargs):
            changed = original_softmax(input, *soft_args, **soft_kwargs)
            chosen, statistics = factor_probabilities(
                native_capture[0][1], changed, self.routing_mode, value_states.dtype
            )
            changed_capture.append((input, chosen, statistics))
            return chosen

        functional.softmax = capture
        try:
            self.original_eager(*args, native_keys, value_states)
            if len(native_capture) != 1:
                raise ValueError("Expected one native softmax")
            functional.softmax = replace
            output = self.original_eager(*args, key_states, value_states)
        finally:
            functional.softmax = original_softmax
        if len(changed_capture) != 1:
            raise ValueError("Expected one changed softmax")
        logits, chosen, statistics = changed_capture[0]
        self.routing_calls += 1
        for key, value in statistics.items():
            self.routing_maxima[key] = max(self.routing_maxima.get(key, 0.0), value)
        if self.calls.get((self.row, "1_k")) == 1:
            if (self.row, LAYER) in self.attention:
                raise ValueError("Duplicate routing capture")
            if self.routing_mode == "rrestore":
                logits = native_capture[0][0]
            self.attention[self.row, LAYER] = attention_summary(logits, chosen, value_states.dtype)
        if not torch.isfinite(output).all():
            raise ValueError("Nonfinite routed output")
        return output

    def finish(self, output):
        import numpy as np

        expected = 0 if self.routing_mode == "native" else 1800
        if self.routing_calls != expected:
            raise ValueError("Routing coverage incomplete")
        with (output / "routing.json").open("x") as stream:
            json.dump(
                dict(
                    mode=self.routing_mode,
                    layer=LAYER,
                    kv_heads=[1, 2],
                    query_heads=list(QUERY_HEADS),
                    calls=self.routing_calls,
                    maxima=self.routing_maxima,
                    original_eager_kernel=True,
                    matched_current_q_and_v=True,
                    logit_summary=(
                        "Hybrids retain pre-mixing changed QK logit fields; "
                        "probability fields are actual"
                    ),
                ),
                stream,
                indent=2,
            )
        with (output / "head-selection.json").open("x") as stream:
            json.dump(
                dict(
                    layer=LAYER,
                    heads=list(self.selected_heads),
                    condition=self.condition,
                    head_dim=64,
                    query_heads_per_kv_head=3,
                ),
                stream,
                indent=2,
            )
        captures = {(row, layer) for row in range(45) for layer in LAYERS}
        replacements = {(row, f"{layer}_k") for row in range(45) for layer in self.selected}
        if set(self.attention) != captures or set(self.transformed) != replacements:
            raise ValueError("Routing attention or replacement coverage incomplete")
        image = np.stack([self.attention[r, layer][0] for r in range(45) for layer in LAYERS])
        summary = np.stack([self.attention[r, layer][1] for r in range(45) for layer in LAYERS])
        with (output / "attention.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                image_probability=image.reshape(45, 8, 15, 64),
                summary=summary.reshape(45, 8, 15, 8),
            )
        record = dict(
            condition=self.condition,
            selected_layers=list(self.selected),
            captures=len(captures),
            noise_index=0,
            denoise_index=0,
            query_reduction="mean over all50 action queries",
            probabilities="actual returned softmax probabilities after native BF16 value cast",
            softmax_return_overridden=self.routing_mode in {"rrestore", "rintra", "rmass"},
            routing_mode=self.routing_mode,
            summary_fields=[
                "image_mass",
                "language_mass",
                "state_mass",
                "padded_camera_mass",
                "entropy",
                "within_image_entropy",
                "image_logit_mean",
                "image_logit_std",
            ],
            attention_sha256=sha(output / "attention.npz"),
            replacement_image_sha256={
                f"{row}:{key}": self.image_digest(value)
                for (row, key), value in self.transformed.items()
            },
            attention_output_returned_unchanged=True,
        )
        with (output / "attention.json").open("x") as stream:
            json.dump(record, stream, indent=2)
        self.attention.clear()
        return KeyFactor.finish(self, output)
