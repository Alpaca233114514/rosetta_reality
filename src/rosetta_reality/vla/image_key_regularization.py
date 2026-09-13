"""Opt-in image-key scene variance objective; no imports load models or data."""

from __future__ import annotations

import hashlib
import json
import math
from functools import wraps

LAYERS = tuple(range(1, 16, 2))
UPSTREAM_SHA256 = "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d"
MARKER = "_rosetta_image_key_scene_regularization"


def validate_scales(scales):
    if (
        not isinstance(scales, dict)
        or any(type(key) is not int for key in scales)
        or set(scales) != set(LAYERS)
    ):
        raise ValueError("Calibration must contain all eight registered K layers")
    for value in scales.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Calibration variance must be numeric")
        if not math.isfinite(value) or value <= 1e-12:
            raise ValueError("Calibration variance is nonfinite or degenerate")


def load_calibration(path, expected_sha256, expected):
    """Hash and parse the same bytes; reject JSON key aliases and bool/int drift."""

    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate calibration JSON key")
            value[key] = item
        return value

    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("K calibration SHA changed")
    report = json.loads(raw, object_pairs_hook=unique_pairs)
    if not isinstance(report, dict) or any(
        type(report.get(key)) is not type(value) or report.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("Fresh-base train-only K calibration identity differs")
    values = report.get("layer_variance")
    if not isinstance(values, dict) or set(values) != {str(layer) for layer in LAYERS}:
        raise ValueError("K calibration layer identity differs")
    scales = {int(key): value for key, value in values.items()}
    validate_scales(scales)
    return scales


def scene_variance(keys):
    """Unbiased variance across scenes, retaining shared positional structure."""
    import torch

    if keys.ndim != 3 or keys.shape[0] < 2 or min(keys.shape[1:]) < 1:
        raise ValueError("Keys require [B>=2, positions, channels]")
    if not keys.is_floating_point() or not bool(torch.isfinite(keys).all()):
        raise ValueError("Keys must be finite floating-point values")
    # Accumulate low-precision activations in FP32, without breaking autograd.
    value = keys.float() if keys.dtype in (torch.float16, torch.bfloat16) else keys
    centered = value - value.mean(dim=0, keepdim=True)
    result = centered.square().sum() / ((value.shape[0] - 1) * value.shape[1] * value.shape[2])
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("Nonfinite scene variance")
    return result


def regularized_forward(policy, native, args, kwargs, scales, coefficient):
    """Capture each full K projection once, add a scalar after native reduction."""
    import torch

    if not policy.training or not torch.is_grad_enabled() or coefficient == 0:
        return native(policy, *args, **kwargs)
    reduction = kwargs.get("reduction", args[3] if len(args) > 3 else "mean")
    if reduction != "mean":
        raise ValueError("Scene regularization requires native scalar mean reduction")
    if getattr(policy, MARKER, False):
        raise RuntimeError("Nested regularized policy forward")
    modules = dict(policy.named_modules())
    projections = {}
    for layer in LAYERS:
        name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.k_proj"
        module = modules.get(name)
        if module is None or tuple(module.weight.shape) != (320, 320) or module.bias is not None:
            raise ValueError("Registered K projection layout differs")
        projections[layer] = module
    captured, handles = {}, []
    original_images = policy.prepare_images
    missing = object()
    previous_images = policy.__dict__.get("prepare_images", missing)
    image_calls = 0
    image_batch_size = None

    def checked_images(batch):
        nonlocal image_calls, image_batch_size
        image_calls += 1
        images, masks = original_images(batch)
        if image_calls != 1 or len(images) != 3 or len(masks) != 3:
            raise ValueError("Exactly one real and two empty camera inputs required")
        if any(
            im.ndim != 4
            or im.shape[1] != 3
            or not im.is_floating_point()
            or not bool(torch.isfinite(im).all())
            for im in images
        ):
            raise ValueError("Camera inputs must be finite BCHW tensors")
        image_batch_size = images[0].shape[0]
        if image_batch_size < 2 or any(im.shape != images[0].shape for im in images):
            raise ValueError("Camera batch dimensions differ or contain fewer than two scenes")
        if any(
            mask.dtype != torch.bool or tuple(mask.shape) != (image_batch_size,) for mask in masks
        ):
            raise ValueError("Camera masks must be boolean batch vectors")
        if not bool(masks[0].all()) or any(bool(mask.any()) for mask in masks[1:]):
            raise ValueError("Real image mask or empty camera mask differs")
        if any(not bool((image == -1).all()) for image in images[1:]):
            raise ValueError("Empty camera values differ")
        vlm = policy.model.vlm_with_expert.get_vlm_model()
        patch, scale = vlm.vision_model.config.patch_size, vlm.connector.scale_factor
        if type(patch) is not int or type(scale) is not int or patch <= 0 or scale <= 0:
            raise ValueError("Invalid registered patch or connector scale")
        if (
            policy.config.add_image_special_tokens
            or policy.config.prefix_length != 0
            or any(
                (im.shape[-2] // patch // scale, im.shape[-1] // patch // scale) != (8, 8)
                or im.shape[-2] % (patch * scale)
                or im.shape[-1] % (patch * scale)
                for im in images
            )
        ):
            raise ValueError("Real image must occupy the registered first 64 tokens")
        return images, masks

    setattr(policy, MARKER, True)

    def capture(layer):
        def hook(_module, inputs, output):
            if layer in captured:
                raise ValueError("Duplicate K projection in one training forward")
            if len(inputs) != 1 or output.ndim != 3 or tuple(output.shape[1:]) != (241, 320):
                raise ValueError("Full 241-token K projection required")
            if image_calls != 1 or output.shape[0] != image_batch_size:
                raise ValueError("Projection does not match the verified camera batch")
            if output.shape != inputs[0].shape or not output.requires_grad:
                raise ValueError("K projection shape or gradient contract changed")
            captured[layer] = scene_variance(output[:, :64]) / scales[layer]
            # Return None: preserve the original full attention input exactly.

        return hook

    try:
        policy.prepare_images = checked_images
        for layer, module in projections.items():
            handles.append(module.register_forward_hook(capture(layer)))
        loss, metrics = native(policy, *args, **kwargs)
        if set(captured) != set(LAYERS) or image_calls != 1:
            raise ValueError("Missing K projection in training forward")
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("Native flow loss must be a finite scalar")
        penalty = torch.stack([captured[layer] for layer in LAYERS]).mean()
        total = loss + coefficient * penalty
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Nonfinite total objective")
        details = dict(metrics)
        details.update(
            loss=float(total.detach()),
            flow_loss=float(loss.detach()),
            image_key_scene_loss=float(penalty.detach()),
            total_loss=float(total.detach()),
        )
        for layer in LAYERS:
            details[f"image_key_scene_layer_{layer}"] = float(captured[layer].detach())
        return total, details
    finally:
        for handle in handles:
            handle.remove()
        setattr(policy, MARKER, False)
        if previous_images is missing:
            delattr(policy, "prepare_images")
        else:
            policy.prepare_images = previous_images


def install(modeling, scales, coefficient, *, upstream_sha256=UPSTREAM_SHA256):
    from pathlib import Path

    from rosetta_reality.experiment import file_sha256

    validate_scales(scales)
    if type(coefficient) not in (int, float) or coefficient not in (0, 0.01):
        raise ValueError("Only preregistered control/treatment coefficients are allowed")
    if getattr(modeling, MARKER, None) is not None:
        raise RuntimeError("Image-key feature is already installed")
    if file_sha256(Path(modeling.__file__)) != upstream_sha256:
        raise ValueError("Pinned upstream implementation changed")
    original = modeling.SmolVLAPolicy.forward
    frozen_scales = dict(scales)

    @wraps(original)
    def forward(self, *args, **kwargs):
        return regularized_forward(self, original, args, kwargs, frozen_scales, coefficient)

    # Control bypass preserves the actual function, gradient graph and RNG.
    if coefficient != 0:
        modeling.SmolVLAPolicy.forward = forward
    setattr(modeling, MARKER, (original, modeling.SmolVLAPolicy.forward))


def restore(modeling):
    state = getattr(modeling, MARKER, None)
    if state is None or modeling.SmolVLAPolicy.forward is not state[1]:
        raise RuntimeError("Missing image-key installation or wrapper order drift")
    modeling.SmolVLAPolicy.forward = state[0]
    delattr(modeling, MARKER)
