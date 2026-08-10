"""Losses for continuous robot actions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor


def smooth_l1_action_loss(
    predicted_actions: Tensor,
    target_actions: Tensor,
    *,
    first_action_weight: float = 0.0,
) -> Tensor:
    """Compute full-chunk loss with an optional executed-first-action term.

    A zero weight is exactly the legacy full-chunk Smooth L1 objective. A
    positive weight forms a normalized mixture, so future chunk actions retain
    gradients while the receding-horizon action receives explicit emphasis.
    """

    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            "predicted_actions and target_actions must have identical shapes, "
            f"but received {tuple(predicted_actions.shape)} and {tuple(target_actions.shape)}."
        )
    if predicted_actions.ndim != 3 or predicted_actions.shape[1] < 1:
        raise ValueError("Action loss requires non-empty [batch, chunk, action] tensors.")
    weight = float(first_action_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("first_action_weight must be finite and non-negative.")
    full_chunk = functional.smooth_l1_loss(predicted_actions, target_actions)
    if weight == 0.0:
        return full_chunk
    first_action = functional.smooth_l1_loss(
        predicted_actions[:, :1], target_actions[:, :1]
    )
    return (full_chunk + weight * first_action) / (1.0 + weight)


def smooth_l1_action_loss_per_sample(
    predicted_actions: Tensor,
    target_actions: Tensor,
) -> Tensor:
    """Return one full-chunk Smooth L1 value per batch element."""

    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            "predicted_actions and target_actions must have identical shapes, "
            f"but received {tuple(predicted_actions.shape)} and {tuple(target_actions.shape)}."
        )
    if predicted_actions.ndim != 3 or predicted_actions.shape[1] < 1:
        raise ValueError("Action loss requires non-empty [batch, chunk, action] tensors.")
    return functional.smooth_l1_loss(
        predicted_actions,
        target_actions,
        reduction="none",
    ).mean(dim=(1, 2))


def globally_normalized_scoped_first_action_loss(
    predicted_actions: Tensor,
    target_actions: Tensor,
    scope_mask: Tensor,
    *,
    global_scale: float,
) -> Tensor:
    """Return a batch contribution whose epoch mean equals the scoped mean.

    ``global_scale`` is the immutable train-sample count divided by the
    immutable scoped-sample count. Dividing by the full batch size, rather
    than by the number selected in this batch, keeps the effective objective
    independent of shuffle composition and gives finite zero contributions to
    batches that contain no scoped sample.
    """

    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            "predicted_actions and target_actions must have identical shapes, "
            f"but received {tuple(predicted_actions.shape)} and {tuple(target_actions.shape)}."
        )
    if (
        predicted_actions.ndim != 3
        or predicted_actions.shape[0] < 1
        or predicted_actions.shape[1] < 1
        or predicted_actions.shape[2] < 1
    ):
        raise ValueError("Scoped action loss requires non-empty [batch, chunk, action] tensors.")
    if (
        scope_mask.shape != (predicted_actions.shape[0],)
        or scope_mask.dtype != torch.bool
    ):
        raise ValueError("scope_mask must be a boolean tensor with one value per batch row.")
    scale = float(global_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("global_scale must be finite and positive.")
    per_sample = functional.smooth_l1_loss(
        predicted_actions[:, 0],
        target_actions[:, 0],
        reduction="none",
    ).mean(dim=1)
    return scale * (per_sample * scope_mask.to(per_sample.dtype)).sum() / (
        predicted_actions.shape[0]
    )
