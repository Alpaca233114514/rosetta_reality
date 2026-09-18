"""Action-space metrics that preserve physical contract semantics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from rosetta_reality.data.normalization import NormalizationStats, normalize
from rosetta_reality.sim import ActionContract


def action_metrics(
    predicted: Tensor,
    target: Tensor,
    contract: ActionContract,
    action_stats: NormalizationStats,
    *,
    raw_predicted: Tensor | None = None,
) -> dict[str, Any]:
    """Compute aggregate, per-dimension, per-step, and validity metrics."""

    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("Predicted and target actions must share [sample, chunk, action] shape.")
    if predicted.shape[-1] != contract.dimension:
        raise ValueError("Action metric width differs from the physical Action Contract.")
    raw = predicted if raw_predicted is None else raw_predicted
    if raw.shape != predicted.shape:
        raise ValueError("Raw and projected predictions must share the same shape.")
    error = predicted - target
    absolute = error.abs()
    finite = torch.isfinite(predicted)
    lower = contract.lower_bounds.to(predicted).view(1, 1, -1)
    upper = contract.upper_bounds.to(predicted).view(1, 1, -1)
    violations = (~finite) | (predicted < lower) | (predicted > upper)
    invalid_samples = violations.reshape(violations.shape[0], -1).any(dim=1)
    raw_finite = torch.isfinite(raw)
    raw_violations = (~raw_finite) | (raw < lower) | (raw > upper)
    raw_invalid_samples = raw_violations.reshape(raw_violations.shape[0], -1).any(dim=1)
    projection = (raw - predicted).abs()
    normalized_prediction = normalize(predicted, action_stats)
    normalized_target = normalize(target, action_stats)
    return {
        "samples": predicted.shape[0],
        "action_mae": float(absolute.mean()),
        "first_action_mae": float(absolute[:, 0].mean()),
        "action_rmse": float(error.square().mean().sqrt()),
        "normalized_smooth_l1": float(
            functional.smooth_l1_loss(normalized_prediction, normalized_target)
        ),
        "invalid_action_rate": float(invalid_samples.to(torch.float32).mean()),
        "limit_violation_rate": float(violations.to(torch.float32).mean()),
        "raw_invalid_action_rate": float(raw_invalid_samples.to(torch.float32).mean()),
        "raw_limit_violation_rate": float(raw_violations.to(torch.float32).mean()),
        "projection_element_rate": float(projection.ne(0).to(torch.float32).mean()),
        "maximum_projection_magnitude": float(projection.max()),
        "per_dimension_mae": {
            name: float(value)
            for name, value in zip(contract.dimension_names, absolute.mean(dim=(0, 1)))
        },
        "per_chunk_step_mae": [float(value) for value in absolute.mean(dim=(0, 2))],
    }
