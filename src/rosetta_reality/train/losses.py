"""Losses for continuous robot actions."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor


def smooth_l1_action_loss(predicted_actions: Tensor, target_actions: Tensor) -> Tensor:
    """Compute Huber/Smooth L1 loss for equally shaped action chunks."""

    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            "predicted_actions and target_actions must have identical shapes, "
            f"but received {tuple(predicted_actions.shape)} and {tuple(target_actions.shape)}."
        )
    return functional.smooth_l1_loss(predicted_actions, target_actions)

