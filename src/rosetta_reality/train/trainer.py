"""A deliberately small training-step implementation for M0 smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer

from rosetta_reality.models.backbones.base import BackboneBatch
from rosetta_reality.models.vla import VLAPolicy
from rosetta_reality.train.losses import smooth_l1_action_loss


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    """Scalar and shape information from one optimizer step."""

    loss: float
    prediction_shape: tuple[int, ...]


def train_step(
    model: VLAPolicy,
    optimizer: Optimizer,
    observations: BackboneBatch,
    robot_state: Tensor,
    target_actions: Tensor,
) -> TrainStepResult:
    """Run forward, Smooth L1 loss, backward, and one optimizer update."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    predicted_actions = model(observations, robot_state)
    loss = smooth_l1_action_loss(predicted_actions, target_actions)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Training loss is not finite.")
    loss.backward()
    optimizer.step()
    return TrainStepResult(
        loss=float(loss.detach().cpu()),
        prediction_shape=predicted_actions.shape,
    )
