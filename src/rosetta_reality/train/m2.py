"""Shared construction and evaluation helpers for the cached M2 policy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from rosetta_reality.data.normalization import (
    DatasetStatistics,
    denormalize,
    normalize,
)
from rosetta_reality.models import ContinuousActionHead, StateEncoder, VLAPolicy
from rosetta_reality.models.backbones import CachedBackbone, VLABackbone
from rosetta_reality.sim import ActionContract


def build_policy_with_backbone(
    experiment: dict[str, Any],
    backbone: VLABackbone,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    statistics: DatasetStatistics | None = None,
) -> VLAPolicy:
    """Build the shared state/fusion/action path around any compatible backbone."""

    configured = experiment["action_expert"]
    fusion_dim = int(configured["fusion_dim"])
    parameterization = str(configured.get("prediction_parameterization", "absolute"))
    if parameterization == "absolute":
        state_to_action_scale = None
        state_to_action_offset = None
    elif parameterization == "residual_from_current_state":
        if statistics is None:
            raise ValueError("Residual action prediction requires train-only statistics.")
        if state_dim != action_dim:
            raise ValueError("Residual action prediction requires state_dim == action_dim.")
        if statistics.state.mean.shape != (state_dim,) or statistics.action.mean.shape != (
            action_dim,
        ):
            raise ValueError("Normalization statistics differ from the model dimensions.")
        state_to_action_scale = statistics.state.std / statistics.action.std
        state_to_action_offset = (
            statistics.state.mean - statistics.action.mean
        ) / statistics.action.std
    else:
        raise ValueError(f"Unsupported action prediction parameterization: {parameterization!r}.")
    return VLAPolicy(
        backbone=backbone,
        state_encoder=StateEncoder(
            state_dim=state_dim,
            hidden_dim=int(configured["state_hidden_dim"]),
            num_layers=int(configured["state_layers"]),
            dropout=float(configured["state_dropout"]),
        ),
        action_head=ContinuousActionHead(
            input_dim=fusion_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=int(configured["head_hidden_dim"]),
        ),
        state_to_action_scale=state_to_action_scale,
        state_to_action_offset=state_to_action_offset,
    )


def build_cached_policy(
    experiment: dict[str, Any],
    *,
    feature_dim: int,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    statistics: DatasetStatistics | None = None,
) -> VLAPolicy:
    """Build only the trainable downstream components for a frozen cache."""

    return build_policy_with_backbone(
        experiment,
        CachedBackbone(feature_dim),
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        statistics=statistics,
    )


def normalized_batch(
    batch: dict[str, Tensor],
    statistics: DatasetStatistics,
    *,
    device: str | torch.device | None = None,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Normalize state and targets while leaving frozen features unchanged."""

    target_device = torch.device(device) if device is not None else None

    def converted(value: Tensor) -> Tensor:
        if target_device is None:
            return value.to(torch.float32)
        return value.to(device=target_device, dtype=torch.float32)

    return (
        {"features": converted(batch["features"])},
        normalize(converted(batch["robot_state"]), statistics.state),
        normalize(converted(batch["actions"]), statistics.action),
    )


@torch.inference_mode()
def predict_denormalized(
    model: VLAPolicy,
    batches: Iterable[dict[str, Tensor]],
    statistics: DatasetStatistics,
    contract: ActionContract,
    *,
    device: str | torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Predict raw and contract-projected physical actions for evaluation."""

    predictions: list[Tensor] = []
    raw_predictions: list[Tensor] = []
    targets: list[Tensor] = []
    model.eval()
    for batch in batches:
        observations, state, target = normalized_batch(batch, statistics, device=device)
        normalized_prediction = model(observations, state)
        if not bool(torch.isfinite(normalized_prediction).all()):
            raise FloatingPointError("Policy prediction contains NaN or Inf.")
        raw_prediction = denormalize(normalized_prediction, statistics.action).cpu()
        projected_prediction, _ = contract.clip(raw_prediction)
        raw_predictions.append(raw_prediction)
        predictions.append(projected_prediction)
        targets.append(denormalize(target, statistics.action).cpu())
    if not predictions:
        raise ValueError("Evaluation loader yielded no batches.")
    return torch.cat(predictions), torch.cat(targets), torch.cat(raw_predictions)
