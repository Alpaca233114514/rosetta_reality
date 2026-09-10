"""Experimental same-state visual supervision; not installed in any trainer.

The caller supplies two verified train observations at the same frame offset,
with identical state/language and different images. Actions must already be
projected through the Action Contract and normalized with train-only statistics.
At flow time 1, both observations receive the same independently drawn noise z.
The velocity targets are z - a; their difference is a_right - a_left.

The paired term therefore cannot be satisfied by a deterministic
state/noisy-action-only predictor when the expert actions differ. The caller
must also rule out different stochastic-layer masks between the two forwards.
The ordinary positive-target term
anchors the common action component. No wrong-image prediction is deliberately
made worse. This module does not pair/load data, sample noise, call a model,
install hooks, change policy flags or create an optimizer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisualPairContext:
    """Paired prepared inputs, all with leading axes [pair_count, 2].

    Tensor suffixes: state [state_dim], language [tokens], noisy_actions
    [horizon, action_dim], time []. Identity pairs come from verified samples;
    this arithmetic helper cannot establish their pixel/label provenance.
    """

    state: Any
    language_tokens: Any
    language_masks: Any
    noisy_actions: Any
    time: Any
    episode_ids: tuple[tuple[int, int], ...]
    frame_ids: tuple[tuple[int, int], ...]
    image_sha256: tuple[tuple[str, str], ...]


def visual_pair_flow_loss(
    predicted_velocity,
    actions,
    valid,
    context: VisualPairContext,
    *,
    train_episode_ids: tuple[int, ...],
    pair_weight: float,
    minimum_action_difference: float,
) -> dict[str, Any]:
    """Return differentiable anchored and paired MSE in normalized action space.

    Velocity/actions/valid have shape [pairs, 2, horizon, action_dim]. ``valid``
    selects actual, non-padding action coordinates. Each pair contributes
    equally after its own valid-count normalization. A zero pair weight is the
    explicit control arm, using exactly the same paired data and time/noise.
    This research objective has no real-model efficacy evidence yet.
    """
    import torch

    for name, value, positive in (
        ("pair_weight", pair_weight, False),
        ("minimum_action_difference", minimum_action_difference, True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
            or (positive and value == 0)
        ):
            raise ValueError(
                f"{name} must be finite and {'positive' if positive else 'nonnegative'}."
            )

    if (
        not isinstance(predicted_velocity, torch.Tensor)
        or predicted_velocity.ndim != 4
        or predicted_velocity.shape[1] != 2
        or any(size == 0 for size in predicted_velocity.shape)
    ):
        raise ValueError("Expected velocity shape [pairs, 2, horizon, action_dim].")
    pair_count = predicted_velocity.shape[0]
    for name, tensor in (
        ("velocity", predicted_velocity),
        ("actions", actions),
        ("noisy_actions", context.noisy_actions),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != predicted_velocity.shape
            or tensor.device != predicted_velocity.device
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"{name} must be finite floating point in the paired action space.")
    if (
        not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or valid.shape != actions.shape
        or valid.device != actions.device
    ):
        raise ValueError("The valid action mask must match every paired action coordinate.")
    for name, tensor in (
        ("state", context.state),
        ("language_tokens", context.language_tokens),
        ("language_masks", context.language_masks),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 3
            or tensor.shape[:2] != (pair_count, 2)
            or tensor.shape[2] == 0
            or not torch.isfinite(tensor).all()
            or not torch.equal(tensor[:, 0], tensor[:, 1])
        ):
            raise ValueError(f"Both images require identical finite {name} inputs.")
    if (
        not context.state.is_floating_point()
        or context.language_tokens.dtype not in (torch.int32, torch.int64)
        or context.language_masks.dtype != torch.bool
        or context.language_tokens.shape != context.language_masks.shape
        or not context.language_masks.any(dim=-1).all()
    ):
        raise ValueError("Paired instructions require valid tokens and nonempty boolean masks.")
    if (
        not isinstance(context.time, torch.Tensor)
        or context.time.shape != (pair_count, 2)
        or not context.time.is_floating_point()
        or not torch.equal(context.time, torch.ones_like(context.time))
        or not torch.equal(context.noisy_actions[:, 0], context.noisy_actions[:, 1])
    ):
        raise ValueError("Visual pairs require time 1 and identical full-noise action inputs.")

    if not train_episode_ids or any(type(ep) is not int or ep < 0 for ep in train_episode_ids):
        raise ValueError("A nonempty, explicit train episode allowlist is required.")
    for identities in (context.episode_ids, context.frame_ids, context.image_sha256):
        if len(identities) != pair_count or any(len(pair) != 2 for pair in identities):
            raise ValueError("Every visual pair needs two complete sample identities.")
    for episodes, frames, images in zip(
        context.episode_ids, context.frame_ids, context.image_sha256, strict=True
    ):
        if (
            any(type(ep) is not int or ep not in train_episode_ids for ep in episodes)
            or episodes[0] == episodes[1]
            or any(type(frame) is not int or frame < 0 for frame in frames)
            or frames[0] != frames[1]
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in images
            )
            or images[0] == images[1]
        ):
            raise ValueError("Pairs need distinct train episodes/images at the same frame offset.")

    shared_valid = valid[:, 0] & valid[:, 1]
    shared_counts = shared_valid.sum(dim=(1, 2))
    if (shared_counts == 0).any():
        raise ValueError("Each pair needs shared non-padding action coordinates.")
    # Labels and supplied noise are supervision, never optimization variables.
    labels = actions.detach().float()
    noise = context.noisy_actions.detach().float()
    velocity = predicted_velocity.float()
    target_difference = labels[:, 1] - labels[:, 0]
    target_rms = (
        target_difference.square().masked_fill(~shared_valid, 0).sum(dim=(1, 2)) / shared_counts
    ).sqrt()
    if not torch.isfinite(target_rms).all():
        raise FloatingPointError("Action contrast overflowed its loss dtype.")
    if (target_rms < minimum_action_difference).any():
        raise ValueError("Action targets do not contain the registered minimum action contrast.")
    target_velocity = noise - labels
    anchor_error = (velocity - target_velocity).square().masked_fill(~valid, 0)
    anchor = (anchor_error.sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3))).mean()
    difference_error = ((velocity[:, 0] - velocity[:, 1]) - target_difference).square()
    paired = (difference_error.masked_fill(~shared_valid, 0).sum(dim=(1, 2)) / shared_counts).mean()
    total = anchor + pair_weight * paired
    if not torch.isfinite(total):
        raise FloatingPointError("Visual pair loss is non-finite.")
    return {
        "loss": total,
        "anchor_loss": anchor,
        "paired_loss": paired,
        "target_difference_rms": target_rms.detach(),
        "pair_count": pair_count,
    }
