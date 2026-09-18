"""Deployment-aligned temporal weighting for the pinned SmolVLA flow loss.

The pinned ``lerobot`` flow-matching objective computes the elementwise
``[batch, chunk, action]`` MSE and reduces it with a uniform mean over all
valid entries.  The Rosetta Action Contract executes only the first predicted
action of each chunk (``receding_horizon_first_action``), so a registered
experiment may install a temporal weight profile before that reduction.

This module is an explicit local extension of the pinned trainer:

- it never edits the dependency cache;
- every installation is checksum-bound to the pinned upstream implementation,
  so a silently changed ``modeling_smolvla.py`` fails closed instead of
  diverging;
- temporal weighting is normalized over the selected, non-padding entries.
  This preserves the loss/gradient scale instead of silently dividing it by
  the 50-step chunk length;
- the active profile must be registered in the hash-bound formal plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rosetta_reality.experiment import file_sha256

# SHA-256 of `lerobot/policies/smolvla/modeling_smolvla.py` at the pinned
# LeRobot revision c903b114a90e703b3f7d0c46cb38727c328c55ff (lerobot 0.6.2).
UPSTREAM_IMPLEMENTATION_SHA256 = (
    "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d"
)

PROFILE_FIRST_ACTION_ONLY = "first_action_only"
NORMALIZATION_SELECTED_VALID_MEAN = "mean_over_selected_valid_entries"

_MODEL_ACTIVE_MARKER = "_rosetta_horizon_model_weight_profile"
_POLICY_ACTIVE_MARKER = "_rosetta_horizon_policy_weight_profile"


@dataclass(frozen=True)
class HorizonWeightProfile:
    """A registered per-timestep weight profile over the action chunk."""

    name: str
    chunk_size: int
    weights: tuple[float, ...]
    normalization: str = NORMALIZATION_SELECTED_VALID_MEAN

    def __post_init__(self) -> None:
        if self.name != PROFILE_FIRST_ACTION_ONLY:
            raise ValueError("Unsupported temporal weight profile.")
        if self.normalization != NORMALIZATION_SELECTED_VALID_MEAN:
            raise ValueError("Unsupported temporal loss normalization.")
        if self.chunk_size <= 0:
            raise ValueError("The temporal weight chunk size must be positive.")
        if len(self.weights) != self.chunk_size:
            raise ValueError("Temporal weights must cover the full action chunk.")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not bool(torch.isfinite(torch.tensor(float(weight))))
            or float(weight) < 0.0
            for weight in self.weights
        ):
            raise ValueError("Temporal weights must be finite and non-negative.")
        if sum(self.weights) <= 0.0:
            raise ValueError("Temporal weights must not be all zero.")

    def tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(self.weights, device=device, dtype=dtype)


def profile_from_plan(
    plan: dict[str, Any], policy_chunk_size: int
) -> HorizonWeightProfile:
    """Build the registered profile from the hash-bound formal plan."""
    contract = plan.get("loss_contract")
    if not isinstance(contract, dict):
        raise ValueError("The registered plan has no temporal loss contract.")
    if contract.get("upstream_implementation_sha256") != UPSTREAM_IMPLEMENTATION_SHA256:
        raise ValueError(
            "The temporal loss contract targets a different upstream implementation."
        )
    chunk_size = int(contract.get("chunk_size", -1))
    if chunk_size != policy_chunk_size:
        raise ValueError("The temporal loss chunk size differs from the policy chunk.")
    profile_name = str(contract.get("profile", ""))
    normalization = str(contract.get("normalization", ""))
    if normalization != NORMALIZATION_SELECTED_VALID_MEAN:
        raise ValueError("Unsupported temporal loss normalization.")
    if profile_name == PROFILE_FIRST_ACTION_ONLY:
        return HorizonWeightProfile(
            name=profile_name,
            chunk_size=chunk_size,
            weights=(1.0,) + (0.0,) * (chunk_size - 1),
            normalization=normalization,
        )
    raise ValueError("Unsupported temporal weight profile.")


def install_horizon_weight_profile(
    modeling_module: Any,
    profile: HorizonWeightProfile,
    *,
    upstream_sha256: str = UPSTREAM_IMPLEMENTATION_SHA256,
) -> None:
    """Wrap the pinned model and policy forwards with temporal weights.

    The model wrapper applies the registered weights to the unreduced
    ``[batch, chunk, action]`` loss.  The policy wrapper then replaces the
    upstream all-valid-entry denominator with the sum of selected valid
    weights.  Both parts are required: weighting only at the model boundary
    would shrink an unpadded first-action-only loss by ``chunk_size`` and would
    bias padded samples according to their remaining horizon length.
    """
    model_class = getattr(modeling_module, "VLAFlowMatching", None)
    policy_class = getattr(modeling_module, "SmolVLAPolicy", None)
    if model_class is None or policy_class is None:
        raise ValueError("The active SmolVLA module lacks a required policy class.")
    original_model_forward = getattr(model_class, "forward")
    original_policy_forward = getattr(policy_class, "forward")
    if (
        getattr(original_model_forward, _MODEL_ACTIVE_MARKER, None) is not None
        or getattr(original_policy_forward, _POLICY_ACTIVE_MARKER, None) is not None
    ):
        raise RuntimeError("A temporal weight profile is already installed.")
    source = Path(modeling_module.__file__).resolve()
    if file_sha256(source) != upstream_sha256:
        raise ValueError(
            "The installed upstream implementation differs from the registered "
            "temporal loss contract."
        )

    def weighted_forward(
        self: Any,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor | None,
        time: torch.Tensor | None,
    ) -> torch.Tensor:
        losses = original_model_forward(
            self,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
        )
        weights = profile.tensor(losses.device, losses.dtype)
        if losses.ndim != 3 or losses.shape[1] != profile.chunk_size:
            raise ValueError("Upstream SmolVLA loss shape differs from the registered profile.")
        return losses * weights.view(1, -1, 1)

    def normalized_policy_forward(
        self: Any,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if reduction not in {"mean", "none"}:
            raise ValueError("Unsupported SmolVLA loss reduction.")
        loss, loss_dict = original_policy_forward(
            self,
            batch,
            noise=noise,
            time=time,
            reduction=reduction,
        )
        actions = batch.get("action")
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
            raise ValueError("SmolVLA batch has no three-dimensional action chunk.")
        if actions.shape[1] != profile.chunk_size:
            raise ValueError("SmolVLA batch chunk differs from the registered profile.")
        weights = profile.tensor(actions.device, loss.dtype)
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is None:
            valid = torch.ones(
                actions.shape[:2], device=actions.device, dtype=loss.dtype
            )
        else:
            if (
                not isinstance(action_is_pad, torch.Tensor)
                or action_is_pad.shape != actions.shape[:2]
                or action_is_pad.dtype is not torch.bool
            ):
                raise ValueError("SmolVLA action padding mask differs from the action chunk.")
            valid = (~action_is_pad).to(dtype=loss.dtype)
        selected = valid * weights.view(1, -1)
        selected_per_sample = selected.sum(dim=1)
        if bool((selected_per_sample <= 0).any().item()):
            raise ValueError("A batch sample has no selected non-padding action.")
        valid_per_sample = valid.sum(dim=1)
        if reduction == "none":
            if loss.ndim != 1 or loss.shape[0] != actions.shape[0]:
                raise ValueError("Upstream per-sample SmolVLA loss shape changed.")
            loss = loss * (valid_per_sample / selected_per_sample)
        else:
            if loss.ndim != 0:
                raise ValueError("Upstream mean SmolVLA loss is not scalar.")
            loss = loss * (valid_per_sample.sum() / selected_per_sample.sum())
        loss_dict["loss"] = float(loss.detach().mean().item())
        return loss, loss_dict

    setattr(weighted_forward, _MODEL_ACTIVE_MARKER, profile.name)
    setattr(weighted_forward, "_rosetta_horizon_original", original_model_forward)
    setattr(normalized_policy_forward, _POLICY_ACTIVE_MARKER, profile.name)
    setattr(
        normalized_policy_forward,
        "_rosetta_horizon_original",
        original_policy_forward,
    )
    model_class.forward = weighted_forward
    policy_class.forward = normalized_policy_forward


def restore_horizon_weight_profile(modeling_module: Any) -> None:
    """Remove the installed wrapper.  Intended for tests and diagnostics."""
    model_class = getattr(modeling_module, "VLAFlowMatching", None)
    policy_class = getattr(modeling_module, "SmolVLAPolicy", None)
    if model_class is None or policy_class is None:
        raise ValueError("The active SmolVLA module lacks a required policy class.")
    model_forward = getattr(model_class, "forward")
    policy_forward = getattr(policy_class, "forward")
    if (
        getattr(model_forward, _MODEL_ACTIVE_MARKER, None) is None
        or getattr(policy_forward, _POLICY_ACTIVE_MARKER, None) is None
    ):
        raise RuntimeError("No temporal weight profile is installed.")
    model_class.forward = getattr(model_forward, "_rosetta_horizon_original")
    policy_class.forward = getattr(policy_forward, "_rosetta_horizon_original")
