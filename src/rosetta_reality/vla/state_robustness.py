"""Training-only normalized state perturbation for local recovery robustness.

SmolVLA receives train-normalized proprioception.  A registered experiment may
add small Gaussian noise to that tensor during optimizer forwards while keeping
the absolute expert action target, validation inputs and deployment inputs
unchanged.  This is local behavior-cloning augmentation, not DAgger and not a
state-conditioned recovery oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rosetta_reality.experiment import file_sha256

UPSTREAM_IMPLEMENTATION_SHA256 = (
    "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d"
)
PROFILE_NORMALIZED_GAUSSIAN_STATE_JITTER = "normalized_gaussian_state_jitter"
STATE_FEATURE = "observation.state"
_ACTIVE_MARKER = "_rosetta_state_robustness_profile"


@dataclass(frozen=True)
class StateRobustnessProfile:
    """One registered training-only state perturbation profile."""

    name: str
    normalized_standard_deviation: float
    training_only: bool = True
    target_semantics: str = "unchanged_absolute_expert_action"

    def __post_init__(self) -> None:
        if self.name != PROFILE_NORMALIZED_GAUSSIAN_STATE_JITTER:
            raise ValueError("Unsupported state-robustness profile.")
        standard_deviation = self.normalized_standard_deviation
        if (
            isinstance(standard_deviation, bool)
            or not isinstance(standard_deviation, int | float)
            or not bool(torch.isfinite(torch.tensor(float(standard_deviation))))
            or not 0.0 < float(standard_deviation) < 1.0
        ):
            raise ValueError("Normalized state jitter must be finite and in (0, 1).")
        if self.training_only is not True:
            raise ValueError("State jitter must be training-only.")
        if self.target_semantics != "unchanged_absolute_expert_action":
            raise ValueError("State jitter requires unchanged absolute expert targets.")


def profile_from_plan(plan: dict[str, Any]) -> StateRobustnessProfile:
    """Build a state-robustness profile from a hash-bound experiment plan."""

    contract = plan.get("state_robustness_contract")
    if not isinstance(contract, dict):
        raise ValueError("The registered plan has no state-robustness contract.")
    if contract.get("upstream_implementation_sha256") != UPSTREAM_IMPLEMENTATION_SHA256:
        raise ValueError("The state-robustness contract targets a different upstream.")
    if contract.get("input_space") != "train_normalized_observation_state":
        raise ValueError("State jitter must operate in train-normalized state space.")
    return StateRobustnessProfile(
        name=str(contract.get("profile", "")),
        normalized_standard_deviation=contract.get(
            "normalized_standard_deviation", float("nan")
        ),
        training_only=contract.get("training_only"),
        target_semantics=str(contract.get("target_semantics", "")),
    )


def install_state_robustness_profile(
    modeling_module: Any,
    profile: StateRobustnessProfile,
    *,
    upstream_sha256: str = UPSTREAM_IMPLEMENTATION_SHA256,
) -> None:
    """Install jitter on optimizer forwards without touching inference methods."""

    policy_class = getattr(modeling_module, "SmolVLAPolicy", None)
    if policy_class is None:
        raise ValueError("The active SmolVLA module lacks its policy class.")
    original_forward = getattr(policy_class, "forward")
    if getattr(original_forward, _ACTIVE_MARKER, None) is not None:
        raise RuntimeError("A state-robustness profile is already installed.")
    source = Path(modeling_module.__file__).resolve()
    if file_sha256(source) != upstream_sha256:
        raise ValueError(
            "The installed upstream implementation differs from the registered "
            "state-robustness contract."
        )

    def robust_forward(
        self: Any,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        active_batch = batch
        if self.training:
            state = batch.get(STATE_FEATURE)
            if not isinstance(state, torch.Tensor) or state.ndim not in {2, 3}:
                raise ValueError("SmolVLA optimizer batch has no valid normalized state.")
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError("SmolVLA normalized state contains NaN or Inf.")
            active_batch = dict(batch)
            jitter = torch.randn_like(state) * float(
                profile.normalized_standard_deviation
            )
            perturbed = state + jitter
            if not bool(torch.isfinite(perturbed).all()):
                raise FloatingPointError("State jitter produced NaN or Inf.")
            active_batch[STATE_FEATURE] = perturbed
        return original_forward(
            self,
            active_batch,
            noise=noise,
            time=time,
            reduction=reduction,
        )

    setattr(robust_forward, _ACTIVE_MARKER, profile.name)
    setattr(robust_forward, "_rosetta_state_robustness_original", original_forward)
    policy_class.forward = robust_forward


def restore_state_robustness_profile(modeling_module: Any) -> None:
    """Remove the installed profile. Intended for tests and diagnostics."""

    policy_class = getattr(modeling_module, "SmolVLAPolicy", None)
    if policy_class is None:
        raise ValueError("The active SmolVLA module lacks its policy class.")
    current = getattr(policy_class, "forward")
    if getattr(current, _ACTIVE_MARKER, None) is None:
        raise RuntimeError("No state-robustness profile is installed.")
    policy_class.forward = getattr(current, "_rosetta_state_robustness_original")
