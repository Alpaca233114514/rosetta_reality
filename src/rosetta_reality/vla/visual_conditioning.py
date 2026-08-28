"""Training-only state dropout for a controlled visual-conditioning axis.

The intervention removes the complete train-normalized proprioceptive vector
for a registered fraction of samples.  It deliberately uses a dedicated CPU
generator so the treatment does not consume the global RNG stream used by the
SmolVLA flow noise or dataloader.  Validation and deployment are untouched.
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
PROFILE_SAMPLEWISE_NORMALIZED_STATE_DROPOUT = (
    "samplewise_normalized_state_dropout"
)
STATE_FEATURE = "observation.state"
_ACTIVE_MARKER = "_rosetta_visual_conditioning_profile"


@dataclass(frozen=True)
class VisualConditioningProfile:
    """One preregistered state-dropout treatment."""

    name: str
    dropout_probability: float
    generator_seed: int
    input_space: str = "train_normalized_observation_state"
    granularity: str = "whole_sample"
    replacement: str = "normalized_zero"
    rescale_retained_state: bool = False
    generator: str = "dedicated_cpu_generator"
    training_only: bool = True
    target_semantics: str = "unchanged_absolute_expert_action"

    def __post_init__(self) -> None:
        if self.name != PROFILE_SAMPLEWISE_NORMALIZED_STATE_DROPOUT:
            raise ValueError("Unsupported visual-conditioning profile.")
        probability = self.dropout_probability
        if (
            isinstance(probability, bool)
            or not isinstance(probability, int | float)
            or not bool(torch.isfinite(torch.tensor(float(probability))))
            or not 0.0 < float(probability) < 1.0
        ):
            raise ValueError("State-dropout probability must be finite and in (0, 1).")
        if (
            isinstance(self.generator_seed, bool)
            or not isinstance(self.generator_seed, int)
            or self.generator_seed < 0
        ):
            raise ValueError("State-dropout generator seed must be non-negative.")
        expected = {
            "input_space": "train_normalized_observation_state",
            "granularity": "whole_sample",
            "replacement": "normalized_zero",
            "generator": "dedicated_cpu_generator",
            "target_semantics": "unchanged_absolute_expert_action",
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"State-dropout {field} differs from the contract.")
        if self.rescale_retained_state is not False:
            raise ValueError("Retained normalized state must not be rescaled.")
        if self.training_only is not True:
            raise ValueError("State dropout must be training-only.")


def profile_from_plan(plan: dict[str, Any]) -> VisualConditioningProfile:
    """Build the treatment from a hash-bound version-2 training plan."""

    contract = plan.get("visual_conditioning_contract")
    if not isinstance(contract, dict):
        raise ValueError("The registered plan has no visual-conditioning contract.")
    if contract.get("upstream_implementation_sha256") != UPSTREAM_IMPLEMENTATION_SHA256:
        raise ValueError("The visual-conditioning contract targets a different upstream.")
    return VisualConditioningProfile(
        name=str(contract.get("profile", "")),
        dropout_probability=contract.get("dropout_probability", float("nan")),
        generator_seed=contract.get("generator_seed", -1),
        input_space=str(contract.get("input_space", "")),
        granularity=str(contract.get("granularity", "")),
        replacement=str(contract.get("replacement", "")),
        rescale_retained_state=contract.get("rescale_retained_state"),
        generator=str(contract.get("generator", "")),
        training_only=contract.get("training_only"),
        target_semantics=str(contract.get("target_semantics", "")),
    )


def install_visual_conditioning_profile(
    modeling_module: Any,
    profile: VisualConditioningProfile,
    *,
    upstream_sha256: str = UPSTREAM_IMPLEMENTATION_SHA256,
) -> None:
    """Install sample-wise state dropout only on optimizer forwards."""

    policy_class = getattr(modeling_module, "SmolVLAPolicy", None)
    if policy_class is None:
        raise ValueError("The active SmolVLA module lacks its policy class.")
    original_forward = getattr(policy_class, "forward")
    if getattr(original_forward, _ACTIVE_MARKER, None) is not None:
        raise RuntimeError("A visual-conditioning profile is already installed.")
    source = Path(modeling_module.__file__).resolve()
    if file_sha256(source) != upstream_sha256:
        raise ValueError(
            "The installed upstream implementation differs from the registered "
            "visual-conditioning contract."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(profile.generator_seed)

    def visual_conditioning_forward(
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
            batch_size = int(state.shape[0])
            if batch_size < 2:
                raise ValueError(
                    "Sample-wise state dropout requires a training batch of at least two."
                )
            drop_count = round(float(profile.dropout_probability) * batch_size)
            if not 0 < drop_count < batch_size:
                raise ValueError(
                    "Registered state dropout degenerates to all-kept or all-dropped."
                )
            dropped_cpu = torch.randperm(batch_size, generator=generator)[:drop_count]
            dropped = dropped_cpu.to(device=state.device)
            perturbed = state.clone()
            perturbed.index_fill_(0, dropped, 0.0)
            if not bool(torch.isfinite(perturbed).all()):
                raise FloatingPointError("State dropout produced NaN or Inf.")
            active_batch = dict(batch)
            active_batch[STATE_FEATURE] = perturbed
        return original_forward(
            self,
            active_batch,
            noise=noise,
            time=time,
            reduction=reduction,
        )

    setattr(visual_conditioning_forward, _ACTIVE_MARKER, profile.name)
    setattr(
        visual_conditioning_forward,
        "_rosetta_visual_conditioning_original",
        original_forward,
    )
    policy_class.forward = visual_conditioning_forward


def restore_visual_conditioning_profile(modeling_module: Any) -> None:
    """Remove the installed treatment. Intended for tests and diagnostics."""

    policy_class = getattr(modeling_module, "SmolVLAPolicy", None)
    if policy_class is None:
        raise ValueError("The active SmolVLA module lacks its policy class.")
    current = getattr(policy_class, "forward")
    if getattr(current, _ACTIVE_MARKER, None) is None:
        raise RuntimeError("No visual-conditioning profile is installed.")
    policy_class.forward = getattr(
        current, "_rosetta_visual_conditioning_original"
    )
