"""Serializable LeRobot processor steps for Rosetta's SmolVLA action boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

from rosetta_reality.sim.action_contract import ActionContract

REGISTRY_NAME = "rosetta_action_contract_projection_processor"
PI_ALOHA_PREPROCESSOR_REGISTRY_NAME = "rosetta_pi_aloha_preprocessor"
PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME = "rosetta_pi_aloha_postprocessor"
_ALOHA_DIMENSION = 14
_FLIPPED_JOINTS = (1, 2, 8, 9)
_GRIPPERS = (6, 13)
PI_ALOHA_ACTION_ADAPTER = "rosetta_pi_aloha"
BOUNDED_SINE_ACTION_ADAPTER = "rosetta_pi_aloha_arms_bounded_sine_grippers"
_ACTION_ADAPTERS = {PI_ALOHA_ACTION_ADAPTER, BOUNDED_SINE_ACTION_ADAPTER}


def _validate_aloha_tensor(value: torch.Tensor, label: str) -> None:
    if value.ndim < 1 or value.shape[-1] != _ALOHA_DIMENSION:
        raise ValueError(f"{label} must have 14 ALOHA features on its last dimension.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains NaN or Inf.")


def standard_aloha_state_to_pi(state: torch.Tensor) -> torch.Tensor:
    """Map raw standard-ALOHA state features to pi-Aloha on the last axis."""

    from lerobot.policies.smolvla.modeling_smolvla import aloha_gripper_to_angular

    _validate_aloha_tensor(state, "Standard-ALOHA state")
    result = state.clone()
    for index in _FLIPPED_JOINTS:
        result[..., index] *= -1
    for index in _GRIPPERS:
        result[..., index] = aloha_gripper_to_angular(result[..., index])
    return result


def standard_aloha_action_to_pi(action: torch.Tensor) -> torch.Tensor:
    """Map raw standard-ALOHA actions to pi-Aloha on the last axis."""

    from lerobot.policies.smolvla.modeling_smolvla import (
        aloha_gripper_from_angular_inv,
    )

    _validate_aloha_tensor(action, "Standard-ALOHA action")
    result = action.clone()
    for index in _FLIPPED_JOINTS:
        result[..., index] *= -1
    for index in _GRIPPERS:
        result[..., index] = aloha_gripper_from_angular_inv(result[..., index])
    return result


def pi_aloha_action_to_standard(action: torch.Tensor) -> torch.Tensor:
    """Map raw pi-Aloha actions back to the standard ALOHA contract."""

    from lerobot.policies.smolvla.modeling_smolvla import aloha_gripper_from_angular

    _validate_aloha_tensor(action, "pi-Aloha action")
    result = action.clone()
    for index in _FLIPPED_JOINTS:
        result[..., index] *= -1
    for index in _GRIPPERS:
        result[..., index] = aloha_gripper_from_angular(result[..., index])
    return result


def standard_aloha_action_to_bounded_sine(action: torch.Tensor) -> torch.Tensor:
    """Encode actions with pi-Aloha arm signs and exactly bounded gripper angles."""

    _validate_aloha_tensor(action, "Standard-ALOHA action")
    result = action.clone()
    for index in _FLIPPED_JOINTS:
        result[..., index] *= -1
    for index in _GRIPPERS:
        normalized = result[..., index].mul(2).sub(1).clamp(-1, 1)
        result[..., index] = torch.asin(normalized)
    return result


def bounded_sine_action_to_standard(action: torch.Tensor) -> torch.Tensor:
    """Decode arbitrary model outputs to standard actions with grippers in [0, 1]."""

    _validate_aloha_tensor(action, "Bounded-sine ALOHA action")
    result = action.clone()
    for index in _FLIPPED_JOINTS:
        result[..., index] *= -1
    for index in _GRIPPERS:
        result[..., index] = result[..., index].sin().add(1).mul(0.5)
    return result


def standard_aloha_action_to_model(
    action: torch.Tensor, representation_adapter: str
) -> torch.Tensor:
    """Dispatch the registered raw action encoder."""

    if representation_adapter == PI_ALOHA_ACTION_ADAPTER:
        return standard_aloha_action_to_pi(action)
    if representation_adapter == BOUNDED_SINE_ACTION_ADAPTER:
        return standard_aloha_action_to_bounded_sine(action)
    raise ValueError("Unsupported Rosetta action representation adapter.")


def model_action_to_standard(
    action: torch.Tensor, representation_adapter: str
) -> torch.Tensor:
    """Dispatch the registered raw action decoder."""

    if representation_adapter == PI_ALOHA_ACTION_ADAPTER:
        return pi_aloha_action_to_standard(action)
    if representation_adapter == BOUNDED_SINE_ACTION_ADAPTER:
        return bounded_sine_action_to_standard(action)
    raise ValueError("Unsupported Rosetta action representation adapter.")


@ProcessorStepRegistry.register(REGISTRY_NAME)
@dataclass
class ActionContractProjectionProcessorStep(ProcessorStep):
    """Project raw dataset targets before normalization and reject corrupt overshoot."""

    lower_bounds: list[float]
    upper_bounds: list[float]
    source_overshoot_tolerances: list[float]
    dimension_names: list[str]
    action_contract_sha256: str
    stage: str = "before_normalization"

    def __post_init__(self) -> None:
        dimension = len(self.dimension_names)
        if (
            dimension <= 0
            or len(self.lower_bounds) != dimension
            or len(self.upper_bounds) != dimension
            or len(self.source_overshoot_tolerances) != dimension
            or self.stage != "before_normalization"
            or len(self.action_contract_sha256) != 64
        ):
            raise ValueError("Rosetta action projection processor configuration is invalid.")
        if any(
            lower >= upper or tolerance < 0
            for lower, upper, tolerance in zip(
                self.lower_bounds,
                self.upper_bounds,
                self.source_overshoot_tolerances,
                strict=True,
            )
        ):
            raise ValueError("Rosetta action projection limits are invalid.")

    @classmethod
    def from_contract(
        cls, contract: ActionContract, *, action_contract_sha256: str
    ) -> ActionContractProjectionProcessorStep:
        return cls(
            lower_bounds=[float(value) for value in contract.lower_bounds.tolist()],
            upper_bounds=[float(value) for value in contract.upper_bounds.tolist()],
            source_overshoot_tolerances=[
                float(value) for value in contract.source_overshoot_tolerances.tolist()
            ],
            dimension_names=list(contract.dimension_names),
            action_contract_sha256=action_contract_sha256,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        if not isinstance(action, torch.Tensor):
            raise ValueError("Rosetta action projection requires a torch action tensor.")
        if action.ndim < 1 or action.shape[-1] != len(self.dimension_names):
            raise ValueError("Rosetta action projection received the wrong action dimension.")
        if not bool(torch.isfinite(action).all()):
            raise ValueError("Rosetta action projection received NaN or Inf.")
        lower = torch.as_tensor(self.lower_bounds, device=action.device, dtype=action.dtype)
        upper = torch.as_tensor(self.upper_bounds, device=action.device, dtype=action.dtype)
        tolerances = torch.as_tensor(
            self.source_overshoot_tolerances,
            device=action.device,
            dtype=action.dtype,
        )
        overshoot = torch.maximum(lower - action, action - upper).clamp_min(0)
        if bool((overshoot > tolerances + 1e-6).any()):
            raise ValueError(
                "A source action exceeds the Action Contract overshoot tolerance."
            )
        new_transition[TransitionKey.ACTION] = torch.maximum(
            torch.minimum(action, upper), lower
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "lower_bounds": self.lower_bounds,
            "upper_bounds": self.upper_bounds,
            "source_overshoot_tolerances": self.source_overshoot_tolerances,
            "dimension_names": self.dimension_names,
            "action_contract_sha256": self.action_contract_sha256,
            "stage": self.stage,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(PI_ALOHA_PREPROCESSOR_REGISTRY_NAME)
@dataclass
class PiAlohaPreprocessorStep(ProcessorStep):
    """Convert raw state and projected targets before train-only normalization."""

    dimension_names: list[str]
    upstream_revision: str
    action_representation_adapter: str = PI_ALOHA_ACTION_ADAPTER
    stage: str = "after_target_projection_before_normalization"

    def __post_init__(self) -> None:
        if (
            len(self.dimension_names) != _ALOHA_DIMENSION
            or self.dimension_names[6] != "left_gripper"
            or self.dimension_names[13] != "right_gripper"
            or len(self.upstream_revision) != 40
            or self.action_representation_adapter not in _ACTION_ADAPTERS
            or self.stage != "after_target_projection_before_normalization"
        ):
            raise ValueError("pi-Aloha preprocessor configuration is invalid.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            if not isinstance(observation, dict):
                raise ValueError("pi-Aloha preprocessing requires a mapped observation.")
            state = observation.get("observation.state")
            if state is not None:
                if not isinstance(state, torch.Tensor):
                    raise ValueError("pi-Aloha preprocessing requires a tensor state.")
                new_observation = observation.copy()
                new_observation["observation.state"] = standard_aloha_state_to_pi(state)
                new_transition[TransitionKey.OBSERVATION] = new_observation
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, torch.Tensor):
                raise ValueError("pi-Aloha preprocessing requires a tensor action.")
            new_transition[TransitionKey.ACTION] = standard_aloha_action_to_model(
                action, self.action_representation_adapter
            )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "dimension_names": self.dimension_names,
            "upstream_revision": self.upstream_revision,
            "action_representation_adapter": self.action_representation_adapter,
            "stage": self.stage,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME)
@dataclass
class PiAlohaPostprocessorStep(ProcessorStep):
    """Decode unnormalized model actions and enforce the standard ALOHA limits."""

    lower_bounds: list[float]
    upper_bounds: list[float]
    dimension_names: list[str]
    upstream_revision: str
    action_representation_adapter: str = PI_ALOHA_ACTION_ADAPTER
    stage: str = "after_unnormalization"

    def __post_init__(self) -> None:
        if (
            len(self.dimension_names) != _ALOHA_DIMENSION
            or len(self.lower_bounds) != _ALOHA_DIMENSION
            or len(self.upper_bounds) != _ALOHA_DIMENSION
            or self.dimension_names[6] != "left_gripper"
            or self.dimension_names[13] != "right_gripper"
            or len(self.upstream_revision) != 40
            or self.action_representation_adapter not in _ACTION_ADAPTERS
            or self.stage != "after_unnormalization"
            or any(
                lower >= upper
                for lower, upper in zip(self.lower_bounds, self.upper_bounds, strict=True)
            )
        ):
            raise ValueError("pi-Aloha postprocessor configuration is invalid.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        if not isinstance(action, torch.Tensor):
            raise ValueError("pi-Aloha postprocessing requires a tensor action.")
        standard = model_action_to_standard(
            action, self.action_representation_adapter
        )
        lower = torch.as_tensor(
            self.lower_bounds, device=standard.device, dtype=standard.dtype
        )
        upper = torch.as_tensor(
            self.upper_bounds, device=standard.device, dtype=standard.dtype
        )
        new_transition[TransitionKey.ACTION] = torch.maximum(
            torch.minimum(standard, upper), lower
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "lower_bounds": self.lower_bounds,
            "upper_bounds": self.upper_bounds,
            "dimension_names": self.dimension_names,
            "upstream_revision": self.upstream_revision,
            "action_representation_adapter": self.action_representation_adapter,
            "stage": self.stage,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def ensure_action_contract_projection(
    preprocessor: Any,
    contract: ActionContract,
    *,
    action_contract_sha256: str,
) -> None:
    """Insert exactly one projection immediately before LeRobot normalization."""

    steps = getattr(preprocessor, "steps", None)
    if not isinstance(steps, list):
        raise ValueError("SmolVLA preprocessor does not expose a mutable step list.")
    existing = [
        (index, step)
        for index, step in enumerate(steps)
        if getattr(step.__class__, "_registry_name", None) == REGISTRY_NAME
    ]
    normalizers = [
        index
        for index, step in enumerate(steps)
        if getattr(step.__class__, "_registry_name", None) == "normalizer_processor"
    ]
    if len(normalizers) != 1:
        raise ValueError("SmolVLA requires exactly one normalizer processor step.")
    expected = ActionContractProjectionProcessorStep.from_contract(
        contract, action_contract_sha256=action_contract_sha256
    )
    if existing:
        between = [
            getattr(step.__class__, "_registry_name", None)
            for step in steps[existing[0][0] + 1 : normalizers[0]]
        ]
        if len(existing) != 1 or between not in (
            [],
            [PI_ALOHA_PREPROCESSOR_REGISTRY_NAME],
        ):
            raise ValueError("Saved Rosetta action projection is not before normalization.")
        if existing[0][1].get_config() != expected.get_config():
            raise ValueError("Saved Rosetta action projection identity differs from the contract.")
        return
    steps.insert(normalizers[0], expected)


def ensure_pi_aloha_processors(
    preprocessor: Any,
    postprocessor: Any,
    contract: ActionContract,
    *,
    upstream_revision: str,
    action_representation_adapter: str = PI_ALOHA_ACTION_ADAPTER,
) -> None:
    """Install a symmetric raw-feature pi-Aloha boundary around normalization."""

    pre_steps = getattr(preprocessor, "steps", None)
    post_steps = getattr(postprocessor, "steps", None)
    if not isinstance(pre_steps, list) or not isinstance(post_steps, list):
        raise ValueError("SmolVLA processors must expose mutable step lists.")
    projection = [
        index
        for index, step in enumerate(pre_steps)
        if getattr(step.__class__, "_registry_name", None) == REGISTRY_NAME
    ]
    normalizer = [
        index
        for index, step in enumerate(pre_steps)
        if getattr(step.__class__, "_registry_name", None) == "normalizer_processor"
    ]
    existing_pre = [
        (index, step)
        for index, step in enumerate(pre_steps)
        if getattr(step.__class__, "_registry_name", None)
        == PI_ALOHA_PREPROCESSOR_REGISTRY_NAME
    ]
    if len(projection) != 1 or len(normalizer) != 1:
        raise ValueError("pi-Aloha preprocessing requires projection and normalization.")
    expected_pre = PiAlohaPreprocessorStep(
        dimension_names=list(contract.dimension_names),
        upstream_revision=upstream_revision,
        action_representation_adapter=action_representation_adapter,
    )
    if existing_pre:
        if len(existing_pre) != 1 or existing_pre[0][1].get_config() != expected_pre.get_config():
            raise ValueError("Saved pi-Aloha preprocessor identity differs from the contract.")
    else:
        pre_steps.insert(normalizer[0], expected_pre)
    projection_index = next(
        index
        for index, step in enumerate(pre_steps)
        if getattr(step.__class__, "_registry_name", None) == REGISTRY_NAME
    )
    adapter_index = next(
        index
        for index, step in enumerate(pre_steps)
        if getattr(step.__class__, "_registry_name", None)
        == PI_ALOHA_PREPROCESSOR_REGISTRY_NAME
    )
    normalizer_index = next(
        index
        for index, step in enumerate(pre_steps)
        if getattr(step.__class__, "_registry_name", None) == "normalizer_processor"
    )
    if not (projection_index + 1 == adapter_index and adapter_index + 1 == normalizer_index):
        raise ValueError("SmolVLA raw-feature processor ordering is invalid.")

    unnormalizer = [
        index
        for index, step in enumerate(post_steps)
        if getattr(step.__class__, "_registry_name", None) == "unnormalizer_processor"
    ]
    existing_post = [
        (index, step)
        for index, step in enumerate(post_steps)
        if getattr(step.__class__, "_registry_name", None)
        == PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME
    ]
    if len(unnormalizer) != 1:
        raise ValueError("pi-Aloha postprocessing requires exactly one unnormalizer.")
    expected_post = PiAlohaPostprocessorStep(
        lower_bounds=[float(value) for value in contract.lower_bounds.tolist()],
        upper_bounds=[float(value) for value in contract.upper_bounds.tolist()],
        dimension_names=list(contract.dimension_names),
        upstream_revision=upstream_revision,
        action_representation_adapter=action_representation_adapter,
    )
    if existing_post:
        if (
            len(existing_post) != 1
            or existing_post[0][1].get_config() != expected_post.get_config()
        ):
            raise ValueError("Saved pi-Aloha postprocessor identity differs from the contract.")
    else:
        post_steps.insert(unnormalizer[0] + 1, expected_post)
    unnormalizer_index = next(
        index
        for index, step in enumerate(post_steps)
        if getattr(step.__class__, "_registry_name", None) == "unnormalizer_processor"
    )
    post_adapter_index = next(
        index
        for index, step in enumerate(post_steps)
        if getattr(step.__class__, "_registry_name", None)
        == PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME
    )
    if post_adapter_index != unnormalizer_index + 1:
        raise ValueError("SmolVLA pi-Aloha postprocessor ordering is invalid.")


def ensure_smolvla_action_boundary(
    preprocessor: Any,
    postprocessor: Any,
    contract: ActionContract,
    action_space: Any,
    *,
    action_contract_sha256: str,
    upstream_revision: str,
) -> None:
    """Install the complete registered standard-ALOHA to model-space boundary."""

    if (
        action_space.target_projection != "action_contract_clip"
        or action_space.representation_adapter not in _ACTION_ADAPTERS
        or action_space.adapt_to_pi_aloha
    ):
        raise ValueError("SmolVLA action-space contract would double or omit adaptation.")
    ensure_action_contract_projection(
        preprocessor,
        contract,
        action_contract_sha256=action_contract_sha256,
    )
    ensure_pi_aloha_processors(
        preprocessor,
        postprocessor,
        contract,
        upstream_revision=upstream_revision,
        action_representation_adapter=action_space.representation_adapter,
    )


def processor_state_path(
    pretrained_dir: Path,
    *,
    pipeline_config_filename: str,
    registry_name: str,
) -> Path:
    """Resolve a processor state file from its serialized pipeline config."""

    config_path = pretrained_dir / pipeline_config_filename
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
        raise ValueError("Saved processor pipeline config is invalid.")
    matches = [
        step.get("state_file")
        for step in value["steps"]
        if isinstance(step, dict) and step.get("registry_name") == registry_name
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError(f"Saved processor state is missing or ambiguous: {registry_name}.")
    relative = Path(matches[0])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError("Saved processor state path is unsafe.")
    return pretrained_dir / relative
