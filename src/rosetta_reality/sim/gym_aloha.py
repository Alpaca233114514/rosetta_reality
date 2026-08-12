"""Lazy Gym-ALOHA adapter behind the simulator-neutral Rosetta boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from rosetta_reality.sim.action_contract import ActionContract
from rosetta_reality.sim.env import SimulationEnvironment


class GymAlohaEnvironment(SimulationEnvironment):
    """Adapt a registered Gym-ALOHA task to Rosetta observations and actions."""

    def __init__(
        self,
        contract: ActionContract,
        *,
        observation_type: str = "pixels_agent_pos",
        render_mode: str | None = None,
        maximum_episode_steps: int | None = None,
        environment: Any | None = None,
    ) -> None:
        self.contract = contract
        self._last_clip_mask = torch.zeros(contract.dimension, dtype=torch.bool)
        if maximum_episode_steps is not None and maximum_episode_steps <= 0:
            raise ValueError("Simulator maximum episode steps must be positive.")
        if environment is None:
            try:
                import gym_aloha  # noqa: F401
                import gymnasium as gym
            except ImportError as error:
                raise RuntimeError(
                    "GymAlohaEnvironment requires the optional 'sim' dependencies."
                ) from error
            make_kwargs: dict[str, Any] = {
                "obs_type": observation_type,
                "render_mode": render_mode,
            }
            if maximum_episode_steps is not None:
                make_kwargs["max_episode_steps"] = maximum_episode_steps
            environment = gym.make(contract.environment_id, **make_kwargs)
        self._environment = environment

    @property
    def raw_environment(self) -> Any:
        """Expose the wrapped environment for read-only diagnostic metrics."""

        return self._environment

    @property
    def last_clip_mask(self) -> Tensor:
        """Return the logical fields clipped at the latest step."""

        return self._last_clip_mask.clone()

    def contact_pairs(self) -> tuple[tuple[str, str], ...]:
        """Return current MuJoCo geom contacts when the wrapped backend exposes them."""

        unwrapped = getattr(self._environment, "unwrapped", self._environment)
        control_environment = getattr(unwrapped, "_env", None)
        physics = getattr(control_environment, "physics", None)
        if physics is None:
            return ()
        pairs: list[tuple[str, str]] = []
        for index in range(int(physics.data.ncon)):
            contact = physics.data.contact[index]
            first = physics.model.id2name(int(contact.geom1), "geom") or "unknown"
            second = physics.model.id2name(int(contact.geom2), "geom") or "unknown"
            pairs.append((str(first), str(second)))
        return tuple(pairs)

    def unexpected_collision_count(self) -> int:
        """Count robot self/table/object contacts outside intended gripper-object contact."""

        return sum(
            self.is_unexpected_collision_pair(first, second)
            for first, second in self.contact_pairs()
        )

    def state_limit_violation_count(self) -> int:
        """Count limited MuJoCo joints outside their adapter-owned physical ranges."""

        unwrapped = getattr(self._environment, "unwrapped", self._environment)
        control_environment = getattr(unwrapped, "_env", None)
        physics = getattr(control_environment, "physics", None)
        if physics is None:
            return 0
        model = physics.model
        data = physics.data
        violations = 0
        for joint_id in range(int(model.njnt)):
            if not bool(model.jnt_limited[joint_id]):
                continue
            qpos_address = int(model.jnt_qposadr[joint_id])
            lower, upper = model.jnt_range[joint_id]
            value = float(data.qpos[qpos_address])
            violations += int(value < float(lower) - 1e-5 or value > float(upper) + 1e-5)
        return violations

    @staticmethod
    def is_unexpected_collision_pair(first: str, second: str) -> bool:
        """Classify one MuJoCo geom pair without hiding cross-arm gripper collisions."""

        def is_robot(name: str) -> bool:
            return name.startswith("vx300s_")

        def is_gripper(name: str) -> bool:
            return "gripper_finger" in name

        def arm_namespace(name: str) -> str:
            return name.split("/", maxsplit=1)[0]

        internal_gripper_contact = (
            is_gripper(first)
            and is_gripper(second)
            and arm_namespace(first) == arm_namespace(second)
        )
        if is_robot(first) and is_robot(second):
            return not internal_gripper_contact
        if first == "table" and is_robot(second) and not is_gripper(second):
            return True
        if second == "table" and is_robot(first) and not is_gripper(first):
            return True
        if is_robot(first) and not is_gripper(first) and not is_robot(second):
            return True
        return bool(is_robot(second) and not is_gripper(second) and not is_robot(first))

    @staticmethod
    def _observation(value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("Gym-ALOHA observation must be a mapping.")
        state_value = value.get("agent_pos", value.get("state"))
        if state_value is None:
            raise KeyError("Gym-ALOHA observation is missing 'agent_pos'.")
        state = torch.as_tensor(state_value, dtype=torch.float32)
        if state.ndim != 1:
            raise ValueError(f"Simulator state must be rank one, received {tuple(state.shape)}.")

        raw_pixels = value.get("pixels", value.get("images", {}))
        images: dict[str, Tensor] = {}
        if isinstance(raw_pixels, Mapping):
            for name, image_value in raw_pixels.items():
                image = torch.as_tensor(image_value)
                if image.ndim != 3:
                    raise ValueError(
                        f"Simulator image {name!r} must be rank three, "
                        f"received {tuple(image.shape)}."
                    )
                if image.shape[-1] in (1, 3, 4):
                    image = image.permute(2, 0, 1)
                images[str(name)] = (
                    image.to(torch.float32).div(255)
                    if image.dtype == torch.uint8
                    else image.to(torch.float32)
                )
        return {"robot_state": state, "images": images, "raw": value}

    def reset(self, *, seed: int | None = None) -> Mapping[str, Any]:
        """Reset and convert the first observation."""

        observation, info = self._environment.reset(seed=seed)
        converted = dict(self._observation(observation))
        converted["info"] = dict(info)
        return converted

    def step(self, action: Tensor) -> tuple[Mapping[str, Any], float, bool, dict[str, Any]]:
        """Clip a logical action, step once, and preserve termination details."""

        self.contract.validate_tensor(action, allow_chunk=False)
        clipped, mask = self.contract.clip(action.detach().to(torch.float32).cpu())
        self._last_clip_mask = mask
        observation, reward, terminated, truncated, info = self._environment.step(
            clipped.numpy()
        )
        result_info = dict(info)
        result_info.update(
            clipped_fields=[
                name for name, was_clipped in zip(self.contract.dimension_names, mask.tolist())
                if was_clipped
            ],
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        return (
            self._observation(observation),
            float(reward),
            bool(terminated or truncated),
            result_info,
        )

    def close(self) -> None:
        """Release the underlying MuJoCo environment."""

        self._environment.close()
