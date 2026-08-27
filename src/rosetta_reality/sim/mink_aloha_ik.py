"""Joint-limit-aware ALOHA differential IK backed by Mink's constrained QP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class MinkAlohaIkSettings:
    """Frozen numerical contract for the upstream Mink/DAQP solve."""

    integration_timestep_s: float
    maximum_iterations: int
    position_cost: float
    orientation_cost: float
    posture_cost: float
    frame_lm_damping: float
    solver_damping: float
    maximum_joint_velocity_rad_s: float
    configuration_limit_gain: float
    joint_limit_margin_rad: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "integration_timestep_s": self.integration_timestep_s,
            "maximum_iterations": self.maximum_iterations,
            "position_cost": self.position_cost,
            "orientation_cost": self.orientation_cost,
            "maximum_joint_velocity_rad_s": self.maximum_joint_velocity_rad_s,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative = {
            "posture_cost": self.posture_cost,
            "frame_lm_damping": self.frame_lm_damping,
            "solver_damping": self.solver_damping,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not 0.0 < self.configuration_limit_gain <= 1.0:
            raise ValueError("configuration_limit_gain must be in (0, 1].")
        if (
            not math.isfinite(self.joint_limit_margin_rad)
            or self.joint_limit_margin_rad < 0.0
        ):
            raise ValueError("joint_limit_margin_rad must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class MinkAlohaIkResult:
    """A QP solution and the unweighted task-space residuals it achieved."""

    qpos: np.ndarray
    iterations: int
    position_errors_m: tuple[float, float]
    orientation_errors_rad: tuple[float, float]


class MinkAlohaIkSolver:
    """Thin ALOHA adapter around Mink's standard constrained differential IK."""

    _ARM_JOINT_NAMES = (
        "vx300s_left/waist",
        "vx300s_left/shoulder",
        "vx300s_left/elbow",
        "vx300s_left/forearm_roll",
        "vx300s_left/wrist_angle",
        "vx300s_left/wrist_rotate",
        "vx300s_right/waist",
        "vx300s_right/shoulder",
        "vx300s_right/elbow",
        "vx300s_right/forearm_roll",
        "vx300s_right/wrist_angle",
        "vx300s_right/wrist_rotate",
    )

    def __init__(
        self,
        model_path: Path,
        *,
        left_site: str,
        right_site: str,
        settings: MinkAlohaIkSettings,
    ) -> None:
        import mink
        import mujoco

        self._mink = mink
        self.settings = settings
        self.model_path = model_path.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        for name in self._ARM_JOINT_NAMES:
            joint_id = int(self.model.joint(name).id)
            if not bool(self.model.jnt_limited[joint_id]):
                raise ValueError(f"ALOHA arm joint is unexpectedly unlimited: {name}.")
            minimum, maximum = self.model.jnt_range[joint_id]
            if 2.0 * settings.joint_limit_margin_rad >= maximum - minimum:
                raise ValueError(f"joint_limit_margin_rad collapses joint range: {name}.")
            self.model.jnt_range[joint_id] = (
                minimum + settings.joint_limit_margin_rad,
                maximum - settings.joint_limit_margin_rad,
            )
        self.configuration = mink.Configuration(self.model)
        self._tasks = (
            mink.FrameTask(
                frame_name=left_site,
                frame_type="site",
                position_cost=settings.position_cost,
                orientation_cost=settings.orientation_cost,
                lm_damping=settings.frame_lm_damping,
            ),
            mink.FrameTask(
                frame_name=right_site,
                frame_type="site",
                position_cost=settings.position_cost,
                orientation_cost=settings.orientation_cost,
                lm_damping=settings.frame_lm_damping,
            ),
        )
        self._posture_task = mink.PostureTask(
            self.model,
            cost=settings.posture_cost,
        )
        self._posture_target_initialized = False
        velocities = {
            name: settings.maximum_joint_velocity_rad_s
            for name in self._ARM_JOINT_NAMES
        }
        arm_dof_indices = {
            int(self.model.jnt_dofadr[self.model.joint(name).id])
            for name in self._ARM_JOINT_NAMES
        }
        frozen_dof_indices = sorted(set(range(self.model.nv)) - arm_dof_indices)
        self._limited_joint_ids = tuple(
            joint_id
            for joint_id in range(self.model.njnt)
            if bool(self.model.jnt_limited[joint_id])
        )
        self._constraints = (
            mink.DofFreezingTask(
                model=self.model,
                dof_indices=frozen_dof_indices,
            ),
        )
        self._limits = (
            mink.ConfigurationLimit(
                model=self.model,
                gain=settings.configuration_limit_gain,
            ),
            mink.VelocityLimit(self.model, velocities),
        )

    @classmethod
    def from_gym_aloha(
        cls,
        *,
        left_site: str,
        right_site: str,
        settings: MinkAlohaIkSettings,
    ) -> MinkAlohaIkSolver:
        """Load the exact insertion MJCF shipped by the pinned Gym-ALOHA."""

        from gym_aloha.constants import ASSETS_DIR

        return cls(
            ASSETS_DIR / "bimanual_viperx_insertion.xml",
            left_site=left_site,
            right_site=right_site,
            settings=settings,
        )

    def solve(
        self,
        current_qpos: np.ndarray,
        *,
        left_position: np.ndarray,
        left_quaternion_wxyz: np.ndarray,
        right_position: np.ndarray,
        right_quaternion_wxyz: np.ndarray,
    ) -> MinkAlohaIkResult:
        """Solve two full-pose tasks subject to native MJCF joint limits."""

        mink = self._mink
        qpos = np.asarray(current_qpos, dtype=np.float64)
        if qpos.shape != (16,):
            raise ValueError(
                f"Expected expanded ALOHA robot qpos shape {(16,)}, got {qpos.shape}."
            )
        if not np.isfinite(qpos).all():
            raise ValueError("Current ALOHA qpos must be finite.")
        native_qpos = np.asarray(self.configuration.q, dtype=np.float64).copy()
        native_qpos[:16] = qpos
        for joint_id in self._limited_joint_ids:
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            minimum, maximum = self.model.jnt_range[joint_id]
            native_qpos[qpos_address] = np.clip(
                native_qpos[qpos_address],
                minimum,
                maximum,
            )
        self.configuration.update(native_qpos)
        if not self._posture_target_initialized:
            self._posture_task.set_target_from_configuration(self.configuration)
            self._posture_target_initialized = True
        targets = (
            (left_position, left_quaternion_wxyz),
            (right_position, right_quaternion_wxyz),
        )
        for task, (position, quaternion) in zip(self._tasks, targets):
            position_array = np.asarray(position, dtype=np.float64)
            quaternion_array = np.asarray(quaternion, dtype=np.float64)
            if position_array.shape != (3,) or quaternion_array.shape != (4,):
                raise ValueError("Mink ALOHA targets require position[3] and wxyz[4].")
            if not np.isfinite(position_array).all() or not np.isfinite(
                quaternion_array
            ).all():
                raise ValueError("Mink ALOHA targets must be finite.")
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(quaternion_array),
                    position_array,
                )
            )

        position_errors = (math.inf, math.inf)
        orientation_errors = (math.inf, math.inf)
        for iteration in range(1, self.settings.maximum_iterations + 1):
            try:
                velocity = mink.solve_ik(
                    self.configuration,
                    [*self._tasks, self._posture_task],
                    dt=self.settings.integration_timestep_s,
                    solver="daqp",
                    damping=self.settings.solver_damping,
                    safety_break=True,
                    limits=self._limits,
                    constraints=self._constraints,
                )
            except (mink.NoSolutionFound, mink.NotWithinConfigurationLimits) as error:
                raise RuntimeError(f"Mink/DAQP constrained IK failed: {error}") from error
            if not np.isfinite(velocity).all():
                raise RuntimeError("Mink/DAQP returned a non-finite joint velocity.")
            self.configuration.integrate_inplace(
                velocity,
                self.settings.integration_timestep_s,
            )
            errors = tuple(task.compute_error(self.configuration) for task in self._tasks)
            position_errors = tuple(float(np.linalg.norm(error[:3])) for error in errors)
            orientation_errors = tuple(float(np.linalg.norm(error[3:])) for error in errors)
            if max(position_errors) <= 1e-6 and max(orientation_errors) <= 1e-6:
                break

        solved_qpos = np.asarray(self.configuration.q, dtype=np.float64).copy()
        if not np.isfinite(solved_qpos).all():
            raise RuntimeError("Mink/DAQP returned a non-finite configuration.")
        self.configuration.check_limits(safety_break=True)
        return MinkAlohaIkResult(
            qpos=solved_qpos,
            iterations=iteration,
            position_errors_m=position_errors,
            orientation_errors_rad=orientation_errors,
        )
