"""Thin JSON-lines client for the official MoveIt 2/OMPL ALOHA sidecar."""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

EXPECTED_JOINT_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
)
MOVEIT_JOINT_GOAL_TOLERANCE_RAD = 1e-6


class MoveItAlohaPlannerError(RuntimeError):
    """Base error for the isolated planner process boundary."""


class MoveItAlohaPlanningError(MoveItAlohaPlannerError):
    """The official planning pipeline rejected or could not solve a request."""

    def __init__(self, reason: str, response: dict[str, Any]) -> None:
        super().__init__(f"MoveIt ALOHA planning failed: {reason}")
        self.reason = reason
        self.response = response


@dataclass(frozen=True, slots=True)
class MoveItAlohaPlannerSettings:
    """Frozen process and official-backend identity contract."""

    executable: Path
    urdf: Path
    srdf: Path
    stderr_log: Path
    ompl_seed: int = 2210
    response_timeout_s: float = 5.0
    launcher: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("executable", "urdf", "srdf"):
            path = getattr(self, name)
            if not path.is_file():
                raise FileNotFoundError(f"MoveIt ALOHA {name} is not a file: {path}")
        if self.stderr_log.exists():
            raise FileExistsError(
                f"Refusing to overwrite MoveIt stderr log: {self.stderr_log}"
            )
        if self.ompl_seed < 0:
            raise ValueError("ompl_seed must be nonnegative.")
        if not math.isfinite(self.response_timeout_s) or self.response_timeout_s <= 0.0:
            raise ValueError("response_timeout_s must be finite and positive.")
        if any(not value for value in self.launcher):
            raise ValueError("MoveIt ALOHA launcher entries must be nonempty.")

    @classmethod
    def from_environment(
        cls,
        *,
        stderr_log: Path,
        ompl_seed: int = 2210,
        response_timeout_s: float = 5.0,
    ) -> MoveItAlohaPlannerSettings:
        """Resolve container-local paths without embedding host paths in plans."""

        names = {
            "executable": "ROSETTA_ALOHA_MOVEIT_EXECUTABLE",
            "urdf": "ROSETTA_ALOHA_MOVEIT_URDF",
            "srdf": "ROSETTA_ALOHA_MOVEIT_SRDF",
        }
        values: dict[str, Path] = {}
        for field, variable in names.items():
            raw = os.environ.get(variable)
            if not raw:
                raise OSError(f"{variable} must identify the pinned MoveIt artifact.")
            values[field] = Path(raw)
        return cls(
            **values,
            stderr_log=stderr_log,
            ompl_seed=ompl_seed,
            response_timeout_s=response_timeout_s,
        )


@dataclass(frozen=True, slots=True)
class MoveItAlohaPlanResult:
    """Validated official OMPL path result in registered arm-joint order."""

    goal: tuple[float, ...]
    next: tuple[float, ...]
    planning_time_s: float
    waypoint_count: int
    path_length_rad: float
    path_maximum_waypoint_joint_delta_rad: float
    first_segment_interpolation: float
    maximum_goal_position_error_m: float
    maximum_goal_orientation_error_rad: float
    maximum_goal_weighted_error: float
    ik_task_mode: str
    maximum_orientation_relaxation_rad: float
    joint_limit_margin_rad: float
    physical_joint_limit_margin_rad: float
    start_state_satisfies_joint_path_constraint: bool
    start_state_path_constraint_recovery: bool
    adapter_added_state_indices: tuple[int, ...]
    adapter_prefix_waypoint_count: int
    minimum_recovery_progress_rad: float
    minimum_start_joint_limit_margin_rad: float
    minimum_goal_joint_limit_margin_rad: float
    minimum_path_joint_limit_margin_rad: float
    minimum_constrained_path_joint_limit_margin_rad: float
    minimum_adapter_prefix_physical_joint_limit_margin_rad: float
    minimum_next_joint_limit_margin_rad: float
    start_bound_reconciliations: tuple[str, ...]
    maximum_start_bound_reconciliation_rad: float
    maximum_requested_start_to_next_joint_delta_rad: float
    ik_search_mode: str
    ik_candidate_selection_mode: str
    ik_seed: int
    ik_maximum_attempts: int
    ik_attempts_used: int
    valid_ik_candidate_count: int
    selected_ik_attempt: int
    selected_ik_minimum_joint_limit_margin_rad: float
    selected_ik_maximum_start_delta_rad: float
    ik_outer_timeout_s: float
    trajectory: tuple[tuple[float, ...], ...]
    raw_response: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MoveItAlohaTrajectoryCommand:
    """One bounded controller target sampled from a retained MoveIt trajectory."""

    positions: tuple[float, ...]
    waypoint_index: int
    waypoint_count: int
    waypoint_advanced: bool
    waypoint_l1_distance_rad: float
    interpolation: float
    reference_reused: bool
    terminal_handoff_ready: bool
    terminal_control_active: bool
    terminal_control_activated: bool


@dataclass(slots=True)
class MoveItAlohaTrajectoryExecutor:
    """Retain and sample an official global trajectory like MoveIt SimpleSampler.

    MoveIt 2.5.9's Hybrid Planning ``SimpleSampler`` keeps a reference
    trajectory, advances by one waypoint when the current joint state is within
    an L1 tolerance, and otherwise continues forwarding the same waypoint.  The
    final interpolation below is the simulator controller boundary: it preserves
    the registered per-command joint-target delta while following that retained
    reference instead of replanning from a new redundant IK solution every step.
    """

    waypoint_l1_tolerance_rad: float
    maximum_joint_step_rad: float
    _phase: str | None = None
    _plan: MoveItAlohaPlanResult | None = None
    _next_waypoint_index: int = 0
    _command_count: int = 0
    _terminal_positions: tuple[float, ...] | None = None
    _terminal_command_count: int = 0

    def __post_init__(self) -> None:
        for name in ("waypoint_l1_tolerance_rad", "maximum_joint_step_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

    @property
    def plan_result(self) -> MoveItAlohaPlanResult | None:
        return self._plan

    def active_for(self, phase: str) -> bool:
        return self._plan is not None and self._phase == phase

    def terminal_control_active_for(self, phase: str) -> bool:
        return self.active_for(phase) and self._terminal_positions is not None

    def reset(self) -> None:
        self._phase = None
        self._plan = None
        self._next_waypoint_index = 0
        self._command_count = 0
        self._terminal_positions = None
        self._terminal_command_count = 0

    def install(self, plan: MoveItAlohaPlanResult, *, phase: str) -> None:
        if not phase:
            raise ValueError("MoveIt trajectory phase must be nonempty.")
        if len(plan.trajectory) != plan.waypoint_count:
            raise MoveItAlohaPlannerError(
                "MoveIt retained trajectory does not match its waypoint count."
            )
        self.reset()
        self._phase = phase
        self._plan = plan

    def activate_terminal_control(
        self,
        positions: Sequence[float],
        *,
        phase: str,
    ) -> None:
        """Latch one bounded terminal controller reference for the active phase."""

        if not self.active_for(phase) or self._plan is None:
            raise MoveItAlohaPlannerError(
                "MoveIt terminal control requires an active retained trajectory."
            )
        if self._next_waypoint_index != len(self._plan.trajectory) - 1:
            raise MoveItAlohaPlannerError(
                "MoveIt terminal control cannot activate before the final waypoint."
            )
        terminal = _finite_vector(list(positions), 12, "terminal positions")
        self._terminal_positions = terminal
        self._terminal_command_count = 0

    def complete_terminal_control(
        self,
        current: Sequence[float],
        *,
        phase: str,
        goal_l1_tolerance_rad: float,
    ) -> bool:
        """Reset a completed terminal reference like MoveIt's local planner.

        The terminal controller may command a feedforward-shifted position, so
        completion is measured against the original, uncompensated MoveIt goal.
        """

        tolerance = float(goal_l1_tolerance_rad)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "MoveIt terminal goal L1 tolerance must be finite and positive."
            )
        if not self.terminal_control_active_for(phase) or self._plan is None:
            return False
        current_positions = _finite_vector(list(current), 12, "current")
        distance = sum(
            abs(value - observed)
            for value, observed in zip(self._plan.goal, current_positions)
        )
        if distance > tolerance:
            return False
        self.reset()
        return True

    def command(
        self,
        current: Sequence[float],
        *,
        phase: str,
    ) -> MoveItAlohaTrajectoryCommand:
        if not self.active_for(phase) or self._plan is None:
            raise MoveItAlohaPlannerError(
                "MoveIt retained trajectory is not active for this phase."
            )
        current_positions = _finite_vector(list(current), 12, "current")
        trajectory = self._plan.trajectory
        terminal_control_active = self._terminal_positions is not None
        terminal_control_activated = (
            terminal_control_active and self._terminal_command_count == 0
        )
        advanced = False
        if terminal_control_active:
            desired = self._terminal_positions
            distance = sum(
                abs(value - observed)
                for value, observed in zip(desired, current_positions)
            )
        else:
            desired = trajectory[self._next_waypoint_index]
            distance = sum(
                abs(value - observed)
                for value, observed in zip(desired, current_positions)
            )
            if distance <= self.waypoint_l1_tolerance_rad:
                next_index = min(self._next_waypoint_index + 1, len(trajectory) - 1)
                advanced = next_index != self._next_waypoint_index
                self._next_waypoint_index = next_index
                desired = trajectory[self._next_waypoint_index]
                distance = sum(
                    abs(value - observed)
                    for value, observed in zip(desired, current_positions)
                )

        terminal_handoff_ready = (
            not terminal_control_active
            and self._next_waypoint_index == len(trajectory) - 1
            and distance <= self.waypoint_l1_tolerance_rad
        )

        deltas = tuple(
            value - observed
            for value, observed in zip(desired, current_positions)
        )
        maximum_delta = max(abs(value) for value in deltas)
        interpolation = (
            1.0
            if maximum_delta <= self.maximum_joint_step_rad
            else self.maximum_joint_step_rad / maximum_delta
        )
        positions = tuple(
            observed + interpolation * delta
            for observed, delta in zip(current_positions, deltas)
        )
        command = MoveItAlohaTrajectoryCommand(
            positions=positions,
            waypoint_index=self._next_waypoint_index,
            waypoint_count=len(trajectory),
            waypoint_advanced=advanced,
            waypoint_l1_distance_rad=distance,
            interpolation=interpolation,
            reference_reused=self._command_count > 0,
            terminal_handoff_ready=terminal_handoff_ready,
            terminal_control_active=terminal_control_active,
            terminal_control_activated=terminal_control_activated,
        )
        self._command_count += 1
        if terminal_control_active:
            self._terminal_command_count += 1
        return command


def _finite_vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise MoveItAlohaPlannerError(f"{name} must contain {size} values.")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise MoveItAlohaPlannerError(f"{name} contains a non-finite value.")
    return result


def _target_payload(
    position: Sequence[float],
    quaternion_wxyz: Sequence[float],
    name: str,
) -> dict[str, list[float]]:
    position_value = _finite_vector(list(position), 3, f"{name}.position")
    quaternion_value = _finite_vector(
        list(quaternion_wxyz), 4, f"{name}.quaternion_wxyz"
    )
    norm = math.sqrt(sum(value * value for value in quaternion_value))
    if norm <= 1e-12:
        raise MoveItAlohaPlannerError(f"{name}.quaternion_wxyz has zero norm.")
    return {
        "position": list(position_value),
        "quaternion_wxyz": [value / norm for value in quaternion_value],
    }


class MoveItAlohaPlanner:
    """Own one sidecar process and serialize bounded planning requests."""

    def __init__(self, settings: MoveItAlohaPlannerSettings) -> None:
        self.settings = settings
        self.settings.stderr_log.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_stream = self.settings.stderr_log.open(
            "x", encoding="utf-8", newline="\n"
        )
        try:
            self._process = subprocess.Popen(
                [
                    *settings.launcher,
                    str(settings.executable),
                    str(settings.urdf),
                    str(settings.srdf),
                    str(settings.ompl_seed),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_stream,
                text=True,
                encoding="utf-8",
                bufsize=1,
                close_fds=True,
            )
        except BaseException:
            self._stderr_stream.close()
            raise
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise MoveItAlohaPlannerError("MoveIt sidecar pipes were not created.")
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="moveit-aloha-stdout",
            daemon=True,
        )
        self._stdout_thread.start()
        self._request_sequence = 0
        self._closed = False
        try:
            self.identity = self._request({"command": "identity"})
            self._validate_identity(self.identity)
        except BaseException:
            self.close()
            raise

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            for line in self._process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if getattr(self, "_closed", False):
            raise MoveItAlohaPlannerError("MoveIt sidecar is closed.")
        if self._process.poll() is not None:
            raise MoveItAlohaPlannerError(
                f"MoveIt sidecar exited with code {self._process.returncode}."
            )
        self._request_sequence += 1
        request_id = f"request-{self._request_sequence:08d}"
        request = dict(payload)
        request["request_id"] = request_id
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        try:
            line = self._responses.get(timeout=self.settings.response_timeout_s)
        except queue.Empty as error:
            raise MoveItAlohaPlannerError(
                f"MoveIt sidecar timed out after {self.settings.response_timeout_s}s."
            ) from error
        if line is None:
            raise MoveItAlohaPlannerError(
                f"MoveIt sidecar closed stdout with code {self._process.poll()}."
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise MoveItAlohaPlannerError("MoveIt sidecar returned invalid JSON.") from error
        if not isinstance(response, dict):
            raise MoveItAlohaPlannerError("MoveIt sidecar response must be an object.")
        if response.get("request_id") != request_id:
            raise MoveItAlohaPlannerError("MoveIt sidecar response identity is out of order.")
        return response

    @staticmethod
    def _validate_identity(identity: dict[str, Any]) -> None:
        expected = {
            "status": "ok",
            "backend": "moveit2_ompl",
            "ros_distro": "humble",
            "moveit_version": "2.5.9",
            "ompl_version": "1.7.0",
            "planner_plugin": "ompl_interface/OMPLPlanner",
            "planner_id": "RRTConnect",
            "planner_type": "geometric::RRTConnect",
            "kinematics_plugin": "lma_kinematics_plugin/LMAKinematicsPlugin",
            "ik_group_selection_mode": "explicit_registered_groups_v1",
            "full_pose_groups": ["left_arm", "right_arm"],
            "ik_task_modes": ["full_pose", "position_priority"],
            "position_priority_kinematics_plugin": (
                "lma_kinematics_plugin/LMAKinematicsPlugin"
            ),
            "position_priority_groups": [
                "left_arm_position_priority",
                "right_arm_position_priority",
            ],
            "position_priority_orientation_weight": 0.0,
            "position_priority_ompl_seed_reset_per_request": True,
            "position_priority_terminal_goal_normalization_limit_rad": 1e-5,
            "ik_search_mode": "deterministic_seeded_moveit_subgroup_multistart_v1",
            "ik_candidate_selection_mode": (
                "deterministic_maximum_minimum_joint_limit_margin_v1"
            ),
            "ik_solver_base_frames": [
                "vx300s_left/base_link",
                "vx300s_right/base_link",
            ],
            "ik_solver_tip_frames": [
                "vx300s_left/ee_gripper_link",
                "vx300s_right/ee_gripper_link",
            ],
            "planning_group": "bimanual",
            "planning_frame": "world",
            "joint_path_constraint_type": "moveit_msgs/JointConstraint",
            "planning_request_adapters": [
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ],
        }
        for name, value in expected.items():
            if identity.get(name) != value:
                raise MoveItAlohaPlannerError(
                    f"MoveIt identity mismatch for {name}: {identity.get(name)!r}."
                )
        if tuple(identity.get("joint_names", ())) != EXPECTED_JOINT_NAMES:
            raise MoveItAlohaPlannerError("MoveIt arm-joint ordering differs.")
        for name in ("collision_geometry_link_count", "collision_geometry_shape_count"):
            value = identity.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MoveItAlohaPlannerError(
                    f"MoveIt identity has no loaded collision geometry for {name}."
                )

    def plan(
        self,
        *,
        start: Sequence[float],
        finger_positions: Sequence[float],
        left_position: Sequence[float],
        left_quaternion_wxyz: Sequence[float],
        right_position: Sequence[float],
        right_quaternion_wxyz: Sequence[float],
        allowed_planning_time_s: float,
        ik_timeout_s: float,
        ik_search_mode: str,
        ik_seed: int,
        ik_maximum_attempts: int,
        maximum_joint_step_rad: float,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        rotation_weight: float,
        maximum_accepted_error: float,
        maximum_accepted_projected_error: float,
        start_bound_reconciliation_tolerance_rad: float,
        physical_joint_limit_margin_rad: float,
        joint_limit_margin_rad: float,
        ik_task_mode: str = "full_pose",
        maximum_orientation_relaxation_rad: float = 0.04,
        include_trajectory: bool = False,
    ) -> MoveItAlohaPlanResult:
        """Request one collision-checked official RRTConnect path."""

        positive = {
            "allowed_planning_time_s": allowed_planning_time_s,
            "ik_timeout_s": ik_timeout_s,
            "maximum_joint_step_rad": maximum_joint_step_rad,
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "rotation_weight": rotation_weight,
            "maximum_accepted_error": maximum_accepted_error,
            "maximum_accepted_projected_error": maximum_accepted_projected_error,
            "start_bound_reconciliation_tolerance_rad": (
                start_bound_reconciliation_tolerance_rad
            ),
            "physical_joint_limit_margin_rad": physical_joint_limit_margin_rad,
            "joint_limit_margin_rad": joint_limit_margin_rad,
            "maximum_orientation_relaxation_rad": (
                maximum_orientation_relaxation_rad
            ),
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if ik_search_mode != "deterministic_seeded_moveit_subgroup_multistart_v1":
            raise ValueError("ik_search_mode differs from the registered mode.")
        if ik_task_mode not in {"full_pose", "position_priority"}:
            raise ValueError("ik_task_mode must be full_pose or position_priority.")
        if maximum_orientation_relaxation_rad > 0.04 + 1e-12:
            raise ValueError("maximum_orientation_relaxation_rad exceeds 0.04 rad.")
        if isinstance(ik_seed, bool) or not isinstance(ik_seed, int) or ik_seed < 0:
            raise ValueError("ik_seed must be a nonnegative integer.")
        if (
            isinstance(ik_maximum_attempts, bool)
            or not isinstance(ik_maximum_attempts, int)
            or not 1 <= ik_maximum_attempts <= 4096
        ):
            raise ValueError("ik_maximum_attempts must be within [1, 4096].")
        if physical_joint_limit_margin_rad >= joint_limit_margin_rad:
            raise ValueError(
                "physical_joint_limit_margin_rad must be smaller than "
                "joint_limit_margin_rad."
            )
        response = self._request(
            {
                "command": "plan",
                "start": list(_finite_vector(list(start), 12, "start")),
                "finger_positions": list(
                    _finite_vector(list(finger_positions), 2, "finger_positions")
                ),
                "targets": {
                    "left": _target_payload(
                        left_position, left_quaternion_wxyz, "targets.left"
                    ),
                    "right": _target_payload(
                        right_position, right_quaternion_wxyz, "targets.right"
                    ),
                },
                **positive,
                "ik_task_mode": ik_task_mode,
                "ik_search_mode": ik_search_mode,
                "ik_seed": ik_seed,
                "ik_maximum_attempts": ik_maximum_attempts,
                "include_trajectory": include_trajectory,
            }
        )
        if response.get("status") != "ok":
            reason = response.get("reason")
            if not isinstance(reason, str):
                reason = "invalid_error_response"
            if reason in {
                "bimanual_lma_ik_failed",
                "bimanual_position_priority_lma_ik_failed",
            }:
                attempts = response.get("ik_attempts_used")
                valid_candidates = response.get("valid_ik_candidate_count")
                if (
                    response.get("ik_task_mode") != ik_task_mode
                    or
                    response.get("ik_search_mode") != ik_search_mode
                    or response.get("ik_candidate_selection_mode")
                    != "deterministic_maximum_minimum_joint_limit_margin_v1"
                    or response.get("ik_seed") != ik_seed
                    or response.get("ik_maximum_attempts")
                    != ik_maximum_attempts
                    or isinstance(attempts, bool)
                    or not isinstance(attempts, int)
                    or not 1 <= attempts <= ik_maximum_attempts
                    or isinstance(valid_candidates, bool)
                    or not isinstance(valid_candidates, int)
                    or not 0 <= valid_candidates <= attempts
                    or not math.isclose(
                        float(response.get("ik_outer_timeout_s", math.nan)),
                        ik_timeout_s,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ):
                    raise MoveItAlohaPlannerError(
                        "MoveIt failed IK without its registered deterministic evidence."
                    )
            raise MoveItAlohaPlanningError(reason, response)
        if response.get("backend") != "moveit2_ompl":
            raise MoveItAlohaPlannerError("Planning response backend identity differs.")
        if response.get("ik_task_mode") != ik_task_mode:
            raise MoveItAlohaPlannerError("Planning response IK task mode differs.")
        if not math.isclose(
            float(response.get("maximum_orientation_relaxation_rad", math.nan)),
            maximum_orientation_relaxation_rad,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise MoveItAlohaPlannerError(
                "Planning response orientation-relaxation bound differs."
            )
        if response.get("planner_plugin") != "ompl_interface/OMPLPlanner":
            raise MoveItAlohaPlannerError("Planning response plugin identity differs.")
        if response.get("planner_id") != "RRTConnect":
            raise MoveItAlohaPlannerError("Planning response planner identity differs.")
        if tuple(response.get("joint_names", ())) != EXPECTED_JOINT_NAMES:
            raise MoveItAlohaPlannerError("Planning response joint ordering differs.")
        response_ik_attempts = response.get("ik_attempts_used")
        valid_ik_candidate_count = response.get("valid_ik_candidate_count")
        selected_ik_attempt = response.get("selected_ik_attempt")
        if (
            response.get("ik_search_mode") != ik_search_mode
            or response.get("ik_candidate_selection_mode")
            != "deterministic_maximum_minimum_joint_limit_margin_v1"
            or response.get("ik_seed") != ik_seed
            or response.get("ik_maximum_attempts") != ik_maximum_attempts
            or isinstance(response_ik_attempts, bool)
            or not isinstance(response_ik_attempts, int)
            or not 1 <= response_ik_attempts <= ik_maximum_attempts
            or isinstance(valid_ik_candidate_count, bool)
            or not isinstance(valid_ik_candidate_count, int)
            or not 1 <= valid_ik_candidate_count <= response_ik_attempts
            or isinstance(selected_ik_attempt, bool)
            or not isinstance(selected_ik_attempt, int)
            or not 1 <= selected_ik_attempt <= response_ik_attempts
            or not math.isclose(
                float(response.get("ik_outer_timeout_s", math.nan)),
                ik_timeout_s,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt deterministic IK search evidence differs."
            )
        reconciliations = response.get("start_bound_reconciliations")
        if not isinstance(reconciliations, list):
            raise MoveItAlohaPlannerError(
                "MoveIt response omits start-bound reconciliation evidence."
            )
        reconciled_names: list[str] = []
        reconciled_deltas: list[float] = []
        for item in reconciliations:
            if not isinstance(item, dict):
                raise MoveItAlohaPlannerError(
                    "MoveIt start-bound reconciliation entry is not an object."
                )
            name = item.get("joint_name")
            if name not in EXPECTED_JOINT_NAMES or name in reconciled_names:
                raise MoveItAlohaPlannerError(
                    "MoveIt start-bound reconciliation joint identity differs."
                )
            requested = float(item.get("requested_position_rad", math.nan))
            reconciled = float(item.get("reconciled_position_rad", math.nan))
            delta = float(item.get("delta_rad", math.nan))
            if (
                not all(math.isfinite(value) for value in (requested, reconciled, delta))
                or delta <= 0.0
                or not math.isclose(
                    abs(reconciled - requested), delta, rel_tol=0.0, abs_tol=1e-12
                )
                or delta > start_bound_reconciliation_tolerance_rad + 1e-12
            ):
                raise MoveItAlohaPlannerError(
                    "MoveIt start-bound reconciliation exceeds its registered tolerance."
                )
            reconciled_names.append(name)
            reconciled_deltas.append(delta)
        maximum_reconciliation = float(
            response.get("maximum_start_bound_reconciliation_rad", math.nan)
        )
        observed_maximum_reconciliation = max(reconciled_deltas, default=0.0)
        if (
            not math.isfinite(maximum_reconciliation)
            or maximum_reconciliation < 0.0
            or not math.isclose(
                maximum_reconciliation,
                observed_maximum_reconciliation,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or maximum_reconciliation
            > start_bound_reconciliation_tolerance_rad + 1e-12
            or not math.isclose(
                float(
                    response.get("start_bound_reconciliation_tolerance_rad", math.nan)
                ),
                start_bound_reconciliation_tolerance_rad,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt response start-bound reconciliation summary differs."
            )
        goal = _finite_vector(response.get("goal"), 12, "goal")
        next_positions = _finite_vector(response.get("next"), 12, "next")
        start_positions = _finite_vector(list(start), 12, "start")
        observed_maximum_delta = max(
            abs(value - current)
            for value, current in zip(next_positions, start_positions)
        )
        if observed_maximum_delta > maximum_joint_step_rad + 1e-9:
            raise MoveItAlohaPlannerError(
                "MoveIt execution waypoint exceeds the registered joint step."
            )
        scalar_names = (
            "planning_time_s",
            "path_length_rad",
            "path_maximum_waypoint_joint_delta_rad",
            "first_segment_interpolation",
            "maximum_goal_position_error_m",
            "maximum_goal_orientation_error_rad",
            "maximum_goal_weighted_error",
            "maximum_terminal_goal_normalization_rad",
            "maximum_requested_start_to_next_joint_delta_rad",
            "minimum_recovery_progress_rad",
            "minimum_start_joint_limit_margin_rad",
            "minimum_goal_joint_limit_margin_rad",
            "minimum_path_joint_limit_margin_rad",
            "minimum_constrained_path_joint_limit_margin_rad",
            "minimum_adapter_prefix_physical_joint_limit_margin_rad",
            "minimum_next_joint_limit_margin_rad",
            "selected_ik_minimum_joint_limit_margin_rad",
            "selected_ik_maximum_start_delta_rad",
        )
        scalars = {name: float(response.get(name, math.nan)) for name in scalar_names}
        if not all(math.isfinite(value) and value >= 0.0 for value in scalars.values()):
            raise MoveItAlohaPlannerError("MoveIt response contains invalid path metrics.")
        if scalars["selected_ik_minimum_joint_limit_margin_rad"] + 1e-12 < (
            joint_limit_margin_rad
        ):
            raise MoveItAlohaPlannerError(
                "Selected MoveIt IK candidate violates the registered joint margin."
            )
        observed_selected_start_delta = max(
            abs(value - current) for value, current in zip(goal, start_positions)
        )
        if not math.isclose(
            scalars["selected_ik_maximum_start_delta_rad"],
            observed_selected_start_delta,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise MoveItAlohaPlannerError(
                "Selected MoveIt IK candidate start delta differs."
            )
        position_priority = ik_task_mode == "position_priority"
        if (
            response.get("position_priority_ompl_seed_reset_per_request")
            is not position_priority
            or response.get("position_priority_terminal_goal_normalized")
            is not position_priority
            or not math.isclose(
                float(
                    response.get(
                        "terminal_goal_normalization_limit_rad", math.nan
                    )
                ),
                1e-5,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or scalars["maximum_terminal_goal_normalization_rad"] > 1e-5 + 1e-12
            or (
                not position_priority
                and not math.isclose(
                    scalars["maximum_terminal_goal_normalization_rad"],
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt position-priority terminal normalization evidence differs."
            )
        if scalars["maximum_goal_position_error_m"] > position_tolerance_m + 1e-12:
            raise MoveItAlohaPlannerError(
                "MoveIt response exceeds the registered position tolerance."
            )
        if ik_task_mode == "position_priority":
            if (
                scalars["maximum_goal_orientation_error_rad"]
                > maximum_orientation_relaxation_rad + 1e-12
            ):
                raise MoveItAlohaPlannerError(
                    "MoveIt position-priority response exceeds its orientation bound."
                )
        elif (
            scalars["maximum_goal_orientation_error_rad"]
            > orientation_tolerance_rad + 1e-12
            or scalars["maximum_goal_weighted_error"]
            > min(maximum_accepted_error, maximum_accepted_projected_error) + 1e-12
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt full-pose response exceeds its registered goal tolerance."
            )
        if not math.isclose(
            scalars["maximum_requested_start_to_next_joint_delta_rad"],
            observed_maximum_delta,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt response requested-start waypoint delta differs."
            )
        if response.get("joint_path_constraint_type") != (
            "moveit_msgs/JointConstraint"
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt response joint path-constraint type differs."
            )
        if response.get("joint_path_constraint_count") != len(EXPECTED_JOINT_NAMES):
            raise MoveItAlohaPlannerError(
                "MoveIt response joint path-constraint count differs."
            )
        if response.get("planning_request_adapters") != [
            "default_planner_request_adapters/FixStartStatePathConstraints"
        ]:
            raise MoveItAlohaPlannerError(
                "MoveIt response planning-request adapter identity differs."
            )
        if not math.isclose(
            float(response.get("joint_limit_margin_rad", math.nan)),
            joint_limit_margin_rad,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt response joint-limit margin differs."
            )
        if not math.isclose(
            float(response.get("physical_joint_limit_margin_rad", math.nan)),
            physical_joint_limit_margin_rad,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt response physical joint-limit margin differs."
            )
        for name in (
            "minimum_path_joint_limit_margin_rad",
            "minimum_adapter_prefix_physical_joint_limit_margin_rad",
            "minimum_next_joint_limit_margin_rad",
        ):
            if scalars[name] + 1e-12 < physical_joint_limit_margin_rad:
                raise MoveItAlohaPlannerError(
                    f"MoveIt response {name} violates the physical joint-limit margin."
                )
        for name in (
            "minimum_goal_joint_limit_margin_rad",
            "minimum_constrained_path_joint_limit_margin_rad",
        ):
            if scalars[name] + 1e-12 < joint_limit_margin_rad:
                raise MoveItAlohaPlannerError(
                    f"MoveIt response {name} violates the registered joint-limit margin."
                )
        waypoint_count = int(response.get("waypoint_count", 0))
        if waypoint_count < 2:
            raise MoveItAlohaPlannerError("MoveIt path contains fewer than two waypoints.")
        raw_trajectory = response.get("trajectory")
        if include_trajectory:
            if not isinstance(raw_trajectory, list) or len(raw_trajectory) != waypoint_count:
                raise MoveItAlohaPlannerError(
                    "MoveIt response omits the registered full trajectory."
                )
            trajectory = tuple(
                _finite_vector(value, 12, f"trajectory[{index}]")
                for index, value in enumerate(raw_trajectory)
            )
            computed_length = 0.0
            computed_maximum_delta = 0.0
            for previous, current in zip(trajectory, trajectory[1:]):
                deltas = tuple(
                    value - prior for value, prior in zip(current, previous)
                )
                computed_length += math.sqrt(sum(value * value for value in deltas))
                computed_maximum_delta = max(
                    computed_maximum_delta,
                    *(abs(value) for value in deltas),
                )
            if not math.isclose(
                computed_length,
                scalars["path_length_rad"],
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or not math.isclose(
                computed_maximum_delta,
                scalars["path_maximum_waypoint_joint_delta_rad"],
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise MoveItAlohaPlannerError(
                    "MoveIt full trajectory metrics differ from the validated path."
                )
            if any(
                abs(value - expected) > start_bound_reconciliation_tolerance_rad + 1e-12
                for value, expected in zip(trajectory[0], start_positions)
            ):
                raise MoveItAlohaPlannerError(
                    "MoveIt full trajectory start exceeds reconciliation tolerance."
                )
            if any(
                abs(value - expected) > MOVEIT_JOINT_GOAL_TOLERANCE_RAD + 1e-12
                for value, expected in zip(trajectory[-1], goal)
            ):
                raise MoveItAlohaPlannerError(
                    "MoveIt full trajectory endpoint differs from the validated goal."
                )
            expected_next = tuple(
                first
                + scalars["first_segment_interpolation"] * (second - first)
                for first, second in zip(trajectory[0], trajectory[1])
            )
            if any(
                abs(value - expected) > 1e-9
                for value, expected in zip(next_positions, expected_next)
            ):
                raise MoveItAlohaPlannerError(
                    "MoveIt bounded first command differs from the full trajectory."
                )
        else:
            if raw_trajectory is not None:
                raise MoveItAlohaPlannerError(
                    "MoveIt returned an unrequested full trajectory."
                )
            trajectory = ()
        start_satisfies = response.get("start_state_satisfies_joint_path_constraint")
        recovery = response.get("start_state_path_constraint_recovery")
        if not isinstance(start_satisfies, bool) or not isinstance(recovery, bool):
            raise MoveItAlohaPlannerError(
                "MoveIt response omits start-state path-constraint recovery state."
            )
        raw_adapter_indices = response.get("adapter_added_state_indices")
        prefix_count = response.get("adapter_prefix_waypoint_count")
        if (
            not isinstance(raw_adapter_indices, list)
            or isinstance(prefix_count, bool)
            or not isinstance(prefix_count, int)
            or prefix_count < 0
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt response adapter-prefix evidence is invalid."
            )
        adapter_indices = tuple(int(index) for index in raw_adapter_indices)
        if (
            any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in raw_adapter_indices
            )
            or adapter_indices != tuple(range(prefix_count))
            or prefix_count > waypoint_count
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt response adapter-added state indices differ."
            )
        if recovery:
            if (
                start_satisfies
                or prefix_count < 2
                or scalars["minimum_start_joint_limit_margin_rad"] + 1e-12
                >= joint_limit_margin_rad
                or scalars["minimum_recovery_progress_rad"] <= 0.0
            ):
                raise MoveItAlohaPlannerError(
                    "MoveIt start-state path-constraint recovery contract differs."
                )
        elif (
            not start_satisfies
            or prefix_count != 0
            or adapter_indices
            or not math.isclose(
                scalars["minimum_recovery_progress_rad"],
                0.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or scalars["minimum_start_joint_limit_margin_rad"] + 1e-12
            < joint_limit_margin_rad
            or scalars["minimum_next_joint_limit_margin_rad"] + 1e-12
            < joint_limit_margin_rad
        ):
            raise MoveItAlohaPlannerError(
                "MoveIt normal path unexpectedly used start-state recovery."
            )
        return MoveItAlohaPlanResult(
            goal=goal,
            next=next_positions,
            planning_time_s=scalars["planning_time_s"],
            waypoint_count=waypoint_count,
            path_length_rad=scalars["path_length_rad"],
            path_maximum_waypoint_joint_delta_rad=scalars[
                "path_maximum_waypoint_joint_delta_rad"
            ],
            first_segment_interpolation=scalars["first_segment_interpolation"],
            maximum_goal_position_error_m=scalars["maximum_goal_position_error_m"],
            maximum_goal_orientation_error_rad=scalars[
                "maximum_goal_orientation_error_rad"
            ],
            maximum_goal_weighted_error=scalars["maximum_goal_weighted_error"],
            ik_task_mode=ik_task_mode,
            maximum_orientation_relaxation_rad=(
                maximum_orientation_relaxation_rad
            ),
            joint_limit_margin_rad=joint_limit_margin_rad,
            physical_joint_limit_margin_rad=physical_joint_limit_margin_rad,
            start_state_satisfies_joint_path_constraint=start_satisfies,
            start_state_path_constraint_recovery=recovery,
            adapter_added_state_indices=adapter_indices,
            adapter_prefix_waypoint_count=prefix_count,
            minimum_recovery_progress_rad=scalars[
                "minimum_recovery_progress_rad"
            ],
            minimum_start_joint_limit_margin_rad=scalars[
                "minimum_start_joint_limit_margin_rad"
            ],
            minimum_goal_joint_limit_margin_rad=scalars[
                "minimum_goal_joint_limit_margin_rad"
            ],
            minimum_path_joint_limit_margin_rad=scalars[
                "minimum_path_joint_limit_margin_rad"
            ],
            minimum_constrained_path_joint_limit_margin_rad=scalars[
                "minimum_constrained_path_joint_limit_margin_rad"
            ],
            minimum_adapter_prefix_physical_joint_limit_margin_rad=scalars[
                "minimum_adapter_prefix_physical_joint_limit_margin_rad"
            ],
            minimum_next_joint_limit_margin_rad=scalars[
                "minimum_next_joint_limit_margin_rad"
            ],
            start_bound_reconciliations=tuple(reconciled_names),
            maximum_start_bound_reconciliation_rad=maximum_reconciliation,
            maximum_requested_start_to_next_joint_delta_rad=scalars[
                "maximum_requested_start_to_next_joint_delta_rad"
            ],
            ik_search_mode=ik_search_mode,
            ik_candidate_selection_mode=(
                "deterministic_maximum_minimum_joint_limit_margin_v1"
            ),
            ik_seed=ik_seed,
            ik_maximum_attempts=ik_maximum_attempts,
            ik_attempts_used=response_ik_attempts,
            valid_ik_candidate_count=valid_ik_candidate_count,
            selected_ik_attempt=selected_ik_attempt,
            selected_ik_minimum_joint_limit_margin_rad=scalars[
                "selected_ik_minimum_joint_limit_margin_rad"
            ],
            selected_ik_maximum_start_delta_rad=scalars[
                "selected_ik_maximum_start_delta_rad"
            ],
            ik_outer_timeout_s=ik_timeout_s,
            trajectory=trajectory,
            raw_response=response,
        )

    def close(self) -> None:
        """Close only the process created by this client."""

        if getattr(self, "_closed", True):
            return
        try:
            if self._process.poll() is None:
                try:
                    self._request({"command": "shutdown"})
                except MoveItAlohaPlannerError:
                    self._process.terminate()
            try:
                self._process.wait(timeout=self.settings.response_timeout_s)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=self.settings.response_timeout_s)
        finally:
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
            self._stdout_thread.join(timeout=self.settings.response_timeout_s)
            self._stderr_stream.close()

    def __enter__(self) -> MoveItAlohaPlanner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
