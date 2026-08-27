"""Event-driven object-geometry teacher for ALOHA insertion recovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class GeometryTeacherError(RuntimeError):
    """Raised when the teacher cannot safely provide a task-space target."""


def _finite_vector(value: Tensor, *, dimension: int, name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().clone()
    if tensor.shape != (dimension,):
        raise ValueError(
            f"{name} must have shape ({dimension},), received {tuple(tensor.shape)}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf.")
    return tensor


def _normalized_quaternion(value: Tensor, *, name: str) -> Tensor:
    quaternion = _finite_vector(value, dimension=4, name=name)
    norm = float(torch.linalg.vector_norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"{name} must have non-zero finite norm.")
    quaternion = quaternion / norm
    if float(quaternion[0]) < 0:
        quaternion = -quaternion
    return quaternion


def quaternion_multiply(first: Tensor, second: Tensor) -> Tensor:
    """Multiply two scalar-first quaternions and return a normalized result."""

    left = _normalized_quaternion(first, name="Left quaternion")
    right = _normalized_quaternion(second, name="Right quaternion")
    aw, ax, ay, az = left
    bw, bx, by, bz = right
    return _normalized_quaternion(
        torch.stack(
            (
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            )
        ),
        name="Quaternion product",
    )


def quaternion_conjugate(value: Tensor) -> Tensor:
    """Return the conjugate of a normalized scalar-first quaternion."""

    quaternion = _normalized_quaternion(value, name="Quaternion")
    result = quaternion.clone()
    result[1:] = -result[1:]
    return result


def quaternion_rotate(value: Tensor, vector: Tensor) -> Tensor:
    """Rotate one three-vector by a scalar-first quaternion."""

    quaternion = _normalized_quaternion(value, name="Rotation quaternion")
    point = _finite_vector(vector, dimension=3, name="Rotated vector")
    scalar = quaternion[:1]
    axis = quaternion[1:]
    return (
        2.0 * torch.dot(axis, point) * axis
        + (scalar.square().item() - torch.dot(axis, axis).item()) * point
        + 2.0 * scalar.item() * torch.cross(axis, point, dim=0)
    )


@dataclass(frozen=True, slots=True)
class GeometryPose:
    """Finite world or relative rigid pose using scalar-first quaternions."""

    position: Tensor
    quaternion: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _finite_vector(self.position, dimension=3, name="Pose position"),
        )
        object.__setattr__(
            self,
            "quaternion",
            _normalized_quaternion(self.quaternion, name="Pose quaternion"),
        )


def compose_pose(parent: GeometryPose, child: GeometryPose) -> GeometryPose:
    """Compose ``world_from_parent`` with ``parent_from_child``."""

    return GeometryPose(
        position=parent.position + quaternion_rotate(parent.quaternion, child.position),
        quaternion=quaternion_multiply(parent.quaternion, child.quaternion),
    )


def inverse_pose(value: GeometryPose) -> GeometryPose:
    """Invert a rigid pose."""

    quaternion = quaternion_conjugate(value.quaternion)
    return GeometryPose(
        position=quaternion_rotate(quaternion, -value.position),
        quaternion=quaternion,
    )


def relative_pose(parent: GeometryPose, child: GeometryPose) -> GeometryPose:
    """Return ``parent_from_child`` for two world poses."""

    return compose_pose(inverse_pose(parent), child)


@dataclass(frozen=True, slots=True)
class InsertionTeacherCalibration:
    """Train-only successful-task geometry used without replaying its timeline."""

    socket_to_left_eef_at_grasp: GeometryPose
    peg_to_right_eef_at_grasp: GeometryPose
    terminal_socket_to_peg: GeometryPose
    insertion_axis_in_socket: Tensor
    source_episode: int
    source_seed: int
    terminal_reward: float
    terminal_success: bool

    def __post_init__(self) -> None:
        axis = _finite_vector(
            self.insertion_axis_in_socket,
            dimension=3,
            name="Insertion axis",
        )
        norm = float(torch.linalg.vector_norm(axis))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError("Insertion axis must have non-zero finite norm.")
        if self.source_episode < 0 or self.source_seed < 0:
            raise ValueError("Calibration episode and seed must be non-negative.")
        if not math.isfinite(self.terminal_reward) or self.terminal_reward != 4.0:
            raise ValueError("Calibration must terminate at reward four.")
        if not self.terminal_success:
            raise ValueError("A failed trajectory cannot calibrate the geometry teacher.")
        object.__setattr__(self, "insertion_axis_in_socket", axis / norm)


@dataclass(frozen=True, slots=True)
class InsertionGeometry:
    """Current robot, object and event state consumed by the teacher."""

    robot_state: Tensor
    left_eef: GeometryPose
    right_eef: GeometryPose
    socket: GeometryPose
    peg: GeometryPose
    observed_reward: float
    socket_grasp_contact: bool
    peg_grasp_contact: bool
    socket_on_table: bool
    peg_on_table: bool
    peg_socket_contact: bool
    pin_contact: bool
    unexpected_collision_count: int

    def __post_init__(self) -> None:
        state = _finite_vector(
            self.robot_state,
            dimension=14,
            name="Insertion robot state",
        )
        if not math.isfinite(self.observed_reward):
            raise ValueError("Observed insertion reward must be finite.")
        if self.unexpected_collision_count < 0:
            raise ValueError("Unexpected collision count must be non-negative.")
        object.__setattr__(self, "robot_state", state)


@dataclass(frozen=True, slots=True)
class InsertionTeacherSettings:
    """Registered geometry thresholds and bounded feedback targets."""

    gripper_open_target: float = 1.0
    gripper_closed_target: float = 0.0
    gripper_open_threshold: float = 0.85
    approach_clearance_m: float = 0.08
    approach_tolerance_m: float = 0.012
    approach_orientation_tolerance_rad: float = 0.04
    grasp_tolerance_m: float = 0.008
    lift_object_height_m: float = 0.17
    lift_tolerance_m: float = 0.012
    lift_feedback_step_m: float = 0.006
    meet_center_x_m: float = 0.0
    meet_center_y_m: float = 0.5
    preinsert_clearance_m: float = 0.10
    coarse_alignment_tolerance_m: float = 0.015
    maximum_cartesian_step_m: float = 0.012
    maximum_orientation_step_rad: float = 0.04
    maximum_grasp_drift_m: float = 0.045
    workspace_x_min_m: float = -0.45
    workspace_x_max_m: float = 0.45
    workspace_y_min_m: float = 0.20
    workspace_y_max_m: float = 0.80
    workspace_z_min_m: float = 0.0
    workspace_z_max_m: float = 0.50

    def __post_init__(self) -> None:
        scalar_names = (
            "gripper_open_target",
            "gripper_closed_target",
            "gripper_open_threshold",
            "approach_clearance_m",
            "approach_tolerance_m",
            "approach_orientation_tolerance_rad",
            "grasp_tolerance_m",
            "lift_object_height_m",
            "lift_tolerance_m",
            "lift_feedback_step_m",
            "meet_center_x_m",
            "meet_center_y_m",
            "preinsert_clearance_m",
            "coarse_alignment_tolerance_m",
            "maximum_cartesian_step_m",
            "maximum_orientation_step_rad",
            "maximum_grasp_drift_m",
            "workspace_x_min_m",
            "workspace_x_max_m",
            "workspace_y_min_m",
            "workspace_y_max_m",
            "workspace_z_min_m",
            "workspace_z_max_m",
        )
        if any(not math.isfinite(getattr(self, name)) for name in scalar_names):
            raise ValueError("Geometry-teacher settings must be finite.")
        if not 0.0 <= self.gripper_closed_target < self.gripper_open_threshold:
            raise ValueError("Closed target must be below the gripper-open threshold.")
        if not self.gripper_open_threshold <= self.gripper_open_target <= 1.0:
            raise ValueError("Open target must reach the registered open threshold.")
        for name in (
            "approach_clearance_m",
            "approach_tolerance_m",
            "approach_orientation_tolerance_rad",
            "grasp_tolerance_m",
            "lift_tolerance_m",
            "lift_feedback_step_m",
            "preinsert_clearance_m",
            "coarse_alignment_tolerance_m",
            "maximum_cartesian_step_m",
            "maximum_orientation_step_rad",
            "maximum_grasp_drift_m",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not (
            self.workspace_x_min_m < self.workspace_x_max_m
            and self.workspace_y_min_m < self.workspace_y_max_m
            and self.workspace_z_min_m < self.workspace_z_max_m
        ):
            raise ValueError("Geometry-teacher workspace bounds are invalid.")
        if not self.workspace_z_min_m < self.lift_object_height_m < self.workspace_z_max_m:
            raise ValueError("Lift height must lie inside the registered workspace.")
        if not 0.0 < self.lift_feedback_step_m <= self.maximum_cartesian_step_m:
            raise ValueError(
                "Lift feedback step must be positive and no larger than the "
                "registered Cartesian step."
            )


class InsertionTeacherPhase(str, Enum):
    """Monotonic event-driven task phases; no phase is selected by time."""

    OPEN = "open"
    APPROACH = "approach"
    ORIENT = "orient"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    COARSE_ALIGN = "coarse_align"
    INSERT = "insert"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class InsertionTaskSpaceTarget:
    """Auditable Cartesian target to be converted by a simulator-owned IK adapter."""

    left_eef: GeometryPose
    right_eef: GeometryPose
    left_gripper: float
    right_gripper: float
    phase: InsertionTeacherPhase
    phase_changed: bool
    maximum_position_error_m: float
    best_observed_reward: float
    lift_feedback_anchor: bool = False


def _position_distance(first: GeometryPose, second: GeometryPose) -> float:
    return float(torch.linalg.vector_norm(first.position - second.position))


def _orientation_distance(first: GeometryPose, second: GeometryPose) -> float:
    dot = abs(float(torch.dot(first.quaternion, second.quaternion)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _with_world_z_offset(value: GeometryPose, offset: float) -> GeometryPose:
    position = value.position.clone()
    position[2] += offset
    return GeometryPose(position=position, quaternion=value.quaternion)


def _bounded_target(
    current: GeometryPose,
    desired: GeometryPose,
    maximum_step: float,
    maximum_orientation_step: float,
) -> GeometryPose:
    delta = desired.position - current.position
    distance = float(torch.linalg.vector_norm(delta))
    if distance > maximum_step:
        delta = delta * (maximum_step / distance)
    current_quaternion = current.quaternion
    desired_quaternion = desired.quaternion
    dot = float(torch.dot(current_quaternion, desired_quaternion))
    if dot < 0.0:
        desired_quaternion = -desired_quaternion
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    angle = 2.0 * math.acos(dot)
    if angle <= maximum_orientation_step:
        quaternion = desired_quaternion
    else:
        fraction = maximum_orientation_step / angle
        sine = math.sin(angle * 0.5)
        if abs(sine) <= 1e-8:
            quaternion = current_quaternion.lerp(desired_quaternion, fraction)
        else:
            quaternion = (
                current_quaternion
                * (math.sin((1.0 - fraction) * angle * 0.5) / sine)
                + desired_quaternion * (math.sin(fraction * angle * 0.5) / sine)
            )
    return GeometryPose(position=current.position + delta, quaternion=quaternion)


class ObjectGeometryInsertionTeacher:
    """Stateful task-space teacher driven only by observed geometry and events."""

    def __init__(
        self,
        calibration: InsertionTeacherCalibration,
        settings: InsertionTeacherSettings,
    ) -> None:
        self.calibration = calibration
        self.settings = settings
        self.reset()

    def reset(self) -> None:
        """Reset teacher history without changing the caller-owned simulator."""

        self._phase = InsertionTeacherPhase.OPEN
        self._best_reward = 0.0
        self._runtime_socket_to_left_eef: GeometryPose | None = None
        self._runtime_peg_to_right_eef: GeometryPose | None = None
        self._lift_socket: GeometryPose | None = None
        self._lift_peg: GeometryPose | None = None
        self._lift_feedback_socket_z: float | None = None
        self._lift_feedback_peg_z: float | None = None
        self._coarse_socket: GeometryPose | None = None
        self._coarse_peg: GeometryPose | None = None
        self._coarse_feedback_socket: GeometryPose | None = None
        self._coarse_feedback_peg: GeometryPose | None = None
        self._terminal_socket: GeometryPose | None = None
        self._terminal_peg: GeometryPose | None = None

    @property
    def phase(self) -> InsertionTeacherPhase:
        """Current monotonic geometry/event phase."""

        return self._phase

    @property
    def best_observed_reward(self) -> float:
        """Highest simulator reward observed by the teacher."""

        return self._best_reward

    def _validate_geometry(self, geometry: InsertionGeometry) -> None:
        settings = self.settings
        for name, pose in (("socket", geometry.socket), ("peg", geometry.peg)):
            x, y, z = (float(value) for value in pose.position)
            inside = (
                settings.workspace_x_min_m <= x <= settings.workspace_x_max_m
                and settings.workspace_y_min_m <= y <= settings.workspace_y_max_m
                and settings.workspace_z_min_m <= z <= settings.workspace_z_max_m
            )
            if not inside:
                raise GeometryTeacherError(
                    f"Observed {name} pose is outside the registered workspace."
                )
        if geometry.unexpected_collision_count:
            raise GeometryTeacherError(
                "Unexpected collision prevents geometry-teacher supervision."
            )

    def _grasp_targets(self, geometry: InsertionGeometry) -> tuple[GeometryPose, GeometryPose]:
        return (
            compose_pose(
                geometry.socket,
                self.calibration.socket_to_left_eef_at_grasp,
            ),
            compose_pose(
                geometry.peg,
                self.calibration.peg_to_right_eef_at_grasp,
            ),
        )

    def _captured_grasp_targets(
        self,
        socket: GeometryPose,
        peg: GeometryPose,
    ) -> tuple[GeometryPose, GeometryPose]:
        if (
            self._runtime_socket_to_left_eef is None
            or self._runtime_peg_to_right_eef is None
        ):
            raise GeometryTeacherError("Geometry teacher has no observed grasp transform.")
        return (
            compose_pose(socket, self._runtime_socket_to_left_eef),
            compose_pose(peg, self._runtime_peg_to_right_eef),
        )

    def _capture_grasp(self, geometry: InsertionGeometry) -> None:
        self._runtime_socket_to_left_eef = relative_pose(
            geometry.socket,
            geometry.left_eef,
        )
        self._runtime_peg_to_right_eef = relative_pose(
            geometry.peg,
            geometry.right_eef,
        )
        self._lift_feedback_socket_z = float(geometry.socket.position[2])
        self._lift_feedback_peg_z = float(geometry.peg.position[2])
        self._lift_socket = GeometryPose(
            position=torch.tensor(
                (
                    float(geometry.socket.position[0]),
                    float(geometry.socket.position[1]),
                    self.settings.lift_object_height_m,
                )
            ),
            quaternion=geometry.socket.quaternion,
        )
        self._lift_peg = GeometryPose(
            position=torch.tensor(
                (
                    float(geometry.peg.position[0]),
                    float(geometry.peg.position[1]),
                    self.settings.lift_object_height_m,
                )
            ),
            quaternion=geometry.peg.quaternion,
        )

    def _capture_alignment_targets(self, geometry: InsertionGeometry) -> None:
        desired_socket_quaternion = geometry.socket.quaternion
        terminal_relative = self.calibration.terminal_socket_to_peg
        terminal_delta = quaternion_rotate(
            desired_socket_quaternion,
            terminal_relative.position,
        )
        approach_delta = quaternion_rotate(
            desired_socket_quaternion,
            self.calibration.insertion_axis_in_socket
            * self.settings.preinsert_clearance_m,
        )
        terminal_socket = GeometryPose(
            position=geometry.socket.position.clone(),
            quaternion=desired_socket_quaternion,
        )
        terminal_peg = compose_pose(terminal_socket, terminal_relative)
        coarse_delta = terminal_delta + approach_delta
        coarse_socket_position = terminal_socket.position - approach_delta * 0.5
        coarse_peg_position = terminal_socket.position + terminal_delta + approach_delta * 0.5
        coarse_socket_position[2] = self.settings.lift_object_height_m
        coarse_peg_position[2] = self.settings.lift_object_height_m
        self._coarse_socket = GeometryPose(
            position=coarse_socket_position,
            quaternion=desired_socket_quaternion,
        )
        self._coarse_peg = GeometryPose(
            position=coarse_peg_position,
            quaternion=terminal_peg.quaternion,
        )
        self._coarse_feedback_socket = GeometryPose(
            position=geometry.socket.position.clone(),
            quaternion=geometry.socket.quaternion,
        )
        self._coarse_feedback_peg = GeometryPose(
            position=geometry.peg.position.clone(),
            quaternion=geometry.peg.quaternion,
        )
        self._terminal_socket = terminal_socket
        self._terminal_peg = terminal_peg

    def _feedback_lift_targets(
        self,
        geometry: InsertionGeometry,
    ) -> tuple[GeometryPose, GeometryPose]:
        if self._lift_feedback_socket_z is None or self._lift_feedback_peg_z is None:
            raise GeometryTeacherError("Geometry teacher has no persistent lift target.")
        step_m = self.settings.lift_feedback_step_m
        lead_m = self.settings.maximum_cartesian_step_m
        next_socket_z = min(
            self.settings.lift_object_height_m,
            self._lift_feedback_socket_z + step_m,
        )
        next_peg_z = min(
            self.settings.lift_object_height_m,
            self._lift_feedback_peg_z + step_m,
        )
        socket_position = geometry.socket.position.clone()
        peg_position = geometry.peg.position.clone()
        used_socket_z = max(
            float(socket_position[2]),
            min(next_socket_z, float(socket_position[2]) + lead_m),
        )
        used_peg_z = max(
            float(peg_position[2]),
            min(next_peg_z, float(peg_position[2]) + lead_m),
        )
        socket_position[2] = used_socket_z
        peg_position[2] = used_peg_z
        socket_target = GeometryPose(
            position=socket_position,
            quaternion=geometry.socket.quaternion,
        )
        peg_target = GeometryPose(
            position=peg_position,
            quaternion=geometry.peg.quaternion,
        )
        self._lift_feedback_socket_z = next_socket_z
        self._lift_feedback_peg_z = next_peg_z
        return (
            compose_pose(
                socket_target,
                relative_pose(geometry.socket, geometry.left_eef),
            ),
            compose_pose(
                peg_target,
                relative_pose(geometry.peg, geometry.right_eef),
            ),
        )

    def _validate_grasp_drift(self, geometry: InsertionGeometry) -> None:
        if (
            self._runtime_socket_to_left_eef is None
            or self._runtime_peg_to_right_eef is None
        ):
            return
        current_left = relative_pose(geometry.socket, geometry.left_eef)
        current_right = relative_pose(geometry.peg, geometry.right_eef)
        drift = max(
            float(
                torch.linalg.vector_norm(
                    current_left.position - self._runtime_socket_to_left_eef.position
                )
            ),
            float(
                torch.linalg.vector_norm(
                    current_right.position - self._runtime_peg_to_right_eef.position
                )
            ),
        )
        if drift > self.settings.maximum_grasp_drift_m:
            raise GeometryTeacherError(
                "Observed object-to-end-effector transform exceeded the grasp-drift limit."
            )

    def decide(self, geometry: InsertionGeometry) -> InsertionTaskSpaceTarget:
        """Return a bounded target without consuming a time index or timestamp."""

        self._validate_geometry(geometry)
        self._best_reward = max(self._best_reward, float(geometry.observed_reward))
        initial_phase = self._phase
        if self._best_reward >= 4.0 or geometry.pin_contact:
            self._phase = InsertionTeacherPhase.COMPLETE

        desired_left = geometry.left_eef
        desired_right = geometry.right_eef
        gripper = self.settings.gripper_closed_target
        lift_feedback_anchor = False

        for _ in range(len(InsertionTeacherPhase)):
            if self._phase is InsertionTeacherPhase.COMPLETE:
                break

            grasp_left, grasp_right = self._grasp_targets(geometry)
            if self._phase is InsertionTeacherPhase.OPEN:
                gripper = self.settings.gripper_open_target
                if (
                    float(geometry.robot_state[6]) >= self.settings.gripper_open_threshold
                    and float(geometry.robot_state[13])
                    >= self.settings.gripper_open_threshold
                ):
                    self._phase = InsertionTeacherPhase.APPROACH
                    continue
                break

            if self._phase is InsertionTeacherPhase.APPROACH:
                gripper = self.settings.gripper_open_target
                approach_left = _with_world_z_offset(
                    grasp_left,
                    self.settings.approach_clearance_m,
                )
                approach_right = _with_world_z_offset(
                    grasp_right,
                    self.settings.approach_clearance_m,
                )
                desired_left = GeometryPose(
                    position=approach_left.position,
                    quaternion=geometry.left_eef.quaternion,
                )
                desired_right = GeometryPose(
                    position=approach_right.position,
                    quaternion=geometry.right_eef.quaternion,
                )
                if max(
                    _position_distance(geometry.left_eef, desired_left),
                    _position_distance(geometry.right_eef, desired_right),
                ) <= self.settings.approach_tolerance_m:
                    self._phase = InsertionTeacherPhase.ORIENT
                    continue
                break

            if self._phase is InsertionTeacherPhase.ORIENT:
                gripper = self.settings.gripper_open_target
                desired_left = _with_world_z_offset(
                    grasp_left,
                    self.settings.approach_clearance_m,
                )
                desired_right = _with_world_z_offset(
                    grasp_right,
                    self.settings.approach_clearance_m,
                )
                position_aligned = max(
                    _position_distance(geometry.left_eef, desired_left),
                    _position_distance(geometry.right_eef, desired_right),
                ) <= self.settings.approach_tolerance_m
                orientation_aligned = max(
                    _orientation_distance(geometry.left_eef, desired_left),
                    _orientation_distance(geometry.right_eef, desired_right),
                ) <= self.settings.approach_orientation_tolerance_rad
                if position_aligned and orientation_aligned:
                    self._phase = InsertionTeacherPhase.DESCEND
                    continue
                break

            if self._phase is InsertionTeacherPhase.DESCEND:
                gripper = self.settings.gripper_open_target
                desired_left, desired_right = grasp_left, grasp_right
                if max(
                    _position_distance(geometry.left_eef, desired_left),
                    _position_distance(geometry.right_eef, desired_right),
                ) <= self.settings.grasp_tolerance_m:
                    self._phase = InsertionTeacherPhase.GRASP
                    continue
                break

            if self._phase is InsertionTeacherPhase.GRASP:
                gripper = self.settings.gripper_closed_target
                desired_left, desired_right = grasp_left, grasp_right
                if (
                    self._best_reward >= 1.0
                    or geometry.socket_grasp_contact
                    and geometry.peg_grasp_contact
                ):
                    self._capture_grasp(geometry)
                    self._phase = InsertionTeacherPhase.LIFT
                    continue
                break

            self._validate_grasp_drift(geometry)
            if self._phase is InsertionTeacherPhase.LIFT:
                desired_left, desired_right = self._feedback_lift_targets(geometry)
                lift_feedback_anchor = True
                lifted = (
                    self._best_reward >= 2.0
                    and not geometry.socket_on_table
                    and not geometry.peg_on_table
                    and float(geometry.socket.position[2])
                    >= self.settings.lift_object_height_m - self.settings.lift_tolerance_m
                    and float(geometry.peg.position[2])
                    >= self.settings.lift_object_height_m - self.settings.lift_tolerance_m
                )
                if lifted:
                    self._capture_alignment_targets(geometry)
                    self._phase = InsertionTeacherPhase.COARSE_ALIGN
                    continue
                break

            if self._phase is InsertionTeacherPhase.COARSE_ALIGN:
                if (
                    self._coarse_socket is None
                    or self._coarse_peg is None
                    or self._coarse_feedback_socket is None
                    or self._coarse_feedback_peg is None
                ):
                    raise GeometryTeacherError("Geometry teacher has no coarse alignment target.")
                coarse_step_m = 0.002
                socket_remaining = (
                    self._coarse_socket.position
                    - self._coarse_feedback_socket.position
                )
                socket_remaining_distance = float(
                    torch.linalg.vector_norm(socket_remaining)
                )
                next_socket_position = self._coarse_feedback_socket.position.clone()
                if socket_remaining_distance > coarse_step_m:
                    next_socket_position += socket_remaining * (
                        coarse_step_m / socket_remaining_distance
                    )
                else:
                    next_socket_position += socket_remaining
                peg_remaining = (
                    self._coarse_peg.position
                    - self._coarse_feedback_peg.position
                )
                peg_remaining_distance = float(
                    torch.linalg.vector_norm(peg_remaining)
                )
                next_peg_position = self._coarse_feedback_peg.position.clone()
                if peg_remaining_distance > coarse_step_m:
                    next_peg_position += peg_remaining * (
                        coarse_step_m / peg_remaining_distance
                    )
                else:
                    next_peg_position += peg_remaining
                lead_m = 0.006

                def _bounded_lead(
                    current_position: Tensor,
                    requested_position: Tensor,
                ) -> Tensor:
                    lead_delta = requested_position - current_position
                    lead_distance = float(torch.linalg.vector_norm(lead_delta))
                    if lead_distance <= lead_m:
                        return requested_position.clone()
                    return current_position + lead_delta * (lead_m / lead_distance)

                used_socket_position = _bounded_lead(
                    geometry.socket.position,
                    next_socket_position,
                )
                used_peg_position = _bounded_lead(
                    geometry.peg.position,
                    next_peg_position,
                )
                socket_target = GeometryPose(
                    position=used_socket_position,
                    quaternion=geometry.socket.quaternion,
                )
                peg_target = GeometryPose(
                    position=used_peg_position,
                    quaternion=geometry.peg.quaternion,
                )
                desired_left = compose_pose(
                    socket_target,
                    relative_pose(geometry.socket, geometry.left_eef),
                )
                desired_right = compose_pose(
                    peg_target,
                    relative_pose(geometry.peg, geometry.right_eef),
                )
                self._coarse_feedback_socket = GeometryPose(
                    position=next_socket_position,
                    quaternion=geometry.socket.quaternion,
                )
                self._coarse_feedback_peg = GeometryPose(
                    position=next_peg_position,
                    quaternion=geometry.peg.quaternion,
                )
                aligned = max(
                    _position_distance(geometry.socket, self._coarse_socket),
                    _position_distance(geometry.peg, self._coarse_peg),
                ) <= self.settings.coarse_alignment_tolerance_m
                if aligned:
                    self._phase = InsertionTeacherPhase.INSERT
                    continue
                break

            if self._phase is InsertionTeacherPhase.INSERT:
                if self._terminal_socket is None or self._terminal_peg is None:
                    raise GeometryTeacherError("Geometry teacher has no terminal insertion target.")
                desired_left, desired_right = self._captured_grasp_targets(
                    self._terminal_socket,
                    self._terminal_peg,
                )
                break

        bounded_left = _bounded_target(
            geometry.left_eef,
            desired_left,
            self.settings.maximum_cartesian_step_m,
            self.settings.maximum_orientation_step_rad,
        )
        bounded_right = _bounded_target(
            geometry.right_eef,
            desired_right,
            self.settings.maximum_cartesian_step_m,
            self.settings.maximum_orientation_step_rad,
        )
        error = max(
            _position_distance(geometry.left_eef, desired_left),
            _position_distance(geometry.right_eef, desired_right),
        )
        return InsertionTaskSpaceTarget(
            left_eef=bounded_left,
            right_eef=bounded_right,
            left_gripper=gripper,
            right_gripper=gripper,
            phase=self._phase,
            phase_changed=self._phase is not initial_phase,
            maximum_position_error_m=error,
            best_observed_reward=self._best_reward,
            lift_feedback_anchor=lift_feedback_anchor,
        )
