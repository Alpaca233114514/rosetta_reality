"""State-conditioned ALOHA insertion scripted teacher (upstream-faithful).

First real teacher candidate of the frozen staged gate
``m2-t2-teacher-gate-001``.  The task policy is adapted from the upstream ALOHA
ACT repository — ``tonyzhaozh/act`` ``scripted_policy.py`` ``InsertionPolicy``,
the generator of the ``aloha_sim_insertion_scripted`` demonstration data — and
bound to the frozen ``InsertionGeometry`` contract:

- The upstream policy plans one time-indexed, open-loop end-effector waypoint
  trajectory from the first observation; ``record_sim_episodes.py`` then
  replays the resulting joint trajectory as the dataset actions.  This adapter
  converts every time-indexed waypoint into a state-conditioned stage whose
  advancement is a pure function of the observed geometry, gripper state and
  contact events.  No stage reads wall-clock time, a step index, an episode or
  a seed.
- Upstream end-effector targets are converted to absolute standard-space
  joint-position actions (Action Contract: 14-D, 50 Hz, normalized grippers,
  ``0_closed_1_open``) with the registered :class:`MinkAlohaIkSolver` and the
  evaluator-proven numerical settings, followed by a registered per-step
  arm-joint target clamp that keeps commanded motion near upstream smoothness.
- The registered grasp orientations are world-frame tilts: the upstream mocap
  bodies carry no quat attribute (identity), so the scripted policy tilts each
  gripper from the world frame, not from the arm's home orientation.  The
  registered tilts are +60 degrees (left) and +120 degrees (right) about world
  Y; the right-arm alternatives (-60 upstream literal, +60 mirrored) stall the
  arm 5-8 cm short of the meet targets in the constrained QP (diagnostic tilt
  matrix), while +120 converges at every stage target, and the tilt leaves the
  finger separation axis (world Y) and the carried objects' world alignment
  unchanged.
- The upstream descend waypoint commands the grasp point 3 cm below the
  object center; the upstream weld-dragged environment tolerates the fingers
  sweeping past the object onto the table, while this gate refuses unexpected
  collisions.  The registered adaptation therefore commands the object center
  itself (``descend_depth_m = 0``) so the closing fingers squeeze-compliantly
  grasp beside the object; the stall exit remains as the contact safety net.
- The IK chase glides: each decide solves toward the stage target bounded to
  one registered Cartesian/orientation step of the current end-effector pose,
  mirroring the upstream policy's per-step linear waypoint interpolation.
  Solving the full stage target directly from a lagging state lets the
  orientation gradient dominate the QP and sacrifice position error
  (diagnostic evidence: position residuals growing to 0.26 m).
- The alignment uses a deterministic fixed-order wrist-branch
  multistart selected at the transport freeze (registered branch diagnostic
  ``m2-smolvla-t2-scripted-teacher-align-branch-diagnostic-2026-08-30``): the
  constrained-IK chase from the frozen state can sit in a wrist branch that
  stalls 0.03-0.17 m short of the parallel align target (tuning seeds
  1900/1902/1904) while the same site position converges with the wrist
  orientation flipped 180 degrees about world Z -- the finger separation axis
  stays on the socket side walls and the upright socket body target is
  unchanged, so the task geometry is identical and only the IK branch moves.
  The probe chases each candidate's align target from the frozen state and
  keeps the first convergence in the registered order parallel -> flipped;
  when neither converges the parallel candidate is kept and the gate rollout
  judges the outcome. The flipped wrist cannot reach the +0.10 m hover climb
  (registered probe: 0.03-0.12 m stall), so the flipped candidate glides
  directly from the freeze pose to its align target; the cage re-settles the
  socket under the yawed wrist, which is why the flipped candidate keeps the
  parallel-composed site position instead of recomposing the cage transform
  through the flipped frame (measured: rigid recomposition stalls). The
  probe reads no time, index, episode or seed input.
- Off-support states refuse explicitly (``unsafe_state`` for unexpected
  collisions, ``workspace_violation`` for object poses outside the registered
  workspace, ``out_of_support`` for a constrained-IK failure).  Corrupted
  geometry cannot reach this class: the frozen observation constructor itself
  rejects non-finite inputs.  Nothing degrades silently.

The workspace bounds are a candidate-owned sanity guard chosen to cover the
registered G0 probe suite and the task spawn region; they are not a gate
threshold and widen nothing in the frozen protocol.  Fidelity of the
time-to-state conversion is validated by the G1/G2 rollout evidence, never
assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rosetta_reality.sim.geometry_teacher import (
    GeometryPose,
    InsertionGeometry,
    compose_pose,
    quaternion_multiply,
    relative_pose,
)
from rosetta_reality.sim.mink_aloha_ik import (
    MinkAlohaIkSettings,
    MinkAlohaIkSolver,
)
from rosetta_reality.sim.teacher_gate import (
    TeacherDecision,
    TeacherGateError,
    TeacherRefusal,
    TeacherRefusalReason,
)

UPSTREAM_SOURCE = {
    "repository": "https://github.com/tonyzhaozh/act",
    "commit": "742c753c0d4a5d87076c8f69e5628c79a8cc5488",
    "path": "scripted_policy.py",
    "policy_class": "InsertionPolicy",
    "supporting_files": ("sim_env.py", "record_sim_episodes.py", "constants.py"),
    "license": "MIT",
    "verified_at": "2026-08-29",
}

LEFT_EEF_SITE = "cali_left_site1"
RIGHT_EEF_SITE = "cali_right_site1"


@dataclass(frozen=True, slots=True)
class ScriptedTeacherSettings:
    """Registered constants of the scripted adapter and its sanity guards."""

    # Gripper targets sit inside the finger joints' physical range: normalized
    # 1.0 unnormalizes to 0.058 rad, past the 0.057 finger joint limit, and
    # normalized 0.0 to 0.01844, under the 0.021 limit; riding the limits
    # registers as joint-limit violations in the gate rollouts.
    gripper_open_target: float = 0.90
    gripper_closed_target: float = 0.09
    gripper_open_threshold: float = 0.85
    approach_clearance_m: float = 0.08
    descend_depth_m: float = 0.0
    meet_center_x_m: float = 0.0
    meet_center_y_m: float = 0.5
    meet_center_z_m: float = 0.15
    transport_left_offset_m: tuple[float, float, float] = (-0.15, 0.0, 0.0)
    # Insertion geometry along the peg's own axis: TRANSPORT parks the peg
    # 0.16 m from the socket center (socket-relative, loose); ALIGN and INSERT
    # then move the SOCKET onto the parked peg -- the left arm owns the
    # precision -- with the socket center at -0.16 m (park) and -0.10 m
    # (insert: peg tip touching the pin) in the peg's frame, composed with the
    # captured socket-to-site grasp transform.
    peg_park_offset_m: float = 0.16
    peg_insert_offset_m: float = 0.09
    transport_tolerance_m: float = 0.04
    align_tolerance_m: float = 0.012
    socket_approach_z_m: float = 0.10
    socket_approach_tolerance_m: float = 0.02
    # Deterministic alignment multistart: the wrist-orientation flip of the
    # flipped candidate and the frozen-state constrained-IK probe thresholds
    # that select the first converging candidate in the registered
    # parallel -> flipped order. See the module docstring for the branch
    # evidence.
    alignment_flip_degrees: float = 180.0
    alignment_probe_maximum_iterations: int = 40
    alignment_probe_position_tolerance_m: float = 0.008
    alignment_probe_orientation_tolerance_rad: float = 0.05
    grasp_orientation_degrees_left: float = 60.0
    # 120 degrees on the right arm: the upstream literal -60 and the mirrored
    # +60 both stall the right arm 5-8 cm short of the transport/insert meet
    # targets (diagnostic tilt matrix), while +120 converges at the approach,
    # transport and insert targets.  The tilt does not change the finger
    # separation axis (world Y under any Y-tilt) and the carried objects stay
    # world-aligned inside the closing cage, so this branch choice costs no
    # task geometry.
    grasp_orientation_degrees_right: float = 120.0
    approach_position_tolerance_m: float = 0.02
    approach_orientation_tolerance_rad: float = 0.04
    grasp_position_tolerance_m: float = 0.008
    transport_position_tolerance_m: float = 0.012
    descend_stall_progress_m: float = 0.0015
    descend_stall_band_m: float = 0.02
    descend_floor_z_m: float = 0.008
    maximum_cartesian_step_m: float = 0.025
    maximum_orientation_step_rad: float = 0.12
    joint_target_gain: float = 0.7
    maximum_joint_target_delta_rad: float = 0.06
    workspace_x_min_m: float = -0.45
    workspace_x_max_m: float = 0.45
    workspace_y_min_m: float = 0.10
    workspace_y_max_m: float = 0.90
    workspace_z_min_m: float = 0.0
    workspace_z_max_m: float = 0.55

    def __post_init__(self) -> None:
        scalars = (
            "gripper_open_target",
            "gripper_closed_target",
            "gripper_open_threshold",
            "approach_clearance_m",
            "descend_depth_m",
            "meet_center_x_m",
            "meet_center_y_m",
            "meet_center_z_m",
            "grasp_orientation_degrees_left",
            "grasp_orientation_degrees_right",
            "approach_position_tolerance_m",
            "approach_orientation_tolerance_rad",
            "grasp_position_tolerance_m",
            "transport_position_tolerance_m",
            "transport_tolerance_m",
            "align_tolerance_m",
            "socket_approach_tolerance_m",
            "socket_approach_z_m",
            "socket_approach_tolerance_m",
            "alignment_flip_degrees",
            "alignment_probe_maximum_iterations",
            "alignment_probe_position_tolerance_m",
            "alignment_probe_orientation_tolerance_rad",
            "peg_park_offset_m",
            "peg_insert_offset_m",
            "descend_stall_progress_m",
            "descend_stall_band_m",
            "descend_floor_z_m",
            "maximum_cartesian_step_m",
            "maximum_orientation_step_rad",
            "joint_target_gain",
            "maximum_joint_target_delta_rad",
            "workspace_x_min_m",
            "workspace_x_max_m",
            "workspace_y_min_m",
            "workspace_y_max_m",
            "workspace_z_min_m",
            "workspace_z_max_m",
        )
        if any(not math.isfinite(getattr(self, name)) for name in scalars):
            raise ValueError("Scripted-teacher settings must be finite.")
        for name in ("transport_left_offset_m",):
            if any(not math.isfinite(float(value)) for value in getattr(self, name)):
                raise ValueError(f"Scripted-teacher setting {name} must be finite.")
        if not 0.0 <= self.gripper_closed_target < self.gripper_open_threshold:
            raise ValueError("Closed target must be below the gripper-open threshold.")
        if not self.gripper_open_threshold <= self.gripper_open_target <= 1.0:
            raise ValueError("Open target must reach the registered open threshold.")
        for name in (
            "approach_position_tolerance_m",
            "approach_orientation_tolerance_rad",
            "grasp_position_tolerance_m",
            "transport_position_tolerance_m",
            "transport_tolerance_m",
            "align_tolerance_m",
            "alignment_probe_maximum_iterations",
            "alignment_probe_position_tolerance_m",
            "alignment_probe_orientation_tolerance_rad",
            "maximum_cartesian_step_m",
            "maximum_orientation_step_rad",
            "maximum_joint_target_delta_rad",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 < self.joint_target_gain <= 1.0:
            raise ValueError("joint_target_gain must be in (0, 1].")
        if not (
            self.workspace_x_min_m < self.workspace_x_max_m
            and self.workspace_y_min_m < self.workspace_y_max_m
            and self.workspace_z_min_m < self.workspace_z_max_m
        ):
            raise ValueError("Scripted-teacher workspace bounds are invalid.")


def default_ik_settings() -> MinkAlohaIkSettings:
    """The evaluator-proven constrained-IK numerical contract."""

    return MinkAlohaIkSettings(
        integration_timestep_s=0.005,
        maximum_iterations=15,
        position_cost=1.0,
        orientation_cost=1.0,
        posture_cost=1e-4,
        frame_lm_damping=1.0,
        solver_damping=1e-5,
        maximum_joint_velocity_rad_s=math.pi,
        configuration_limit_gain=0.95,
        joint_limit_margin_rad=0.04540462255477905,
    )


class ScriptedInsertionStage(str, Enum):
    """Monotonic state-conditioned stages mirroring the upstream waypoints."""

    OPEN = "open"
    APPROACH = "approach"
    ORIENT = "orient"
    DESCEND = "descend"
    GRASP = "grasp"
    TRANSPORT = "transport"
    SOCKET_APPROACH = "socket_approach"
    ALIGN = "align"
    INSERT = "insert"
    COMPLETE = "complete"


def _axis_angle_quaternion(axis: tuple[float, float, float], degrees: float) -> Tensor:
    axis_tensor = torch.as_tensor(axis, dtype=torch.float64)
    norm = float(torch.linalg.vector_norm(axis_tensor))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Rotation axis must have non-zero finite norm.")
    axis_tensor = axis_tensor / norm
    half_radians = math.radians(degrees) / 2.0
    return torch.tensor(
        [
            math.cos(half_radians),
            *(float(value) * math.sin(half_radians) for value in axis_tensor),
        ],
        dtype=torch.float32,
    )


def _expanded_robot_qpos(logical: Tensor) -> np.ndarray:
    """Expand the 14-D logical state into the 16-D native ALOHA qpos."""

    from gym_aloha.tasks.sim import unnormalize_puppet_gripper_position

    value = logical.detach().to(torch.float64).cpu().numpy()
    left_gripper = float(unnormalize_puppet_gripper_position(float(value[6])))
    right_gripper = float(unnormalize_puppet_gripper_position(float(value[13])))
    return np.concatenate(
        (
            value[:6],
            np.asarray([left_gripper, -left_gripper]),
            value[7:13],
            np.asarray([right_gripper, -right_gripper]),
        )
    )


@dataclass(frozen=True, slots=True)
class ScriptedTeacherCapture:
    """Home and object references captured from observations, never from time.

    The object positions are captured when the OPEN stage exits; they are
    ``None`` until the teacher has observed the grippers open.
    """

    home_left: GeometryPose
    home_right: GeometryPose
    grasp_left_quaternion: Tensor
    grasp_right_quaternion: Tensor
    socket_position: Tensor | None
    peg_position: Tensor | None


class ScriptedInsertionTeacher:
    """Deterministic state-conditioned adapter of the upstream insertion policy."""

    identity = "gym-aloha-scripted-insertion-adapter-001"

    def __init__(
        self,
        *,
        settings: ScriptedTeacherSettings | None = None,
        ik_settings: MinkAlohaIkSettings | None = None,
        solver: Any = None,
    ) -> None:
        self.settings = settings if settings is not None else ScriptedTeacherSettings()
        # The grasp orientations are world-frame constants: the upstream mocap
        # bodies carry no quat attribute, so the scripted policy tilts each
        # gripper from identity.  See the settings comments for the branch
        # evidence behind the per-arm tilt values.
        self._grasp_left_quaternion = _axis_angle_quaternion(
            (0.0, 1.0, 0.0), self.settings.grasp_orientation_degrees_left
        )
        self._grasp_right_quaternion = _axis_angle_quaternion(
            (0.0, 1.0, 0.0), self.settings.grasp_orientation_degrees_right
        )
        if solver is None:
            solver = MinkAlohaIkSolver.from_gym_aloha(
                left_site=LEFT_EEF_SITE,
                right_site=RIGHT_EEF_SITE,
                settings=ik_settings if ik_settings is not None else default_ik_settings(),
            )
        self._solver = solver
        self.reset()

    def reset(self) -> None:
        """Clear teacher-internal captures; the simulator stays caller-owned."""

        self._phase = ScriptedInsertionStage.OPEN
        self._home_left: GeometryPose | None = None
        self._home_right: GeometryPose | None = None
        self._orient_left: GeometryPose | None = None
        self._orient_right: GeometryPose | None = None
        self._socket_position: Tensor | None = None
        self._peg_position: Tensor | None = None
        self._peg_from_right_site: GeometryPose | None = None
        self._socket_from_left_site: GeometryPose | None = None
        self._hold_right_site: GeometryPose | None = None
        self._peg_hold_pose: GeometryPose | None = None
        self._alignment_flipped: bool = False
        self._previous_left_z: float | None = None
        self._previous_right_z: float | None = None

    @property
    def phase(self) -> ScriptedInsertionStage:
        """Current monotonic stage (observability only; not a teacher input)."""

        return self._phase

    @property
    def capture(self) -> ScriptedTeacherCapture:
        """The captured references behind the current decisions."""

        if self._home_left is None or self._home_right is None:
            raise TeacherGateError("Scripted teacher has captured no home pose yet.")
        assert self._grasp_left_quaternion is not None
        assert self._grasp_right_quaternion is not None
        return ScriptedTeacherCapture(
            home_left=self._home_left,
            home_right=self._home_right,
            grasp_left_quaternion=self._grasp_left_quaternion,
            grasp_right_quaternion=self._grasp_right_quaternion,
            socket_position=self._socket_position,
            peg_position=self._peg_position,
        )

    def decide(self, observation: InsertionGeometry) -> TeacherDecision:
        refusal = self._validate_support(observation)
        if refusal is not None:
            return refusal
        if self._home_left is None:
            self._capture_home(observation)
        left_target, right_target, gripper = self._advance(observation)
        decision = self._task_target_action(
            observation, left_target, right_target, gripper
        )
        self._previous_left_z = float(observation.left_eef.position[2])
        self._previous_right_z = float(observation.right_eef.position[2])
        return decision

    # ------------------------------------------------------------ support

    def _validate_support(self, observation: InsertionGeometry) -> TeacherDecision | None:
        if observation.unexpected_collision_count:
            return TeacherDecision(
                refusal=TeacherRefusal(
                    reason=TeacherRefusalReason.UNSAFE_STATE,
                    detail=(
                        "unexpected collision count "
                        f"{observation.unexpected_collision_count} prevents scripted "
                        "supervision"
                    ),
                )
            )
        settings = self.settings
        for name, pose in (("socket", observation.socket), ("peg", observation.peg)):
            x, y, z = (float(value) for value in pose.position)
            inside = (
                settings.workspace_x_min_m <= x <= settings.workspace_x_max_m
                and settings.workspace_y_min_m <= y <= settings.workspace_y_max_m
                and settings.workspace_z_min_m <= z <= settings.workspace_z_max_m
            )
            if not inside:
                return TeacherDecision(
                    refusal=TeacherRefusal(
                        reason=TeacherRefusalReason.WORKSPACE_VIOLATION,
                        detail=(
                            f"observed {name} pose ({x:.4f}, {y:.4f}, {z:.4f}) is "
                            "outside the scripted-teacher workspace"
                        ),
                    )
                )
        return None

    def _capture_home(self, observation: InsertionGeometry) -> None:
        self._home_left = observation.left_eef
        self._home_right = observation.right_eef

    def _capture_objects(self, observation: InsertionGeometry) -> None:
        self._socket_position = observation.socket.position.clone()
        self._peg_position = observation.peg.position.clone()

    def _capture_orient(self) -> None:
        """Freeze the approach hold points the ORIENT stage rotates in place at."""

        approach_left, approach_right = self._approach_targets()
        self._orient_left = approach_left
        self._orient_right = approach_right

    # ------------------------------------------------------------- stages

    def _grippers_open(self, observation: InsertionGeometry) -> bool:
        threshold = self.settings.gripper_open_threshold
        return (
            float(observation.robot_state[6]) >= threshold
            and float(observation.robot_state[13]) >= threshold
        )

    def _meet_center(self) -> Tensor:
        return torch.tensor(
            (
                self.settings.meet_center_x_m,
                self.settings.meet_center_y_m,
                self.settings.meet_center_z_m,
            ),
            dtype=torch.float32,
        )

    def _offset_pose(
        self,
        position: Tensor,
        offset: tuple[float, float, float],
        quaternion: Tensor,
    ) -> GeometryPose:
        return GeometryPose(
            position=position + torch.tensor(offset, dtype=torch.float32),
            quaternion=quaternion,
        )

    def _approach_targets(self) -> tuple[GeometryPose, GeometryPose]:
        assert self._socket_position is not None and self._peg_position is not None
        assert self._grasp_left_quaternion is not None
        assert self._grasp_right_quaternion is not None
        clearance = (0.0, 0.0, self.settings.approach_clearance_m)
        return (
            self._offset_pose(self._socket_position, clearance, self._grasp_left_quaternion),
            self._offset_pose(self._peg_position, clearance, self._grasp_right_quaternion),
        )

    def _descend_targets(self) -> tuple[GeometryPose, GeometryPose]:
        assert self._socket_position is not None and self._peg_position is not None
        assert self._grasp_left_quaternion is not None
        assert self._grasp_right_quaternion is not None
        depth = (0.0, 0.0, -self.settings.descend_depth_m)
        return (
            self._offset_pose(self._socket_position, depth, self._grasp_left_quaternion),
            self._offset_pose(self._peg_position, depth, self._grasp_right_quaternion),
        )

    def _meet_left_target(self) -> GeometryPose:
        assert self._grasp_left_quaternion is not None
        meet = self._meet_center()
        return self._offset_pose(
            meet,
            self.settings.transport_left_offset_m,
            self._grasp_left_quaternion,
        )

    def _peg_park_body(self, observation: InsertionGeometry) -> GeometryPose:
        """Peg body pose parked 0.16 m from the socket center along its axis."""

        return compose_pose(
            observation.socket,
            GeometryPose(
                position=torch.tensor(
                    [self.settings.peg_park_offset_m, 0.0, 0.0], dtype=torch.float32
                ),
                quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            ),
        )

    def _flip_pose(self) -> GeometryPose:
        """Wrist-orientation flip of the flipped alignment candidate."""

        return GeometryPose(
            position=torch.zeros(3, dtype=torch.float32),
            quaternion=_axis_angle_quaternion(
                (0.0, 0.0, 1.0), self.settings.alignment_flip_degrees
            ),
        )

    def _peg_local_frame(self, offset_m: float, height_m: float = 0.0) -> GeometryPose:
        assert self._peg_hold_pose is not None
        return compose_pose(
            self._peg_hold_pose,
            GeometryPose(
                position=torch.tensor(
                    [-offset_m, 0.0, height_m], dtype=torch.float32
                ),
                quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            ),
        )

    def _socket_body_target(self, observation: InsertionGeometry, offset_m: float) -> GeometryPose:
        """Socket body pose at ``offset_m`` along the observed peg's own axis.

        The offset composes from the frozen peg pose, so the channel axis aligns
        with the peg exactly as it sits in its compliant cage; no heuristic
        aim is needed and the channel guides the insertion. The flip flag only
        rotates the frame about world Z; the position -- the only part the
        stage conditions read -- is identical for both candidates."""

        frame = self._peg_local_frame(offset_m)
        if self._alignment_flipped:
            frame = compose_pose(frame, self._flip_pose())
        return frame

    def _alignment_probe_converges(
        self,
        observation: InsertionGeometry,
        left: GeometryPose,
        right: GeometryPose,
    ) -> bool:
        """Chase one alignment candidate from the frozen state to convergence."""

        settings = self.settings
        current = np.asarray(_expanded_robot_qpos(observation.robot_state), dtype=np.float64)
        result = None
        for _ in range(settings.alignment_probe_maximum_iterations):
            try:
                result = self._solver.solve(
                    current,
                    left_position=left.position.numpy(),
                    left_quaternion_wxyz=left.quaternion.numpy(),
                    right_position=right.position.numpy(),
                    right_quaternion_wxyz=right.quaternion.numpy(),
                )
            except (RuntimeError, ValueError):
                return False
            moved = float(np.abs(np.asarray(result.qpos[:16]) - current).max())
            current = np.asarray(result.qpos[:16], dtype=np.float64)
            if (
                result.position_errors_m[0]
                <= settings.alignment_probe_position_tolerance_m
                and result.orientation_errors_rad[0]
                <= settings.alignment_probe_orientation_tolerance_rad
            ) or moved <= 1e-9:
                break
        assert result is not None
        return bool(
            result.position_errors_m[0] <= settings.alignment_probe_position_tolerance_m
            and result.orientation_errors_rad[0]
            <= settings.alignment_probe_orientation_tolerance_rad
        )

    def _select_alignment_candidate(self, observation: InsertionGeometry) -> bool:
        """Fixed-order alignment multistart, selected from the frozen state.

        Returns whether the flipped wrist candidate was registered for the
        episode. The parallel candidate is tried first so a passing trajectory
        is unchanged; when neither candidate converges the parallel candidate
        is kept and the gate rollout judges the outcome.
        """

        assert self._hold_right_site is not None
        for flipped in (False, True):
            self._alignment_flipped = flipped
            left, right = self._align_targets(observation)
            if self._alignment_probe_converges(observation, left, right):
                return flipped
        self._alignment_flipped = False
        return False

    def _transport_targets(
        self, observation: InsertionGeometry
    ) -> tuple[GeometryPose, GeometryPose]:
        assert self._peg_from_right_site is not None
        peg_park = self._peg_park_body(observation)
        return (
            self._meet_left_target(),
            compose_pose(peg_park, self._peg_from_right_site),
        )

    def _socket_body_approach(
        self, observation: InsertionGeometry, offset_m: float
    ) -> GeometryPose:
        """Socket approach pose hovering above the insertion axis.

        Only the parallel candidate enters SOCKET_APPROACH: the flipped wrist
        branch cannot reach the hover climb (registered probe evidence) and
        bypasses the stage, so this composition stays the registered
        peg-local offset.
        """

        return self._peg_local_frame(offset_m, self.settings.socket_approach_z_m)

    def _socket_approach_targets(
        self, observation: InsertionGeometry
    ) -> tuple[GeometryPose, GeometryPose]:
        assert self._socket_from_left_site is not None
        assert self._hold_right_site is not None
        socket_hover = self._socket_body_approach(
            observation, self.settings.peg_park_offset_m
        )
        return (
            compose_pose(socket_hover, self._socket_from_left_site),
            self._hold_right_site,
        )

    def _site_target(self, observation: InsertionGeometry, offset_m: float) -> GeometryPose:
        """Left-site target for the socket body at ``offset_m`` on the peg axis.

        Parallel candidate: the rigid composition through the frozen cage
        transform. Flipped candidate: the measured convergent branch keeps the
        parallel-composed site position (the compliant cage re-settles the
        socket under the yawed wrist, so recomposing the cage transform
        through the flipped frame is the wrong target) and flips the site
        orientation 180 degrees about world Z; the finger separation axis
        stays on the socket side walls and the upright socket body target is
        unchanged.
        """

        assert self._socket_from_left_site is not None
        assert self._peg_hold_pose is not None
        body = self._socket_body_target(observation, offset_m)
        if not self._alignment_flipped:
            return compose_pose(body, self._socket_from_left_site)
        parallel_body = GeometryPose(
            position=body.position, quaternion=self._peg_hold_pose.quaternion
        )
        parallel_site = compose_pose(parallel_body, self._socket_from_left_site)
        return GeometryPose(
            position=parallel_site.position,
            quaternion=quaternion_multiply(
                self._flip_pose().quaternion, parallel_site.quaternion
            ),
        )

    def _align_targets(
        self, observation: InsertionGeometry
    ) -> tuple[GeometryPose, GeometryPose]:
        assert self._socket_from_left_site is not None
        assert self._hold_right_site is not None
        return (
            self._site_target(observation, self.settings.peg_park_offset_m),
            self._hold_right_site,
        )

    def _insert_targets(
        self, observation: InsertionGeometry
    ) -> tuple[GeometryPose, GeometryPose]:
        assert self._socket_from_left_site is not None
        assert self._hold_right_site is not None
        return (
            self._site_target(observation, self.settings.peg_insert_offset_m),
            self._hold_right_site,
        )

    @staticmethod
    def _position_distance(first: GeometryPose, second: GeometryPose) -> float:
        return float(torch.linalg.vector_norm(first.position - second.position))

    @staticmethod
    def _orientation_distance(first: GeometryPose, second: GeometryPose) -> float:
        dot = abs(float(torch.dot(first.quaternion, second.quaternion)))
        return 2.0 * math.acos(min(1.0, max(-1.0, dot)))

    def _reached(
        self,
        observation: InsertionGeometry,
        left: GeometryPose,
        right: GeometryPose,
        position_tolerance_m: float,
        orientation_tolerance_rad: float | None,
    ) -> bool:
        position_reached = (
            self._position_distance(observation.left_eef, left) <= position_tolerance_m
            and self._position_distance(observation.right_eef, right) <= position_tolerance_m
        )
        if not position_reached or orientation_tolerance_rad is None:
            return position_reached
        return (
            self._orientation_distance(observation.left_eef, left) <= orientation_tolerance_rad
            and self._orientation_distance(observation.right_eef, right)
            <= orientation_tolerance_rad
        )

    def _descent_stalled(
        self,
        observation: InsertionGeometry,
        left: GeometryPose,
        right: GeometryPose,
    ) -> bool:
        """Both arms stopped descending while already below the approach band.

        The upstream descend waypoint commands the grasp point below the table
        surface; contact stops the real descent above the commanded setpoint,
        so the state-conditioned exit is "reached the setpoint or stopped
        making vertical progress while low".
        """

        settings = self.settings
        if self._previous_left_z is None or self._previous_right_z is None:
            return False
        approach_left, approach_right = self._approach_targets()
        left_z = float(observation.left_eef.position[2])
        right_z = float(observation.right_eef.position[2])
        left_below = (
            left_z
            <= float(approach_left.position[2]) - settings.descend_stall_band_m
        )
        right_below = (
            right_z
            <= float(approach_right.position[2]) - settings.descend_stall_band_m
        )
        left_stalled = (
            abs(left_z - self._previous_left_z) <= settings.descend_stall_progress_m
        )
        right_stalled = (
            abs(right_z - self._previous_right_z)
            <= settings.descend_stall_progress_m
        )
        return left_below and right_below and left_stalled and right_stalled

    def _advance(
        self, observation: InsertionGeometry
    ) -> tuple[GeometryPose, GeometryPose, float]:
        settings = self.settings
        for _ in range(len(ScriptedInsertionStage)):
            if self._phase is ScriptedInsertionStage.OPEN:
                if self._grippers_open(observation):
                    self._capture_objects(observation)
                    self._phase = ScriptedInsertionStage.APPROACH
                    continue
                assert self._home_left is not None and self._home_right is not None
                return (
                    self._home_left,
                    self._home_right,
                    settings.gripper_open_target,
                )
            if self._phase is ScriptedInsertionStage.APPROACH:
                left, right = self._approach_targets()
                # Position-only travel: the orientation target follows the live
                # pose so no orientation task force couples into the position
                # chase; ORIENT re-engages the orientation afterwards.
                held_left = GeometryPose(
                    position=left.position, quaternion=observation.left_eef.quaternion
                )
                held_right = GeometryPose(
                    position=right.position,
                    quaternion=observation.right_eef.quaternion,
                )
                if self._reached(
                    observation,
                    held_left,
                    held_right,
                    settings.approach_position_tolerance_m,
                    None,
                ):
                    self._capture_orient()
                    self._phase = ScriptedInsertionStage.ORIENT
                    continue
                return held_left, held_right, settings.gripper_open_target
            if self._phase is ScriptedInsertionStage.ORIENT:
                assert self._orient_left is not None and self._orient_right is not None
                # Rotate to the grasp tilt with the captured approach position
                # pinned: the entry position error is within the travel
                # tolerance, so the position task stays strong while the
                # orientation glides.
                if self._reached(
                    observation,
                    self._orient_left,
                    self._orient_right,
                    settings.approach_position_tolerance_m,
                    settings.approach_orientation_tolerance_rad,
                ):
                    self._phase = ScriptedInsertionStage.DESCEND
                    continue
                return (
                    self._orient_left,
                    self._orient_right,
                    settings.gripper_open_target,
                )
            if self._phase is ScriptedInsertionStage.DESCEND:
                left, right = self._descend_targets()
                reached = self._reached(
                    observation,
                    left,
                    right,
                    settings.grasp_position_tolerance_m,
                    settings.approach_orientation_tolerance_rad,
                )
                floor = (
                    float(observation.left_eef.position[2])
                    <= settings.descend_floor_z_m
                    or float(observation.right_eef.position[2])
                    <= settings.descend_floor_z_m
                )
                if reached or floor or self._descent_stalled(observation, left, right):
                    self._phase = ScriptedInsertionStage.GRASP
                    continue
                return left, right, settings.gripper_open_target
            if self._phase is ScriptedInsertionStage.GRASP:
                if observation.socket_grasp_contact and observation.peg_grasp_contact:
                    self._peg_from_right_site = relative_pose(
                        observation.peg, observation.right_eef
                    )
                    self._phase = ScriptedInsertionStage.TRANSPORT
                    continue
                left, right = self._descend_targets()
                return left, right, settings.gripper_closed_target
            if self._phase is ScriptedInsertionStage.TRANSPORT:
                left, right = self._transport_targets(observation)
                left_reached = (
                    self._position_distance(observation.left_eef, left)
                    <= settings.transport_position_tolerance_m
                )
                peg_park = self._peg_park_body(observation)
                peg_parked = (
                    self._position_distance(observation.peg, peg_park)
                    <= settings.transport_tolerance_m
                )
                if left_reached and peg_parked:
                    # Freeze the right arm's carry pose: the peg hold target
                    # must not chase the live socket pose while the socket
                    # approaches, or the two targets lift each other in a
                    # positive feedback loop.
                    self._hold_right_site = observation.right_eef
                    # Freeze the peg reference too: a target composed from the
                    # live peg pose creeps forever as the compliant cage
                    # settles under the changing force balance.  The socket's
                    # in-cage relative transform is re-captured here for the
                    # same reason: the grasp-time capture goes stale as the
                    # socket settles, and composing insert targets with it
                    # flips the site's tilt sign into an unreachable branch.
                    self._peg_hold_pose = observation.peg
                    self._socket_from_left_site = relative_pose(
                        observation.socket, observation.left_eef
                    )
                    self._alignment_flipped = self._select_alignment_candidate(
                        observation
                    )
                    if self._alignment_flipped:
                        # The flipped wrist branch cannot reach the +0.10 m
                        # hover climb (registered probe: 0.03-0.12 m stall);
                        # the socket glides directly from the freeze pose to
                        # its flipped align target.
                        self._phase = ScriptedInsertionStage.ALIGN
                    else:
                        self._phase = ScriptedInsertionStage.SOCKET_APPROACH
                    continue
                return left, right, settings.gripper_closed_target
            if self._phase is ScriptedInsertionStage.SOCKET_APPROACH:
                left, right = self._socket_approach_targets(observation)
                socket_hover = self._socket_body_approach(
                    observation, settings.peg_park_offset_m
                )
                hover_reached = (
                    self._position_distance(observation.socket, socket_hover)
                    <= settings.socket_approach_tolerance_m
                )
                if hover_reached:
                    self._phase = ScriptedInsertionStage.ALIGN
                    continue
                return left, right, settings.gripper_closed_target
            if self._phase is ScriptedInsertionStage.ALIGN:
                left, right = self._align_targets(observation)
                socket_park = self._socket_body_target(
                    observation, settings.peg_park_offset_m
                )
                socket_aligned = (
                    self._position_distance(observation.socket, socket_park)
                    <= settings.align_tolerance_m
                )
                if socket_aligned:
                    self._phase = ScriptedInsertionStage.INSERT
                    continue
                return left, right, settings.gripper_closed_target
            if self._phase is ScriptedInsertionStage.INSERT:
                if observation.pin_contact or observation.observed_reward >= 4.0:
                    self._phase = ScriptedInsertionStage.COMPLETE
                    continue
                left, right = self._insert_targets(observation)
                return left, right, settings.gripper_closed_target
            left, right = self._insert_targets(observation)
            return left, right, settings.gripper_closed_target
        raise TeacherGateError("Scripted stage advance did not terminate.")

    # --------------------------------------------------------------- action

    @staticmethod
    def _bounded_target(
        current: GeometryPose,
        desired: GeometryPose,
        maximum_step_m: float,
        maximum_orientation_step_rad: float,
    ) -> GeometryPose:
        """Glide the stage target toward the current pose (upstream interpolation).

        Solving the full stage target directly from a lagging state lets the
        orientation gradient dominate the QP and sacrifice position error; a
        target kept within one registered Cartesian/orientation step of the
        current pose mirrors the upstream policy's per-step linear waypoint
        interpolation and keeps every solve small and locally feasible.
        """

        delta = desired.position - current.position
        distance = float(torch.linalg.vector_norm(delta))
        if distance > maximum_step_m:
            delta = delta * (maximum_step_m / distance)
        desired_quaternion = desired.quaternion
        current_quaternion = current.quaternion
        dot = float(torch.dot(current_quaternion, desired_quaternion))
        if dot < 0.0:
            desired_quaternion = -desired_quaternion
            dot = -dot
        dot = min(1.0, max(-1.0, dot))
        angle = 2.0 * math.acos(dot)
        if angle <= maximum_orientation_step_rad:
            quaternion = desired_quaternion
        else:
            fraction = maximum_orientation_step_rad / angle
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

    def _task_target_action(
        self,
        observation: InsertionGeometry,
        left_target: GeometryPose,
        right_target: GeometryPose,
        gripper: float,
    ) -> TeacherDecision:
        current = observation.robot_state
        settings = self.settings
        glided_left = self._bounded_target(
            observation.left_eef,
            left_target,
            settings.maximum_cartesian_step_m,
            settings.maximum_orientation_step_rad,
        )
        glided_right = self._bounded_target(
            observation.right_eef,
            right_target,
            settings.maximum_cartesian_step_m,
            settings.maximum_orientation_step_rad,
        )
        try:
            result = self._solver.solve(
                _expanded_robot_qpos(current),
                left_position=glided_left.position.numpy(),
                left_quaternion_wxyz=glided_left.quaternion.numpy(),
                right_position=glided_right.position.numpy(),
                right_quaternion_wxyz=glided_right.quaternion.numpy(),
            )
        except (RuntimeError, ValueError) as error:
            return TeacherDecision(
                refusal=TeacherRefusal(
                    reason=TeacherRefusalReason.OUT_OF_SUPPORT,
                    detail=f"constrained IK refused the scripted target: {error}",
                )
            )
        raw = current.clone()
        raw[:6] = torch.as_tensor(result.qpos[:6], dtype=torch.float32)
        raw[7:13] = torch.as_tensor(result.qpos[8:14], dtype=torch.float32)
        raw[6] = gripper
        raw[13] = gripper
        # Damped pursuit: command a fraction of the solved joint correction so
        # the discrete position targets cannot ring against the actuator loop.
        gain = self.settings.joint_target_gain
        maximum_delta = self.settings.maximum_joint_target_delta_rad
        for index in (*range(6), *range(7, 13)):
            delta = gain * (float(raw[index]) - float(current[index]))
            if abs(delta) > maximum_delta:
                delta = math.copysign(maximum_delta, delta)
            raw[index] = float(current[index]) + delta
        return TeacherDecision(action=raw)


def build_teacher(
    *,
    settings: ScriptedTeacherSettings | None = None,
    ik_settings: MinkAlohaIkSettings | None = None,
) -> ScriptedInsertionTeacher:
    """Registration factory: one scripted-teacher candidate instance."""

    return ScriptedInsertionTeacher(settings=settings, ik_settings=ik_settings)
