"""Mink geometric insertion teacher: scripted task plan, direct multistart IK.

Teacher candidate line of the frozen staged gate ``m2-t2-teacher-gate-001``.
The current registered candidate is ``mink-geometric-insertion-teacher-003``
(``reports/training/m2-smolvla-t2-geometric-teacher-003-design-2026-09-03.md``):
the task plan is reused verbatim from the registered scripted candidate, the
command layer solves the exact stage target with the constrained Mink/DAQP
QP and commands the solved arm joints directly (full gain, one
direction-preserving common scale), and the candidate-002 closure's coupled
pose-constants are replaced by **stall-triggered, state-conditioned
escapes** over teacher-internal EEF-position history (the contract-legal
pattern of the scripted descend-stall guard — no time, index, episode or
seed input):

- transport stall unlock — static arms with the peg genuinely parked hold
  their poses so the registered stage exit fires through its own condition
  (the exit tolerance stays untouched for non-stalling poses);
- align stall branch escape — shifted multistart seeds engage under
  acceptance-only selection, with the scripted glide bounding the wrist
  flip, only after ALIGN stalls;
- insert stall press — with the socket touching the peg and the arms
  stalled, the target presses deeper along the peg axis through the
  compliant cage.

Predecessors: candidate 001 (closed 2026-09-02, G1 never passed) and
candidate 002 (closed 2026-09-03, G1 passed / G2 2/5 with the coupling
table that motivates this design). Refusals stay explicit; nothing
degrades silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from rosetta_reality.sim.geometry_teacher import (
    GeometryPose,
    InsertionGeometry,
    relative_pose,
)
from rosetta_reality.sim.mink_aloha_ik import MinkAlohaIkResult
from rosetta_reality.sim.scripted_teacher import (
    ScriptedInsertionStage,
    ScriptedInsertionTeacher,
    ScriptedTeacherSettings,
    _expanded_robot_qpos,
)
from rosetta_reality.sim.teacher_gate import (
    TeacherDecision,
    TeacherRefusal,
    TeacherRefusalReason,
)

# Expanded 16-D ALOHA qpos arm joints: 0-5 left arm, 8-13 right arm (6/14 are
# the normalized grippers, 7/15 their mirrored partners).
_EXPANDED_ARM_JOINT_INDICES = frozenset({0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13})


@dataclass(frozen=True, slots=True)
class GeometricCommandSettings:
    """Registered constants of the direct multistart command layer."""

    ik_seed_shifts: tuple[tuple[int, float], ...] = (
        (5, math.pi / 2),
        (5, -math.pi / 2),
        (13, math.pi / 2),
        (13, -math.pi / 2),
        (5, math.pi),
        (13, math.pi),
        (3, math.pi),
        (11, math.pi),
    )
    rotation_weight: float = 0.2
    seed_acceptance_position_m: float = 0.004
    seed_acceptance_orientation_rad: float = 0.05
    # Candidate 002 repairs (registered 2026-09-03), each targeting one
    # measured candidate-001 mechanism: the transport-entry table press and
    # the descend side-loading that levered the left finger joint past its
    # limit. See m2-smolvla-t2-geometric-teacher-002-design-2026-09-03.md.
    transport_height_guard: bool = True
    contact_phase_scale_cap: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.rotation_weight) or self.rotation_weight < 0.0:
            raise ValueError("rotation_weight must be finite and nonnegative.")
        for name in (
            "seed_acceptance_position_m",
            "seed_acceptance_orientation_rad",
            "contact_phase_scale_cap",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.contact_phase_scale_cap > 1.0:
            raise ValueError("contact_phase_scale_cap must not exceed one.")
        for joint_index, shift in self.ik_seed_shifts:
            if joint_index not in _EXPANDED_ARM_JOINT_INDICES:
                raise ValueError(
                    f"IK seed shift joint index {joint_index} is not an expanded "
                    "ALOHA arm joint."
                )
            if not math.isfinite(shift) or shift == 0.0:
                raise ValueError("IK seed shifts must be finite and nonzero.")


@dataclass(frozen=True, slots=True)
class GeometricEscapeSettings:
    """Registered constants of the stall-triggered escape state machine.

    Candidate 003 axis: each escape engages only when the observed geometry
    shows the system has stopped making progress, so poses that never stall
    keep the measured candidate-002 G2-001 behaviour. See
    ``m2-smolvla-t2-geometric-teacher-003-design-2026-09-03.md``.
    """

    stall_window_steps: int = 12
    stall_progress_m: float = 0.0015
    insert_press_through_m: float = 0.02
    # Escape-2 acceptance margin: when ALIGN has demonstrably converged to
    # the QP tracking floor, the achieved socket alignment is accepted if it
    # sits within this margin of the park pose (anchored to the ~0.03 m
    # residual floor the scripted dock sweep measured as physically
    # workable through the channel-guided insert).
    align_accept_margin_m: float = 0.03
    # Candidate-004 first-step fidelity: a stage target that already
    # coincides with the live end-effector pose is a HOLD, and the correct
    # hold command is the current joint state — re-solving it through the
    # QP lands on a different redundancy configuration (~0.12 rad mean
    # commanded jump at step 0, measured 2026-09-03 on the seed-10
    # calibration), putting every episode onto the wrong configuration
    # manifold before the first real motion.  The hold branch commands
    # zero motion, so it legitimately precedes the IK-failure refusal
    # (no solve is needed); the observation-level support refusals
    # (collisions, workspace) still run first in decide().
    hold_enabled: bool = True
    hold_position_epsilon_m: float = 1e-3
    hold_orientation_epsilon_rad: float = 1e-2

    def __post_init__(self) -> None:
        if self.stall_window_steps < 2:
            raise ValueError("stall_window_steps must be at least two.")
        if not math.isfinite(self.stall_progress_m) or self.stall_progress_m <= 0.0:
            raise ValueError("stall_progress_m must be finite and positive.")
        if not math.isfinite(self.insert_press_through_m) or (
            self.insert_press_through_m < 0.0
        ):
            raise ValueError("insert_press_through_m must be finite and nonnegative.")
        if not math.isfinite(self.align_accept_margin_m) or (
            self.align_accept_margin_m <= 0.0
        ):
            raise ValueError("align_accept_margin_m must be finite and positive.")
        for name in ("hold_position_epsilon_m", "hold_orientation_epsilon_rad"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True, slots=True)
class CommandDiagnostic:
    """Observability record of one multistart decision (not a teacher input)."""

    seeds_attempted: int
    seed_accepted_within_tolerance: bool
    residual_weighted_max: float
    left_position_residual_m: float
    right_position_residual_m: float
    left_orientation_residual_rad: float
    right_orientation_residual_rad: float
    command_scale: float


class MinkGeometricInsertionTeacher(ScriptedInsertionTeacher):
    """Scripted insertion task plan with the direct multistart command layer."""

    identity = "mink-geometric-insertion-teacher-003"

    def __init__(
        self,
        *,
        command_settings: GeometricCommandSettings | None = None,
        escape_settings: GeometricEscapeSettings | None = None,
        settings: ScriptedTeacherSettings | None = None,
        ik_settings: Any = None,
        solver: Any = None,
    ) -> None:
        super().__init__(settings=settings, ik_settings=ik_settings, solver=solver)
        self.command_settings = (
            command_settings
            if command_settings is not None
            else GeometricCommandSettings()
        )
        self.escape_settings = (
            escape_settings
            if escape_settings is not None
            else GeometricEscapeSettings()
        )
        self._last_command_diagnostic: CommandDiagnostic | None = None
        self._reset_escapes()

    def _reset_escapes(self) -> None:
        self._stall_phase: ScriptedInsertionStage | None = None
        self._stall_history: list[tuple[Tensor, Tensor]] = []
        self._align_escape_engaged = False
        self._insert_press_engaged = False
        self._latest_observation: InsertionGeometry | None = None

    @property
    def last_command_diagnostic(self) -> CommandDiagnostic | None:
        """Diagnostics of the most recent decision (observability only)."""

        return self._last_command_diagnostic

    def reset(self) -> None:
        super().reset()
        self._last_command_diagnostic = None
        self._reset_escapes()

    def decide(self, observation: InsertionGeometry) -> TeacherDecision:
        self._latest_observation = observation
        self._record_stall_history(observation)
        if self._movement_stalled():
            if self._phase is ScriptedInsertionStage.ALIGN:
                self._align_escape_engaged = True
                # Repair round 2 (cage recapture): the measured align stall
                # tracks the frozen site composition to the QP floor while
                # the socket body hangs ~1.5 cm off the park pose — the
                # in-cage socket-to-site transform drifted since the
                # transport freeze.  Re-capturing it through the current
                # pose re-aims the site target so the socket lands on the
                # park pose (the same state-conditioned recapture primitive
                # the transport freeze itself uses).
                self._socket_from_left_site = relative_pose(
                    observation.socket, observation.left_eef
                )
            elif (
                self._phase is ScriptedInsertionStage.INSERT
                and observation.peg_socket_contact
            ):
                self._insert_press_engaged = True
        return super().decide(observation)

    def _meet_left_target(self) -> GeometryPose:
        """Escape 1 (transport stall unlock), applied at the exit layer.

        The registered stage exit reads the meet target through this method,
        so holding the current left end-effector pose here — only when both
        arms are stalled AND the peg is parked within the unchanged
        transport tolerance — lets the exit fire through its own condition
        on the next observation and run the normal freeze captures.  The
        relative freeze captures absorb the residual meet offset (measured:
        the parked-peg deadlock sits ~1.2 cm outside the 0.012 exit check
        at the dual-arm loaded QP floor).
        """

        observation = self._latest_observation
        if (
            observation is not None
            and self._phase is ScriptedInsertionStage.TRANSPORT
            and self._movement_stalled()
            and self._peg_park_distance(observation)
            <= self.settings.transport_tolerance_m
        ):
            return observation.left_eef
        return super()._meet_left_target()

    # ---------------------------------------------------- stall detection

    def _record_stall_history(self, observation: InsertionGeometry) -> None:
        """Track observed EEF positions within the current phase only.

        The scripted descend-stall guard set the contract-legal precedent:
        teacher-internal history over observations carries no time, index,
        episode or seed input and is deterministic under replay."""

        if self._stall_phase is not self._phase:
            self._stall_phase = self._phase
            self._stall_history.clear()
        self._stall_history.append(
            (
                observation.left_eef.position.clone(),
                observation.right_eef.position.clone(),
            )
        )
        window = self.escape_settings.stall_window_steps
        if len(self._stall_history) > window:
            del self._stall_history[:-window]

    def _movement_stalled(self) -> bool:
        window = self.escape_settings.stall_window_steps
        if len(self._stall_history) < window:
            return False
        oldest_left, oldest_right = self._stall_history[0]
        newest_left, newest_right = self._stall_history[-1]
        bound = window * self.escape_settings.stall_progress_m
        return (
            float(torch.linalg.vector_norm(newest_left - oldest_left)) <= bound
            and float(torch.linalg.vector_norm(newest_right - oldest_right)) <= bound
        )

    def _socket_body_target(
        self, observation: InsertionGeometry, offset_m: float
    ) -> GeometryPose:
        """Escape 2 (align acceptance at the proven QP floor).

        The registered align exit reads the park reference through this
        method.  Measured on 1904: the carrying arm tracks the site
        composition to the 8 mm QP floor while the socket body hangs 14 mm
        off the park pose against a 12 mm exit tolerance — the demanded
        precision is below the achievable tracking floor in this loaded
        configuration.  When ALIGN has stalled with the escape engaged and
        the achieved gap is within the registered physical margin, the
        achieved socket pose becomes the alignment reference: the exit
        fires through its own condition and the insert channel (composing
        from the frozen peg frame) absorbs the residual offset — the same
        accept-when-proven semantics as the transport unlock.
        """

        if (
            math.isclose(offset_m, self.settings.peg_park_offset_m)
            and self._phase is ScriptedInsertionStage.ALIGN
            and self._align_escape_engaged
            and self._movement_stalled()
        ):
            gap = float(
                torch.linalg.vector_norm(
                    observation.socket.position
                    - super()._socket_body_target(observation, offset_m).position
                )
            )
            if gap <= self.escape_settings.align_accept_margin_m:
                return observation.socket
        return super()._socket_body_target(observation, offset_m)

    def _peg_park_distance(self, observation: InsertionGeometry) -> float:
        """Distance of the observed peg to the registered park body pose."""

        return self._position_distance(
            observation.peg, self._peg_park_body(observation)
        )

    # ------------------------------------------------------- command layer

    #: Stages whose targets are subject to the measured insert-branch IK
    #: trap: INSERT always engages the shifted multistart seeds under the
    #: registered lowest-residual rule (the candidate-002 G2-001 behaviour
    #: that solves 1903/1904).  ALIGN engages them only after its
    #: stall-triggered escape fires (candidate 003): the always-on variant
    #: pinned 1901 at a bad branch while 1900 needs the escape.
    _BRANCH_SEED_STAGES = frozenset({ScriptedInsertionStage.INSERT})

    #: Contact-adjacent stages whose command speed is bounded by the
    #: registered contact-phase scale cap (candidate 002 repair 2): the
    #: seed-10 diagnostic measured the descend bottom squeeze levering the
    #: left finger joint past its limit at full direct-command speed.
    _CONTACT_PHASE_STAGES = frozenset(
        {
            ScriptedInsertionStage.DESCEND,
            ScriptedInsertionStage.GRASP,
            ScriptedInsertionStage.TRANSPORT,
        }
    )

    #: Contact-adjacent stages that inherit the registered scripted glide
    #: (candidate 002 repair round 2): the direct full-target command keeps
    #: pressing into contact because its target never yields — the measured
    #: wedge (finger levered past its limit) and the transport-entry press
    #: both persisted under speed caps, and the carried objects swing out of
    #: the cage under unbounded wrist rotations.  The glide bounds the demand
    #: to one registered step of the achieved pose, so a blocked arm sees its
    #: target collapse onto itself and stops pressing while a wrist rotation
    #: walks gradually; stage exits still judge the exact targets, so
    #: precision is preserved.
    _GLIDE_PHASE_STAGES = frozenset(
        {
            ScriptedInsertionStage.DESCEND,
            ScriptedInsertionStage.GRASP,
            ScriptedInsertionStage.TRANSPORT,
        }
    )

    def _apply_contact_phase_glide(
        self,
        observation: InsertionGeometry,
        left_target: GeometryPose,
        right_target: GeometryPose,
    ) -> tuple[GeometryPose, GeometryPose]:
        """Bound the stage target to one registered step of the live pose in
        the contact-adjacent stages (the scripted protective semantics)."""

        if self._phase not in self._GLIDE_PHASE_STAGES:
            return left_target, right_target
        settings = self.settings
        return (
            self._bounded_target(
                observation.left_eef,
                left_target,
                settings.maximum_cartesian_step_m,
                settings.maximum_orientation_step_rad,
            ),
            self._bounded_target(
                observation.right_eef,
                right_target,
                settings.maximum_cartesian_step_m,
                settings.maximum_orientation_step_rad,
            ),
        )

    def _apply_carrying_height_guard(
        self,
        observation: InsertionGeometry,
        left_target: GeometryPose,
        right_target: GeometryPose,
    ) -> tuple[GeometryPose, GeometryPose]:
        """No downward target while the carried object rests on the table.

        Candidate 002 repair 1: the transport park target sits below the
        carried pose at entry, and the full-gain command pressed both right
        fingers into the table. While the observed object-on-table contact
        flag holds, the carrying arm's target z is clamped to its current z;
        the clamp is a no-op for upward targets and releases when the object
        lifts.
        """

        if not self.command_settings.transport_height_guard:
            return left_target, right_target
        if self._phase is not ScriptedInsertionStage.TRANSPORT:
            # DESCEND/GRASP legitimately command downward motion toward the
            # objects resting on the table; the guard exists only for the
            # carrying drag.
            return left_target, right_target
        if observation.peg_on_table:
            position = right_target.position.clone()
            current_z = float(observation.right_eef.position[2])
            if float(position[2]) < current_z:
                position[2] = current_z
                right_target = GeometryPose(
                    position=position, quaternion=right_target.quaternion
                )
        if observation.socket_on_table:
            position = left_target.position.clone()
            current_z = float(observation.left_eef.position[2])
            if float(position[2]) < current_z:
                position[2] = current_z
                left_target = GeometryPose(
                    position=position, quaternion=left_target.quaternion
                )
        return left_target, right_target

    def _ik_seeds(self, expanded_qpos: Any) -> list[Any]:
        seeds = [expanded_qpos]
        engage_align = (
            self._phase is ScriptedInsertionStage.ALIGN and self._align_escape_engaged
        )
        if self._phase in self._BRANCH_SEED_STAGES or engage_align:
            for joint_index, shift in self.command_settings.ik_seed_shifts:
                seed = expanded_qpos.copy()
                seed[joint_index] += shift
                seeds.append(seed)
        return seeds

    def _weighted_residual(self, result: MinkAlohaIkResult) -> float:
        weight = self.command_settings.rotation_weight
        return max(
            position + weight * orientation
            for position, orientation in zip(
                result.position_errors_m, result.orientation_errors_rad
            )
        )

    def _accepted_within_tolerance(self, result: MinkAlohaIkResult) -> bool:
        command = self.command_settings
        return all(
            position <= command.seed_acceptance_position_m
            and orientation <= command.seed_acceptance_orientation_rad
            for position, orientation in zip(
                result.position_errors_m, result.orientation_errors_rad
            )
        )

    def _task_target_action(
        self,
        observation: InsertionGeometry,
        left_target: GeometryPose,
        right_target: GeometryPose,
        gripper: float,
    ) -> TeacherDecision:
        current = observation.robot_state
        if (
            self._phase is ScriptedInsertionStage.ALIGN
            and self._align_escape_engaged
        ):
            # Escape 2 (align stall branch escape): the shifted seeds are
            # engaged (see _ik_seeds) and the target inherits the scripted
            # glide so the wrist branch flip rotates gradually instead of
            # sweeping the socket out of the cage.
            left_target, right_target = self._apply_contact_phase_glide(
                observation, left_target, right_target
            )
        if (
            self._phase is ScriptedInsertionStage.INSERT
            and self._insert_press_engaged
        ):
            # Escape 3 (insert stall press): the free track has stalled with
            # the socket touching the peg; press deeper along the peg axis
            # so the compliant cage converts the command into insertion
            # force.
            left_target = self._site_target(
                observation,
                self.settings.peg_insert_offset_m
                - self.escape_settings.insert_press_through_m,
            )
        left_target, right_target = self._apply_contact_phase_glide(
            observation, left_target, right_target
        )
        left_target, right_target = self._apply_carrying_height_guard(
            observation, left_target, right_target
        )
        escape = self.escape_settings
        if (
            escape.hold_enabled
            and self._position_distance(observation.left_eef, left_target)
            <= escape.hold_position_epsilon_m
            and self._position_distance(observation.right_eef, right_target)
            <= escape.hold_position_epsilon_m
            and self._orientation_distance(observation.left_eef, left_target)
            <= escape.hold_orientation_epsilon_rad
            and self._orientation_distance(observation.right_eef, right_target)
            <= escape.hold_orientation_epsilon_rad
        ):
            # Candidate-004 first-step fidelity: a target that already
            # coincides with the live pose is a HOLD — command the current
            # joints instead of re-solving the QP (whose redundancy
            # resolution lands ~0.12 rad away at episode start).
            raw = current.clone()
            raw[6] = gripper
            raw[13] = gripper
            self._last_command_diagnostic = CommandDiagnostic(
                seeds_attempted=0,
                seed_accepted_within_tolerance=True,
                residual_weighted_max=0.0,
                left_position_residual_m=0.0,
                right_position_residual_m=0.0,
                left_orientation_residual_rad=0.0,
                right_orientation_residual_rad=0.0,
                command_scale=1.0,
            )
            return TeacherDecision(action=raw)
        expanded = _expanded_robot_qpos(current)
        solver_errors: list[str] = []
        best: tuple[float, MinkAlohaIkResult] | None = None
        accepted = False
        seeds_attempted = 0
        for seed in self._ik_seeds(expanded):
            seeds_attempted += 1
            try:
                result = self._solver.solve(
                    seed,
                    left_position=left_target.position.numpy(),
                    left_quaternion_wxyz=left_target.quaternion.numpy(),
                    right_position=right_target.position.numpy(),
                    right_quaternion_wxyz=right_target.quaternion.numpy(),
                )
            except (RuntimeError, ValueError) as error:
                solver_errors.append(str(error))
                continue
            residual = self._weighted_residual(result)
            if best is None or residual < best[0]:
                best = (residual, result)
            if self._accepted_within_tolerance(result):
                accepted = True
                break
        # Branch selection: the lowest weighted residual across the engaged
        # seed family, with an early break when a seed converges within the
        # strict acceptance tolerance (the registered candidate-001 rule).
        # Under the align stall escape the family includes the shifted
        # seeds, so a stalled unshifted branch is replaced by the best
        # converging alternative instead of re-freezing the deadlock.
        if best is None:
            result = None
        else:
            result = best[1]
        if result is None:
            detail = "; ".join(solver_errors[-2:]) if solver_errors else "no seed ran"
            return TeacherDecision(
                refusal=TeacherRefusal(
                    reason=TeacherRefusalReason.OUT_OF_SUPPORT,
                    detail=(
                        "geometric teacher multistart refused the stage target on "
                        f"every seed: {detail}"
                    ),
                )
            )
        residual = self._weighted_residual(result)
        raw = current.clone()
        raw[:6] = torch.as_tensor(result.qpos[:6], dtype=torch.float32)
        raw[7:13] = torch.as_tensor(result.qpos[8:14], dtype=torch.float32)
        raw[6] = gripper
        raw[13] = gripper
        # Direct pursuit (repair round 1): the solved configuration is
        # commanded at full gain and the registered smoothness bound is
        # applied as one direction-preserving scale s = min(1, bound /
        # max|joint delta|).  A per-joint clamp distorts the joint-space
        # direction (measured: 8-10 of 12 joints saturated per step and the
        # arms flailed into the socket within 71 steps on seed 10), while the
        # proportional scale keeps the commanded path on the solved IK
        # direction at the same worst-case per-joint speed bound.
        maximum_delta = self.settings.maximum_joint_target_delta_rad
        arm_indices = (*range(6), *range(7, 13))
        deltas = torch.tensor(
            [float(raw[index]) - float(current[index]) for index in arm_indices],
            dtype=torch.float32,
        )
        maximum_absolute = float(deltas.abs().max()) if deltas.numel() else 0.0
        scale = (
            min(1.0, maximum_delta / maximum_absolute)
            if maximum_absolute > 1e-12
            else 1.0
        )
        if self._phase in self._CONTACT_PHASE_STAGES:
            scale = min(scale, self.command_settings.contact_phase_scale_cap)
        for offset, index in enumerate(arm_indices):
            raw[index] = float(current[index]) + scale * float(deltas[offset])
        self._last_command_diagnostic = CommandDiagnostic(
            seeds_attempted=seeds_attempted,
            seed_accepted_within_tolerance=accepted,
            residual_weighted_max=residual,
            left_position_residual_m=result.position_errors_m[0],
            right_position_residual_m=result.position_errors_m[1],
            left_orientation_residual_rad=result.orientation_errors_rad[0],
            right_orientation_residual_rad=result.orientation_errors_rad[1],
            command_scale=scale,
        )
        return TeacherDecision(action=raw)


def build_teacher(
    *,
    settings: ScriptedTeacherSettings | None = None,
    command_settings: GeometricCommandSettings | None = None,
    escape_settings: GeometricEscapeSettings | None = None,
    ik_settings: Any = None,
) -> MinkGeometricInsertionTeacher:
    """Registration factory: one geometric-teacher candidate instance.

    The registered candidate-003 composition: the candidate-002 G2-001
    command layer (scripted stage settings unchanged, contact-phase glide,
    carrying height guard, contact-phase scale cap, INSERT branch seeds
    under lowest-residual selection) plus the three stall-triggered
    escapes. See the module docstring and the candidate-003 design.
    """

    return MinkGeometricInsertionTeacher(
        settings=settings if settings is not None else ScriptedTeacherSettings(),
        command_settings=command_settings,
        escape_settings=escape_settings,
        ik_settings=ik_settings,
    )
