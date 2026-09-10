"""Geometric-teacher candidate tests: multistart command layer, refusals.

Covers the candidate of
``reports/training/m2-smolvla-t2-geometric-teacher-candidate-design-2026-09-02.md``
against the frozen teacher contract: the deterministic multistart seed family,
first-convergence acceptance, the direct full-gain command clamp, explicit
refusal when every seed fails, replay determinism and reset hygiene. The
constrained solver is replaced by a deterministic double; the real solver and
the live-environment observation path are exercised by the gate stages and the
scripted-teacher container tests.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPOSITORY_ROOT / "src"),):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from rosetta_reality.sim.geometric_teacher import (  # noqa: E402
    GeometricCommandSettings,
    GeometricEscapeSettings,
    MinkGeometricInsertionTeacher,
)
from rosetta_reality.sim.geometry_teacher import (  # noqa: E402
    GeometryPose,
    InsertionGeometry,
    relative_pose,
)
from rosetta_reality.sim.mink_aloha_ik import MinkAlohaIkResult  # noqa: E402
from rosetta_reality.sim.scripted_teacher import (  # noqa: E402
    ScriptedInsertionStage,
    _expanded_robot_qpos,
)
from rosetta_reality.sim.teacher_gate import (  # noqa: E402
    TeacherRefusalReason,
    probe_observation,
    run_conformance_probe,
)


@dataclass
class SolverDouble:
    """Deterministic IK double: records seeds, holds the current pose.

    ``offsets`` shift solved arm joints 0..n away from the seed so the
    direct-command scaling is measurable. ``fail_calls`` raises on the given
    zero-based call ordinals. Returned residuals are configurable.
    """

    offsets: tuple[float, ...] = ()
    failure: str | None = None
    fail_calls: tuple[int, ...] = ()
    position_errors_m: tuple[float, float] = (0.0, 0.0)
    orientation_errors_rad: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []
        self.invocations = 0

    def solve(
        self,
        current_qpos: np.ndarray,
        *,
        left_position: np.ndarray,
        left_quaternion_wxyz: np.ndarray,
        right_position: np.ndarray,
        right_quaternion_wxyz: np.ndarray,
    ) -> MinkAlohaIkResult:
        ordinal = self.invocations
        self.invocations += 1
        if self.failure is not None or ordinal in self.fail_calls:
            raise RuntimeError(self.failure or "seed solve failed")
        self.calls.append(
            {
                "current_qpos": np.array(current_qpos, copy=True),
                "left_position": np.array(left_position, copy=True),
                "right_position": np.array(right_position, copy=True),
            }
        )
        qpos = np.array(current_qpos, dtype=np.float64).copy()
        for joint_index, offset in enumerate(self.offsets):
            qpos[joint_index] += offset
        return MinkAlohaIkResult(
            qpos=qpos,
            iterations=1,
            position_errors_m=self.position_errors_m,
            orientation_errors_rad=self.orientation_errors_rad,
        )


def build_teacher_with(solver: SolverDouble) -> MinkGeometricInsertionTeacher:
    return MinkGeometricInsertionTeacher(solver=solver)


def build_teacher_no_hold(solver: SolverDouble) -> MinkGeometricInsertionTeacher:
    """Legacy direct-command construction: the hold branch disabled.

    Used by the tests that exercise the solver path with fixtures whose
    stage targets coincide with the live pose (OPEN holds by design now)."""

    return MinkGeometricInsertionTeacher(
        solver=solver,
        escape_settings=GeometricEscapeSettings(hold_enabled=False),
    )


def prime_stage_captures(
    teacher: MinkGeometricInsertionTeacher, observation: InsertionGeometry
) -> None:
    """Populate the stage machine's capture state for a forced late stage."""

    teacher._socket_position = observation.socket.position.clone()
    teacher._peg_position = observation.peg.position.clone()
    teacher._orient_left = observation.left_eef
    teacher._orient_right = observation.right_eef
    teacher._peg_from_right_site = relative_pose(observation.peg, observation.right_eef)
    teacher._hold_right_site = observation.right_eef
    teacher._peg_hold_pose = observation.peg
    teacher._socket_from_left_site = relative_pose(
        observation.socket, observation.left_eef
    )
    teacher._alignment_flipped = False


def test_multistart_tries_registered_seed_family_in_insert_stage() -> None:
    solver = SolverDouble(
        position_errors_m=(0.05, 0.05), orientation_errors_rad=(0.2, 0.2)
    )
    teacher = build_teacher_with(solver)
    observation = probe_observation()
    prime_stage_captures(teacher, observation)
    teacher._phase = ScriptedInsertionStage.INSERT
    decision = teacher.decide(observation)
    assert decision.refusal is None
    assert len(solver.calls) == 1 + len(GeometricCommandSettings().ik_seed_shifts)
    base = solver.calls[0]["current_qpos"]
    expected_shifts = (
        (5, math.pi / 2),
        (5, -math.pi / 2),
        (13, math.pi / 2),
        (13, -math.pi / 2),
        (5, math.pi),
        (13, math.pi),
        (3, math.pi),
        (11, math.pi),
    )
    for call_index, (joint_index, shift) in enumerate(expected_shifts, start=1):
        seed = solver.calls[call_index]["current_qpos"]
        assert seed[joint_index] == pytest.approx(base[joint_index] + shift)
        # Every other entry stays on the unshifted expanded state.
        mutated = np.zeros_like(seed, dtype=bool)
        mutated[joint_index] = True
        assert np.allclose(
            seed[~mutated], np.delete(base, joint_index)
        ), f"seed {call_index} must shift only joint {joint_index}"
    diagnostic = teacher.last_command_diagnostic
    assert diagnostic is not None
    assert diagnostic.seeds_attempted == len(expected_shifts) + 1
    assert diagnostic.seed_accepted_within_tolerance is False


def test_carrying_and_early_stages_solve_only_the_current_branch() -> None:
    """Repair round 2: the branch switch while carrying dragged the held peg
    out of grasp; only ALIGN/INSERT may engage the shifted seeds."""

    for stage in (
        ScriptedInsertionStage.OPEN,
        ScriptedInsertionStage.APPROACH,
        ScriptedInsertionStage.ORIENT,
        ScriptedInsertionStage.DESCEND,
        ScriptedInsertionStage.GRASP,
        ScriptedInsertionStage.TRANSPORT,
        ScriptedInsertionStage.ALIGN,
    ):
        solver = SolverDouble(
            position_errors_m=(0.05, 0.05), orientation_errors_rad=(0.2, 0.2)
        )
        teacher = build_teacher_no_hold(solver)
        observation = probe_observation()
        prime_stage_captures(teacher, observation)
        teacher._phase = stage
        decision = teacher.decide(observation)
        assert decision.refusal is None
        assert solver.invocations == 1, f"stage {stage} must solve unshifted"
        assert teacher.last_command_diagnostic is not None
        assert teacher.last_command_diagnostic.seeds_attempted == 1


def test_first_converging_seed_is_accepted_immediately() -> None:
    solver = SolverDouble(fail_calls=(0,))
    teacher = build_teacher_with(solver)
    observation = probe_observation()
    prime_stage_captures(teacher, observation)
    teacher._phase = ScriptedInsertionStage.INSERT
    decision = teacher.decide(observation)
    assert decision.refusal is None
    assert solver.invocations == 2
    assert len(solver.calls) == 1
    diagnostic = teacher.last_command_diagnostic
    assert diagnostic is not None
    assert diagnostic.seeds_attempted == 2
    assert diagnostic.seed_accepted_within_tolerance is True
    assert diagnostic.residual_weighted_max == pytest.approx(0.0)


def test_command_is_direct_with_registered_smoothness_scale() -> None:
    current = probe_observation().robot_state
    expanded = _expanded_robot_qpos(current)

    near_solver = SolverDouble(offsets=(0.05,))
    near_decision = build_teacher_no_hold(near_solver).decide(probe_observation())
    assert near_decision.action is not None
    assert float(near_decision.action[0]) == pytest.approx(
        float(current[0]) + 0.05
    ), "direct command must apply the solved joint at full gain when within bound"

    far_solver = SolverDouble(offsets=(0.5,))
    teacher = build_teacher_no_hold(far_solver)
    far_decision = teacher.decide(probe_observation())
    assert far_decision.action is not None
    maximum_delta = teacher.settings.maximum_joint_target_delta_rad
    assert float(far_decision.action[0]) == pytest.approx(
        float(current[0]) + maximum_delta
    )
    assert far_solver.calls[0]["current_qpos"][0] == pytest.approx(
        float(expanded[0])
    ), "the IK target must be solved from the exact expanded state (no glide)"
    diagnostic = teacher.last_command_diagnostic
    assert diagnostic is not None
    assert diagnostic.command_scale == pytest.approx(maximum_delta / 0.5)


def test_command_scale_preserves_joint_space_direction() -> None:
    current = probe_observation().robot_state
    solver = SolverDouble(offsets=(0.5, 0.3))
    decision = build_teacher_no_hold(solver).decide(probe_observation())
    assert decision.action is not None
    maximum_delta = build_teacher_with(SolverDouble()).settings.maximum_joint_target_delta_rad
    scale = maximum_delta / 0.5
    assert float(decision.action[0]) == pytest.approx(float(current[0]) + 0.5 * scale)
    assert float(decision.action[1]) == pytest.approx(
        float(current[1]) + 0.3 * scale
    ), "proportional scaling must preserve the solved joint-space direction"


def test_every_seed_failing_refuses_out_of_support() -> None:
    teacher = build_teacher_no_hold(SolverDouble(failure="seed solve failed"))
    decision = teacher.decide(probe_observation())
    assert decision.refusal is not None
    assert decision.refusal.reason is TeacherRefusalReason.OUT_OF_SUPPORT
    assert "multistart" in decision.refusal.detail


def test_hold_commands_zero_delta_without_solving() -> None:
    """Candidate-004 first-step fidelity: when the stage target coincides
    with the live pose (the OPEN-phase home hold, exactly the measured
    step-0 yank scenario), the command is the current joints — no IK, no
    first-step configuration jump."""

    observation = probe_observation()
    solver = SolverDouble(failure="must not be called on a hold")
    teacher = build_teacher_with(solver)
    decision = teacher.decide(observation)
    assert decision.refusal is None
    assert decision.action is not None
    assert solver.invocations == 0, "a hold must not re-solve through the QP"
    assert torch.allclose(decision.action[:6], observation.robot_state[:6])
    assert torch.allclose(decision.action[7:13], observation.robot_state[7:13])
    assert float(decision.action[6]) == pytest.approx(0.90)
    assert float(decision.action[13]) == pytest.approx(0.90)
    diagnostic = teacher.last_command_diagnostic
    assert diagnostic is not None
    assert diagnostic.seeds_attempted == 0


def test_hold_rejects_when_orientation_differs() -> None:
    """A target at the same position but a different orientation is not a
    hold: the solver path must run (the wrist still has to rotate)."""

    observation = probe_observation()
    tilted = math.radians(45.0)
    teacher = build_teacher_with(SolverDouble())
    teacher.decide(observation)  # captures home
    home = teacher._home_left
    tilted_pose = GeometryPose(
        position=home.position,
        quaternion=torch.tensor(
            [
                math.cos(tilted / 2.0),
                0.0,
                math.sin(tilted / 2.0),
                0.0,
            ],
            dtype=torch.float32,
        ),
    )
    teacher._home_left = tilted_pose
    decision = teacher.decide(observation)
    assert decision.refusal is None


def test_probe_suite_passes_on_the_geometric_candidate() -> None:
    result = run_conformance_probe(
        build_teacher_with(SolverDouble()),
        probe_observations=[
            probe_observation(),
            probe_observation(observed_reward=2.0),
        ],
        off_support_observation=dataclasses.replace(
            probe_observation(), unexpected_collision_count=1
        ),
    )
    assert result.deterministic is True
    assert result.refusal_region_non_degenerate is True


def test_reset_clears_command_diagnostic() -> None:
    teacher = build_teacher_with(SolverDouble())
    teacher.decide(probe_observation())
    assert teacher.last_command_diagnostic is not None
    teacher.reset()
    assert teacher.last_command_diagnostic is None
    assert teacher._orient_left is None
    assert teacher._orient_right is None


def test_command_settings_validation() -> None:
    with pytest.raises(ValueError, match="expanded"):
        GeometricCommandSettings(ik_seed_shifts=((6, math.pi),))
    with pytest.raises(ValueError, match="nonzero"):
        GeometricCommandSettings(ik_seed_shifts=((5, 0.0),))
    with pytest.raises(ValueError, match="nonnegative"):
        GeometricCommandSettings(rotation_weight=-0.1)
    with pytest.raises(ValueError, match="positive"):
        GeometricCommandSettings(seed_acceptance_position_m=0.0)
    with pytest.raises(ValueError, match="not exceed"):
        GeometricCommandSettings(contact_phase_scale_cap=1.5)
    with pytest.raises(ValueError, match="at least two"):
        GeometricEscapeSettings(stall_window_steps=1)
    with pytest.raises(ValueError, match="positive"):
        GeometricEscapeSettings(stall_progress_m=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        GeometricEscapeSettings(insert_press_through_m=-0.01)


def _parked_observation() -> InsertionGeometry:
    """A probe observation whose peg sits at the registered park offset."""

    observation = probe_observation()
    return dataclasses.replace(
        observation,
        peg=GeometryPose(
            position=observation.socket.position
            + torch.tensor([0.16, 0.0, 0.0], dtype=torch.float32),
            quaternion=observation.peg.quaternion,
        ),
    )


def test_transport_stall_unlock_holds_the_meet_target() -> None:
    """Escape 1: a static TRANSPORT with the peg genuinely parked holds the
    meet target at the current pose — the registered stage exit reads the
    target through the same method and fires by itself."""

    observation = _parked_observation()
    solver = SolverDouble(
        position_errors_m=(0.05, 0.05), orientation_errors_rad=(0.2, 0.2)
    )
    teacher = build_teacher_with(solver)
    prime_stage_captures(teacher, observation)
    teacher._phase = ScriptedInsertionStage.TRANSPORT
    window = teacher.escape_settings.stall_window_steps
    for _ in range(window - 1):
        teacher.decide(observation)
    # Pre-stall, the meet target drives the arms toward the registered pose.
    current_left = observation.left_eef.position.numpy()
    traveling_left = np.array(solver.calls[0]["left_position"], copy=True)
    assert np.linalg.norm(traveling_left - current_left) > 1e-3

    decision = teacher.decide(observation)
    assert decision.refusal is None
    # The held meet target satisfies the registered exit within the same
    # decision: the stage machine advances and runs the freeze captures.
    assert teacher.phase is not ScriptedInsertionStage.TRANSPORT
    assert teacher._hold_right_site is not None
    assert teacher._socket_from_left_site is not None


def test_transport_stall_requires_a_parked_peg() -> None:
    """A static TRANSPORT whose peg never parked must keep chasing the meet
    targets (the escape is earned by the task condition, not by stalling
    alone)."""

    observation = probe_observation()  # peg 0.20 m from the park body
    solver = SolverDouble(
        position_errors_m=(0.05, 0.05), orientation_errors_rad=(0.2, 0.2)
    )
    teacher = build_teacher_with(solver)
    prime_stage_captures(teacher, observation)
    teacher._phase = ScriptedInsertionStage.TRANSPORT
    window = teacher.escape_settings.stall_window_steps
    for _ in range(window + 2):
        teacher.decide(observation)
    assert teacher.phase is ScriptedInsertionStage.TRANSPORT, (
        "without the parked peg the stall escape must not fire"
    )


def test_align_stall_engages_acceptance_only_branch_seeds() -> None:
    """Escape 2: ALIGN solves unshifted until it stalls; the escape engages
    the shifted seed family under acceptance-only selection."""

    observation = probe_observation()
    solver = SolverDouble(
        position_errors_m=(0.05, 0.05), orientation_errors_rad=(0.2, 0.2)
    )
    teacher = build_teacher_with(solver)
    prime_stage_captures(teacher, observation)
    teacher._phase = ScriptedInsertionStage.ALIGN
    window = teacher.escape_settings.stall_window_steps
    for _ in range(window - 1):
        teacher.decide(observation)
    assert solver.invocations == window - 1, "pre-stall ALIGN stays unshifted"
    teacher.decide(observation)
    assert teacher._align_escape_engaged is True
    assert solver.invocations == window - 1 + 1 + len(
        GeometricCommandSettings().ik_seed_shifts
    ), (
        "the escape decision runs the full seed family; without a converging "
        "seed the best-residual alternative replaces the stalled branch"
    )


def test_align_escape_accepts_a_converging_shifted_seed() -> None:
    """With the escape engaged, a shifted seed that converges within the
    registered tolerances is accepted immediately."""

    observation = probe_observation()
    solver = SolverDouble(fail_calls=(0,))
    teacher = build_teacher_with(solver)
    prime_stage_captures(teacher, observation)
    teacher._phase = ScriptedInsertionStage.ALIGN
    teacher._align_escape_engaged = True
    decision = teacher.decide(observation)
    assert decision.refusal is None
    assert solver.invocations == 2  # unshifted refused, first shift accepted


def test_insert_stall_press_latches_on_peg_socket_contact() -> None:
    """Escape 3: a stalled INSERT with the socket touching the peg presses
    deeper along the peg axis; without the contact flag it never engages."""

    observation = probe_observation()
    solver = SolverDouble()
    teacher = build_teacher_with(solver)
    prime_stage_captures(teacher, observation)
    teacher._phase = ScriptedInsertionStage.INSERT
    window = teacher.escape_settings.stall_window_steps
    for _ in range(window + 2):
        teacher.decide(observation)
    assert teacher._insert_press_engaged is False, (
        "peg_socket_contact gates the press escape"
    )

    contact = dataclasses.replace(observation, peg_socket_contact=True)
    teacher.decide(contact)
    assert teacher._insert_press_engaged is True
    pressed_left = np.array(solver.calls[-1]["left_position"], copy=True)
    teacher.decide(contact)
    latched_left = np.array(solver.calls[-1]["left_position"], copy=True)
    assert np.allclose(pressed_left, latched_left), "the press is latched"


def test_reset_clears_escape_state() -> None:
    observation = probe_observation()
    teacher = build_teacher_with(SolverDouble())
    teacher.decide(observation)
    teacher._align_escape_engaged = True
    teacher._insert_press_engaged = True
    teacher._stall_history.append(
        (observation.left_eef.position, observation.right_eef.position)
    )
    teacher.reset()
    assert teacher._align_escape_engaged is False
    assert teacher._insert_press_engaged is False
    assert teacher._stall_history == []
    assert teacher._stall_phase is None
    assert teacher._last_command_diagnostic is None


def test_carrying_height_guard_blocks_downward_transport_target() -> None:
    """Candidate 002 repair 1: while the peg rests on the table in TRANSPORT,
    a park target below the carried pose is clamped to the current height."""

    observation = probe_observation()
    solver = SolverDouble()
    teacher = build_teacher_with(solver)
    prime_stage_captures(teacher, observation)
    # Force the carry transform to demand a right-arm target below the
    # current end-effector height (mimicking the measured transport entry).
    teacher._peg_from_right_site = GeometryPose(
        position=torch.tensor([0.0, 0.0, -0.02], dtype=torch.float32),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    teacher._phase = ScriptedInsertionStage.TRANSPORT
    decision = teacher.decide(observation)
    assert decision.refusal is None
    current_right_z = float(observation.right_eef.position[2])
    assert float(solver.calls[0]["right_position"][2]) == pytest.approx(
        current_right_z
    ), "the downward transport target must be clamped while the peg is on the table"
    # The upward left target (meet center) survives the guard: the glide
    # bounds the 3-D demand to one step (its z component still rises), and
    # the clamp — which only blocks downward targets — is a no-op for it.
    current_left = observation.left_eef.position
    recorded_left = solver.calls[0]["left_position"]
    assert float(recorded_left[2]) > float(current_left[2])
    assert float(
        torch.linalg.vector_norm(
            torch.as_tensor(recorded_left, dtype=torch.float32) - current_left
        )
    ) <= teacher.settings.maximum_cartesian_step_m + 1e-6


def test_carrying_height_guard_is_transport_scoped() -> None:
    """DESCEND commands downward motion but only through the registered
    glide step: the demand is bounded, never clamped by the guard, and never
    pressing at full range toward the table object."""

    observation = probe_observation()
    solver = SolverDouble()
    teacher = build_teacher_with(solver)
    prime_stage_captures(teacher, observation)
    teacher._phase = ScriptedInsertionStage.DESCEND
    decision = teacher.decide(observation)
    assert decision.refusal is None
    current_right_z = float(observation.right_eef.position[2])
    glide_step = teacher.settings.maximum_cartesian_step_m
    recorded_z = float(solver.calls[0]["right_position"][2])
    assert recorded_z == pytest.approx(current_right_z - glide_step)
    assert recorded_z > float(observation.peg.position[2]), (
        "the glide must bound the descend demand instead of pressing toward "
        "the object center at full range"
    )


def test_contact_phase_glide_bounds_the_demand_to_one_step() -> None:
    """Repair round 2: contact-adjacent stages inherit the scripted glide so
    a blocked arm sees its target collapse onto itself and stops pressing."""

    observation = probe_observation()
    solver = SolverDouble(offsets=(0.4,))
    teacher = build_teacher_with(solver)
    prime_stage_captures(teacher, observation)
    # A carry transform demanding a far right-arm target (position and z).
    teacher._peg_from_right_site = GeometryPose(
        position=torch.tensor([0.3, 0.0, -0.1], dtype=torch.float32),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    teacher._phase = ScriptedInsertionStage.TRANSPORT
    decision = teacher.decide(observation)
    assert decision.refusal is None
    current_right = observation.right_eef.position
    recorded = solver.calls[0]["right_position"]
    distance = float(
        torch.linalg.vector_norm(
            torch.as_tensor(recorded, dtype=torch.float32) - current_right
        )
    )
    assert distance <= teacher.settings.maximum_cartesian_step_m + 1e-6, (
        "the transport demand must be glided to at most one registered step"
    )
    assert distance > 0.0, "the lateral carry demand must survive"
    assert float(recorded[2]) == pytest.approx(
        float(current_right[2])
    ), "the height guard must hold the carried height on top of the glide"


def test_contact_phase_scale_cap_binds_in_transport_only() -> None:
    """Candidate 002 repair 2: contact-adjacent stages run at half command
    scale; free-space phases keep the full direct-command speed."""

    observation = probe_observation()
    capped_solver = SolverDouble(offsets=(0.05,))
    capped_teacher = build_teacher_with(capped_solver)
    prime_stage_captures(capped_teacher, observation)
    capped_teacher._phase = ScriptedInsertionStage.TRANSPORT
    capped_decision = capped_teacher.decide(observation)
    assert capped_decision.action is not None
    current = observation.robot_state
    assert float(capped_decision.action[0]) == pytest.approx(
        float(current[0]) + 0.025
    ), "transport command scale must be capped at the registered 0.5"

    free_solver = SolverDouble(offsets=(0.05,))
    free_teacher = build_teacher_with(free_solver)
    prime_stage_captures(free_teacher, observation)
    # Hold the ORIENT stage: a far in-place rotation target keeps the stage
    # from advancing into the capped DESCEND.
    far_pose = GeometryPose(
        position=observation.left_eef.position
        + torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32),
        quaternion=observation.left_eef.quaternion,
    )
    free_teacher._orient_left = far_pose
    free_teacher._orient_right = far_pose
    free_teacher._phase = ScriptedInsertionStage.ORIENT
    free_decision = free_teacher.decide(observation)
    assert free_decision.action is not None
    assert float(free_decision.action[0]) == pytest.approx(float(current[0]) + 0.05)


def test_factory_registers_the_measured_composition() -> None:
    """The registered candidate-003 composition: the candidate-002 G2-001
    command layer plus the stall-triggered escapes; INSERT-only branch
    stages, ALIGN escapes only on stall."""

    from rosetta_reality.sim.geometric_teacher import build_teacher
    from rosetta_reality.sim.scripted_teacher import ScriptedTeacherSettings

    teacher = build_teacher()
    assert teacher.identity == "mink-geometric-insertion-teacher-003"
    assert teacher.settings.transport_position_tolerance_m == pytest.approx(0.012)
    assert teacher.settings.align_tolerance_m == pytest.approx(0.012)
    assert teacher._BRANCH_SEED_STAGES == frozenset(
        {ScriptedInsertionStage.INSERT}
    )
    assert teacher.escape_settings.stall_window_steps >= 2
    assert ScriptedTeacherSettings().transport_position_tolerance_m == pytest.approx(
        0.012
    ), "the scripted class default must stay untouched"
