"""Scripted-teacher candidate tests: stage machine, refusals, determinism.

Covers the candidate of
``reports/training/m2-smolvla-t2-scripted-teacher-candidate-design-2026-08-29.md``
against the frozen teacher contract: state-conditioned stage advancement with
no time input, explicit refusal regions, the registered joint-target clamp,
replay determinism, and — inside the pinned container — the real constrained
solver and the live-environment observer semantics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPOSITORY_ROOT / "src"),):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import rosetta_reality.sim.scripted_teacher as scripted_teacher  # noqa: E402
from rosetta_reality.sim.geometry_teacher import (  # noqa: E402
    GeometryPose,
    InsertionGeometry,
    compose_pose,
    quaternion_multiply,
    relative_pose,
)
from rosetta_reality.sim.mink_aloha_ik import MinkAlohaIkResult  # noqa: E402
from rosetta_reality.sim.scripted_teacher import (  # noqa: E402
    ScriptedInsertionStage,
    ScriptedInsertionTeacher,
    ScriptedTeacherSettings,
)
from rosetta_reality.sim.teacher_gate import (  # noqa: E402
    TeacherRefusalReason,
    assert_observation_contract,
)


@dataclass
class RecordingSolver:
    """Deterministic solver double: records targets, holds the current pose."""

    failure: str | None = None
    offset: float = 0.0

    def __post_init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def solve(
        self,
        current_qpos: np.ndarray,
        *,
        left_position: np.ndarray,
        left_quaternion_wxyz: np.ndarray,
        right_position: np.ndarray,
        right_quaternion_wxyz: np.ndarray,
    ) -> MinkAlohaIkResult:
        if self.failure is not None:
            raise RuntimeError(self.failure)
        self.calls.append(
            {
                "current_qpos": np.array(current_qpos, copy=True),
                "left_position": np.array(left_position, copy=True),
                "left_quaternion_wxyz": np.array(left_quaternion_wxyz, copy=True),
                "right_position": np.array(right_position, copy=True),
                "right_quaternion_wxyz": np.array(right_quaternion_wxyz, copy=True),
            }
        )
        qpos = np.array(current_qpos, dtype=np.float64).copy()
        qpos[0] += self.offset
        return MinkAlohaIkResult(
            qpos=qpos,
            iterations=1,
            position_errors_m=(0.0, 0.0),
            orientation_errors_rad=(0.0, 0.0),
        )


def _stub_expanded(logical: torch.Tensor) -> np.ndarray:
    value = logical.detach().to(torch.float64).cpu().numpy()
    return np.concatenate((value[:6], [0.021, -0.021], value[7:13], [0.057, -0.057]))


class BranchSelectiveSolver:
    """Deterministic solver double with per-orientation branch stalls.

    Requests whose left quaternion matches a stall entry return a persistent
    5 cm residual (the IK-branch trap measured by the registered alignment
    diagnostic); every other request converges exactly.
    """

    def __init__(
        self,
        stall_quaternions: list[torch.Tensor] | None = None,
        *,
        stall_all: bool = False,
        offset: float = 0.0,
    ) -> None:
        self.stall_quaternions = stall_quaternions or []
        self.stall_all = stall_all
        self.offset = offset
        self.calls: list[dict[str, np.ndarray]] = []

    def solve(
        self,
        current_qpos: np.ndarray,
        *,
        left_position: np.ndarray,
        left_quaternion_wxyz: np.ndarray,
        right_position: np.ndarray,
        right_quaternion_wxyz: np.ndarray,
    ) -> MinkAlohaIkResult:
        requested = torch.as_tensor(left_quaternion_wxyz, dtype=torch.float32)
        stalled = self.stall_all or any(
            abs(float(torch.dot(requested, stall_quaternion))) > 0.999
            for stall_quaternion in self.stall_quaternions
        )
        self.calls.append(
            {
                "current_qpos": np.array(current_qpos, copy=True),
                "left_position": np.array(left_position, copy=True),
                "left_quaternion_wxyz": np.array(left_quaternion_wxyz, copy=True),
                "right_position": np.array(right_position, copy=True),
                "right_quaternion_wxyz": np.array(right_quaternion_wxyz, copy=True),
            }
        )
        qpos = np.array(current_qpos, dtype=np.float64).copy()
        qpos[0] += self.offset
        if stalled:
            return MinkAlohaIkResult(
                qpos=qpos,
                iterations=1,
                position_errors_m=(0.05, 0.0),
                orientation_errors_rad=(0.05, 0.0),
            )
        return MinkAlohaIkResult(
            qpos=qpos,
            iterations=1,
            position_errors_m=(0.0, 0.0),
            orientation_errors_rad=(0.0, 0.0),
        )


@pytest.fixture()
def stub_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scripted_teacher, "_expanded_robot_qpos", _stub_expanded)


def _pose(x: float, y: float, z: float) -> GeometryPose:
    return GeometryPose(
        position=torch.tensor([x, y, z], dtype=torch.float32),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )


def _oriented_pose(x: float, y: float, z: float, quaternion: torch.Tensor) -> GeometryPose:
    return GeometryPose(
        position=torch.tensor([x, y, z], dtype=torch.float32),
        quaternion=quaternion,
    )


SOCKET_POSITION = (-0.10, 0.10, 0.02)
PEG_POSITION = (0.10, 0.10, 0.03)
# Registered world-frame grasp tilts: +60 degrees (left) and +120 degrees
# (right) about world Y.
GRASP_LEFT_QUATERNION = scripted_teacher._axis_angle_quaternion(
    (0.0, 1.0, 0.0), 60.0
)
GRASP_RIGHT_QUATERNION = scripted_teacher._axis_angle_quaternion(
    (0.0, 1.0, 0.0), 120.0
)


def _geometry(**overrides) -> InsertionGeometry:
    fields: dict = {
        "robot_state": torch.zeros(14, dtype=torch.float32),
        "left_eef": _pose(-0.10, 0.10, 0.05),
        "right_eef": _pose(0.10, 0.10, 0.05),
        "socket": _pose(*SOCKET_POSITION),
        "peg": _pose(*PEG_POSITION),
        "observed_reward": 0.0,
        "socket_grasp_contact": False,
        "peg_grasp_contact": False,
        "socket_on_table": True,
        "peg_on_table": True,
        "peg_socket_contact": False,
        "pin_contact": False,
        "unexpected_collision_count": 0,
    }
    fields.update(overrides)
    return InsertionGeometry(**fields)


def _state_with_grippers(left: float, right: float) -> torch.Tensor:
    state = torch.zeros(14, dtype=torch.float32)
    state[6] = left
    state[13] = right
    return state


_OPEN_STATE = _state_with_grippers(0.0, 0.0)
_OPEN_GRIPPER_STATE = _state_with_grippers(0.9, 0.9)
# Unbounded glide keeps the recorded IK targets exactly at the stage targets;
# the glide bound itself is covered by a dedicated test below.
_TEST_SETTINGS = ScriptedTeacherSettings(
    maximum_cartesian_step_m=10.0,
    maximum_orientation_step_rad=10.0,
)


def _stage_walk_observations() -> list[InsertionGeometry]:
    """Observations that advance the stage machine exactly one stage each."""

    return [
        _geometry(robot_state=_OPEN_STATE),
        _geometry(robot_state=_OPEN_GRIPPER_STATE),
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_pose(-0.10, 0.10, 0.10),
            right_eef=_pose(0.10, 0.10, 0.11),
        ),
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.10, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.10, 0.10, 0.11, GRASP_RIGHT_QUATERNION),
        ),
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.02, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.10, 0.10, 0.03, GRASP_RIGHT_QUATERNION),
        ),
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.02, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.10, 0.10, 0.03, GRASP_RIGHT_QUATERNION),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
        ),
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.15, 0.50, 0.15, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.06, 0.10, 0.035, GRASP_RIGHT_QUATERNION),
            peg=_pose(0.06, 0.10, 0.035),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
        ),
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.15, 0.50, 0.15, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.06, 0.10, 0.02, GRASP_RIGHT_QUATERNION),
            peg=_pose(0.06, 0.10, 0.02),
            socket=_pose(-0.10, 0.10, 0.12),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
        ),
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.02, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.06, 0.10, 0.02, GRASP_RIGHT_QUATERNION),
            peg=_pose(0.06, 0.10, 0.02),
            socket=_pose(-0.10, 0.10, 0.02),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
            pin_contact=True,
        ),
    ]


def _walk_to(teacher: ScriptedInsertionTeacher, stage: ScriptedInsertionStage) -> None:
    """Advance a fresh teacher to the requested stage of the walk."""

    for observation in _stage_walk_observations():
        if teacher.phase is stage:
            return
        teacher.decide(observation)
    if teacher.phase is not stage:
        raise AssertionError(f"walk never reached stage {stage.value}")


def test_observation_contract_stays_frozen() -> None:
    assert_observation_contract()


def test_settings_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        ScriptedTeacherSettings(maximum_joint_target_delta_rad=float("nan"))
    with pytest.raises(ValueError, match="positive"):
        ScriptedTeacherSettings(maximum_joint_target_delta_rad=0.0)
    with pytest.raises(ValueError, match="positive"):
        ScriptedTeacherSettings(alignment_probe_maximum_iterations=0)
    with pytest.raises(ValueError, match="workspace"):
        ScriptedTeacherSettings(workspace_y_min_m=0.9, workspace_y_max_m=0.1)


def test_open_stage_opens_gripper_and_holds_home(stub_expansion) -> None:
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=RecordingSolver())
    decision = teacher.decide(_geometry())
    assert decision.refusal is None
    assert decision.action is not None
    assert teacher.phase is ScriptedInsertionStage.OPEN
    assert float(decision.action[6]) == pytest.approx(0.90, abs=1e-6)
    assert float(decision.action[13]) == pytest.approx(0.90, abs=1e-6)
    assert float(decision.action[0]) == 0.0
    assert float(decision.action[7]) == 0.0


def test_gripper_state_advances_to_approach_with_captured_objects(stub_expansion) -> None:
    solver = RecordingSolver()
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=solver)
    teacher.decide(_geometry())
    decision = teacher.decide(_geometry(robot_state=_OPEN_GRIPPER_STATE))
    assert teacher.phase is ScriptedInsertionStage.APPROACH
    assert decision.action is not None
    assert float(decision.action[6]) == pytest.approx(0.90, abs=1e-6)
    call = solver.calls[-1]
    np.testing.assert_allclose(call["left_position"], (-0.10, 0.10, 0.10), atol=1e-6)
    np.testing.assert_allclose(call["right_position"], (0.10, 0.10, 0.11), atol=1e-6)
    capture = teacher.capture
    np.testing.assert_allclose(
        capture.socket_position.numpy(), SOCKET_POSITION, atol=1e-6
    )
    np.testing.assert_allclose(capture.peg_position.numpy(), PEG_POSITION, atol=1e-6)
    np.testing.assert_allclose(
        capture.grasp_left_quaternion.numpy(),
        GRASP_LEFT_QUATERNION.numpy(),
        atol=1e-6,
    )


def test_orient_requires_orientation_before_descending(stub_expansion) -> None:
    solver = RecordingSolver()
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=solver)
    _walk_to(teacher, ScriptedInsertionStage.ORIENT)
    decision = teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_pose(-0.10, 0.10, 0.10),
            right_eef=_pose(0.10, 0.10, 0.11),
        )
    )
    assert teacher.phase is ScriptedInsertionStage.ORIENT
    assert decision.action is not None
    call = solver.calls[-1]
    np.testing.assert_allclose(call["left_position"], (-0.10, 0.10, 0.10), atol=1e-6)
    decision = teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.10, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.10, 0.10, 0.11, GRASP_RIGHT_QUATERNION),
        )
    )
    assert teacher.phase is ScriptedInsertionStage.DESCEND
    call = solver.calls[-1]
    np.testing.assert_allclose(call["left_position"], (-0.10, 0.10, 0.02), atol=1e-6)
    np.testing.assert_allclose(call["right_position"], (0.10, 0.10, 0.03), atol=1e-6)


def test_grasp_contacts_advance_to_transport(stub_expansion) -> None:
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=RecordingSolver())
    _walk_to(teacher, ScriptedInsertionStage.DESCEND)
    decision = teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.02, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.10, 0.10, 0.03, GRASP_RIGHT_QUATERNION),
        )
    )
    assert teacher.phase is ScriptedInsertionStage.GRASP
    assert decision.action is not None
    assert float(decision.action[6]) == pytest.approx(0.09, abs=1e-6)
    decision = teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.02, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.10, 0.10, 0.03, GRASP_RIGHT_QUATERNION),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
        )
    )
    assert teacher.phase is ScriptedInsertionStage.TRANSPORT
    assert decision.action is not None
    assert float(decision.action[6]) == pytest.approx(0.09, abs=1e-6)


def test_descent_stall_advances_to_grasp(stub_expansion) -> None:
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=RecordingSolver())
    _walk_to(teacher, ScriptedInsertionStage.DESCEND)
    low = _geometry(
        robot_state=_OPEN_GRIPPER_STATE,
        left_eef=_oriented_pose(-0.10, 0.10, 0.05, GRASP_LEFT_QUATERNION),
        right_eef=_oriented_pose(0.10, 0.10, 0.05, GRASP_RIGHT_QUATERNION),
    )
    teacher.decide(low)
    assert teacher.phase is ScriptedInsertionStage.DESCEND
    decision = teacher.decide(low)
    assert teacher.phase is ScriptedInsertionStage.GRASP
    assert decision.action is not None
    assert float(decision.action[6]) == pytest.approx(0.09, abs=1e-6)


def test_socket_approach_align_insert_move_socket_onto_peg(stub_expansion) -> None:
    solver = RecordingSolver()
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=solver)
    _walk_to(teacher, ScriptedInsertionStage.TRANSPORT)
    decision = teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.15, 0.50, 0.15, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.06, 0.10, 0.02, GRASP_RIGHT_QUATERNION),
            peg=_pose(0.06, 0.10, 0.02),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
        )
    )
    assert teacher.phase is ScriptedInsertionStage.SOCKET_APPROACH
    call = solver.calls[-1]
    # SOCKET_APPROACH: the socket hovers above the insertion axis, composed
    # from the frozen peg reference and the in-cage transform captured at the
    # stage entry.
    entry_left = _pose(-0.15, 0.50, 0.15)
    entry_socket = _pose(-0.10, 0.10, 0.02)
    entry_peg = _pose(0.06, 0.10, 0.02)
    expected_hover = compose_pose(
        compose_pose(
            entry_peg,
            GeometryPose(
                position=torch.tensor([-0.16, 0.0, 0.10], dtype=torch.float32),
                quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            ),
        ),
        relative_pose(entry_socket, entry_left),
    )
    np.testing.assert_allclose(
        call["left_position"], expected_hover.position.numpy(), atol=1e-6
    )
    np.testing.assert_allclose(call["right_position"], (0.06, 0.10, 0.02), atol=1e-6)
    decision = teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.12, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.06, 0.10, 0.02, GRASP_RIGHT_QUATERNION),
            peg=_pose(0.06, 0.10, 0.02),
            socket=_pose(-0.10, 0.10, 0.12),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
        )
    )
    assert teacher.phase is ScriptedInsertionStage.ALIGN
    call = solver.calls[-1]
    # ALIGN: the socket descends onto the peg's axis at the park distance.
    expected_park = compose_pose(
        compose_pose(
            entry_peg,
            GeometryPose(
                position=torch.tensor([-0.16, 0.0, 0.0], dtype=torch.float32),
                quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            ),
        ),
        relative_pose(entry_socket, entry_left),
    )
    np.testing.assert_allclose(
        call["left_position"], expected_park.position.numpy(), atol=1e-6
    )
    decision = teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.02, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.06, 0.10, 0.02, GRASP_RIGHT_QUATERNION),
            peg=_pose(0.06, 0.10, 0.02),
            socket=_pose(-0.10, 0.10, 0.02),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
        )
    )
    assert teacher.phase is ScriptedInsertionStage.INSERT
    call = solver.calls[-1]
    # INSERT: the socket slides to peg - 0.09 along the axis; the peg tip
    # presses the pin seated at the socket center.
    expected_insert = compose_pose(
        compose_pose(
            entry_peg,
            GeometryPose(
                position=torch.tensor([-0.09, 0.0, 0.0], dtype=torch.float32),
                quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            ),
        ),
        relative_pose(entry_socket, entry_left),
    )
    np.testing.assert_allclose(
        call["left_position"], expected_insert.position.numpy(), atol=1e-6
    )
    decision = teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.02, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.06, 0.10, 0.02, GRASP_RIGHT_QUATERNION),
            peg=_pose(0.06, 0.10, 0.02),
            socket=_pose(-0.10, 0.10, 0.02),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
            pin_contact=True,
        )
    )
    assert teacher.phase is ScriptedInsertionStage.COMPLETE
    assert decision.action is not None
    assert float(decision.action[6]) == pytest.approx(0.09, abs=1e-6)
    decision = teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.10, 0.10, 0.02, GRASP_LEFT_QUATERNION),
            right_eef=_oriented_pose(0.06, 0.10, 0.02, GRASP_RIGHT_QUATERNION),
            peg=_pose(0.06, 0.10, 0.02),
            socket=_pose(-0.10, 0.10, 0.02),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
            observed_reward=4.0,
        )
    )
    assert teacher.phase is ScriptedInsertionStage.COMPLETE
    assert decision.action is not None


_FLIP_QUATERNION = scripted_teacher._axis_angle_quaternion((0.0, 0.0, 1.0), 180.0)
_ENTRY_LEFT = _oriented_pose(-0.15, 0.50, 0.15, GRASP_LEFT_QUATERNION)
_ENTRY_PEG = _pose(0.06, 0.10, 0.02)


def _transport_exit_observation(socket: GeometryPose) -> InsertionGeometry:
    """Transport-exit walk observation.

    The socket position selects how far the stage walk cascades within one
    decide: at the park position the flipped candidate would advance past
    ALIGN, so each test passes the pose that holds the stage it asserts.
    """

    return _geometry(
        robot_state=_OPEN_GRIPPER_STATE,
        left_eef=_ENTRY_LEFT,
        right_eef=_oriented_pose(0.06, 0.10, 0.02, GRASP_RIGHT_QUATERNION),
        peg=_pose(0.06, 0.10, 0.02),
        socket=socket,
        socket_grasp_contact=True,
        peg_grasp_contact=True,
    )


def _expected_parallel_site(offset_m: float, entry_socket: GeometryPose) -> GeometryPose:
    cage = relative_pose(entry_socket, _ENTRY_LEFT)
    park = compose_pose(
        _ENTRY_PEG,
        GeometryPose(
            position=torch.tensor([-offset_m, 0.0, 0.0], dtype=torch.float32),
            quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        ),
    )
    return compose_pose(park, cage)


def _same_rotation(first: torch.Tensor, second: torch.Tensor) -> bool:
    return abs(float(torch.dot(first, second))) > 1.0 - 1e-6


def test_alignment_multistart_keeps_parallel_when_it_converges(stub_expansion) -> None:
    solver = RecordingSolver()
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=solver)
    _walk_to(teacher, ScriptedInsertionStage.TRANSPORT)
    decision = teacher.decide(
        _transport_exit_observation(socket=_pose(-0.10, 0.10, 0.02))
    )
    assert decision.refusal is None
    assert teacher.phase is ScriptedInsertionStage.SOCKET_APPROACH
    assert teacher._alignment_flipped is False


def test_alignment_multistart_flips_when_parallel_stalls(stub_expansion) -> None:
    solver = BranchSelectiveSolver(
        stall_quaternions=[torch.tensor([1.0, 0.0, 0.0, 0.0]), GRASP_LEFT_QUATERNION]
    )
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=solver)
    _walk_to(teacher, ScriptedInsertionStage.TRANSPORT)
    decision = teacher.decide(
        _transport_exit_observation(socket=_pose(-0.10, 0.10, 0.05))
    )
    assert decision.refusal is None
    # The flipped wrist bypasses the unreachable hover climb.
    assert teacher.phase is ScriptedInsertionStage.ALIGN
    assert teacher._alignment_flipped is True
    parallel_site = _expected_parallel_site(0.16, _pose(-0.10, 0.10, 0.05))
    flipped_quaternion = quaternion_multiply(_FLIP_QUATERNION, parallel_site.quaternion)
    call = solver.calls[-1]
    # The measured convergent branch: parallel-composed site position, wrist
    # orientation flipped 180 degrees about world Z.
    np.testing.assert_allclose(
        call["left_position"], parallel_site.position.numpy(), atol=1e-6
    )
    assert _same_rotation(
        torch.as_tensor(call["left_quaternion_wxyz"]), flipped_quaternion
    )


def test_flipped_alignment_insert_keeps_parallel_position(stub_expansion) -> None:
    solver = BranchSelectiveSolver(
        stall_quaternions=[torch.tensor([1.0, 0.0, 0.0, 0.0]), GRASP_LEFT_QUATERNION]
    )
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=solver)
    _walk_to(teacher, ScriptedInsertionStage.TRANSPORT)
    teacher.decide(_transport_exit_observation(socket=_pose(-0.10, 0.10, 0.02)))
    teacher.decide(
        _geometry(
            robot_state=_OPEN_GRIPPER_STATE,
            left_eef=_oriented_pose(-0.15, 0.50, 0.15, _FLIP_QUATERNION),
            right_eef=_oriented_pose(0.06, 0.10, 0.02, GRASP_RIGHT_QUATERNION),
            socket=_pose(-0.10, 0.10, 0.02),
            peg=_pose(0.06, 0.10, 0.02),
            socket_grasp_contact=True,
            peg_grasp_contact=True,
        )
    )
    assert teacher.phase is ScriptedInsertionStage.INSERT
    parallel_site = _expected_parallel_site(0.09, _pose(-0.10, 0.10, 0.02))
    flipped_quaternion = quaternion_multiply(_FLIP_QUATERNION, parallel_site.quaternion)
    call = solver.calls[-1]
    np.testing.assert_allclose(
        call["left_position"], parallel_site.position.numpy(), atol=1e-6
    )
    assert _same_rotation(
        torch.as_tensor(call["left_quaternion_wxyz"]), flipped_quaternion
    )


def test_alignment_multistart_keeps_parallel_when_neither_converges(stub_expansion) -> None:
    solver = BranchSelectiveSolver(stall_all=True)
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=solver)
    _walk_to(teacher, ScriptedInsertionStage.TRANSPORT)
    decision = teacher.decide(
        _transport_exit_observation(socket=_pose(-0.10, 0.10, 0.02))
    )
    assert decision.refusal is None
    assert teacher.phase is ScriptedInsertionStage.SOCKET_APPROACH
    assert teacher._alignment_flipped is False


def test_ik_target_glides_toward_stage_target(stub_expansion) -> None:
    solver = RecordingSolver()
    teacher = ScriptedInsertionTeacher(settings=ScriptedTeacherSettings(), solver=solver)
    _walk_to(teacher, ScriptedInsertionStage.APPROACH)
    observation = _geometry(robot_state=_OPEN_GRIPPER_STATE)
    decision = teacher.decide(observation)
    assert decision.action is not None
    left_target, _right_target = teacher._approach_targets()
    glided = solver.calls[-1]["left_position"]
    current = observation.left_eef.position.numpy()
    stage = left_target.position.numpy()
    step = float(np.linalg.norm(glided - current))
    assert step <= ScriptedTeacherSettings().maximum_cartesian_step_m + 1e-6
    assert step > 0.0
    remaining_before = float(np.linalg.norm(stage - current))
    remaining_after = float(np.linalg.norm(stage - glided))
    assert remaining_after < remaining_before


def test_collision_refuses_unsafe_state_without_advancing(stub_expansion) -> None:
    solver = RecordingSolver()
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=solver)
    decision = teacher.decide(_geometry(unexpected_collision_count=2))
    assert decision.refusal is not None
    assert decision.refusal.reason is TeacherRefusalReason.UNSAFE_STATE
    assert decision.action is None
    assert teacher.phase is ScriptedInsertionStage.OPEN
    assert solver.calls == []


def test_workspace_violation_refuses(stub_expansion) -> None:
    teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=RecordingSolver())
    decision = teacher.decide(_geometry(socket=_pose(0.0, 5.0, 0.0)))
    assert decision.refusal is not None
    assert decision.refusal.reason is TeacherRefusalReason.WORKSPACE_VIOLATION
    assert "socket" in decision.refusal.detail


def test_ik_failure_refuses_out_of_support(stub_expansion) -> None:
    teacher = ScriptedInsertionTeacher(
        settings=_TEST_SETTINGS, solver=RecordingSolver(failure="qp boom")
    )
    decision = teacher.decide(_geometry(robot_state=_OPEN_GRIPPER_STATE))
    assert decision.refusal is not None
    assert decision.refusal.reason is TeacherRefusalReason.OUT_OF_SUPPORT
    assert "qp boom" in decision.refusal.detail


def test_joint_target_delta_is_clamped(stub_expansion) -> None:
    settings = ScriptedTeacherSettings()
    teacher = ScriptedInsertionTeacher(
        solver=RecordingSolver(offset=1.0), settings=settings
    )
    decision = teacher.decide(_geometry(robot_state=_OPEN_GRIPPER_STATE))
    assert decision.action is not None
    assert float(decision.action[0]) == pytest.approx(
        settings.maximum_joint_target_delta_rad, abs=1e-6
    )
    assert float(decision.action[7]) == 0.0


def test_replay_is_byte_deterministic(stub_expansion) -> None:
    def run_sequence() -> list[bytes]:
        teacher = ScriptedInsertionTeacher(settings=_TEST_SETTINGS, solver=RecordingSolver())
        trace: list[bytes] = []
        observations = _stage_walk_observations() + [
            _geometry(
                robot_state=_OPEN_GRIPPER_STATE,
                left_eef=_oriented_pose(-0.15, 0.50, 0.15, GRASP_LEFT_QUATERNION),
                right_eef=_oriented_pose(0.06, 0.10, 0.035, GRASP_RIGHT_QUATERNION),
                socket_grasp_contact=True,
                peg_grasp_contact=True,
                pin_contact=True,
            ),
            _geometry(
                robot_state=_OPEN_GRIPPER_STATE,
                left_eef=_oriented_pose(-0.15, 0.50, 0.15, GRASP_LEFT_QUATERNION),
                right_eef=_oriented_pose(0.06, 0.10, 0.035, GRASP_RIGHT_QUATERNION),
                socket_grasp_contact=True,
                peg_grasp_contact=True,
                observed_reward=4.0,
            ),
        ]
        for observation in observations:
            decision = teacher.decide(observation)
            assert decision.action is not None
            trace.append(decision.action.numpy().tobytes())
            trace.append(teacher.phase.value.encode())
        return trace

    assert run_sequence() == run_sequence()


def test_real_constrained_solver_is_deterministic() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    pytest.importorskip("gym_aloha")

    teacher = ScriptedInsertionTeacher(solver=None)
    decision_first = teacher.decide(_geometry(robot_state=_OPEN_GRIPPER_STATE))
    assert decision_first.action is not None
    assert bool(torch.isfinite(decision_first.action).all())
    teacher.reset()
    decision_second = teacher.decide(_geometry(robot_state=_OPEN_GRIPPER_STATE))
    assert decision_second.action is not None
    assert torch.equal(decision_first.action, decision_second.action)


def test_observer_extracts_frozen_fields_from_live_environment() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gym_aloha")

    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment
    from rosetta_reality.sim.scripted_teacher_observer import build_observation

    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=4)
    try:
        environment.reset(seed=10)
        geometry = build_observation(environment, contract)
        assert geometry.robot_state.shape == (14,)
        assert bool(torch.isfinite(geometry.robot_state).all())
        assert 0.0 <= float(geometry.robot_state[6]) <= 1.0
        for pose in (geometry.left_eef, geometry.right_eef, geometry.socket, geometry.peg):
            assert bool(torch.isfinite(pose.position).all())
            assert bool(torch.isfinite(pose.quaternion).all())
        assert geometry.observed_reward >= 0.0

        hold = geometry.robot_state.clone()
        observation, reward, _done, _info = environment.step(hold)
        geometry_after = build_observation(environment, contract)
        assert geometry_after.observed_reward == pytest.approx(float(reward), abs=1e-6)
        assert torch.allclose(
            geometry_after.robot_state, observation["robot_state"], atol=1e-5
        )
    finally:
        environment.close()


def test_observer_rejects_foreign_contract() -> None:
    from rosetta_reality.sim.scripted_teacher_observer import build_observation

    with pytest.raises(ValueError, match="14-D"):
        build_observation(object(), SimpleNamespace(dimension=7))


def test_reset_clears_orient_capture_targets() -> None:
    teacher = ScriptedInsertionTeacher(solver=RecordingSolver())
    pose = GeometryPose(
        position=torch.zeros(3, dtype=torch.float32),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    teacher._orient_left = pose
    teacher._orient_right = pose
    teacher.reset()
    assert teacher._orient_left is None
    assert teacher._orient_right is None
