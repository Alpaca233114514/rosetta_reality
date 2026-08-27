"""Object-geometry-conditioned insertion teacher tests."""

import math

import pytest
import torch

from rosetta_reality.sim.geometry_teacher import (
    GeometryPose,
    GeometryTeacherError,
    InsertionGeometry,
    InsertionTeacherCalibration,
    InsertionTeacherPhase,
    InsertionTeacherSettings,
    ObjectGeometryInsertionTeacher,
    compose_pose,
    inverse_pose,
    relative_pose,
)
from scripts.evaluate_aloha_geometry_teacher import (
    _bounded_orientation_waypoint,
    _cartesian_waypoint,
    _quaternion_distance,
)

IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _pose(x: float, y: float, z: float) -> GeometryPose:
    return GeometryPose(torch.tensor([x, y, z]), IDENTITY)


def _calibration() -> InsertionTeacherCalibration:
    return InsertionTeacherCalibration(
        socket_to_left_eef_at_grasp=_pose(0.0, 0.0, 0.0),
        peg_to_right_eef_at_grasp=_pose(0.0, 0.0, 0.0),
        terminal_socket_to_peg=_pose(0.0, 0.0, 0.0),
        insertion_axis_in_socket=torch.tensor([1.0, 0.0, 0.0]),
        source_episode=2,
        source_seed=10,
        terminal_reward=4.0,
        terminal_success=True,
    )


def _geometry(
    *,
    left: GeometryPose | None = None,
    right: GeometryPose | None = None,
    socket: GeometryPose | None = None,
    peg: GeometryPose | None = None,
    left_gripper: float = 0.0,
    right_gripper: float = 0.0,
    reward: float = 0.0,
    grasped: bool = False,
    on_table: bool = True,
    pin_contact: bool = False,
    collisions: int = 0,
) -> InsertionGeometry:
    state = torch.zeros(14)
    state[6] = left_gripper
    state[13] = right_gripper
    return InsertionGeometry(
        robot_state=state,
        left_eef=left or _pose(-0.2, 0.5, 0.3),
        right_eef=right or _pose(0.2, 0.5, 0.3),
        socket=socket or _pose(-0.2, 0.5, 0.05),
        peg=peg or _pose(0.2, 0.5, 0.05),
        observed_reward=reward,
        socket_grasp_contact=grasped,
        peg_grasp_contact=grasped,
        socket_on_table=on_table,
        peg_on_table=on_table,
        peg_socket_contact=False,
        pin_contact=pin_contact,
        unexpected_collision_count=collisions,
    )


def test_pose_composition_and_inverse_round_trip() -> None:
    parent = GeometryPose(
        torch.tensor([0.2, -0.1, 0.3]),
        torch.tensor([0.9238795, 0.0, 0.0, 0.3826834]),
    )
    child = _pose(0.1, 0.0, 0.0)

    world = compose_pose(parent, child)

    assert torch.allclose(relative_pose(parent, world).position, child.position, atol=1e-6)
    assert torch.allclose(
        compose_pose(inverse_pose(parent), parent).position,
        torch.zeros(3),
        atol=1e-6,
    )


def test_teacher_progresses_only_from_observed_geometry_and_events() -> None:
    settings = InsertionTeacherSettings(maximum_cartesian_step_m=1.0)
    teacher = ObjectGeometryInsertionTeacher(_calibration(), settings)

    opening = teacher.decide(_geometry())
    assert opening.phase is InsertionTeacherPhase.OPEN
    assert opening.left_gripper == 1.0

    socket = _pose(-0.2, 0.5, 0.05)
    peg = _pose(0.2, 0.5, 0.05)
    left_approach = _pose(-0.2, 0.5, 0.13)
    right_approach = _pose(0.2, 0.5, 0.13)
    descending = teacher.decide(
        _geometry(
            left=left_approach,
            right=right_approach,
            socket=socket,
            peg=peg,
            left_gripper=1.0,
            right_gripper=1.0,
        )
    )
    assert descending.phase is InsertionTeacherPhase.DESCEND
    assert torch.allclose(descending.left_eef.position, socket.position)

    grasp = teacher.decide(
        _geometry(
            left=socket,
            right=peg,
            socket=socket,
            peg=peg,
            left_gripper=1.0,
            right_gripper=1.0,
        )
    )
    assert grasp.phase is InsertionTeacherPhase.GRASP
    assert grasp.left_gripper == 0.0

    lift = teacher.decide(
        _geometry(
            left=socket,
            right=peg,
            socket=socket,
            peg=peg,
            reward=1.0,
            grasped=True,
        )
    )
    assert lift.phase is InsertionTeacherPhase.LIFT
    assert lift.left_eef.position[2] == pytest.approx(
        0.05 + settings.lift_feedback_step_m
    )
    assert lift.lift_feedback_anchor is True

    lifted_socket = _pose(-0.2, 0.5, settings.lift_object_height_m)
    lifted_peg = _pose(0.2, 0.5, settings.lift_object_height_m)
    coarse = teacher.decide(
        _geometry(
            left=lifted_socket,
            right=lifted_peg,
            socket=lifted_socket,
            peg=lifted_peg,
            reward=2.0,
            grasped=True,
            on_table=False,
        )
    )
    assert coarse.phase is InsertionTeacherPhase.COARSE_ALIGN
    assert coarse.left_eef.position[0] == pytest.approx(-0.05)
    assert coarse.right_eef.position[0] == pytest.approx(0.05)

    coarse_socket = _pose(-0.05, 0.5, settings.lift_object_height_m)
    coarse_peg = _pose(0.05, 0.5, settings.lift_object_height_m)
    insert = teacher.decide(
        _geometry(
            left=coarse_socket,
            right=coarse_peg,
            socket=coarse_socket,
            peg=coarse_peg,
            reward=2.0,
            grasped=True,
            on_table=False,
        )
    )
    assert insert.phase is InsertionTeacherPhase.INSERT
    assert insert.left_eef.position[0] == pytest.approx(0.0)
    assert insert.right_eef.position[0] == pytest.approx(0.0)

    complete = teacher.decide(
        _geometry(
            left=_pose(0.0, 0.5, settings.lift_object_height_m),
            right=_pose(0.0, 0.5, settings.lift_object_height_m),
            socket=_pose(0.0, 0.5, settings.lift_object_height_m),
            peg=_pose(0.0, 0.5, settings.lift_object_height_m),
            reward=4.0,
            grasped=True,
            on_table=False,
            pin_contact=True,
        )
    )
    assert complete.phase is InsertionTeacherPhase.COMPLETE


def test_teacher_fails_closed_on_unregistered_geometry_or_collision() -> None:
    teacher = ObjectGeometryInsertionTeacher(_calibration(), InsertionTeacherSettings())

    with pytest.raises(GeometryTeacherError, match="workspace"):
        teacher.decide(_geometry(socket=_pose(-2.0, 0.5, 0.05)))
    with pytest.raises(GeometryTeacherError, match="collision"):
        teacher.decide(_geometry(collisions=1))


def test_teacher_bounds_orientation_feedback_per_decision() -> None:
    rotation = torch.tensor([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])
    calibration = InsertionTeacherCalibration(
        socket_to_left_eef_at_grasp=GeometryPose(torch.zeros(3), rotation),
        peg_to_right_eef_at_grasp=GeometryPose(torch.zeros(3), rotation),
        terminal_socket_to_peg=_pose(0.0, 0.0, 0.0),
        insertion_axis_in_socket=torch.tensor([1.0, 0.0, 0.0]),
        source_episode=2,
        source_seed=10,
        terminal_reward=4.0,
        terminal_success=True,
    )
    settings = InsertionTeacherSettings(maximum_orientation_step_rad=0.04)
    teacher = ObjectGeometryInsertionTeacher(calibration, settings)

    target = teacher.decide(
        _geometry(
            left=_pose(-0.2, 0.5, 0.13),
            right=_pose(0.2, 0.5, 0.13),
            left_gripper=1.0,
            right_gripper=1.0,
        )
    )

    assert target.phase is InsertionTeacherPhase.ORIENT
    angle = 2.0 * math.acos(float(torch.dot(IDENTITY, target.left_eef.quaternion)))
    assert angle == pytest.approx(settings.maximum_orientation_step_rad, abs=1e-5)


def test_path_planner_bounds_orientation_relaxation_before_ik() -> None:
    feasible = torch.tensor([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])

    waypoint, relaxation = _bounded_orientation_waypoint(IDENTITY, feasible, 0.04)

    assert relaxation == pytest.approx(0.04)
    assert _quaternion_distance(IDENTITY, waypoint) == pytest.approx(0.04, abs=1e-5)


def test_path_planner_backs_off_position_without_changing_requested_endpoint() -> None:
    current = _pose(0.0, 0.0, 0.0)
    requested = _pose(0.012, 0.0, 0.0)

    waypoint = _cartesian_waypoint(current, requested, 0.25, IDENTITY)

    assert waypoint.position.tolist() == pytest.approx([0.003, 0.0, 0.0])
    assert requested.position.tolist() == pytest.approx([0.012, 0.0, 0.0])
