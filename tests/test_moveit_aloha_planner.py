"""Protocol tests for the isolated official MoveIt ALOHA process client."""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

import scripts.evaluate_aloha_geometry_teacher as geometry_evaluator
from rosetta_reality.sim.action_contract import load_action_contract
from rosetta_reality.sim.geometry_teacher import (
    GeometryPose,
    InsertionTaskSpaceTarget,
    InsertionTeacherPhase,
)
from rosetta_reality.sim.moveit_aloha_planner import (
    EXPECTED_JOINT_NAMES,
    MoveItAlohaPlanner,
    MoveItAlohaPlannerError,
    MoveItAlohaPlannerSettings,
    MoveItAlohaPlanningError,
    MoveItAlohaPlanResult,
    MoveItAlohaTrajectoryExecutor,
)
from rosetta_reality.sim.mujoco_position_feedforward import (
    MujocoPositionFeedforwardResult,
)
from scripts.evaluate_aloha_geometry_teacher import (
    IkActionResult,
    _ik_action,
    _moveit_path_action,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

FAKE_SIDECAR = r'''#!/usr/bin/env python3
import json
import sys

JOINTS = __JOINTS__

for line in sys.stdin:
    request = json.loads(line)
    command = request["command"]
    request_id = request["request_id"]
    if command == "identity":
        response = {
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
            "joint_names": JOINTS,
            "collision_geometry_link_count": 28,
            "collision_geometry_shape_count": 30,
        }
    elif command == "plan":
        if request["targets"]["left"]["position"][0] < 0.0:
            response = {
                "status": "error",
                "reason": "bimanual_lma_ik_failed",
                "ik_task_mode": request["ik_task_mode"],
                "ik_search_mode": request["ik_search_mode"],
                "ik_candidate_selection_mode": (
                    "deterministic_maximum_minimum_joint_limit_margin_v1"
                ),
                "ik_seed": request["ik_seed"],
                "ik_maximum_attempts": request["ik_maximum_attempts"],
                "ik_attempts_used": request["ik_maximum_attempts"],
                "valid_ik_candidate_count": 0,
                "ik_outer_timeout_s": request["ik_timeout_s"],
                "maximum_orientation_relaxation_rad": request[
                    "maximum_orientation_relaxation_rad"
                ],
                "position_priority_ompl_seed_reset_per_request": (
                    request["ik_task_mode"] == "position_priority"
                ),
                "position_priority_terminal_goal_normalized": (
                    request["ik_task_mode"] == "position_priority"
                ),
                "maximum_terminal_goal_normalization_rad": (
                    1e-6 if request["ik_task_mode"] == "position_priority" else 0.0
                ),
                "terminal_goal_normalization_limit_rad": 1e-5,
            }
        else:
            requested_start = request["start"]
            start = list(requested_start)
            reconciliations = []
            if start[5] > 3.14158:
                reconciled = 3.14158
                delta = start[5] - reconciled
                reconciliations.append({
                    "joint_name": "left_wrist_rotate",
                    "requested_position_rad": start[5],
                    "reconciled_position_rad": reconciled,
                    "delta_rad": delta,
                })
                start[5] = reconciled
            next_positions = list(start)
            recovery = start[11] < -3.13
            if recovery:
                next_positions[11] += 0.001
            else:
                next_positions[0] += 0.1
            minimum_start_margin = 0.00658 if recovery else 0.5
            minimum_next_margin = 0.00758 if recovery else 0.2
            selected_maximum_start_delta = max(
                abs(0.2 - value) for value in requested_start
            )
            response = {
                "status": "ok",
                "backend": "moveit2_ompl",
                "planner_plugin": "ompl_interface/OMPLPlanner",
                "planner_id": "RRTConnect",
                "joint_names": JOINTS,
                "ik_task_mode": request["ik_task_mode"],
                "maximum_orientation_relaxation_rad": request[
                    "maximum_orientation_relaxation_rad"
                ],
                "position_priority_ompl_seed_reset_per_request": (
                    request["ik_task_mode"] == "position_priority"
                ),
                "position_priority_terminal_goal_normalized": (
                    request["ik_task_mode"] == "position_priority"
                ),
                "maximum_terminal_goal_normalization_rad": (
                    1e-6 if request["ik_task_mode"] == "position_priority" else 0.0
                ),
                "terminal_goal_normalization_limit_rad": 1e-5,
                "ik_search_mode": request["ik_search_mode"],
                "ik_candidate_selection_mode": (
                    "deterministic_maximum_minimum_joint_limit_margin_v1"
                ),
                "ik_seed": request["ik_seed"],
                "ik_maximum_attempts": request["ik_maximum_attempts"],
                "ik_attempts_used": 3,
                "valid_ik_candidate_count": 2,
                "selected_ik_attempt": 2,
                "selected_ik_minimum_joint_limit_margin_rad": 0.25,
                "selected_ik_maximum_start_delta_rad": selected_maximum_start_delta,
                "ik_outer_timeout_s": request["ik_timeout_s"],
                "goal": [0.2] * 12,
                "next": next_positions,
                "planning_time_s": 0.01,
                "waypoint_count": 4 if recovery else 2,
                "path_length_rad": 0.3,
                "path_maximum_waypoint_joint_delta_rad": 0.2,
                "first_segment_interpolation": 0.5,
                "maximum_goal_position_error_m": 1e-6,
                "maximum_goal_orientation_error_rad": 2e-6,
                "maximum_goal_weighted_error": 1.4e-6,
                "joint_path_constraint_type": "moveit_msgs/JointConstraint",
                "joint_path_constraint_count": 12,
                "joint_limit_margin_rad": request["joint_limit_margin_rad"],
                "physical_joint_limit_margin_rad": request[
                    "physical_joint_limit_margin_rad"
                ],
                "start_state_satisfies_joint_path_constraint": not recovery,
                "start_state_path_constraint_recovery": recovery,
                "planning_request_adapters": [
                    "default_planner_request_adapters/FixStartStatePathConstraints"
                ],
                "adapter_added_state_indices": [0, 1] if recovery else [],
                "adapter_prefix_waypoint_count": 2 if recovery else 0,
                "minimum_recovery_progress_rad": 0.001 if recovery else 0.0,
                "minimum_start_joint_limit_margin_rad": minimum_start_margin,
                "minimum_goal_joint_limit_margin_rad": 0.25,
                "minimum_path_joint_limit_margin_rad": (
                    minimum_start_margin if recovery else 0.1
                ),
                "minimum_constrained_path_joint_limit_margin_rad": 0.1,
                "minimum_adapter_prefix_physical_joint_limit_margin_rad": (
                    minimum_start_margin if recovery else 0.5
                ),
                "minimum_next_joint_limit_margin_rad": minimum_next_margin,
                "maximum_requested_start_to_next_joint_delta_rad": max(
                    abs(value - requested)
                    for value, requested in zip(next_positions, requested_start)
                ),
                "start_bound_reconciliations": reconciliations,
                "maximum_start_bound_reconciliation_rad": max(
                    (item["delta_rad"] for item in reconciliations), default=0.0
                ),
                "start_bound_reconciliation_tolerance_rad": request[
                    "start_bound_reconciliation_tolerance_rad"
                ],
            }
            if request.get("include_trajectory", False):
                goal = response["goal"]
                trajectory = [start, next_positions, goal]
                segment_deltas = [
                    [value - previous for value, previous in zip(current, prior)]
                    for prior, current in zip(trajectory, trajectory[1:])
                ]
                response["trajectory"] = trajectory
                response["waypoint_count"] = len(trajectory)
                response["path_length_rad"] = sum(
                    sum(value * value for value in delta) ** 0.5
                    for delta in segment_deltas
                )
                response["path_maximum_waypoint_joint_delta_rad"] = max(
                    abs(value) for delta in segment_deltas for value in delta
                )
                response["first_segment_interpolation"] = 1.0
    elif command == "shutdown":
        response = {"status": "ok", "shutdown": True}
    else:
        response = {"status": "error", "reason": "unknown_command"}
    response["request_id"] = request_id
    print(json.dumps(response), flush=True)
    if command == "shutdown":
        break
'''.replace("__JOINTS__", repr(list(EXPECTED_JOINT_NAMES)))


def _settings(tmp_path: Path) -> MoveItAlohaPlannerSettings:
    executable = tmp_path / "fake_moveit.py"
    executable.write_text(FAKE_SIDECAR, encoding="utf-8", newline="\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    urdf = tmp_path / "robot.urdf"
    srdf = tmp_path / "robot.srdf"
    urdf.write_text("<robot name='fake'/>", encoding="utf-8")
    srdf.write_text("<robot name='fake'/>", encoding="utf-8")
    return MoveItAlohaPlannerSettings(
        executable=executable,
        urdf=urdf,
        srdf=srdf,
        stderr_log=tmp_path / "moveit.stderr.log",
        response_timeout_s=1.0,
        launcher=(sys.executable,),
    )


def _plan(
    planner: MoveItAlohaPlanner,
    *,
    left_x: float = 0.1,
    start: list[float] | None = None,
    include_trajectory: bool = False,
):
    return planner.plan(
        start=[0.0] * 12 if start is None else start,
        finger_positions=[0.02, 0.02],
        left_position=[left_x, 0.5, 0.2],
        left_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        right_position=[0.1, 0.4, 0.2],
        right_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        allowed_planning_time_s=0.25,
        ik_timeout_s=0.025,
        ik_search_mode="deterministic_seeded_moveit_subgroup_multistart_v1",
        ik_seed=2210,
        ik_maximum_attempts=256,
        maximum_joint_step_rad=0.2,
        position_tolerance_m=0.001,
        orientation_tolerance_rad=0.003,
        rotation_weight=0.2,
        maximum_accepted_error=0.001,
        maximum_accepted_projected_error=0.003,
        start_bound_reconciliation_tolerance_rad=0.00002,
        physical_joint_limit_margin_rad=0.005,
        joint_limit_margin_rad=0.01,
        include_trajectory=include_trajectory,
    )


def test_moveit_client_validates_identity_and_bounded_plan(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with MoveItAlohaPlanner(settings) as planner:
        result = _plan(planner)

    assert planner.identity["planner_type"] == "geometric::RRTConnect"
    assert result.next[0] == pytest.approx(0.1)
    assert result.waypoint_count == 2
    assert result.maximum_goal_position_error_m == pytest.approx(1e-6)
    assert result.ik_search_mode == (
        "deterministic_seeded_moveit_subgroup_multistart_v1"
    )
    assert result.ik_candidate_selection_mode == (
        "deterministic_maximum_minimum_joint_limit_margin_v1"
    )
    assert result.ik_seed == 2210
    assert result.ik_maximum_attempts == 256
    assert result.ik_attempts_used == 3
    assert result.valid_ik_candidate_count == 2
    assert result.selected_ik_attempt == 2
    assert result.selected_ik_minimum_joint_limit_margin_rad == pytest.approx(0.25)
    assert settings.stderr_log.is_file()


def test_moveit_client_retains_full_official_trajectory_for_simple_sampler(
    tmp_path: Path,
) -> None:
    with MoveItAlohaPlanner(_settings(tmp_path)) as planner:
        result = _plan(planner, include_trajectory=True)

    assert len(result.trajectory) == result.waypoint_count == 3
    executor = MoveItAlohaTrajectoryExecutor(
        waypoint_l1_tolerance_rad=0.2,
        maximum_joint_step_rad=0.05,
    )
    executor.install(result, phase="orient")
    first = executor.command([0.0] * 12, phase="orient")
    second = executor.command([0.0] * 12, phase="orient")
    third = executor.command([0.0] * 12, phase="orient")

    assert first.waypoint_index == 1
    assert first.reference_reused is False
    assert first.positions[0] == pytest.approx(0.05)
    assert second.waypoint_index == 2
    assert second.reference_reused is True
    assert max(abs(value) for value in second.positions) == pytest.approx(0.05)
    assert third.waypoint_index == 2
    assert third.waypoint_advanced is False


def test_moveit_client_accepts_official_joint_goal_tolerance(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sidecar = settings.executable.read_text(encoding="utf-8").replace(
        "trajectory = [start, next_positions, goal]",
        "endpoint = list(goal); endpoint[0] += 9e-7\n"
        "                trajectory = [start, next_positions, endpoint]",
    )
    settings.executable.write_text(sidecar, encoding="utf-8", newline="\n")

    with MoveItAlohaPlanner(settings) as planner:
        result = _plan(planner, include_trajectory=True)

    assert result.trajectory[-1][0] - result.goal[0] == pytest.approx(9e-7)


def test_moveit_executor_latches_terminal_control_without_replanning(
    tmp_path: Path,
) -> None:
    with MoveItAlohaPlanner(_settings(tmp_path)) as planner:
        result = _plan(planner, include_trajectory=True)

    executor = MoveItAlohaTrajectoryExecutor(
        waypoint_l1_tolerance_rad=0.2,
        maximum_joint_step_rad=0.05,
    )
    executor.install(result, phase="approach")
    executor.command(result.trajectory[0], phase="approach")
    final_reference = executor.command(
        result.trajectory[1],
        phase="approach",
    )
    assert final_reference.waypoint_index == result.waypoint_count - 1
    assert final_reference.terminal_handoff_ready is False
    ready = executor.command(result.trajectory[-1], phase="approach")
    assert ready.terminal_handoff_ready is True

    compensated = tuple(value + 0.01 for value in result.trajectory[-1])
    executor.activate_terminal_control(compensated, phase="approach")
    first = executor.command(result.trajectory[-1], phase="approach")
    second = executor.command(first.positions, phase="approach")

    assert executor.terminal_control_active_for("approach") is True
    assert first.terminal_control_active is True
    assert first.terminal_control_activated is True
    assert second.terminal_control_activated is False
    assert second.waypoint_index == result.waypoint_count - 1


def test_moveit_executor_releases_only_completed_original_terminal_goal(
    tmp_path: Path,
) -> None:
    with MoveItAlohaPlanner(_settings(tmp_path)) as planner:
        result = _plan(planner, include_trajectory=True)

    executor = MoveItAlohaTrajectoryExecutor(
        waypoint_l1_tolerance_rad=0.2,
        maximum_joint_step_rad=0.05,
    )
    executor.install(result, phase="approach")
    executor.command(result.trajectory[0], phase="approach")
    executor.command(result.trajectory[1], phase="approach")
    executor.command(result.trajectory[-1], phase="approach")
    executor.activate_terminal_control(
        tuple(value + 0.01 for value in result.goal),
        phase="approach",
    )

    not_reached = list(result.goal)
    not_reached[0] += 0.0011
    assert (
        executor.complete_terminal_control(
            not_reached,
            phase="approach",
            goal_l1_tolerance_rad=0.001,
        )
        is False
    )
    assert executor.terminal_control_active_for("approach") is True
    assert (
        executor.complete_terminal_control(
            result.goal,
            phase="approach",
            goal_l1_tolerance_rad=0.001,
        )
        is True
    )
    assert executor.plan_result is None
    assert executor.terminal_control_active_for("approach") is False


def test_moveit_client_rejects_joint_goal_drift_beyond_official_tolerance(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    sidecar = settings.executable.read_text(encoding="utf-8").replace(
        "trajectory = [start, next_positions, goal]",
        "endpoint = list(goal); endpoint[0] += 1.1e-6\n"
        "                trajectory = [start, next_positions, endpoint]",
    )
    settings.executable.write_text(sidecar, encoding="utf-8", newline="\n")

    with MoveItAlohaPlanner(settings) as planner:
        with pytest.raises(MoveItAlohaPlannerError, match="endpoint differs"):
            _plan(planner, include_trajectory=True)


def test_moveit_client_preserves_official_planning_failure(tmp_path: Path) -> None:
    with MoveItAlohaPlanner(_settings(tmp_path)) as planner:
        with pytest.raises(MoveItAlohaPlanningError, match="bimanual_lma_ik_failed"):
            _plan(planner, left_x=-0.1)


def test_moveit_client_requires_loaded_collision_geometry(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sidecar = settings.executable.read_text(encoding="utf-8").replace(
        '"collision_geometry_shape_count": 30,',
        '"collision_geometry_shape_count": 0,',
    )
    settings.executable.write_text(sidecar, encoding="utf-8", newline="\n")

    with pytest.raises(MoveItAlohaPlannerError, match="no loaded collision geometry"):
        MoveItAlohaPlanner(settings)


def test_moveit_client_preserves_bounded_start_reconciliation(tmp_path: Path) -> None:
    start = [0.0] * 12
    start[5] = 3.141592653589793

    with MoveItAlohaPlanner(_settings(tmp_path)) as planner:
        result = _plan(planner, start=start)

    assert result.start_bound_reconciliations == ("left_wrist_rotate",)
    assert result.maximum_start_bound_reconciliation_rad == pytest.approx(
        3.141592653589793 - 3.14158
    )


def test_moveit_client_accepts_official_start_path_constraint_recovery(
    tmp_path: Path,
) -> None:
    start = [0.0] * 12
    start[11] = -3.135

    with MoveItAlohaPlanner(_settings(tmp_path)) as planner:
        result = _plan(planner, start=start)

    assert result.start_state_satisfies_joint_path_constraint is False
    assert result.start_state_path_constraint_recovery is True
    assert result.adapter_added_state_indices == (0, 1)
    assert result.adapter_prefix_waypoint_count == 2
    assert result.minimum_recovery_progress_rad == pytest.approx(0.001)
    assert result.minimum_start_joint_limit_margin_rad == pytest.approx(0.00658)
    assert result.minimum_next_joint_limit_margin_rad == pytest.approx(0.00758)


def test_moveit_client_rejects_official_adapter_identity_drift(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sidecar = settings.executable.read_text(encoding="utf-8").replace(
        "default_planner_request_adapters/FixStartStatePathConstraints",
        "custom/FixStartStatePathConstraints",
    )
    settings.executable.write_text(sidecar, encoding="utf-8", newline="\n")

    with pytest.raises(MoveItAlohaPlannerError, match="planning_request_adapters"):
        MoveItAlohaPlanner(settings)


def test_moveit_client_rejects_recovery_without_positive_progress(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    sidecar = settings.executable.read_text(encoding="utf-8").replace(
        '"minimum_recovery_progress_rad": 0.001 if recovery else 0.0,',
        '"minimum_recovery_progress_rad": 0.0,',
    )
    settings.executable.write_text(sidecar, encoding="utf-8", newline="\n")
    start = [0.0] * 12
    start[11] = -3.135

    with MoveItAlohaPlanner(settings) as planner:
        with pytest.raises(MoveItAlohaPlannerError, match="recovery contract"):
            _plan(planner, start=start)


def test_moveit_client_rejects_recovery_prefix_below_physical_margin(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    sidecar = settings.executable.read_text(encoding="utf-8").replace(
        "minimum_start_margin if recovery else 0.5\n                ),",
        "0.004 if recovery else 0.5\n                ),",
    )
    settings.executable.write_text(sidecar, encoding="utf-8", newline="\n")
    start = [0.0] * 12
    start[11] = -3.135

    with MoveItAlohaPlanner(settings) as planner:
        with pytest.raises(MoveItAlohaPlannerError, match="physical joint-limit margin"):
            _plan(planner, start=start)


def test_moveit_client_rejects_start_reconciliation_beyond_tolerance(
    tmp_path: Path,
) -> None:
    start = [0.0] * 12
    start[5] = 3.15

    with MoveItAlohaPlanner(_settings(tmp_path)) as planner:
        with pytest.raises(MoveItAlohaPlannerError, match="registered tolerance"):
            _plan(planner, start=start)


def test_moveit_settings_refuse_to_overwrite_stderr_log(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.stderr_log.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MoveItAlohaPlannerSettings(
            executable=settings.executable,
            urdf=settings.urdf,
            srdf=settings.srdf,
            stderr_log=settings.stderr_log,
        )


def test_moveit_client_rejects_execution_step_beyond_contract(tmp_path: Path) -> None:
    with MoveItAlohaPlanner(_settings(tmp_path)) as planner:
        with pytest.raises(MoveItAlohaPlannerError, match="joint step"):
            planner.plan(
                start=[0.0] * 12,
                finger_positions=[0.02, 0.02],
                left_position=[0.1, 0.5, 0.2],
                left_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
                right_position=[0.1, 0.4, 0.2],
                right_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
                allowed_planning_time_s=0.25,
                ik_timeout_s=0.025,
                ik_search_mode=(
                    "deterministic_seeded_moveit_subgroup_multistart_v1"
                ),
                ik_seed=2210,
                ik_maximum_attempts=256,
                maximum_joint_step_rad=0.05,
                position_tolerance_m=0.001,
                orientation_tolerance_rad=0.003,
                rotation_weight=0.2,
                maximum_accepted_error=0.001,
                maximum_accepted_projected_error=0.003,
                start_bound_reconciliation_tolerance_rad=0.00002,
                physical_joint_limit_margin_rad=0.005,
                joint_limit_margin_rad=0.01,
            )


def test_moveit_client_rejects_path_below_registered_joint_margin(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    sidecar = settings.executable.read_text(encoding="utf-8").replace(
        '"minimum_constrained_path_joint_limit_margin_rad": 0.1,',
        '"minimum_constrained_path_joint_limit_margin_rad": 0.009,',
    )
    settings.executable.write_text(sidecar, encoding="utf-8", newline="\n")

    with MoveItAlohaPlanner(settings) as planner:
        with pytest.raises(MoveItAlohaPlannerError, match="registered joint-limit margin"):
            _plan(planner)


class _FakePlanner:
    def __init__(
        self,
        failure: str | None = None,
        failure_response: dict[str, Any] | None = None,
        next_index: int = 5,
        next_delta: float = 0.1,
    ) -> None:
        self.failure = failure
        self.failure_response = failure_response
        self.next_index = next_index
        self.next_delta = next_delta
        self.request: dict[str, Any] | None = None
        self.calls = 0

    def plan(self, **request: Any) -> MoveItAlohaPlanResult:
        self.calls += 1
        self.request = request
        if self.failure is not None:
            raise MoveItAlohaPlanningError(
                self.failure,
                self.failure_response
                or {"status": "error", "reason": self.failure},
            )
        start = tuple(float(value) for value in request["start"])
        next_positions = list(start)
        next_positions[self.next_index] += self.next_delta
        return MoveItAlohaPlanResult(
            goal=tuple([0.2] * 12),
            next=tuple(next_positions),
            planning_time_s=0.02,
            waypoint_count=3,
            path_length_rad=0.4,
            path_maximum_waypoint_joint_delta_rad=0.2,
            first_segment_interpolation=0.5,
            maximum_goal_position_error_m=1e-6,
            maximum_goal_orientation_error_rad=2e-6,
            maximum_goal_weighted_error=1.4e-6,
            ik_task_mode=str(request["ik_task_mode"]),
            maximum_orientation_relaxation_rad=float(
                request["maximum_orientation_relaxation_rad"]
            ),
            joint_limit_margin_rad=float(request["joint_limit_margin_rad"]),
            physical_joint_limit_margin_rad=float(
                request["physical_joint_limit_margin_rad"]
            ),
            start_state_satisfies_joint_path_constraint=True,
            start_state_path_constraint_recovery=False,
            adapter_added_state_indices=(),
            adapter_prefix_waypoint_count=0,
            minimum_recovery_progress_rad=0.0,
            minimum_start_joint_limit_margin_rad=0.5,
            minimum_goal_joint_limit_margin_rad=0.25,
            minimum_path_joint_limit_margin_rad=0.1,
            minimum_constrained_path_joint_limit_margin_rad=0.1,
            minimum_adapter_prefix_physical_joint_limit_margin_rad=0.5,
            minimum_next_joint_limit_margin_rad=0.2,
            start_bound_reconciliations=(),
            maximum_start_bound_reconciliation_rad=0.0,
            maximum_requested_start_to_next_joint_delta_rad=abs(self.next_delta),
            ik_search_mode=str(request["ik_search_mode"]),
            ik_candidate_selection_mode=(
                "deterministic_maximum_minimum_joint_limit_margin_v1"
            ),
            ik_seed=int(request["ik_seed"]),
            ik_maximum_attempts=int(request["ik_maximum_attempts"]),
            ik_attempts_used=3,
            valid_ik_candidate_count=2,
            selected_ik_attempt=2,
            selected_ik_minimum_joint_limit_margin_rad=0.25,
            selected_ik_maximum_start_delta_rad=max(
                abs(0.2 - value) for value in start
            ),
            ik_outer_timeout_s=float(request["ik_timeout_s"]),
            trajectory=(
                start,
                tuple(next_positions),
                tuple([0.2] * 12),
            ),
            raw_response={"status": "ok"},
        )


def _moveit_settings() -> dict[str, Any]:
    return {
        "solver_backend": "mink_qp",
        "path_planner_enabled": True,
        "path_planner_backend": "moveit2_ompl",
        "path_planner_phases": ["approach", "orient"],
        "path_planner_allowed_planning_time_s": 0.25,
        "path_planner_ik_timeout_s": 0.025,
        "path_planner_ik_search_mode": (
            "deterministic_seeded_moveit_subgroup_multistart_v1"
        ),
        "path_planner_ik_candidate_selection_mode": (
            "deterministic_maximum_minimum_joint_limit_margin_v1"
        ),
        "path_planner_ik_seed": 2210,
        "path_planner_ik_maximum_attempts": 256,
        "path_planner_maximum_joint_step_rad": 0.2,
        "path_planner_position_tolerance_m": 0.001,
        "path_planner_orientation_tolerance_rad": 0.003,
        "path_planner_start_bound_reconciliation_tolerance_rad": 0.00002,
        "path_planner_physical_joint_limit_margin_rad": 0.005,
        "path_planner_joint_limit_margin_rad": 0.01,
        "path_planner_finger_lower_m": 0.021,
        "path_planner_finger_upper_m": 0.057,
        "path_planner_finger_bound_reconciliation_tolerance_m": 0.001,
        "path_planner_position_priority_enabled": False,
        "path_planner_position_priority_cartesian_backoff_fractions": [
            1.0,
            0.75,
            0.5,
            0.25,
            0.125,
        ],
        "path_planner_position_priority_minimum_cartesian_progress_m": 0.001,
        "path_planner_position_priority_maximum_orientation_relaxation_rad": 0.04,
        "path_planner_include_trajectory": True,
        "path_planner_waypoint_l1_tolerance_rad": 0.2,
        "rotation_weight": 0.2,
        "maximum_accepted_error": 0.001,
        "maximum_accepted_projected_error": 0.003,
        "maximum_joint_target_delta": 0.2,
    }


def _executor() -> MoveItAlohaTrajectoryExecutor:
    return MoveItAlohaTrajectoryExecutor(
        waypoint_l1_tolerance_rad=0.2,
        maximum_joint_step_rad=0.2,
    )


def _target() -> InsertionTaskSpaceTarget:
    pose = GeometryPose(
        position=torch.tensor([0.1, 0.5, 0.2]),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    return InsertionTaskSpaceTarget(
        left_eef=pose,
        right_eef=pose,
        left_gripper=1.0,
        right_gripper=1.0,
        phase=InsertionTeacherPhase.APPROACH,
        phase_changed=False,
        maximum_position_error_m=0.1,
        best_observed_reward=0.0,
    )


def test_moveit_fallback_maps_official_path_waypoint_to_action_contract() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    current = torch.tensor(
        [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 1.0] * 2,
        dtype=torch.float32,
    )
    initial = IkActionResult(
        action=current,
        success=False,
        maximum_error=0.012,
        maximum_projected_error=0.012,
        joint_delta_saturations=0,
        contract_clip_fields=(),
        solver_backend="mink_qp_daqp",
    )
    planner = _FakePlanner()

    result = _moveit_path_action(
        current,
        _target(),
        contract=contract,
        settings=_moveit_settings(),
        initial_result=initial,
        planner=planner,  # type: ignore[arg-type]
        executor=_executor(),
    )

    assert result.success is True
    assert result.action[5] == pytest.approx(0.1)
    assert result.action[6] == pytest.approx(1.0)
    assert result.path_planner_mode == (
        "moveit2_ompl_rrtconnect+simple_sampler+forward_trajectory"
    )
    assert result.path_planner_waypoint_count == 3
    assert result.path_planner_goal_position_error_m == pytest.approx(1e-6)
    assert result.path_planner_goal_weighted_error == pytest.approx(1.4e-6)
    assert result.path_planner_ik_search_mode == (
        "deterministic_seeded_moveit_subgroup_multistart_v1"
    )
    assert result.path_planner_ik_candidate_selection_mode == (
        "deterministic_maximum_minimum_joint_limit_margin_v1"
    )
    assert result.path_planner_ik_seed == 2210
    assert result.path_planner_ik_maximum_attempts == 256
    assert result.path_planner_ik_attempts_used == 3
    assert result.path_planner_valid_ik_candidate_count == 2
    assert result.path_planner_selected_ik_attempt == 2
    assert result.path_planner_selected_ik_minimum_joint_limit_margin_rad == (
        pytest.approx(0.25)
    )
    assert planner.request is not None
    assert planner.request["maximum_accepted_error"] == pytest.approx(0.001)
    assert planner.request["start_bound_reconciliation_tolerance_rad"] == pytest.approx(
        0.00002
    )
    assert planner.request["joint_limit_margin_rad"] == pytest.approx(0.01)
    assert planner.request["finger_positions"] == pytest.approx([0.057, 0.057])
    assert result.path_planner_joint_limit_margin_rad == pytest.approx(0.01)
    assert result.path_planner_minimum_path_joint_limit_margin_rad == pytest.approx(
        0.1
    )


def test_ik_action_reuses_retained_reference_without_replanning() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    current = torch.tensor(
        [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 1.0] * 2,
        dtype=torch.float32,
    )
    initial = IkActionResult(
        action=current,
        success=False,
        maximum_error=0.012,
        maximum_projected_error=0.012,
        joint_delta_saturations=0,
        contract_clip_fields=(),
        solver_backend="mink_qp_daqp",
    )
    planner = _FakePlanner()
    executor = _executor()
    first = _moveit_path_action(
        current,
        _target(),
        contract=contract,
        settings=_moveit_settings(),
        initial_result=initial,
        planner=planner,  # type: ignore[arg-type]
        executor=executor,
    )

    second = _ik_action(
        None,
        first.action,
        _target(),
        contract=contract,
        settings=_moveit_settings(),
        _mink_solver=object(),  # type: ignore[arg-type]
        _moveit_planner=planner,  # type: ignore[arg-type]
        _moveit_executor=executor,
    )

    assert planner.calls == 1
    assert second.path_planner_attempted is False
    assert second.path_planner_used is True
    assert second.path_planner_reference_reused is True
    assert second.path_planner_reference_waypoint_index == 2


def test_ik_action_hands_final_waypoint_to_bounded_terminal_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    current = torch.tensor(
        [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 1.0] * 2,
        dtype=torch.float32,
    )
    initial = IkActionResult(
        action=current,
        success=False,
        maximum_error=0.012,
        maximum_projected_error=0.012,
        joint_delta_saturations=0,
        contract_clip_fields=(),
        solver_backend="mink_qp_daqp",
    )
    planner = _FakePlanner()
    executor = _executor()
    first = _moveit_path_action(
        current,
        _target(),
        contract=contract,
        settings=_moveit_settings(),
        initial_result=initial,
        planner=planner,  # type: ignore[arg-type]
        executor=executor,
    )
    assert first.path_planner_reference_waypoint_index == 1
    planned = executor.plan_result
    assert planned is not None
    final = current.clone()
    final[list(range(6))] = torch.as_tensor(planned.trajectory[-1][:6])
    final[list(range(7, 13))] = torch.as_tensor(planned.trajectory[-1][6:])

    expected = MujocoPositionFeedforwardResult(
        positions=tuple(value + 0.01 for value in planned.trajectory[-1]),
        corrections_rad=(0.01,) * 12,
        maximum_correction_rad=0.01,
        minimum_command_joint_limit_margin_rad=0.2,
    )
    captured_feedforward: dict[str, Any] = {}

    def _feedforward(*args: Any, **kwargs: Any) -> MujocoPositionFeedforwardResult:
        captured_feedforward.update(kwargs)
        return expected

    monkeypatch.setattr(
        geometry_evaluator,
        "static_position_feedforward",
        _feedforward,
    )
    settings = _moveit_settings()
    settings.update(
        path_planner_terminal_control_enabled=True,
        path_planner_terminal_control_joint_limit_margin_rad=0.01,
        path_planner_terminal_control_maximum_correction_rad=0.05,
        path_planner_terminal_control_neutral_reference_tolerance_rad=1e-9,
    )

    next_waypoint = current.clone()
    next_waypoint[list(range(6))] = torch.as_tensor(planned.trajectory[1][:6])
    next_waypoint[list(range(7, 13))] = torch.as_tensor(
        planned.trajectory[1][6:]
    )
    intermediate = _ik_action(
        None,
        next_waypoint,
        _target(),
        contract=contract,
        settings=settings,
        _mink_solver=object(),  # type: ignore[arg-type]
        _moveit_planner=planner,  # type: ignore[arg-type]
        _moveit_executor=executor,
    )
    assert intermediate.path_planner_terminal_control_active is False

    result = _ik_action(
        None,
        final,
        _target(),
        contract=contract,
        settings=settings,
        _mink_solver=object(),  # type: ignore[arg-type]
        _moveit_planner=planner,  # type: ignore[arg-type]
        _moveit_executor=executor,
    )

    assert planner.calls == 1
    assert result.success is True
    assert result.path_planner_attempted is False
    assert result.path_planner_terminal_control_active is True
    assert result.path_planner_terminal_control_activated is True
    assert result.path_planner_terminal_control_maximum_correction_rad == pytest.approx(
        0.01
    )
    assert result.path_planner_terminal_control_minimum_command_margin_rad == (
        pytest.approx(0.2)
    )
    assert captured_feedforward["arm_joint_names"] == (
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

    refreshed = IkActionResult(
        action=final,
        success=True,
        maximum_error=0.0,
        maximum_projected_error=0.0,
        joint_delta_saturations=0,
        contract_clip_fields=(),
        solver_backend="mink_qp_daqp",
    )
    monkeypatch.setattr(
        geometry_evaluator,
        "_mink_ik_action",
        lambda *args, **kwargs: refreshed,
    )
    settings.update(
        path_planner_replan_on_terminal_completion=True,
        path_planner_terminal_completion_goal_l1_tolerance_rad=0.001,
    )
    completed = _ik_action(
        None,
        final,
        _target(),
        contract=contract,
        settings=settings,
        _mink_solver=object(),  # type: ignore[arg-type]
        _moveit_planner=planner,  # type: ignore[arg-type]
        _moveit_executor=executor,
    )

    assert planner.calls == 1
    assert completed.success is True
    assert completed.path_planner_terminal_control_completed is True
    assert executor.plan_result is None
    assert result.solver_backend.endswith(
        "+mujoco_static_inverse_dynamics_position_feedforward"
    )


def test_moveit_fallback_preserves_official_planning_failure() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    current = torch.tensor(
        [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 1.0] * 2,
        dtype=torch.float32,
    )
    initial = IkActionResult(
        action=current,
        success=False,
        maximum_error=0.012,
        maximum_projected_error=0.012,
        joint_delta_saturations=0,
        contract_clip_fields=(),
        solver_backend="mink_qp_daqp",
    )

    result = _moveit_path_action(
        current,
        _target(),
        contract=contract,
        settings=_moveit_settings(),
        initial_result=initial,
        planner=_FakePlanner("bimanual_lma_ik_failed"),  # type: ignore[arg-type]
        executor=_executor(),
    )

    assert result.success is False
    assert result.path_planner_attempted is True
    assert result.path_planner_used is False
    assert result.solver_failure == "bimanual_lma_ik_failed"


def test_moveit_fallback_allows_only_float32_quantization_of_bounded_step() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    current = torch.tensor(
        [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 1.0] * 2,
        dtype=torch.float32,
    )
    current[12] = torch.pi
    initial = IkActionResult(
        action=current,
        success=False,
        maximum_error=0.012,
        maximum_projected_error=0.012,
        joint_delta_saturations=0,
        contract_clip_fields=(),
        solver_backend="mink_qp_daqp",
    )
    maximum_step = _moveit_settings()["maximum_joint_target_delta"]

    result = _moveit_path_action(
        current,
        _target(),
        contract=contract,
        settings=_moveit_settings(),
        initial_result=initial,
        planner=_FakePlanner(  # type: ignore[arg-type]
            next_index=11,
            next_delta=-maximum_step,
        ),
        executor=_executor(),
    )

    observed_delta = abs(float(result.action[12] - current[12]))
    quantization_tolerance = float(torch.finfo(torch.float32).eps) * float(
        current[12]
    )
    assert observed_delta > maximum_step
    assert observed_delta <= maximum_step + quantization_tolerance


def test_moveit_fallback_records_rejected_start_bound_evidence() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    current = torch.tensor(
        [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 1.0] * 2,
        dtype=torch.float32,
    )
    initial = IkActionResult(
        action=current,
        success=False,
        maximum_error=0.012,
        maximum_projected_error=0.012,
        joint_delta_saturations=0,
        contract_clip_fields=(),
        solver_backend="mink_qp_daqp",
    )
    response = {
        "status": "error",
        "reason": "start_state_out_of_bounds",
        "start_bound_violations": [
            {
                "joint_name": "right_wrist_rotate",
                "requested_position_rad": 3.14161,
                "nearest_bound_position_rad": 3.14158,
                "delta_rad": 0.00003,
            }
        ],
        "maximum_start_bound_violation_rad": 0.00003,
        "start_bound_reconciliation_tolerance_rad": 0.00002,
    }

    result = _moveit_path_action(
        current,
        _target(),
        contract=contract,
        settings=_moveit_settings(),
        initial_result=initial,
        planner=_FakePlanner(  # type: ignore[arg-type]
            "start_state_out_of_bounds",
            response,
        ),
        executor=_executor(),
    )

    assert result.success is False
    assert result.path_planner_start_bound_violations == ("right_wrist_rotate",)
    assert result.path_planner_maximum_start_bound_violation_rad == pytest.approx(
        0.00003
    )
