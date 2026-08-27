"""Evaluate an event-driven ALOHA insertion teacher without writing labels."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.data import resolve_prepared_cache  # noqa: E402
from rosetta_reality.data.config import DatasetConfig, load_dataset_config  # noqa: E402
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract  # noqa: E402
from rosetta_reality.sim.action_contract import ActionContract  # noqa: E402
from rosetta_reality.sim.geometry_teacher import (  # noqa: E402
    GeometryPose,
    GeometryTeacherError,
    InsertionGeometry,
    InsertionTaskSpaceTarget,
    InsertionTeacherCalibration,
    InsertionTeacherPhase,
    InsertionTeacherSettings,
    ObjectGeometryInsertionTeacher,
    relative_pose,
)
from rosetta_reality.sim.mink_aloha_ik import (  # noqa: E402
    MinkAlohaIkSettings,
    MinkAlohaIkSolver,
)
from rosetta_reality.sim.moveit_aloha_planner import (  # noqa: E402
    EXPECTED_JOINT_NAMES,
    MoveItAlohaPlanner,
    MoveItAlohaPlannerError,
    MoveItAlohaPlannerSettings,
    MoveItAlohaPlanningError,
    MoveItAlohaPlanResult,
    MoveItAlohaTrajectoryCommand,
    MoveItAlohaTrajectoryExecutor,
)
from rosetta_reality.sim.mujoco_position_feedforward import (  # noqa: E402
    MujocoPositionFeedforwardResult,
    static_position_feedforward,
)

DEFAULT_PLAN = REPOSITORY_ROOT / "configs/sim/aloha_insertion_geometry_teacher_005.yaml"
LEFT_SITE = "cali_left_site1"
RIGHT_SITE = "cali_right_site1"
LEFT_JOINTS = (
    "vx300s_left/waist",
    "vx300s_left/shoulder",
    "vx300s_left/elbow",
    "vx300s_left/forearm_roll",
    "vx300s_left/wrist_angle",
    "vx300s_left/wrist_rotate",
)
RIGHT_JOINTS = (
    "vx300s_right/waist",
    "vx300s_right/shoulder",
    "vx300s_right/elbow",
    "vx300s_right/forearm_roll",
    "vx300s_right/wrist_angle",
    "vx300s_right/wrist_rotate",
)
ARM_ACTION_INDICES = (*range(6), *range(7, 13))
ARM_ACTION_NAMES = (
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


@dataclass(frozen=True, slots=True)
class CalibrationReplay:
    """Successful train-only replay and derived rigid-geometry calibration."""

    calibration: InsertionTeacherCalibration
    steps_executed: int
    first_grasp_step: int
    terminal_step: int
    maximum_reward: float


@dataclass(frozen=True, slots=True)
class IkActionResult:
    """Logical joint target produced by the simulator-specific IK boundary."""

    action: Tensor
    success: bool
    maximum_error: float
    maximum_projected_error: float
    joint_delta_saturations: int
    contract_clip_fields: tuple[str, ...]
    path_planner_attempted: bool = False
    path_planner_used: bool = False
    path_planner_mode: str | None = None
    path_planner_fraction: float | None = None
    path_planner_trust_region_mode: str | None = None
    path_planner_trust_region_basis: str | None = None
    path_planner_trust_region_selection_policy: str | None = None
    path_planner_trust_region_restoration_reference: str | None = None
    path_planner_trust_region_active_arm: str | None = None
    path_planner_trust_region_radius_m: float | None = None
    path_planner_trust_region_direction: tuple[float, float, float] | None = None
    path_planner_trust_region_orientation_fraction: float | None = None
    path_planner_trust_region_orientation_target_rad: float | None = None
    path_planner_trust_region_margin_improvement_rad: float | None = None
    path_planner_trust_region_requested_position_relaxation_m: float | None = None
    path_planner_trust_region_candidates_evaluated: int = 0
    path_planner_trust_region_valid_candidates: int = 0
    path_planner_orientation_relaxation_rad: float = 0.0
    path_planner_initial_projected_error: float | None = None
    path_planner_planning_time_s: float | None = None
    path_planner_waypoint_count: int | None = None
    path_planner_path_length_rad: float | None = None
    path_planner_goal_position_error_m: float | None = None
    path_planner_goal_orientation_error_rad: float | None = None
    path_planner_goal_weighted_error: float | None = None
    path_planner_ik_search_mode: str | None = None
    path_planner_ik_candidate_selection_mode: str | None = None
    path_planner_ik_seed: int | None = None
    path_planner_ik_maximum_attempts: int | None = None
    path_planner_ik_attempts_used: int | None = None
    path_planner_valid_ik_candidate_count: int | None = None
    path_planner_selected_ik_attempt: int | None = None
    path_planner_selected_ik_minimum_joint_limit_margin_rad: float | None = None
    path_planner_selected_ik_maximum_start_delta_rad: float | None = None
    path_planner_ik_outer_timeout_s: float | None = None
    path_planner_joint_limit_margin_rad: float | None = None
    path_planner_physical_joint_limit_margin_rad: float | None = None
    path_planner_start_state_path_constraint_recovery: bool = False
    path_planner_adapter_prefix_waypoint_count: int = 0
    path_planner_minimum_recovery_progress_rad: float | None = None
    path_planner_minimum_start_joint_limit_margin_rad: float | None = None
    path_planner_minimum_goal_joint_limit_margin_rad: float | None = None
    path_planner_minimum_path_joint_limit_margin_rad: float | None = None
    path_planner_minimum_constrained_path_joint_limit_margin_rad: float | None = None
    path_planner_minimum_adapter_prefix_physical_joint_limit_margin_rad: (
        float | None
    ) = None
    path_planner_minimum_next_joint_limit_margin_rad: float | None = None
    path_planner_start_bound_reconciliations: tuple[str, ...] = ()
    path_planner_maximum_start_bound_reconciliation_rad: float = 0.0
    path_planner_start_bound_violations: tuple[str, ...] = ()
    path_planner_maximum_start_bound_violation_rad: float = 0.0
    path_planner_reference_reused: bool = False
    path_planner_reference_waypoint_index: int | None = None
    path_planner_reference_waypoint_advanced: bool = False
    path_planner_reference_waypoint_l1_distance_rad: float | None = None
    path_planner_terminal_control_active: bool = False
    path_planner_terminal_control_activated: bool = False
    path_planner_terminal_control_completed: bool = False
    path_planner_terminal_control_maximum_correction_rad: float | None = None
    path_planner_terminal_control_minimum_command_margin_rad: float | None = None
    solver_backend: str = "dm_control_qpos_from_site_pose"
    solver_iterations: int | None = None
    solver_failure: str | None = None


def _quaternion_distance(first: Tensor, second: Tensor) -> float:
    dot = abs(float(torch.dot(first, second)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _bounded_orientation_waypoint(
    requested: Tensor,
    feasible: Tensor,
    maximum_relaxation_rad: float,
) -> tuple[Tensor, float]:
    """Move toward a feasible orientation without exceeding one planner step."""

    requested = requested.detach().to(torch.float32).cpu()
    feasible = feasible.detach().to(torch.float32).cpu()
    if float(torch.dot(requested, feasible)) < 0.0:
        feasible = -feasible
    angle = _quaternion_distance(requested, feasible)
    if angle <= maximum_relaxation_rad:
        return feasible, angle
    fraction = maximum_relaxation_rad / angle
    half_angle = angle * 0.5
    sine = math.sin(half_angle)
    if abs(sine) <= 1e-8:
        waypoint = requested.lerp(feasible, fraction)
    else:
        waypoint = (
            requested * (math.sin((1.0 - fraction) * half_angle) / sine)
            + feasible * (math.sin(fraction * half_angle) / sine)
        )
    waypoint = waypoint / torch.linalg.vector_norm(waypoint)
    return waypoint, maximum_relaxation_rad


def _cartesian_waypoint(
    current: GeometryPose,
    requested: GeometryPose,
    fraction: float,
    quaternion: Tensor,
) -> GeometryPose:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Path-planner Cartesian fractions must lie in (0, 1].")
    return GeometryPose(
        position=current.position + (requested.position - current.position) * fraction,
        quaternion=quaternion,
    )


def _full_pose_cartesian_waypoint(
    current: GeometryPose,
    requested: GeometryPose,
    fraction: float,
) -> GeometryPose:
    """Interpolate translation and shortest-arc orientation by one fraction."""

    orientation_distance = _quaternion_distance(
        current.quaternion,
        requested.quaternion,
    )
    quaternion, _ = _bounded_orientation_waypoint(
        current.quaternion,
        requested.quaternion,
        orientation_distance * fraction,
    )
    return _cartesian_waypoint(
        current,
        requested,
        fraction,
        quaternion,
    )


def _feedback_aligned_orthonormal_basis(
    current_position: Tensor,
    requested_position: Tensor,
) -> tuple[tuple[float, float, float], ...]:
    """Return deterministic signed radial and tangent directions."""

    radial = (
        requested_position.detach().to(torch.float64).cpu()
        - current_position.detach().to(torch.float64).cpu()
    )
    norm = float(torch.linalg.vector_norm(radial))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Feedback-aligned trust-region basis needs position error.")
    radial = radial / norm
    world_axes = torch.eye(3, dtype=torch.float64)
    reference = min(
        world_axes,
        key=lambda axis: abs(float(torch.dot(radial, axis))),
    )
    tangent_one = torch.linalg.cross(radial, reference)
    tangent_one = tangent_one / torch.linalg.vector_norm(tangent_one)
    tangent_two = torch.linalg.cross(radial, tangent_one)
    axes = (radial, tangent_one, tangent_two)
    return tuple(
        tuple(float(component) * sign for component in axis)
        for axis in axes
        for sign in (1.0, -1.0)
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _load_plan(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    plan = _mapping(raw, "Geometry-teacher plan")
    if plan.get("schema_version") != 1:
        raise ValueError("Geometry-teacher plan must use schema version one.")
    if plan.get("status") != "diagnostic_preregistered_no_label_collection":
        raise ValueError("Geometry-teacher plan is not an isolated diagnostic plan.")
    for relative, expected in _mapping(
        plan.get("implementation_files"),
        "implementation_files",
    ).items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe implementation path: {relative!r}.")
        if expected == "PLACEHOLDER":
            raise ValueError(f"Implementation hash is not frozen: {relative}.")
        if file_sha256(REPOSITORY_ROOT / relative_path) != str(expected):
            raise ValueError(f"Geometry-teacher implementation identity differs: {relative}.")
    return plan


def _repository_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a repository-relative path, received {raw!r}.")
    return REPOSITORY_ROOT / relative


def _run_root() -> Path:
    raw = os.environ.get("ROSETTA_RUN_ROOT")
    if not raw:
        raise OSError("ROSETTA_RUN_ROOT must be defined by the container launcher.")
    root = Path(raw).resolve()
    if not root.is_absolute():
        raise ValueError("ROSETTA_RUN_ROOT must resolve to an absolute path.")
    return root


def _validate_plan_boundaries(plan: dict[str, Any]) -> None:
    output = _mapping(plan.get("output"), "output")
    run_directory = output.get("run_directory")
    if not isinstance(run_directory, str) or not run_directory.strip():
        raise ValueError("Geometry-teacher output run_directory must be nonempty.")
    if Path(run_directory).is_absolute() or ".." in Path(run_directory).parts:
        raise ValueError("Geometry-teacher output run_directory must be relative and safe.")
    if output.get("reports_are_scoped_by_plan_sha256_and_stage") is not True:
        raise ValueError(
            "Geometry-teacher reports must be scoped by plan SHA-256 and stage."
        )

    scope = _mapping(plan.get("scope"), "scope")
    train = {int(value) for value in scope.get("train_episodes", [])}
    validation = {int(value) for value in scope.get("validation_episodes", [])}
    hidden = {int(value) for value in scope.get("hidden_test_episodes", [])}
    if not train or train & validation or train & hidden or validation & hidden:
        raise ValueError("Geometry-teacher dataset splits are empty or overlap.")
    if scope.get("hidden_test_loaded") is not False:
        raise ValueError("Geometry-teacher plan must keep the hidden test sealed.")

    calibration = _mapping(plan.get("calibration"), "calibration")
    calibration_episode = int(calibration["source_episode"])
    if calibration_episode not in train:
        raise ValueError("Geometry-teacher calibration must use a train-only episode.")
    axis = torch.as_tensor(calibration.get("insertion_axis_in_socket"), dtype=torch.float32)
    if axis.shape != (3,) or not bool(torch.isfinite(axis).all()):
        raise ValueError("Geometry-teacher insertion axis must be a finite three-vector.")

    evaluation = _mapping(plan.get("evaluation"), "evaluation")
    if evaluation.get("stages") != "calibration_then_exact_then_tuning_then_development":
        raise ValueError("Geometry-teacher staged evaluation boundary changed.")
    exact = _mapping(evaluation.get("exact_control"), "exact_control")
    if int(exact["source_episode"]) != calibration_episode:
        raise ValueError("Exact control must use the calibration train episode.")
    tuning = {int(value) for value in evaluation.get("tuning_simulator_seeds", [])}
    development = {
        int(value) for value in evaluation.get("development_simulator_seeds", [])
    }
    collection = {
        int(value) for value in evaluation.get("reserved_collection_simulator_seeds", [])
    }
    policy_gate = {
        int(value) for value in evaluation.get("reserved_policy_gate4_seeds", [])
    }
    seed_groups = (tuning, development, collection, policy_gate)
    if any(not group for group in seed_groups):
        raise ValueError(
            "Tuning, development, collection, and policy Gate seeds must be registered."
        )
    if any(
        first & second
        for index, first in enumerate(seed_groups)
        for second in seed_groups[index + 1 :]
    ):
        raise ValueError(
            "Tuning, development, collection, and policy Gate seeds must be disjoint."
        )
    exact_seed = int(exact["simulator_seed"])
    if any(exact_seed in group for group in seed_groups):
        raise ValueError("Exact-control seed must be isolated from every later stage.")

    execution_diagnostics = _mapping(
        plan.get("execution_diagnostics"),
        "execution_diagnostics",
    )
    expected_execution_diagnostic_fields = {
        "schema",
        "joint_limit_margin_rad",
        "arm_dimension_names",
        "record_every_executed_step",
        "record_full_joint_vectors",
        "record_per_joint_margins",
        "affects_action_selection",
    }
    if set(execution_diagnostics) != expected_execution_diagnostic_fields:
        raise ValueError("Execution-diagnostic settings differ from the registered schema.")
    if execution_diagnostics.get("schema") != (
        "commanded_vs_observed_joint_margin_v1"
    ):
        raise ValueError("Execution-diagnostic schema identity differs.")
    if tuple(execution_diagnostics.get("arm_dimension_names", ())) != (
        ARM_ACTION_NAMES
    ):
        raise ValueError("Execution-diagnostic arm ordering differs.")
    for key in (
        "record_every_executed_step",
        "record_full_joint_vectors",
        "record_per_joint_margins",
    ):
        if execution_diagnostics.get(key) is not True:
            raise ValueError(f"Execution diagnostic must enable {key}.")
    if execution_diagnostics.get("affects_action_selection") is not False:
        raise ValueError("Execution diagnostics must not affect action selection.")

    execution_guard = _mapping(plan.get("execution_guard"), "execution_guard")
    expected_execution_guard_fields = {
        "schema",
        "strategy",
        "source_audit",
        "source_audit_sha256",
        "source_exact_report_sha256",
        "reserve_metric",
        "reserve_scope",
        "physical_joint_limit_margin_rad",
        "tracking_reserve_rad",
        "command_joint_limit_margin_rad",
        "applies_to",
        "affects_action_selection",
    }
    guard_schema = execution_guard.get("schema")
    official_recovery_v2 = (
        "robust_joint_limit_constraint_tightening_with_official_start_recovery_v2"
    )
    official_recovery_safe_set_v3 = (
        "robust_joint_limit_constraint_tightening_with_official_"
        "start_recovery_safe_set_v3"
    )
    if guard_schema in {official_recovery_v2, official_recovery_safe_set_v3}:
        expected_execution_guard_fields |= {
            "start_state_path_constraint_adapter",
            "recovery_entry_condition",
            "recovery_prefix_joint_limit_margin_rad",
            "recovery_constrained_suffix_joint_limit_margin_rad",
            "recovery_requires_positive_margin_progress",
            "recovery_forbidden_when_start_satisfies_command_margin",
        }
    if guard_schema == official_recovery_safe_set_v3:
        expected_execution_guard_fields |= {
            "source_plan",
            "source_plan_sha256",
            "source_remote_static_attempt",
            "source_remote_direct_smoke_results_sha256",
            "source_remote_execution_log_sha256",
            "recovery_below_command_margin_requires_monotonic_progress",
            "recovery_after_first_command_margin_entry_must_remain_inside",
        }
    if set(execution_guard) != expected_execution_guard_fields:
        raise ValueError("Execution-guard settings differ from the registered schema.")
    if guard_schema not in {
        "robust_joint_limit_constraint_tightening_v1",
        official_recovery_v2,
        official_recovery_safe_set_v3,
    }:
        raise ValueError("Execution-guard schema identity differs.")
    expected_strategy = (
        "static_uniform_constraint_tightening"
        if guard_schema == "robust_joint_limit_constraint_tightening_v1"
        else (
            "static_uniform_constraint_tightening_plus_official_"
            "start_state_path_constraint_adapter"
        )
    )
    if execution_guard.get("strategy") != expected_strategy:
        raise ValueError("Execution-guard strategy identity differs.")
    if execution_guard.get("reserve_metric") != (
        "maximum_tracking_overshoot_toward_limit_rad"
    ):
        raise ValueError("Execution-guard reserve metric identity differs.")
    if execution_guard.get("reserve_scope") != (
        "all_arm_joints_all_executed_train_exact_steps"
    ):
        raise ValueError("Execution-guard reserve scope identity differs.")
    if tuple(execution_guard.get("applies_to", ())) != (
        "mink.ConfigurationLimit",
        "moveit_msgs/JointConstraint",
    ):
        raise ValueError("Execution guard must reuse the registered official limits.")
    if execution_guard.get("affects_action_selection") is not True:
        raise ValueError("Execution guard must tighten command selection.")
    if guard_schema == official_recovery_safe_set_v3:
        source_plan = str(execution_guard.get("source_plan"))
        if source_plan != "configs/sim/aloha_insertion_geometry_teacher_029.yaml":
            raise ValueError("Execution-guard source plan identity differs.")
        if file_sha256(_repository_path(source_plan)) != str(
            execution_guard.get("source_plan_sha256")
        ):
            raise ValueError("Execution-guard source plan hash differs.")
        if execution_guard.get("source_remote_static_attempt") != (
            "athena-plan029-static-001"
        ):
            raise ValueError("Execution-guard remote static attempt identity differs.")
        for key in (
            "source_remote_direct_smoke_results_sha256",
            "source_remote_execution_log_sha256",
        ):
            value = str(execution_guard.get(key))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"Execution-guard evidence hash is invalid: {key}.")

    source_audit_path = _repository_path(str(execution_guard.get("source_audit")))
    source_audit_sha256 = str(execution_guard.get("source_audit_sha256"))
    if file_sha256(source_audit_path) != source_audit_sha256:
        raise ValueError("Execution-guard source audit identity differs.")
    source_audit = _mapping(
        json.loads(source_audit_path.read_text(encoding="utf-8")),
        "execution-guard source audit",
    )
    expected_source_audit_id = (
        "m2-smolvla-athena-plan027-exact-audit-2026-08-15"
        if guard_schema == "robust_joint_limit_constraint_tightening_v1"
        else "m2-smolvla-athena-plan028-exact-audit-2026-08-15"
    )
    if source_audit.get("audit_id") != expected_source_audit_id:
        raise ValueError("Execution-guard source audit ID differs.")
    source_attempt = (
        "attempt_004"
        if guard_schema == "robust_joint_limit_constraint_tightening_v1"
        else "attempt_001"
    )
    source_exact = _mapping(
        _mapping(source_audit.get("exact"), "source exact").get(source_attempt),
        "source exact attempt",
    )
    if source_exact.get("report_sha256") != execution_guard.get(
        "source_exact_report_sha256"
    ):
        raise ValueError("Execution-guard source exact identity differs.")

    physical_joint_margin = float(
        execution_guard.get("physical_joint_limit_margin_rad", math.nan)
    )
    tracking_reserve = float(
        execution_guard.get("tracking_reserve_rad", math.nan)
    )
    command_joint_margin = float(
        execution_guard.get("command_joint_limit_margin_rad", math.nan)
    )
    source_tracking_reserve = float(
        source_exact.get("maximum_tracking_overshoot_toward_limit_rad", math.nan)
    )
    if not math.isclose(
        physical_joint_margin,
        0.01,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Execution guard changes the physical joint-limit margin.")
    if (
        not math.isfinite(tracking_reserve)
        or tracking_reserve <= 0.0
        or not math.isclose(
            tracking_reserve,
            source_tracking_reserve,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("Execution-guard reserve differs from train-exact evidence.")
    if (
        not math.isfinite(command_joint_margin)
        or command_joint_margin > 0.05
        or not math.isclose(
            command_joint_margin,
            physical_joint_margin + tracking_reserve,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("Execution-guard command margin is not physical plus reserve.")
    if not math.isclose(
        float(execution_diagnostics["joint_limit_margin_rad"]),
        physical_joint_margin,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Execution diagnostics must retain the physical joint margin.")
    official_start_recovery_adapter = (
        "default_planner_request_adapters/FixStartStatePathConstraints"
    )
    if guard_schema in {official_recovery_v2, official_recovery_safe_set_v3}:
        if execution_guard.get("start_state_path_constraint_adapter") != (
            official_start_recovery_adapter
        ):
            raise ValueError("Execution guard does not use the official MoveIt adapter.")
        if execution_guard.get("recovery_entry_condition") != (
            "physical_safe_but_tightened_path_constraint_invalid"
        ):
            raise ValueError("Execution-guard recovery entry condition differs.")
        if not math.isclose(
            float(execution_guard["recovery_prefix_joint_limit_margin_rad"]),
            physical_joint_margin,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Recovery prefix must retain the physical joint margin.")
        if not math.isclose(
            float(
                execution_guard[
                    "recovery_constrained_suffix_joint_limit_margin_rad"
                ]
            ),
            command_joint_margin,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Recovery suffix must re-enter the tightened joint margin.")
        for key in (
            "recovery_requires_positive_margin_progress",
            "recovery_forbidden_when_start_satisfies_command_margin",
        ):
            if execution_guard.get(key) is not True:
                raise ValueError(f"Execution-guard recovery contract must enable {key}.")
        if guard_schema == official_recovery_safe_set_v3:
            for key in (
                "recovery_below_command_margin_requires_monotonic_progress",
                "recovery_after_first_command_margin_entry_must_remain_inside",
            ):
                if execution_guard.get(key) is not True:
                    raise ValueError(
                        f"Execution-guard safe-set recovery must enable {key}."
                    )

    trajectory_execution = _mapping(
        plan.get("trajectory_execution"),
        "trajectory_execution",
    )
    expected_trajectory_execution = {
        "schema": "official_moveit_hybrid_planning_retained_reference_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_030.yaml",
        "source_plan_sha256": (
            "763ba45ed8ca8d84120dae99ca1375d915427e77f290784437d291575ad25f4d"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan030-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "1bdd559c8f61658a23d78e920ee0468ee0e2b34b42a9ceb66e56dd2c1fc81157"
        ),
        "source_exact_report_sha256": (
            "aaa033c6a1740bac40d9589b01372c059e9d242099b2aa6a5ab4c4f8d1029fa3"
        ),
        "observed_global_plan_attempts": 131,
        "observed_terminal_phase": "orient",
        "single_axis": "retain_and_follow_accepted_official_global_trajectory",
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if trajectory_execution != expected_trajectory_execution:
        raise ValueError("MoveIt retained-trajectory source evidence differs.")
    trajectory_source_plan = _repository_path(
        str(trajectory_execution["source_plan"])
    )
    trajectory_source_audit = _repository_path(
        str(trajectory_execution["source_audit"])
    )
    if file_sha256(trajectory_source_plan) != trajectory_execution[
        "source_plan_sha256"
    ]:
        raise ValueError("MoveIt retained-trajectory source plan identity differs.")
    if file_sha256(trajectory_source_audit) != trajectory_execution[
        "source_audit_sha256"
    ]:
        raise ValueError("MoveIt retained-trajectory source audit identity differs.")
    trajectory_source_evidence = _mapping(
        json.loads(trajectory_source_audit.read_text(encoding="utf-8")),
        "MoveIt retained-trajectory source audit",
    )
    trajectory_exact = _mapping(
        _mapping(trajectory_source_evidence.get("exact"), "trajectory exact").get(
            "attempt_002"
        ),
        "trajectory exact attempt",
    )
    if (
        trajectory_source_evidence.get("audit_id")
        != "m2-smolvla-athena-plan030-exact-audit-2026-08-15"
        or trajectory_exact.get("report_sha256")
        != trajectory_execution["source_exact_report_sha256"]
        or trajectory_exact.get("path_planner_attempts")
        != trajectory_execution["observed_global_plan_attempts"]
        or trajectory_exact.get("final_phase")
        != trajectory_execution["observed_terminal_phase"]
    ):
        raise ValueError("MoveIt retained-trajectory exact evidence differs.")

    terminal_control = _mapping(
        plan.get("terminal_control"),
        "terminal_control",
    )
    expected_terminal_control = {
        "schema": "official_mujoco_static_inverse_dynamics_position_feedforward_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_032.yaml",
        "source_plan_sha256": (
            "435630a5f5e3037a3580e4296e38d4bd9b2b5f9ea2c49f55bb4e261dca7e5645"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan032-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "082c5e6dda8665acdfbe87b2572dfc1cbdc016c85ea29273e6af71f6ccbdc244"
        ),
        "source_exact_report_sha256": (
            "f8ea0bb5514afe7c1bf64930b05ac40f52346b9c98ec35e9c2a539dbfc2acf5c"
        ),
        "source_final_reference_waypoint_l1_distance_rad": (
            0.03446431288757733
        ),
        "source_final_target_position_error_m": 0.02851971797645092,
        "handoff": "final_waypoint_within_moveit_simple_sampler_l1_tolerance",
        "single_axis": (
            "static_inverse_dynamics_feedforward_at_retained_final_waypoint"
        ),
        "official_mujoco_actuation_source": (
            "https://mujoco.readthedocs.io/en/stable/computation/"
            "index.html#actuation-model"
        ),
        "official_mujoco_force_balance_source": (
            "https://mujoco.readthedocs.io/en/stable/computation/"
            "index.html#passive-forces"
        ),
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if terminal_control != expected_terminal_control:
        raise ValueError("Terminal position-control source evidence differs.")
    terminal_source_plan = _repository_path(str(terminal_control["source_plan"]))
    terminal_source_audit = _repository_path(str(terminal_control["source_audit"]))
    if file_sha256(terminal_source_plan) != terminal_control["source_plan_sha256"]:
        raise ValueError("Terminal position-control source plan identity differs.")
    if file_sha256(terminal_source_audit) != terminal_control["source_audit_sha256"]:
        raise ValueError("Terminal position-control source audit identity differs.")
    terminal_evidence = _mapping(
        json.loads(terminal_source_audit.read_text(encoding="utf-8")),
        "terminal position-control source audit",
    )
    terminal_exact = _mapping(
        terminal_evidence.get("exact"),
        "terminal position-control exact evidence",
    )
    if (
        terminal_evidence.get("report_id")
        != "m2-smolvla-athena-plan032-exact-audit-2026-08-15"
        or _mapping(terminal_evidence.get("identity"), "terminal identity").get(
            "exact_report_sha256"
        )
        != terminal_control["source_exact_report_sha256"]
        or terminal_exact.get("final_reference_waypoint_l1_distance_rad")
        != terminal_control["source_final_reference_waypoint_l1_distance_rad"]
        or terminal_exact.get("final_target_position_error_m")
        != terminal_control["source_final_target_position_error_m"]
    ):
        raise ValueError("Terminal position-control exact evidence differs.")

    sparse_moment_repair = _mapping(
        plan.get("sparse_actuator_moment_repair"),
        "sparse_actuator_moment_repair",
    )
    expected_sparse_moment_repair = {
        "schema": "official_mujoco_sparse_actuator_moment_storage_repair_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_033.yaml",
        "source_plan_sha256": (
            "3fd522c8bab1c96012c4bf40d1a5aed84aea27ef03e31b1ceb7bdb2ff0887d14"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan033-local-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "89728307f15ebbb4d5d65adcc0a7ea31d1720563b3852ebb3bbf6633fc5efad0"
        ),
        "source_exact_report_sha256": (
            "a7516fb47d386ba3f87d63eda7ef787edf434aa914d1f2975e77eed49af65f52"
        ),
        "observed_mujoco_version": "3.8.1",
        "observed_actuator_moment_storage": "csr",
        "observed_actuator_moment_shape": [16],
        "required_csr_fields": [
            "moment_rownnz",
            "moment_rowadr",
            "moment_colind",
        ],
        "direct_one_dof_row_required": True,
        "dense_storage_remains_tested": True,
        "affects_force_balance": False,
        "affects_planner_or_teacher_behavior": False,
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if sparse_moment_repair != expected_sparse_moment_repair:
        raise ValueError("MuJoCo sparse actuator-moment repair evidence differs.")
    sparse_source_plan = _repository_path(str(sparse_moment_repair["source_plan"]))
    sparse_source_audit = _repository_path(
        str(sparse_moment_repair["source_audit"])
    )
    if file_sha256(sparse_source_plan) != sparse_moment_repair["source_plan_sha256"]:
        raise ValueError("MuJoCo sparse-moment source plan identity differs.")
    if (
        file_sha256(sparse_source_audit)
        != sparse_moment_repair["source_audit_sha256"]
    ):
        raise ValueError("MuJoCo sparse-moment source audit identity differs.")
    sparse_evidence = _mapping(
        json.loads(sparse_source_audit.read_text(encoding="utf-8")),
        "MuJoCo sparse-moment source audit",
    )
    sparse_execution = _mapping(
        sparse_evidence.get("execution"),
        "MuJoCo sparse-moment source execution",
    )
    sparse_failure = _mapping(
        sparse_evidence.get("failure"),
        "MuJoCo sparse-moment source failure",
    )
    if (
        sparse_evidence.get("audit_id")
        != "m2-smolvla-athena-plan033-local-exact-audit-2026-08-15"
        or sparse_execution.get("plan_sha256")
        != sparse_moment_repair["source_plan_sha256"]
        or sparse_execution.get("exact_report_sha256")
        != sparse_moment_repair["source_exact_report_sha256"]
        or sparse_failure.get("observed_actuator_moment_shape")
        != sparse_moment_repair["observed_actuator_moment_shape"]
    ):
        raise ValueError("MuJoCo sparse actuator-moment failure evidence differs.")

    joint_name_repair = _mapping(
        plan.get("joint_name_adapter_repair"),
        "joint_name_adapter_repair",
    )
    expected_joint_name_repair = {
        "schema": "action_contract_to_gym_mujoco_joint_name_adapter_repair_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_034.yaml",
        "source_plan_sha256": (
            "6dc5019b80522eca489b2d1473f0f801ef6c596cd5cbf38acfe1401baaa883ad"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan034-local-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "a96aab19647cb6622d2fff357d99712eccc19fb3ea81012acffd138bdcc3db97"
        ),
        "source_execution_log_sha256": (
            "f1552754b36e1d15588cfb23dcb59a9dbf06f7428be91d7d92dff9b54c26697d"
        ),
        "action_contract_names": list(EXPECTED_JOINT_NAMES),
        "gym_mujoco_names": [*LEFT_JOINTS, *RIGHT_JOINTS],
        "mapping_order_matches_arm_action_indices": True,
        "affects_force_balance": False,
        "affects_planner_or_teacher_behavior": False,
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if joint_name_repair != expected_joint_name_repair:
        raise ValueError("Gym MuJoCo joint-name adapter repair evidence differs.")
    joint_source_plan = _repository_path(str(joint_name_repair["source_plan"]))
    joint_source_audit = _repository_path(str(joint_name_repair["source_audit"]))
    if file_sha256(joint_source_plan) != joint_name_repair["source_plan_sha256"]:
        raise ValueError("Gym joint-name source plan identity differs.")
    if file_sha256(joint_source_audit) != joint_name_repair["source_audit_sha256"]:
        raise ValueError("Gym joint-name source audit identity differs.")
    joint_evidence = _mapping(
        json.loads(joint_source_audit.read_text(encoding="utf-8")),
        "Gym joint-name source audit",
    )
    joint_execution = _mapping(
        joint_evidence.get("execution"),
        "Gym joint-name source execution",
    )
    joint_failure = _mapping(
        joint_evidence.get("failure"),
        "Gym joint-name source failure",
    )
    if (
        joint_evidence.get("audit_id")
        != "m2-smolvla-athena-plan034-local-exact-audit-2026-08-15"
        or joint_execution.get("plan_sha256")
        != joint_name_repair["source_plan_sha256"]
        or joint_execution.get("execution_log_sha256")
        != joint_name_repair["source_execution_log_sha256"]
        or joint_failure.get("action_contract_joint_name")
        != joint_name_repair["action_contract_names"][0]
        or joint_failure.get("gym_joint_name")
        != joint_name_repair["gym_mujoco_names"][0]
    ):
        raise ValueError("Gym joint-name adapter failure evidence differs.")

    terminal_completion = _mapping(
        plan.get("terminal_completion_refresh"),
        "terminal_completion_refresh",
    )
    expected_terminal_completion = {
        "schema": "official_moveit_local_planner_terminal_completion_refresh_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_035.yaml",
        "source_plan_sha256": (
            "e7b048ea99d1ea7fd0eb997ce63baf986ad5a575131d8ee84dc24646d9168197"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan035-local-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "efd1531568f49a172763f3d8fc61b7b4d5644b2f0952d640d8ccad645df05cf7"
        ),
        "source_exact_report_sha256": (
            "2d1fa4e3afd3748a218c82220ea678972cc25a97514f9bacec6d71eca6a6174f"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-moveit-terminal-completion-refresh-"
            "preregistration-2026-08-15.json"
        ),
        "preregistration_sha256": (
            "0a66425598ca2293552633a930c518950f35f2b9f771cb9f7c68db440d56077c"
        ),
        "source_plan_step": 97,
        "source_target_position_error_at_plan_m": 0.0381825380027294,
        "source_teacher_maximum_cartesian_step_m": 0.012,
        "source_final_target_position_error_m": 0.026163499802350998,
        "source_final_arm_joint_l1_to_original_moveit_goal_rad": (
            0.0000296434154734015
        ),
        "terminal_goal_l1_tolerance_rad": 0.001,
        "completion_condition": (
            "terminal_control_active_and_observed_arm_state_within_l1_"
            "tolerance_of_uncompensated_moveit_goal"
        ),
        "post_completion": (
            "reset_reference_then_current_target_mink_then_moveit_fallback"
        ),
        "official_moveit_hybrid_planning_source": (
            "https://moveit.picknik.ai/humble/doc/examples/"
            "hybrid_planning/hybrid_planning_tutorial.html"
        ),
        "official_simple_sampler_source": (
            "https://github.com/moveit/moveit2/blob/2.5.9/moveit_ros/"
            "hybrid_planning/local_planner/trajectory_operator_plugins/"
            "src/simple_sampler.cpp"
        ),
        "official_local_planner_component_source": (
            "https://github.com/moveit/moveit2/blob/2.5.9/moveit_ros/"
            "hybrid_planning/local_planner/local_planner_component/src/"
            "local_planner_component.cpp"
        ),
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if terminal_completion != expected_terminal_completion:
        raise ValueError("MoveIt terminal-completion refresh evidence differs.")
    completion_source_plan = _repository_path(
        str(terminal_completion["source_plan"])
    )
    completion_source_audit = _repository_path(
        str(terminal_completion["source_audit"])
    )
    completion_preregistration = _repository_path(
        str(terminal_completion["preregistration"])
    )
    if file_sha256(completion_source_plan) != terminal_completion[
        "source_plan_sha256"
    ]:
        raise ValueError("MoveIt terminal-completion source plan identity differs.")
    if file_sha256(completion_source_audit) != terminal_completion[
        "source_audit_sha256"
    ]:
        raise ValueError("MoveIt terminal-completion source audit identity differs.")
    if file_sha256(completion_preregistration) != terminal_completion[
        "preregistration_sha256"
    ]:
        raise ValueError("MoveIt terminal-completion preregistration identity differs.")
    completion_evidence = _mapping(
        json.loads(completion_source_audit.read_text(encoding="utf-8")),
        "MoveIt terminal-completion source audit",
    )
    completion_execution = _mapping(
        completion_evidence.get("execution"),
        "MoveIt terminal-completion source execution",
    )
    completion_result = _mapping(
        completion_evidence.get("exact_result"),
        "MoveIt terminal-completion source result",
    )
    completion_preregistered = _mapping(
        json.loads(completion_preregistration.read_text(encoding="utf-8")),
        "MoveIt terminal-completion preregistration",
    )
    preregistered_source = _mapping(
        completion_preregistered.get("source"),
        "MoveIt terminal-completion preregistration source",
    )
    preregistered_failure = _mapping(
        completion_preregistered.get("observed_failure"),
        "MoveIt terminal-completion preregistration failure",
    )
    if (
        completion_evidence.get("audit_id")
        != "m2-smolvla-athena-plan035-local-exact-audit-2026-08-15"
        or completion_execution.get("plan_sha256")
        != terminal_completion["source_plan_sha256"]
        or completion_execution.get("exact_report_sha256")
        != terminal_completion["source_exact_report_sha256"]
        or completion_result.get("final_target_position_error_m")
        != terminal_completion["source_final_target_position_error_m"]
        or preregistered_source.get("exact_report_sha256")
        != terminal_completion["source_exact_report_sha256"]
        or preregistered_failure.get("moveit_plan_step")
        != terminal_completion["source_plan_step"]
        or preregistered_failure.get(
            "final_arm_joint_l1_to_original_moveit_goal_rad"
        )
        != terminal_completion[
            "source_final_arm_joint_l1_to_original_moveit_goal_rad"
        ]
    ):
        raise ValueError("MoveIt terminal-completion exact evidence differs.")

    task_contact_policy = _mapping(
        plan.get("task_contact_policy"),
        "task_contact_policy",
    )
    expected_task_contact_policy = {
        "schema": "phase_scoped_train_demonstrated_task_contact_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_036.yaml",
        "source_plan_sha256": (
            "3bb462191c2229734684ba37a9efa4e3827b3057139b8bb20942cad0e833acd2"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan036-local-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "1d4ca435b5a1d52ccebea228a3d7f423e0fb9558c61ea7b408639c903f1895b9"
        ),
        "source_exact_report_sha256": (
            "ee4c52484af8f07177111ffd755ae92b0ec92810373e97ec8aac51363bf57730"
        ),
        "source_calibration_contact_diagnostic_sha256": (
            "fa05af5c86a57a8db4f956e5806fc4fe71e0e96f59024b1d0cafbf5ac8dd7081"
        ),
        "source_final_contact_diagnostic_sha256": (
            "d89832f01a374ada292918e311d63112467d415b84736ee26fd147effe16eaff"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-phase-scoped-task-contact-"
            "preregistration-2026-08-15.json"
        ),
        "preregistration_sha256": (
            "0329905381c647980a54a59e14635ddacaf27231d74632f0227b954b5af9e398"
        ),
        "phases": ["descend", "grasp"],
        "allowed_unordered_geom_pairs": [
            ["table", "vx300s_right/10_right_gripper_finger"]
        ],
        "evidence_episode": 2,
        "evidence_seed": 10,
        "evidence_contact_steps": [
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            182,
            183,
            185,
        ],
        "evidence_first_grasp_step": 186,
        "evidence_terminal_reward": 4.0,
        "raw_contacts_reported": True,
        "all_other_unexpected_contacts_fail_closed": True,
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if task_contact_policy != expected_task_contact_policy:
        raise ValueError("Phase-scoped task-contact policy evidence differs.")
    contact_source_plan = _repository_path(str(task_contact_policy["source_plan"]))
    contact_source_audit = _repository_path(
        str(task_contact_policy["source_audit"])
    )
    contact_preregistration = _repository_path(
        str(task_contact_policy["preregistration"])
    )
    if file_sha256(contact_source_plan) != task_contact_policy["source_plan_sha256"]:
        raise ValueError("Task-contact source plan identity differs.")
    if file_sha256(contact_source_audit) != task_contact_policy[
        "source_audit_sha256"
    ]:
        raise ValueError("Task-contact source audit identity differs.")
    if file_sha256(contact_preregistration) != task_contact_policy[
        "preregistration_sha256"
    ]:
        raise ValueError("Task-contact preregistration identity differs.")
    contact_evidence = _mapping(
        json.loads(contact_source_audit.read_text(encoding="utf-8")),
        "task-contact source audit",
    )
    contact_execution = _mapping(
        contact_evidence.get("execution"),
        "task-contact source execution",
    )
    contact_diagnosis = _mapping(
        contact_evidence.get("contact_diagnosis"),
        "task-contact source diagnosis",
    )
    contact_preregistered = _mapping(
        json.loads(contact_preregistration.read_text(encoding="utf-8")),
        "task-contact preregistration",
    )
    contact_change = _mapping(
        contact_preregistered.get("registered_change"),
        "task-contact preregistered change",
    )
    if (
        contact_evidence.get("audit_id")
        != "m2-smolvla-athena-plan036-local-exact-audit-2026-08-15"
        or contact_execution.get("plan_sha256")
        != task_contact_policy["source_plan_sha256"]
        or contact_execution.get("exact_report_sha256")
        != task_contact_policy["source_exact_report_sha256"]
        or contact_diagnosis.get("calibration_contact_diagnostic_sha256")
        != task_contact_policy["source_calibration_contact_diagnostic_sha256"]
        or contact_diagnosis.get("unexpected_pair")
        != task_contact_policy["allowed_unordered_geom_pairs"][0]
        or contact_change.get("allowed_pair")
        != task_contact_policy["allowed_unordered_geom_pairs"][0]
        or contact_change.get("allowed_phases") != task_contact_policy["phases"]
    ):
        raise ValueError("Phase-scoped task-contact exact evidence differs.")

    execution_horizon = _mapping(
        plan.get("execution_horizon"),
        "execution_horizon",
    )
    expected_execution_horizon = {
        "schema": "measured_safe_path_execution_horizon_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_037.yaml",
        "source_plan_sha256": (
            "81eb1b480597670af2bbc11dd4e9d9becc3f960e036799fbb72cefde5cdb9263"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan037-local-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "797a8192c5cdb32554f0e6a7c1a27de27162f2116e8d3d7c11d119b53141fc2d"
        ),
        "source_exact_report_sha256": (
            "6b99e77a4b97df52e858ce3b8bde3632038dbc3e1955874f310b7e90d1a3690e"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-execution-horizon-preregistration-2026-08-15.json"
        ),
        "preregistration_sha256": (
            "07f2cd07c12139bb2e1eb4934062e6d419dddba7b9633309ce1a0f6008e0ac34"
        ),
        "source_maximum_steps": 500,
        "source_orient_to_descend_transition_step": 478,
        "source_final_target_position_error_m": 0.00954453926533461,
        "train_calibration_post_grasp_steps": 108,
        "maximum_steps": 750,
        "single_axis": "maximum_episode_steps_500_to_750",
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_planner_or_controller": False,
        "affects_seed_or_label_boundaries": False,
    }
    if execution_horizon != expected_execution_horizon:
        raise ValueError("Measured execution-horizon evidence differs.")
    horizon_source_plan = _repository_path(str(execution_horizon["source_plan"]))
    horizon_source_audit = _repository_path(
        str(execution_horizon["source_audit"])
    )
    horizon_preregistration = _repository_path(
        str(execution_horizon["preregistration"])
    )
    if file_sha256(horizon_source_plan) != execution_horizon["source_plan_sha256"]:
        raise ValueError("Execution-horizon source plan identity differs.")
    if file_sha256(horizon_source_audit) != execution_horizon[
        "source_audit_sha256"
    ]:
        raise ValueError("Execution-horizon source audit identity differs.")
    if file_sha256(horizon_preregistration) != execution_horizon[
        "preregistration_sha256"
    ]:
        raise ValueError("Execution-horizon preregistration identity differs.")
    horizon_evidence = _mapping(
        json.loads(horizon_source_audit.read_text(encoding="utf-8")),
        "execution-horizon source audit",
    )
    horizon_execution = _mapping(
        horizon_evidence.get("execution"),
        "execution-horizon source execution",
    )
    horizon_diagnosis = _mapping(
        horizon_evidence.get("horizon_diagnosis"),
        "execution-horizon source diagnosis",
    )
    horizon_preregistered = _mapping(
        json.loads(horizon_preregistration.read_text(encoding="utf-8")),
        "execution-horizon preregistration",
    )
    horizon_change = _mapping(
        horizon_preregistered.get("registered_change"),
        "execution-horizon preregistered change",
    )
    if (
        horizon_evidence.get("audit_id")
        != "m2-smolvla-athena-plan037-local-exact-audit-2026-08-15"
        or horizon_execution.get("plan_sha256")
        != execution_horizon["source_plan_sha256"]
        or horizon_execution.get("exact_report_sha256")
        != execution_horizon["source_exact_report_sha256"]
        or horizon_diagnosis.get("orient_to_descend_transition_step")
        != execution_horizon["source_orient_to_descend_transition_step"]
        or horizon_change.get("new_maximum_steps")
        != execution_horizon["maximum_steps"]
    ):
        raise ValueError("Measured execution-horizon exact evidence differs.")

    teacher = _mapping(plan.get("teacher"), "teacher")
    expected_teacher_fields = {field.name for field in fields(InsertionTeacherSettings)}
    if set(teacher) != expected_teacher_fields:
        raise ValueError("Geometry-teacher settings do not match the registered schema.")
    teacher_settings = InsertionTeacherSettings(**teacher)

    evaluation = _mapping(plan.get("evaluation"), "evaluation")
    if int(evaluation.get("maximum_steps", -1)) != execution_horizon["maximum_steps"]:
        raise ValueError("Evaluation maximum steps differs from its registered horizon.")

    moveit_ik_budget = _mapping(
        plan.get("moveit_ik_budget"),
        "moveit_ik_budget",
    )
    expected_moveit_ik_budget = {
        "schema": "official_moveit_subgroup_ik_timeout_budget_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_040.yaml",
        "source_plan_sha256": (
            "564e787da05af530241e44dd420e23d87e9dac47a8f1fc59c60b06f2d64b2768"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan040-local-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "5210fd20e3c6a9051e6b0d109aac3ed0fc0e5b67c2935b49325723d2a9fd5571"
        ),
        "source_exact_report_sha256": (
            "e959de95684c4c309b3d563a9984ce31add4967bbc10ff0fe5bfd44915383169"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-moveit-ik-timeout-2s-preregistration-2026-08-15.json"
        ),
        "preregistration_sha256": (
            "fe483740cec044c289bb2b5512785a7595051e87efa2867cffb3a46b3324cc84"
        ),
        "source_failure_step": 249,
        "source_solver_failure": "bimanual_lma_ik_failed",
        "source_ik_timeout_s": 0.5,
        "ik_timeout_s": 2.0,
        "allowed_planning_time_s": 0.25,
        "single_axis": "moveit_lma_ik_timeout_0_50_to_2_00_seconds",
        "official_moveit_robot_state_source": (
            "https://moveit.picknik.ai/humble/api/html/"
            "classmoveit_1_1core_1_1RobotState.html"
        ),
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_planner_or_controller": False,
        "affects_horizon": False,
        "affects_seed_or_label_boundaries": False,
    }
    if moveit_ik_budget != expected_moveit_ik_budget:
        raise ValueError("MoveIt IK-timeout budget evidence differs.")
    ik_budget_source_plan = _repository_path(str(moveit_ik_budget["source_plan"]))
    ik_budget_source_audit = _repository_path(
        str(moveit_ik_budget["source_audit"])
    )
    ik_budget_preregistration = _repository_path(
        str(moveit_ik_budget["preregistration"])
    )
    if file_sha256(ik_budget_source_plan) != moveit_ik_budget["source_plan_sha256"]:
        raise ValueError("MoveIt IK-timeout source plan identity differs.")
    if file_sha256(ik_budget_source_audit) != moveit_ik_budget[
        "source_audit_sha256"
    ]:
        raise ValueError("MoveIt IK-timeout source audit identity differs.")
    if file_sha256(ik_budget_preregistration) != moveit_ik_budget[
        "preregistration_sha256"
    ]:
        raise ValueError("MoveIt IK-timeout preregistration identity differs.")
    ik_budget_evidence = _mapping(
        json.loads(ik_budget_source_audit.read_text(encoding="utf-8")),
        "MoveIt IK-timeout source audit",
    )
    ik_budget_execution = _mapping(
        ik_budget_evidence.get("execution"),
        "MoveIt IK-timeout source execution",
    )
    ik_budget_failure = _mapping(
        ik_budget_evidence.get("failure_boundary"),
        "MoveIt IK-timeout source failure",
    )
    ik_budget_preregistered = _mapping(
        json.loads(ik_budget_preregistration.read_text(encoding="utf-8")),
        "MoveIt IK-timeout preregistration",
    )
    ik_budget_change = _mapping(
        ik_budget_preregistered.get("registered_change"),
        "MoveIt IK-timeout preregistered change",
    )
    if (
        ik_budget_evidence.get("audit_id")
        != "m2-smolvla-athena-plan040-local-exact-audit-2026-08-15"
        or ik_budget_execution.get("plan_sha256")
        != moveit_ik_budget["source_plan_sha256"]
        or ik_budget_execution.get("exact_report_sha256")
        != moveit_ik_budget["source_exact_report_sha256"]
        or ik_budget_failure.get("step")
        != moveit_ik_budget["source_failure_step"]
        or ik_budget_failure.get("solver_failure")
        != moveit_ik_budget["source_solver_failure"]
        or ik_budget_failure.get("registered_moveit_ik_timeout_s")
        != moveit_ik_budget["source_ik_timeout_s"]
        or ik_budget_change.get("new_ik_timeout_s")
        != moveit_ik_budget["ik_timeout_s"]
        or ik_budget_change.get("allowed_planning_time_s")
        != moveit_ik_budget["allowed_planning_time_s"]
    ):
        raise ValueError("MoveIt IK-timeout exact evidence differs.")

    deterministic_moveit_ik = _mapping(
        plan.get("deterministic_moveit_ik"),
        "deterministic_moveit_ik",
    )
    expected_deterministic_moveit_ik = {
        "schema": "official_moveit_deterministic_subgroup_ik_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_041.yaml",
        "source_plan_sha256": (
            "f9a1168f29d0ca59cae2a722b7826e3c35038f966fcaa00fa939249470d38a2c"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan041-local-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "cc0ef524ce966c1549cabdf0ef8463e87b2c71b3d7e2fc4f7ed28acf148c260a"
        ),
        "source_exact_report_sha256": (
            "4d2fbb92b790d9e47b7696539509868644d0530a0b0c657d678d156f62b96813"
        ),
        "plan042_preregistration": (
            "reports/training/"
            "m2-smolvla-deterministic-moveit-subgroup-ik-preregistration-"
            "2026-08-15.json"
        ),
        "plan042_preregistration_sha256": (
            "b70268f5dd4d183086bb375c03819ac144d254e772d79493b97dd842726a5dbb"
        ),
        "plan042_runtime_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan042-runtime-identity-audit-2026-08-15.json"
        ),
        "plan042_runtime_audit_sha256": (
            "0d2653d3f36b3504eae4e6fa117f88b592a8ee14578cdb0e2b47361317ba2fa1"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-deterministic-moveit-subgroup-ik-frame-"
            "preregistration-2026-08-15.json"
        ),
        "preregistration_sha256": (
            "56af21a301d38a5256fa61c2e1dbb945d9b89a6bb24a014c8f81edf961d2084f"
        ),
        "repeat_report": (
            "reports/training/"
            "m2-smolvla-aloha-moveit-deterministic-ik-plan043-2026-08-15.json"
        ),
        "repeat_report_sha256": (
            "52c8cacddd341d0215c60cc469e47d4025265c0d4410ef1ddb12a71b55d5b279"
        ),
        "model_parity_report": (
            "reports/training/"
            "m2-smolvla-aloha-moveit-model-parity-plan043-2026-08-15.json"
        ),
        "model_parity_report_sha256": (
            "e9926845045dd39f28a55a8cd7019495f62bb19b556a0d79b9c8adc0a541eb29"
        ),
        "search_mode": "deterministic_seeded_moveit_subgroup_multistart_v1",
        "seed": 2210,
        "maximum_attempts": 256,
        "outer_timeout_s": 2.0,
        "subgroup_order": ["left_arm", "right_arm"],
        "solver": "lma_kinematics_plugin/LMAKinematicsPlugin",
        "solver_call": "getPositionIK",
        "target_input_frame": "world",
        "solver_base_frames": [
            "vx300s_left/base_link",
            "vx300s_right/base_link",
        ],
        "solver_tip_frames": [
            "vx300s_left/ee_gripper_link",
            "vx300s_right/ee_gripper_link",
        ],
        "target_transform": "moveit::core::RobotState::setToIKSolverFrame",
        "full_state_validation": [
            "satisfiesBounds",
            "joint_path_constraints",
            "self_collision",
        ],
        "official_moveit_robot_state_source": (
            "https://github.com/moveit/moveit2/blob/2.5.9/"
            "moveit_core/robot_state/src/robot_state.cpp#L1671-L1858"
        ),
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_planner_or_controller": False,
        "affects_horizon": False,
        "affects_seed_or_label_boundaries": False,
    }
    if deterministic_moveit_ik != expected_deterministic_moveit_ik:
        raise ValueError("Deterministic MoveIt subgroup-IK evidence differs.")
    deterministic_paths = {
        name: _repository_path(str(deterministic_moveit_ik[name]))
        for name in (
            "source_plan",
            "source_audit",
            "plan042_preregistration",
            "plan042_runtime_audit",
            "preregistration",
            "repeat_report",
            "model_parity_report",
        )
    }
    for name, path in deterministic_paths.items():
        if file_sha256(path) != deterministic_moveit_ik[f"{name}_sha256"]:
            raise ValueError(f"Deterministic MoveIt evidence hash differs for {name}.")
    deterministic_source = _mapping(
        json.loads(deterministic_paths["source_audit"].read_text(encoding="utf-8")),
        "deterministic MoveIt source audit",
    )
    deterministic_failure = _mapping(
        deterministic_source.get("failure_boundary"),
        "deterministic MoveIt source failure",
    )
    plan042_runtime = _mapping(
        json.loads(
            deterministic_paths["plan042_runtime_audit"].read_text(encoding="utf-8")
        ),
        "Plan042 runtime audit",
    )
    deterministic_repeat = _mapping(
        json.loads(deterministic_paths["repeat_report"].read_text(encoding="utf-8")),
        "deterministic MoveIt repeat report",
    )
    repeat_contract = _mapping(
        deterministic_repeat.get("search_contract"),
        "deterministic MoveIt repeat contract",
    )
    repeat_result = _mapping(
        deterministic_repeat.get("repeat_result"),
        "deterministic MoveIt repeat result",
    )
    if (
        deterministic_source.get("audit_id")
        != "m2-smolvla-athena-plan041-local-exact-audit-2026-08-15"
        or deterministic_failure.get("solver_failure")
        != "bimanual_lma_ik_failed"
        or deterministic_failure.get("registered_moveit_ik_timeout_s")
        != deterministic_moveit_ik["outer_timeout_s"]
        or plan042_runtime.get("status") != "failed_before_plan_execution"
        or deterministic_repeat.get("status") != "passed"
        or repeat_contract.get("mode") != deterministic_moveit_ik["search_mode"]
        or repeat_contract.get("seed") != deterministic_moveit_ik["seed"]
        or repeat_contract.get("maximum_attempts")
        != deterministic_moveit_ik["maximum_attempts"]
        or repeat_contract.get("solver_base_frames")
        != deterministic_moveit_ik["solver_base_frames"]
        or repeat_contract.get("solver_tip_frames")
        != deterministic_moveit_ik["solver_tip_frames"]
        or repeat_result.get("both_status_ok") is not True
        or repeat_result.get("goal_vectors_exactly_equal") is not True
        or repeat_result.get("attempt_counts_equal") is not True
    ):
        raise ValueError("Deterministic MoveIt subgroup-IK proof differs.")

    full_pose_group_selection = _mapping(
        plan.get("full_pose_group_selection"),
        "full_pose_group_selection",
    )
    expected_full_pose_group_selection = {
        "schema": "explicit_moveit_registered_full_pose_groups_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_044.yaml",
        "source_plan_sha256": (
            "c573dbf48b0e4d68de24e55e16a01c4c58e173050c4a9085612863d6adb1951e"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan044-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "78342bdde53f439a6663d184fddc271b480a4d2523048162f42c582c7b8ba039"
        ),
        "source_exact_report_sha256": (
            "d61823063721d53c50b0929af710d7512cd0e44afd46b5e0f6074c6d7d32b989"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-explicit-moveit-full-pose-group-selection-"
            "preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "cf6c04b3509358efbbc97e1a85b614173d98108eb0e3213ad8f45452f56d5db1"
        ),
        "runtime_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan045-runtime-identity-audit-2026-08-16.json"
        ),
        "runtime_audit_sha256": (
            "f7703845776229a10c4b98bfb1bc02c279b8302a87690b8d4a3622feb3a3351b"
        ),
        "group_selection_mode": "explicit_registered_groups_v1",
        "full_pose_groups": ["left_arm", "right_arm"],
        "position_priority_groups": [
            "left_arm_position_priority",
            "right_arm_position_priority",
        ],
        "lookup": "moveit_core_robot_model_getJointModelGroup",
        "group_order_is_fixed": True,
        "null_group_fails_closed": True,
        "single_axis": (
            "replace_bimanual_getSubgroups_enumeration_with_explicit_"
            "registered_full_pose_arm_groups"
        ),
        "affects_solver_or_planner": False,
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if full_pose_group_selection != expected_full_pose_group_selection:
        raise ValueError("Explicit MoveIt full-pose group evidence differs.")
    full_pose_source_plan = _repository_path(
        str(full_pose_group_selection["source_plan"])
    )
    full_pose_source_audit_path = _repository_path(
        str(full_pose_group_selection["source_audit"])
    )
    full_pose_preregistration_path = _repository_path(
        str(full_pose_group_selection["preregistration"])
    )
    full_pose_runtime_audit_path = _repository_path(
        str(full_pose_group_selection["runtime_audit"])
    )
    if (
        file_sha256(full_pose_source_plan)
        != full_pose_group_selection["source_plan_sha256"]
        or file_sha256(full_pose_source_audit_path)
        != full_pose_group_selection["source_audit_sha256"]
        or file_sha256(full_pose_preregistration_path)
        != full_pose_group_selection["preregistration_sha256"]
        or file_sha256(full_pose_runtime_audit_path)
        != full_pose_group_selection["runtime_audit_sha256"]
    ):
        raise ValueError("Explicit MoveIt full-pose group evidence hash differs.")
    full_pose_source_audit = _mapping(
        json.loads(full_pose_source_audit_path.read_text(encoding="utf-8")),
        "full-pose group source audit",
    )
    full_pose_source_failure = _mapping(
        full_pose_source_audit.get("failure_boundary"),
        "full-pose group source failure",
    )
    full_pose_runtime_audit = _mapping(
        json.loads(full_pose_runtime_audit_path.read_text(encoding="utf-8")),
        "full-pose group runtime audit",
    )
    full_pose_runtime_identity = _mapping(
        full_pose_runtime_audit.get("identity"),
        "full-pose group runtime identity",
    )
    full_pose_regression = _mapping(
        full_pose_runtime_audit.get("full_pose_regression"),
        "full-pose group regression",
    )
    if (
        full_pose_source_audit.get("audit_id")
        != "m2-smolvla-athena-plan044-local-exact-audit-2026-08-16"
        or full_pose_source_failure.get("solver_failure")
        != "bimanual_subgroup_count_differs"
        or full_pose_runtime_audit.get("status") != "passed_before_exact"
        or full_pose_runtime_identity.get("ik_group_selection_mode")
        != full_pose_group_selection["group_selection_mode"]
        or full_pose_runtime_identity.get("full_pose_groups")
        != full_pose_group_selection["full_pose_groups"]
        or full_pose_runtime_identity.get("position_priority_groups")
        != full_pose_group_selection["position_priority_groups"]
        or full_pose_regression.get("status") != "passed"
        or full_pose_regression.get("both_status_ok") is not True
        or full_pose_regression.get("bimanual_subgroup_count_failure_observed")
        is not False
    ):
        raise ValueError("Explicit MoveIt full-pose group source proof differs.")

    full_pose_cartesian_backoff = _mapping(
        plan.get("full_pose_cartesian_backoff"),
        "full_pose_cartesian_backoff",
    )
    expected_full_pose_cartesian_backoff = {
        "schema": "official_moveit_full_pose_cartesian_backoff_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_045.yaml",
        "source_plan_sha256": (
            "2fcd6b157fede3c76d72bcc3590a8d197d50c74abb3d55508bb12d8ff6b0f488"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan045-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "cf26736f139143b24a6a78819009ef8f6b196db508f0829e9cae9cf69a2d5a63"
        ),
        "source_exact_report_sha256": (
            "5c6cddbdff6e3b47932242c0a8ab00a95e9a1c882a86cec272ef37c563545e7e"
        ),
        "diagnostic_request_response_sha256": (
            "ee23d700eef68cedf30d45eb91faaaca89d74f8c1531b4db23c01c56755cdfb6"
        ),
        "diagnostic_probe_sha256": (
            "add43c591cae700b2c1e0c5c7e22127df596cccc85953b83a9ad81b044fd19cc"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-official-moveit-full-pose-cartesian-backoff-"
            "preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "d20878b9cb3b335d7112944d342430e490dac34d9cf96b893efd4569595d7f95"
        ),
        "activation_phases": ["approach", "orient"],
        "activation_after": ["mink_qp_failed", "direct_full_pose_moveit_lma_failed"],
        "fractions_largest_first": [0.75, 0.5, 0.25, 0.125, 0.1, 0.05],
        "position_interpolation": "linear_current_to_requested",
        "orientation_interpolation": (
            "shortest_arc_quaternion_slerp_current_to_requested"
        ),
        "minimum_linear_progress_m": 0.00025,
        "minimum_angular_progress_rad": 0.001,
        "progress_rule": (
            "maximum_arm_linear_or_angular_progress_meets_its_bound"
        ),
        "waypoint_solver": "lma_kinematics_plugin/LMAKinematicsPlugin",
        "waypoint_solver_mode": "full_pose",
        "global_planner": "official_moveit2_ompl_rrtconnect",
        "feedback_replans_original_target": True,
        "position_priority_fallback_order": (
            "after_full_pose_cartesian_backoff_exhaustion_in_approach_only"
        ),
        "single_axis": (
            "official_full_pose_lma_cartesian_position_and_quaternion_"
            "backoff_before_position_only_fallback"
        ),
        "affects_final_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if full_pose_cartesian_backoff != expected_full_pose_cartesian_backoff:
        raise ValueError("Official MoveIt full-pose Cartesian backoff evidence differs.")
    full_pose_backoff_source_path = _repository_path(
        str(full_pose_cartesian_backoff["source_plan"])
    )
    full_pose_backoff_audit_path = _repository_path(
        str(full_pose_cartesian_backoff["source_audit"])
    )
    full_pose_backoff_preregistration_path = _repository_path(
        str(full_pose_cartesian_backoff["preregistration"])
    )
    if (
        file_sha256(full_pose_backoff_source_path)
        != full_pose_cartesian_backoff["source_plan_sha256"]
        or file_sha256(full_pose_backoff_audit_path)
        != full_pose_cartesian_backoff["source_audit_sha256"]
        or file_sha256(full_pose_backoff_preregistration_path)
        != full_pose_cartesian_backoff["preregistration_sha256"]
    ):
        raise ValueError("Official MoveIt full-pose backoff evidence hash differs.")
    full_pose_backoff_audit = _mapping(
        json.loads(full_pose_backoff_audit_path.read_text(encoding="utf-8")),
        "full-pose Cartesian backoff source audit",
    )
    full_pose_backoff_failure = _mapping(
        full_pose_backoff_audit.get("failure_boundary"),
        "full-pose Cartesian backoff source failure",
    )
    full_pose_backoff_diagnostic = _mapping(
        full_pose_backoff_audit.get("attempt_scoped_diagnostic"),
        "full-pose Cartesian backoff diagnostic",
    )
    if (
        full_pose_backoff_audit.get("audit_id")
        != "m2-smolvla-athena-plan045-local-exact-audit-2026-08-16"
        or full_pose_backoff_failure.get("step") != 225
        or full_pose_backoff_failure.get("phase") != "orient"
        or full_pose_backoff_failure.get("solver_failure")
        != "bimanual_lma_ik_failed"
        or full_pose_backoff_diagnostic.get(
            "full_pose_bimanual_largest_successful_fraction"
        )
        != 0.125
        or full_pose_backoff_diagnostic.get(
            "full_pose_bimanual_successful_fractions"
        )
        != [0.125, 0.1, 0.05]
        or full_pose_backoff_diagnostic.get(
            "full_pose_backoff_uses_official_lma_and_ompl"
        )
        is not True
    ):
        raise ValueError("Official MoveIt full-pose backoff source proof differs.")

    joint_margin_candidate_selection = _mapping(
        plan.get("joint_margin_candidate_selection"),
        "joint_margin_candidate_selection",
    )
    expected_joint_margin_candidate_selection = {
        "schema": "deterministic_moveit_joint_margin_candidate_selection_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_046.yaml",
        "source_plan_sha256": (
            "a80b3490044e45ff2603ada009c4bd764a5e5df60271308ae8d69c3c495f23b6"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan046-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "9101b0787a53f5ae2093f9528c3da1c60b83150117a5641ab2252a442978de9e"
        ),
        "source_exact_report_sha256": (
            "dcf2667b3994d1093c14f162f4a17468cd3c2544d94ff28babaa6d52e04fc1b3"
        ),
        "diagnostic_request_response_sha256": (
            "8f5fb9d4457c04e25a9de7b68ad466f89a19738b137c6d2023e27f15c697d880"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-deterministic-moveit-joint-margin-candidate-selection-"
            "preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "ac299843778da44a7f4b30421098f54e16648e18a04e7694dd7bafc2bf820ddf"
        ),
        "runtime_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan047-runtime-identity-audit-2026-08-16.json"
        ),
        "runtime_audit_sha256": (
            "f7a641fe79d8fc7f6139ccc26b5951c2c325986ec4e0c2747e53801a5c8426a2"
        ),
        "repeat_report": (
            "reports/training/"
            "m2-smolvla-aloha-moveit-joint-margin-selection-repeat-"
            "plan047-2026-08-16.json"
        ),
        "repeat_report_sha256": (
            "30d8cf0ff8c367a7aa7e5b5124ed048825f36ac64d4b88d016e14f4985125dfe"
        ),
        "model_parity_report": (
            "reports/training/"
            "m2-smolvla-aloha-moveit-model-parity-plan047-2026-08-16.json"
        ),
        "model_parity_report_sha256": (
            "e68a6627b65fdd7dd05a7153d5a217a32dc8656aefd23718986f7c9a54222679"
        ),
        "source_failure_step": 257,
        "source_failure_phase": "orient",
        "source_solver_failure": "bimanual_lma_ik_failed",
        "source_pre_step_minimum_margin_joint": "right_wrist_rotate",
        "source_pre_step_minimum_margin_rad": 0.0466891670989988,
        "registered_command_margin_rad": 0.04540462255477905,
        "selection_mode": (
            "deterministic_maximum_minimum_joint_limit_margin_v1"
        ),
        "candidate_generator": (
            "lma_kinematics_plugin/LMAKinematicsPlugin"
        ),
        "task_modes": ["full_pose", "position_priority"],
        "validity_filter": [
            "satisfiesBounds",
            "registered_joint_path_constraints",
            "self_collision_free",
            "registered_task_space_tolerance",
        ],
        "primary_objective": "maximize_minimum_arm_joint_limit_margin_rad",
        "secondary_objective": (
            "minimize_maximum_absolute_arm_joint_delta_from_start_rad"
        ),
        "final_tie_break": "lowest_deterministic_attempt_index",
        "seed": 2210,
        "maximum_attempts": 256,
        "outer_timeout_s": 2.0,
        "global_planner": "official_moveit2_ompl_rrtconnect",
        "selection_diagnostics_required": [
            "valid_ik_candidate_count",
            "selected_ik_attempt",
            "selected_ik_minimum_joint_limit_margin_rad",
            "selected_ik_maximum_start_delta_rad",
        ],
        "single_axis": (
            "replace_first_valid_lma_candidate_with_deterministic_maximum_"
            "minimum_joint_margin_selection"
        ),
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_planner_or_controller": False,
        "affects_horizon": False,
        "affects_seed_or_label_boundaries": False,
    }
    if joint_margin_candidate_selection != expected_joint_margin_candidate_selection:
        raise ValueError("MoveIt joint-margin candidate-selection evidence differs.")
    candidate_source_path = _repository_path(
        str(joint_margin_candidate_selection["source_plan"])
    )
    candidate_audit_path = _repository_path(
        str(joint_margin_candidate_selection["source_audit"])
    )
    candidate_preregistration_path = _repository_path(
        str(joint_margin_candidate_selection["preregistration"])
    )
    candidate_runtime_audit_path = _repository_path(
        str(joint_margin_candidate_selection["runtime_audit"])
    )
    candidate_repeat_report_path = _repository_path(
        str(joint_margin_candidate_selection["repeat_report"])
    )
    candidate_model_parity_path = _repository_path(
        str(joint_margin_candidate_selection["model_parity_report"])
    )
    if (
        file_sha256(candidate_source_path)
        != joint_margin_candidate_selection["source_plan_sha256"]
        or file_sha256(candidate_audit_path)
        != joint_margin_candidate_selection["source_audit_sha256"]
        or file_sha256(candidate_preregistration_path)
        != joint_margin_candidate_selection["preregistration_sha256"]
        or file_sha256(candidate_runtime_audit_path)
        != joint_margin_candidate_selection["runtime_audit_sha256"]
        or file_sha256(candidate_repeat_report_path)
        != joint_margin_candidate_selection["repeat_report_sha256"]
        or file_sha256(candidate_model_parity_path)
        != joint_margin_candidate_selection["model_parity_report_sha256"]
    ):
        raise ValueError("MoveIt joint-margin candidate evidence hash differs.")
    candidate_audit = _mapping(
        json.loads(candidate_audit_path.read_text(encoding="utf-8")),
        "MoveIt joint-margin candidate source audit",
    )
    candidate_execution = _mapping(
        candidate_audit.get("execution"),
        "MoveIt joint-margin candidate source execution",
    )
    candidate_plan = _mapping(
        candidate_audit.get("plan"),
        "MoveIt joint-margin candidate source plan",
    )
    candidate_failure = _mapping(
        candidate_audit.get("failure_boundary"),
        "MoveIt joint-margin candidate source failure",
    )
    candidate_diagnostic = _mapping(
        candidate_audit.get("attempt_scoped_diagnostic"),
        "MoveIt joint-margin candidate source diagnostic",
    )
    candidate_root_cause = _mapping(
        candidate_audit.get("root_cause"),
        "MoveIt joint-margin candidate source root cause",
    )
    candidate_preregistered = _mapping(
        json.loads(candidate_preregistration_path.read_text(encoding="utf-8")),
        "MoveIt joint-margin candidate preregistration",
    )
    candidate_registered_change = _mapping(
        candidate_preregistered.get("registered_change"),
        "MoveIt joint-margin candidate registered change",
    )
    candidate_runtime_audit = _mapping(
        json.loads(candidate_runtime_audit_path.read_text(encoding="utf-8")),
        "MoveIt joint-margin candidate runtime audit",
    )
    candidate_runtime_identity = _mapping(
        candidate_runtime_audit.get("identity"),
        "MoveIt joint-margin candidate runtime identity",
    )
    candidate_repeat_report = _mapping(
        json.loads(candidate_repeat_report_path.read_text(encoding="utf-8")),
        "MoveIt joint-margin candidate repeat report",
    )
    candidate_repeat_result = _mapping(
        candidate_repeat_report.get("repeat_result"),
        "MoveIt joint-margin candidate repeat result",
    )
    candidate_model_parity = _mapping(
        json.loads(candidate_model_parity_path.read_text(encoding="utf-8")),
        "MoveIt joint-margin candidate model parity",
    )
    for key in (
        "selection_mode",
        "candidate_generator",
        "task_modes",
        "validity_filter",
        "primary_objective",
        "secondary_objective",
        "final_tie_break",
        "seed",
        "maximum_attempts",
        "outer_timeout_s",
        "global_planner",
        "selection_diagnostics_required",
    ):
        if candidate_registered_change.get(key) != joint_margin_candidate_selection[key]:
            raise ValueError(
                f"MoveIt joint-margin candidate registration differs for {key}."
            )
    if (
        candidate_audit.get("audit_id")
        != "m2-smolvla-athena-plan046-local-exact-audit-2026-08-16"
        or candidate_plan.get("sha256")
        != joint_margin_candidate_selection["source_plan_sha256"]
        or candidate_execution.get("exact_report_sha256")
        != joint_margin_candidate_selection["source_exact_report_sha256"]
        or candidate_failure.get("step")
        != joint_margin_candidate_selection["source_failure_step"]
        or candidate_failure.get("phase")
        != joint_margin_candidate_selection["source_failure_phase"]
        or candidate_failure.get("solver_failure")
        != joint_margin_candidate_selection["source_solver_failure"]
        or candidate_failure.get("pre_step_minimum_margin_joint")
        != joint_margin_candidate_selection[
            "source_pre_step_minimum_margin_joint"
        ]
        or candidate_failure.get("pre_step_minimum_margin_rad")
        != joint_margin_candidate_selection["source_pre_step_minimum_margin_rad"]
        or candidate_failure.get("registered_command_margin_rad")
        != joint_margin_candidate_selection["registered_command_margin_rad"]
        or candidate_diagnostic.get("request_response_sha256")
        != joint_margin_candidate_selection[
            "diagnostic_request_response_sha256"
        ]
        or candidate_root_cause.get("classification")
        != "first_valid_local_ik_continuation_consumes_remaining_active_joint_margin"
        or candidate_runtime_audit.get("status") != "passed_before_exact"
        or candidate_runtime_identity.get("ik_candidate_selection_mode")
        != joint_margin_candidate_selection["selection_mode"]
        or candidate_repeat_report.get("status") != "passed"
        or candidate_repeat_result.get("goal_vectors_exactly_equal") is not True
        or candidate_repeat_result.get("trajectory_vectors_exactly_equal") is not True
        or candidate_repeat_result.get("selected_attempts_equal") is not True
        or candidate_repeat_result.get("selected_margins_equal") is not True
        or candidate_model_parity.get("status") != "passed"
        or candidate_model_parity.get("sample_count") != 5
    ):
        raise ValueError("MoveIt joint-margin candidate source proof differs.")

    active_set_trust_region = _mapping(
        plan.get("active_set_cartesian_trust_region"),
        "active_set_cartesian_trust_region",
    )
    expected_active_set_trust_region = {
        "schema": "active_set_cartesian_trust_region_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_047.yaml",
        "source_plan_sha256": (
            "bcc564976f07b39d3746d738b33aecda9a2971f3929b28266467bbde7a1020ac"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan047-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "f7d37e750b90d753f3e2a43eb698596e300a09fc54af59f17d51104bc39f7bf8"
        ),
        "source_exact_report_sha256": (
            "5a3db61675ec216d8323dcbd4b9cb29137411335d062a24a61a5cba4f7de4206"
        ),
        "discarded_pick_ik_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan049-pick-ik-global-runtime-audit-2026-08-16.json"
        ),
        "discarded_pick_ik_audit_sha256": (
            "d147b3af1766bb475c85208118aa9c81de791509b4b16fe3e7f71d531d5cff47"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-active-set-cartesian-trust-region-"
            "preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "2bec0d538ce06e8699783a54cdda8f1c76d2e08c51f1354ee5c37f2bbefa9df0"
        ),
        "probe_requests_sha256": (
            "af0a6139239d8bb9fe1fe8191203a398e734b2deedbcd7c4f5d1a3356918bcee"
        ),
        "probe_results_sha256": (
            "a789385cca0e7adec79142c9b0c63572503bbd6a46aed4ca6883a5ca158c4a3d"
        ),
        "probe_summary_sha256": (
            "22a15ede3d7f6983faefdfca80cc71f9971a184ff2b936bdacd1c8310b5d52ff"
        ),
        "activation_phase": "orient",
        "activation_after": [
            "mink_qp_failed",
            "direct_full_pose_moveit_lma_failed",
            "registered_full_pose_cartesian_backoff_exhausted",
        ],
        "active_arm_selection": (
            "arm_owning_current_minimum_action_contract_joint_margin"
        ),
        "passive_arm_position": "current",
        "passive_arm_orientation": "requested",
        "active_arm_position_center": "current",
        "coordinate_directions": [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        "radii_m": [0.0, 0.0015, 0.003, 0.006, 0.009],
        "maximum_requested_position_relaxation_m": 0.012,
        "margin_restoration_orientation_fraction": 0.0,
        "minimum_margin_improvement_rad": 0.001,
        "margin_restoration_selection": (
            "maximum_goal_minimum_joint_limit_margin_then_minimum_radius_"
            "then_direction_order"
        ),
        "orientation_progress_fractions": [0.5, 0.25, 0.125, 0.05],
        "orientation_progress_selection": (
            "largest_fraction_then_maximum_goal_minimum_joint_limit_margin_"
            "then_minimum_radius_then_direction_order"
        ),
        "active_orientation_interpolation": (
            "shortest_arc_quaternion_slerp_current_to_requested"
        ),
        "waypoint_solver": "lma_kinematics_plugin/LMAKinematicsPlugin",
        "global_planner": "official_moveit2_ompl_rrtconnect",
        "feedback_replans_original_target": True,
        "persistent_hidden_state": False,
        "custom_task_direction_enabled": False,
        "affects_final_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if active_set_trust_region != expected_active_set_trust_region:
        raise ValueError("Active-set Cartesian trust-region evidence differs.")
    trust_source_path = _repository_path(str(active_set_trust_region["source_plan"]))
    trust_audit_path = _repository_path(str(active_set_trust_region["source_audit"]))
    trust_pick_audit_path = _repository_path(
        str(active_set_trust_region["discarded_pick_ik_audit"])
    )
    trust_preregistration_path = _repository_path(
        str(active_set_trust_region["preregistration"])
    )
    if (
        file_sha256(trust_source_path) != active_set_trust_region["source_plan_sha256"]
        or file_sha256(trust_audit_path)
        != active_set_trust_region["source_audit_sha256"]
        or file_sha256(trust_pick_audit_path)
        != active_set_trust_region["discarded_pick_ik_audit_sha256"]
        or file_sha256(trust_preregistration_path)
        != active_set_trust_region["preregistration_sha256"]
    ):
        raise ValueError("Active-set Cartesian trust-region evidence hash differs.")
    trust_audit = _mapping(
        json.loads(trust_audit_path.read_text(encoding="utf-8")),
        "active-set trust-region source audit",
    )
    trust_failure = _mapping(
        trust_audit.get("failure_boundary"),
        "active-set trust-region source failure",
    )
    trust_pick_audit = _mapping(
        json.loads(trust_pick_audit_path.read_text(encoding="utf-8")),
        "active-set trust-region discarded Pick IK audit",
    )
    trust_preregistration = _mapping(
        json.loads(trust_preregistration_path.read_text(encoding="utf-8")),
        "active-set trust-region preregistration",
    )
    trust_probe = _mapping(
        trust_preregistration.get("diagnostic_evidence"),
        "active-set trust-region probe evidence",
    )
    if (
        trust_audit.get("status") != "failed_train_only_exact"
        or trust_failure.get("step") != 256
        or trust_failure.get("phase") != "orient"
        or trust_failure.get("pre_step_minimum_margin_joint")
        != "right_wrist_rotate"
        or trust_pick_audit.get("status")
        != "failed_captured_request_runtime_gate"
        or trust_probe.get("requests") != 90
        or trust_probe.get("successful_candidates") != 15
    ):
        raise ValueError("Active-set Cartesian trust-region source proof differs.")

    feedback_basis = _mapping(
        plan.get("feedback_aligned_trust_region_basis"),
        "feedback_aligned_trust_region_basis",
    )
    expected_feedback_basis = {
        "schema": "feedback_aligned_trust_region_basis_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_050.yaml",
        "source_plan_sha256": (
            "1b4731cb1c18739ba2d3a0f6d06966f0f7fefb2eeb961196a71ca0dcfe74f811"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan050-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "934b4d8b50f511ddb1e68b32d1fa69a163d8b81b327099e780df9d5cbdac87c3"
        ),
        "source_exact_report_sha256": (
            "7bb8771d7fc0c3e75394852c242b498eeafff876ad45fb1abdf9bf456859815b"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-feedback-aligned-trust-region-basis-"
            "preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "032aeb898f65f6fbf8fc85e884e73ce78d888ecc0c7bdcc6720afd9e9bb43a1d"
        ),
        "probe_requests_sha256": (
            "174a274a95388fa55559facfe7410954f45ddec3b7c1a10b006ddb0e240e6f09"
        ),
        "probe_results_sha256": (
            "97c397adecfba02cf3c7955d7f0f6519590968cdcdc34f19658dcb8e0ef59ca9"
        ),
        "probe_summary_sha256": (
            "d9df5c91313082a9e8dc70f3f138ab74dd317194e652be4f4d4ca3b2e0da4261"
        ),
        "candidate_basis": "feedback_aligned_orthonormal_v1",
        "radial_axis": "normalize(requested_active_position-current_active_position)",
        "reference_axis": (
            "world_basis_axis_with_minimum_absolute_radial_dot_with_xyz_tie_order"
        ),
        "tangent_one": "normalize(cross(radial_axis,reference_axis))",
        "tangent_two": "cross(radial_axis,tangent_one)",
        "direction_order": [
            "positive_radial",
            "negative_radial",
            "positive_tangent_one",
            "negative_tangent_one",
            "positive_tangent_two",
            "negative_tangent_two",
        ],
        "zero_position_error_behavior": "fail_closed_without_a_feedback_basis",
        "persistent_hidden_state": False,
        "task_specific_direction": False,
        "single_axis": (
            "fixed_world_coordinate_basis_to_deterministic_feedback_aligned_"
            "orthonormal_basis_v1"
        ),
        "affects_radii_or_position_envelope": False,
        "affects_orientation_fractions": False,
        "affects_pose_or_joint_margin_gates": False,
        "affects_seed_or_label_boundaries": False,
    }
    if feedback_basis != expected_feedback_basis:
        raise ValueError("Feedback-aligned trust-region basis evidence differs.")
    feedback_source_path = _repository_path(str(feedback_basis["source_plan"]))
    feedback_audit_path = _repository_path(str(feedback_basis["source_audit"]))
    feedback_preregistration_path = _repository_path(
        str(feedback_basis["preregistration"])
    )
    if (
        file_sha256(feedback_source_path) != feedback_basis["source_plan_sha256"]
        or file_sha256(feedback_audit_path) != feedback_basis["source_audit_sha256"]
        or file_sha256(feedback_preregistration_path)
        != feedback_basis["preregistration_sha256"]
    ):
        raise ValueError("Feedback-aligned trust-region basis hash differs.")
    feedback_audit = _mapping(
        json.loads(feedback_audit_path.read_text(encoding="utf-8")),
        "feedback-aligned trust-region source audit",
    )
    feedback_failure = _mapping(
        feedback_audit.get("failure_boundary"),
        "feedback-aligned trust-region source failure",
    )
    feedback_preregistration = _mapping(
        json.loads(feedback_preregistration_path.read_text(encoding="utf-8")),
        "feedback-aligned trust-region preregistration",
    )
    feedback_probe = _mapping(
        feedback_preregistration.get("diagnostic_evidence"),
        "feedback-aligned trust-region diagnostic evidence",
    )
    if (
        feedback_audit.get("status") != "failed_train_only_exact"
        or feedback_failure.get("step") != 712
        or feedback_failure.get("phase") != "orient"
        or feedback_failure.get("pre_step_minimum_margin_joint")
        != "right_wrist_rotate"
        or feedback_preregistration.get("status")
        != "preregistered_before_plan051_execution"
        or feedback_probe.get("requests") != 65
        or feedback_probe.get("successful_candidates") != 11
    ):
        raise ValueError("Feedback-aligned trust-region basis proof differs.")

    orientation_first_selection = _mapping(
        plan.get("orientation_first_trust_region_selection"),
        "orientation_first_trust_region_selection",
    )
    expected_orientation_first_selection = {
        "schema": "orientation_first_trust_region_selection_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_051.yaml",
        "source_plan_sha256": (
            "e593c3c7f13eba83c3760d315b1acde34c4fe5243069c13bfdbee061f2e82ef8"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan051-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "c31291f4425488f1ed8d282da8e619cf72039a9adb99872298fb0d43916e4a1b"
        ),
        "source_exact_report_sha256": (
            "f80d14eee434c3637330e77ecf6d1b1e8f45aeff0fd123bec06dc3299428c647"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-orientation-first-trust-region-selection-"
            "preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "35b42c14eb96270456c6684c6d87d49f21587516a85a37c4d4738fc5371a89e7"
        ),
        "selection_policy": "orientation_progress_first_v1",
        "selection_order": [
            "largest_feasible_orientation_fraction",
            "maximum_goal_minimum_joint_margin",
            "minimum_radius",
            "direction_order",
            "margin_restoration_fallback",
        ],
        "margin_restoration_fallback_only_when_no_progress_candidate": True,
        "persistent_hidden_state": False,
        "single_axis": (
            "margin_restoration_then_orientation_progress_to_orientation_"
            "progress_then_margin_restoration"
        ),
        "affects_candidate_basis_or_radii": False,
        "affects_position_or_orientation_envelopes": False,
        "affects_pose_or_joint_margin_gates": False,
        "affects_seed_or_label_boundaries": False,
    }
    if orientation_first_selection != expected_orientation_first_selection:
        raise ValueError("Orientation-first trust-region selection evidence differs.")
    orientation_source_path = _repository_path(
        str(orientation_first_selection["source_plan"])
    )
    orientation_audit_path = _repository_path(
        str(orientation_first_selection["source_audit"])
    )
    orientation_preregistration_path = _repository_path(
        str(orientation_first_selection["preregistration"])
    )
    if (
        file_sha256(orientation_source_path)
        != orientation_first_selection["source_plan_sha256"]
        or file_sha256(orientation_audit_path)
        != orientation_first_selection["source_audit_sha256"]
        or file_sha256(orientation_preregistration_path)
        != orientation_first_selection["preregistration_sha256"]
    ):
        raise ValueError("Orientation-first trust-region selection hash differs.")
    orientation_audit = _mapping(
        json.loads(orientation_audit_path.read_text(encoding="utf-8")),
        "orientation-first trust-region source audit",
    )
    orientation_result = _mapping(
        orientation_audit.get("result"),
        "orientation-first trust-region source result",
    )
    orientation_preregistration = _mapping(
        json.loads(orientation_preregistration_path.read_text(encoding="utf-8")),
        "orientation-first trust-region preregistration",
    )
    if (
        orientation_audit.get("status")
        != "failed_train_only_exact_horizon_exhausted"
        or orientation_result.get("steps_executed") != 750
        or orientation_result.get("final_phase") != "orient"
        or orientation_result.get("inverse_kinematics_failures") != 0
        or orientation_result.get("trust_region_margin_restoration_events") != 8
        or orientation_result.get("trust_region_orientation_progress_events") != 0
        or orientation_preregistration.get("status")
        != "preregistered_before_plan052_execution"
    ):
        raise ValueError("Orientation-first trust-region selection proof differs.")

    constraint_restoration = _mapping(
        plan.get("constraint_anchored_restoration"),
        "constraint_anchored_restoration",
    )
    expected_constraint_restoration = {
        "schema": "constraint_anchored_restoration_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_052.yaml",
        "source_plan_sha256": (
            "7b9be243e46731a031d1ac1c65d521144a008fbcb6f8b2046b60eacf5b2158c5"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan052-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "e86769bf601e8c81d17ee943d413b6ad1114256cc168956f1be4ca7c20208114"
        ),
        "source_exact_report_sha256": (
            "d6a15d1b899ceb8f41f69294889b6113e2073a1f92d14b9e105645835c1a4c5e"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-constraint-anchored-restoration-"
            "preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "26a4c55778865c39ff6230f4e5848078ea8c5e367d619a618e36eb0eb358314d"
        ),
        "probe_requests_sha256": (
            "186862ccc5ee450f9af5fcca821a704cdfcc00471652f06d28338aa3576ff7ce"
        ),
        "probe_results_sha256": (
            "b43fb24fb6d33a2844e6775a1c9390c07545a1191fee559163003ddf0a52a9c6"
        ),
        "probe_summary_sha256": (
            "fbf708ac10dc0cb4d83c4bfde215332d96c7e1a54629c57b047d8661332c37d8"
        ),
        "restoration_reference": "command_margin_boundary",
        "restoration_buffer_rad": 0.001,
        "activation": "only_when_no_registered_orientation_progress_candidate_exists",
        "selection": (
            "maximum_goal_minimum_joint_margin_then_minimum_radius_then_"
            "direction_order"
        ),
        "persistent_hidden_state": False,
        "single_axis": (
            "current_relative_0_001_rad_improvement_to_command_boundary_"
            "plus_0_001_rad_interior_buffer"
        ),
        "affects_command_margin_or_pose_gates": False,
        "affects_basis_radii_or_envelopes": False,
        "affects_seed_or_label_boundaries": False,
    }
    if constraint_restoration != expected_constraint_restoration:
        raise ValueError("Constraint-anchored restoration evidence differs.")
    restoration_source_path = _repository_path(
        str(constraint_restoration["source_plan"])
    )
    restoration_audit_path = _repository_path(
        str(constraint_restoration["source_audit"])
    )
    restoration_preregistration_path = _repository_path(
        str(constraint_restoration["preregistration"])
    )
    if (
        file_sha256(restoration_source_path)
        != constraint_restoration["source_plan_sha256"]
        or file_sha256(restoration_audit_path)
        != constraint_restoration["source_audit_sha256"]
        or file_sha256(restoration_preregistration_path)
        != constraint_restoration["preregistration_sha256"]
    ):
        raise ValueError("Constraint-anchored restoration hash differs.")
    restoration_audit = _mapping(
        json.loads(restoration_audit_path.read_text(encoding="utf-8")),
        "constraint-anchored restoration source audit",
    )
    restoration_failure = _mapping(
        restoration_audit.get("failure_boundary"),
        "constraint-anchored restoration source failure",
    )
    restoration_preregistration = _mapping(
        json.loads(restoration_preregistration_path.read_text(encoding="utf-8")),
        "constraint-anchored restoration preregistration",
    )
    restoration_probe = _mapping(
        restoration_preregistration.get("diagnostic_evidence"),
        "constraint-anchored restoration probe evidence",
    )
    if (
        restoration_audit.get("status") != "failed_train_only_exact"
        or restoration_failure.get("step") != 332
        or restoration_failure.get("pre_step_minimum_margin_joint")
        != "right_wrist_rotate"
        or restoration_preregistration.get("status")
        != "preregistered_before_plan053_execution"
        or restoration_probe.get("requests") != 25
        or restoration_probe.get("successful_candidates") != 1
    ):
        raise ValueError("Constraint-anchored restoration proof differs.")

    orientation_target_budget = _mapping(
        plan.get("expanded_orientation_target_budget"),
        "expanded_orientation_target_budget",
    )
    expected_orientation_target_budget = {
        "schema": "expanded_orientation_target_budget_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_053.yaml",
        "source_plan_sha256": (
            "c0a3759f2f40796cdaf8bfc7e3c3f9a15dafe54190c12477d93d127a7d266926"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan053-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "f35c1d2e9c7d9b90c46822ed2b8adc308f1f69d1dab6bd1d9620ee57b7904711"
        ),
        "source_exact_report_sha256": (
            "ffbbf2d475c0ccff6c28374b1c5dda4a4800acd0dbb8cb5f19f97eac24d134af"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-expanded-orientation-target-budget-"
            "preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "c481b94260a1f1bbed8e45ef2abae037718be3e4964b8aad85e4b5768f206668"
        ),
        "probe_generator_sha256": (
            "1ed8672ad651db0e797f422636a8023db8825c844577d860ee8e54f05b9ea7cf"
        ),
        "probe_runner_sha256": (
            "bb02f4f713460d6088e0f61bd7b9772cb5bca268050f8e8f5213654ed987b993"
        ),
        "probe_requests_sha256": (
            "36164f60edbaf7581ad9b3ead2d1102b66e88ad01f67296167094a38f21c5d51"
        ),
        "probe_results_sha256": (
            "c788b98270fbd4285d205220ba8f6d033197542eca54ee22a1b9cef0bb3a52d0"
        ),
        "probe_summary_sha256": (
            "5e4539fa2eff3d83614f28fb47be0f6791984a07f591ffa06f79decec9ffa8d2"
        ),
        "previous_orientation_target_budget_rad": 0.04,
        "orientation_target_budget_rad": 0.2,
        "maximum_registered_fraction": 0.5,
        "maximum_fractional_orientation_target_rad": 0.1,
        "interpretation": (
            "bounded_intermediate_orientation_target_not_acceptance_tolerance"
        ),
        "persistent_hidden_state": False,
        "single_axis": "teacher_maximum_orientation_step_rad_0_04_to_0_20",
        "affects_final_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_waypoint_joint_step_cap": False,
        "affects_horizon": False,
        "affects_seed_or_label_boundaries": False,
    }
    if orientation_target_budget != expected_orientation_target_budget:
        raise ValueError("Expanded orientation target budget evidence differs.")
    orientation_budget_source_path = _repository_path(
        str(orientation_target_budget["source_plan"])
    )
    orientation_budget_audit_path = _repository_path(
        str(orientation_target_budget["source_audit"])
    )
    orientation_budget_preregistration_path = _repository_path(
        str(orientation_target_budget["preregistration"])
    )
    orientation_budget_probe_generator_path = _repository_path(
        "runs/m2-smolvla-aloha-geometry-teacher-054/"
        "local-orientation-step-probe-002/generate_probe_requests.py"
    )
    orientation_budget_probe_runner_path = _repository_path(
        "runs/m2-smolvla-aloha-geometry-teacher-054/"
        "local-orientation-step-probe-002/run-probe.sh"
    )
    if (
        file_sha256(orientation_budget_source_path)
        != orientation_target_budget["source_plan_sha256"]
        or file_sha256(orientation_budget_audit_path)
        != orientation_target_budget["source_audit_sha256"]
        or file_sha256(orientation_budget_preregistration_path)
        != orientation_target_budget["preregistration_sha256"]
        or file_sha256(orientation_budget_probe_generator_path)
        != orientation_target_budget["probe_generator_sha256"]
        or file_sha256(orientation_budget_probe_runner_path)
        != orientation_target_budget["probe_runner_sha256"]
    ):
        raise ValueError("Expanded orientation target budget hash differs.")
    orientation_budget_audit = _mapping(
        json.loads(orientation_budget_audit_path.read_text(encoding="utf-8")),
        "expanded orientation target source audit",
    )
    orientation_budget_progress = _mapping(
        orientation_budget_audit.get("progress_before_failure"),
        "expanded orientation target source progress",
    )
    orientation_budget_root_cause = _mapping(
        orientation_budget_audit.get("root_cause"),
        "expanded orientation target source root cause",
    )
    orientation_budget_frontier = _mapping(
        orientation_budget_audit.get("frontier_probe"),
        "expanded orientation target frontier evidence",
    )
    orientation_budget_probe_002 = _mapping(
        orientation_budget_frontier.get("probe_002"),
        "expanded orientation target frontier probe 002",
    )
    orientation_budget_preregistration = _mapping(
        json.loads(
            orientation_budget_preregistration_path.read_text(encoding="utf-8")
        ),
        "expanded orientation target preregistration",
    )
    orientation_budget_change = _mapping(
        orientation_budget_preregistration.get("single_axis_change"),
        "expanded orientation target registered change",
    )
    if (
        orientation_budget_audit.get("status") != "failed_train_only_exact"
        or orientation_budget_progress.get("steps_executed") != 750
        or orientation_budget_progress.get("final_phase") != "orient"
        or orientation_budget_progress.get("inverse_kinematics_failures") != 0
        or orientation_budget_progress.get(
            "trust_region_orientation_progress_events"
        )
        != 18
        or orientation_budget_root_cause.get("classification")
        != "orientation_target_budget_too_small_for_registered_horizon"
        or orientation_budget_probe_002.get("requests") != 54
        or orientation_budget_probe_002.get("valid") != 30
        or orientation_budget_probe_002.get("full_fraction_valid") != 9
        or orientation_budget_probe_002.get("requests_sha256")
        != orientation_target_budget["probe_requests_sha256"]
        or orientation_budget_probe_002.get("results_sha256")
        != orientation_target_budget["probe_results_sha256"]
        or orientation_budget_probe_002.get("summary_sha256")
        != orientation_target_budget["probe_summary_sha256"]
        or orientation_budget_preregistration.get("status")
        != "preregistered_before_plan054_execution"
        or orientation_budget_change.get("setting")
        != "teacher.maximum_orientation_step_rad"
        or orientation_budget_change.get("from_rad") != 0.04
        or orientation_budget_change.get("to_rad") != 0.2
    ):
        raise ValueError("Expanded orientation target budget proof differs.")

    lift_grasp_feedback = _mapping(
        plan.get("lift_grasp_feedback"),
        "lift_grasp_feedback",
    )
    expected_lift_grasp_feedback = {
        "schema": "lift_grasp_feedback_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_054.yaml",
        "source_plan_sha256": (
            "d8d68745ea7a22c59f924713c93f04bcacce4371fca820b6040645b913c5ef6d"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan054-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "9e5b82fd0126f5652918040734b70fcc8b47a7063d46daee3b134cd5fb1a5847"
        ),
        "source_exact_report_sha256": (
            "dfbe89d5afebbfa63fc0f689a5b2c7015c34de6a88573f8f2defb8b152fe19f3"
        ),
        "source_remote_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan054-remote-exact-audit-2026-08-16.json"
        ),
        "source_remote_audit_sha256": (
            "853bfdaf02ec1a79c3b143d9cf3e41920b7e0f719a3997dbc3d08058ced47f1f"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-lift-grasp-feedback-preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "05f06a02d0051e459b987609769ad0d22994aa0c1e57b98699b0896be4bfecad"
        ),
        "step_m": 0.006,
        "interpretation": (
            "bounded_feedback_object_lift_increment_not_acceptance_tolerance"
        ),
        "persistent_hidden_state": False,
        "single_axis": (
            "fixed_terminal_lift_target_replaced_by_feedback_anchored_increment"
        ),
        "affects_final_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if lift_grasp_feedback != expected_lift_grasp_feedback:
        raise ValueError("Lift grasp-feedback evidence differs.")
    lift_feedback_source_path = _repository_path(
        str(lift_grasp_feedback["source_plan"])
    )
    lift_feedback_audit_path = _repository_path(
        str(lift_grasp_feedback["source_audit"])
    )
    lift_feedback_remote_audit_path = _repository_path(
        str(lift_grasp_feedback["source_remote_audit"])
    )
    lift_feedback_preregistration_path = _repository_path(
        str(lift_grasp_feedback["preregistration"])
    )
    if (
        file_sha256(lift_feedback_source_path)
        != lift_grasp_feedback["source_plan_sha256"]
        or file_sha256(lift_feedback_audit_path)
        != lift_grasp_feedback["source_audit_sha256"]
        or file_sha256(lift_feedback_remote_audit_path)
        != lift_grasp_feedback["source_remote_audit_sha256"]
        or file_sha256(lift_feedback_preregistration_path)
        != lift_grasp_feedback["preregistration_sha256"]
    ):
        raise ValueError("Lift grasp-feedback source identity differs.")
    lift_feedback_audit = _mapping(
        json.loads(lift_feedback_audit_path.read_text(encoding="utf-8")),
        "lift grasp-feedback source audit",
    )
    lift_feedback_failure = _mapping(
        lift_feedback_audit.get("failure_boundary"),
        "lift grasp-feedback source failure boundary",
    )
    lift_feedback_remote_audit = _mapping(
        json.loads(lift_feedback_remote_audit_path.read_text(encoding="utf-8")),
        "lift grasp-feedback remote audit",
    )
    lift_feedback_preregistration = _mapping(
        json.loads(lift_feedback_preregistration_path.read_text(encoding="utf-8")),
        "lift grasp-feedback preregistration",
    )
    lift_feedback_change = _mapping(
        lift_feedback_preregistration.get("single_axis_change"),
        "lift grasp-feedback registered change",
    )
    if (
        lift_feedback_audit.get("status") != "failed_train_only_exact"
        or lift_feedback_failure.get("step") != 422
        or lift_feedback_failure.get("phase") != "lift"
        or lift_feedback_failure.get("failure")
        != "observed_grasp_transform_exceeded_registered_drift"
        or lift_feedback_audit.get("result", {}).get(
            "expanded_orientation_target_budget_events"
        )
        != 0
        or lift_feedback_audit.get("result", {}).get(
            "inverse_kinematics_failures"
        )
        != 0
        or lift_feedback_audit.get("result", {}).get(
            "commanded_margin_breach_events"
        )
        != 0
        or lift_feedback_audit.get("result", {}).get(
            "observed_margin_breach_events"
        )
        != 0
        or lift_feedback_remote_audit.get("status") != "failed_train_only_exact"
        or lift_feedback_remote_audit.get("identity", {}).get(
            "exact_report_sha256"
        )
        != "b20e503fce4adf91d37ff55bdf81ce956d4e7e3c08a188902611294fb6ef2dc2"
        or lift_feedback_preregistration.get("status")
        != "preregistered_before_plan055_execution"
        or lift_feedback_change.get("setting")
        != "teacher.lift_feedback_step_m"
        or lift_feedback_change.get("step_m") != 0.006
    ):
        raise ValueError("Lift grasp-feedback proof differs.")
    if not math.isclose(
        float(teacher_settings.lift_feedback_step_m),
        float(lift_grasp_feedback["step_m"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Lift feedback step differs from teacher.")
    plan_acceptance = _mapping(plan.get("acceptance"), "acceptance")
    if int(
        plan_acceptance.get("minimum_lift_feedback_anchor_commands", -1)
    ) != 1:
        raise ValueError(
            "Lift feedback-anchor command must be an exact-gate event."
        )

    lift_contact_exemption = _mapping(
        plan.get("lift_contact_exemption"),
        "lift_contact_exemption",
    )
    expected_lift_contact_exemption = {
        "schema": "lift_contact_exemption_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_057.yaml",
        "source_plan_sha256": (
            "75d64f6fc048d9fd7d69542691777968254f7cc99025731cfae8de71f66d03f0"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan057-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "3de6ab4e09f934cd5f5487aff73b4de67f4c444d5093aa1854d285c9cc150151"
        ),
        "source_exact_report_sha256": (
            "f5be34fb54215baacbc82edc142aa15934eb642572371058ec6993b9018ee0b9"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-lift-gripper-bar-contact-preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "080803f92af40f6ebb76f8fc2e0862c87cd0b949679c5c462c061fce59ea8ef0"
        ),
        "allowed_unordered_geom_pairs": [
            ["table", "vx300s_right/10_right_gripper_finger"],
            ["table", "vx300s_right/9_gripper_bar"],
        ],
        "phases": ["lift"],
        "interpretation": (
            "same_right_gripper_assembly_observed_lift_contact_"
            "not_new_phase_or_relaxed_gate"
        ),
        "persistent_hidden_state": False,
        "single_axis": (
            "lift_contact_scope_adds_observed_right_gripper_bar_pair_only"
        ),
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if lift_contact_exemption != expected_lift_contact_exemption:
        raise ValueError("Lift contact-exemption evidence differs.")
    lift_contact_source_path = _repository_path(
        str(lift_contact_exemption["source_plan"])
    )
    lift_contact_audit_path = _repository_path(
        str(lift_contact_exemption["source_audit"])
    )
    lift_contact_preregistration_path = _repository_path(
        str(lift_contact_exemption["preregistration"])
    )
    if (
        file_sha256(lift_contact_source_path)
        != lift_contact_exemption["source_plan_sha256"]
        or file_sha256(lift_contact_audit_path)
        != lift_contact_exemption["source_audit_sha256"]
        or file_sha256(lift_contact_preregistration_path)
        != lift_contact_exemption["preregistration_sha256"]
    ):
        raise ValueError("Lift contact-exemption source identity differs.")
    lift_contact_audit = _mapping(
        json.loads(lift_contact_audit_path.read_text(encoding="utf-8")),
        "lift contact-exemption source audit",
    )
    lift_contact_result = _mapping(
        lift_contact_audit.get("result"),
        "lift contact-exemption source result",
    )
    lift_contact_preregistration = _mapping(
        json.loads(lift_contact_preregistration_path.read_text(encoding="utf-8")),
        "lift contact-exemption preregistration",
    )
    lift_contact_change = _mapping(
        lift_contact_preregistration.get("single_axis_change"),
        "lift contact-exemption registered change",
    )
    if (
        lift_contact_audit.get("status") != "failed_train_only_exact"
        or lift_contact_result.get("failure_step") != 444
        or lift_contact_result.get("final_phase") != "lift"
        or lift_contact_result.get("unexpected_collision_pair")
        != ["table", "vx300s_right/9_gripper_bar"]
        or lift_contact_result.get("lift_feedback_anchor_commands")
        != 56
        or lift_contact_result.get("lift_moveit_fallback_events") != 1
        or lift_contact_preregistration.get("status")
        != "preregistered_before_plan058_execution"
        or lift_contact_change.get("setting")
        != "lift_contact_exemption.allowed_unordered_geom_pairs"
        or lift_contact_change.get("to_pairs")
        != [
            ["table", "vx300s_right/10_right_gripper_finger"],
            ["table", "vx300s_right/9_gripper_bar"],
        ]
    ):
        raise ValueError("Lift contact-exemption proof differs.")
    if int(
        plan_acceptance.get("minimum_lift_contact_exemption_events", -1)
    ) != 1:
        raise ValueError(
            "Lift contact-exemption event must be an exact-gate event."
        )
    if int(
        plan_acceptance.get(
            "minimum_lift_gripper_bar_contact_exemption_events",
            -1,
        )
    ) != 1:
        raise ValueError(
            "Lift gripper-bar contact exemption must be an exact-gate event."
        )

    lift_moveit_fallback = _mapping(
        plan.get("lift_moveit_fallback"),
        "lift_moveit_fallback",
    )
    expected_lift_moveit_fallback = {
        "schema": "lift_moveit_fallback_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_056.yaml",
        "source_plan_sha256": (
            "20653da3f6d0ad6d1cb4528a29380f4298051b6841713cab6053a6334981e56e"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan056-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "2d03c505ccad15d2ed6cf4ae29a5eacdb426ada0baf4d91898f1e1ccb339a0b6"
        ),
        "source_exact_report_sha256": (
            "47e399ac6bdbca5d6580e679477cfef7bae45759248ffd813ad03fc27b205acf"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-lift-moveit-fallback-preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "fb00b8070b003ad5017231789c537d1e6eabcbe2f5dbdd2e7975545d33d04494"
        ),
        "activation_phases": ["lift"],
        "activation": "only_after_unchanged_mink_qp_failure",
        "interpretation": (
            "official_moveit_fallback_phase_scope_extension_"
            "not_new_solver_or_relaxed_gate"
        ),
        "persistent_hidden_state": False,
        "single_axis": (
            "official_moveit_fallback_extended_to_lift_after_mink_failure"
        ),
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if lift_moveit_fallback != expected_lift_moveit_fallback:
        raise ValueError("Lift MoveIt fallback evidence differs.")
    lift_moveit_source_path = _repository_path(
        str(lift_moveit_fallback["source_plan"])
    )
    lift_moveit_audit_path = _repository_path(
        str(lift_moveit_fallback["source_audit"])
    )
    lift_moveit_preregistration_path = _repository_path(
        str(lift_moveit_fallback["preregistration"])
    )
    if (
        file_sha256(lift_moveit_source_path)
        != lift_moveit_fallback["source_plan_sha256"]
        or file_sha256(lift_moveit_audit_path)
        != lift_moveit_fallback["source_audit_sha256"]
        or file_sha256(lift_moveit_preregistration_path)
        != lift_moveit_fallback["preregistration_sha256"]
    ):
        raise ValueError("Lift MoveIt fallback source identity differs.")
    lift_moveit_audit = _mapping(
        json.loads(lift_moveit_audit_path.read_text(encoding="utf-8")),
        "lift MoveIt fallback source audit",
    )
    lift_moveit_result = _mapping(
        lift_moveit_audit.get("result"),
        "lift MoveIt fallback source result",
    )
    lift_moveit_preregistration = _mapping(
        json.loads(lift_moveit_preregistration_path.read_text(encoding="utf-8")),
        "lift MoveIt fallback preregistration",
    )
    lift_moveit_change = _mapping(
        lift_moveit_preregistration.get("single_axis_change"),
        "lift MoveIt fallback registered change",
    )
    if (
        lift_moveit_audit.get("status") != "failed_train_only_exact"
        or lift_moveit_result.get("failure_step") != 443
        or lift_moveit_result.get("final_phase") != "lift"
        or lift_moveit_result.get("failure") != "inverse_kinematics_failure"
        or lift_moveit_result.get("maximum_inverse_kinematics_error")
        != 0.0015606494501644484
        or lift_moveit_result.get("inverse_kinematics_failures") != 1
        or lift_moveit_result.get("lift_feedback_anchor_commands") != 56
        or lift_moveit_preregistration.get("status")
        != "preregistered_before_plan057_execution"
        or lift_moveit_change.get("setting")
        != "inverse_kinematics.path_planner_phases"
        or lift_moveit_change.get("to_phases")
        != ["approach", "orient", "lift"]
    ):
        raise ValueError("Lift MoveIt fallback proof differs.")
    if int(plan_acceptance.get("minimum_lift_moveit_fallback_events", -1)) != 1:
        raise ValueError(
            "Lift MoveIt fallback event must be an exact-gate event."
        )

    position_priority_waypoint = _mapping(
        plan.get("position_priority_waypoint"),
        "position_priority_waypoint",
    )
    expected_position_priority_waypoint = {
        "schema": "official_moveit_lma_position_priority_waypoint_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_043.yaml",
        "source_plan_sha256": (
            "0c2baa2dd38fb0639fbdbf9b584e5a4d24125e0575a51177fc76c95ee53e4d40"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan043-local-exact-audit-2026-08-16.json"
        ),
        "source_audit_sha256": (
            "8c54a42449aca7f4e4d36de22e44fc2815175bdf03254edcbec64a1d4010f4b6"
        ),
        "source_exact_report_sha256": (
            "590e4eab19506fd3f9d682e3a3b80c64ed92812870eb67621d8633969fc788ae"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-official-moveit-position-priority-waypoint-"
            "preregistration-2026-08-16.json"
        ),
        "preregistration_sha256": (
            "a1fcc9b8d6aa01045e98e017d9edd43335357bf21176f5de8e5fe9dde2514b4d"
        ),
        "runtime_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan045-runtime-identity-audit-2026-08-16.json"
        ),
        "runtime_audit_sha256": (
            "f7703845776229a10c4b98bfb1bc02c279b8302a87690b8d4a3622feb3a3351b"
        ),
        "repeat_report": (
            "reports/training/"
            "m2-smolvla-aloha-moveit-position-priority-repeat-plan045-"
            "2026-08-16.json"
        ),
        "repeat_report_sha256": (
            "0ec07c6882928afdc6a8cd1cb9c0913d36a9eb79cb575f3bfb6cefd835789117"
        ),
        "activation_phases": ["approach"],
        "solver": "lma_kinematics_plugin/LMAKinematicsPlugin",
        "solver_mode": "position_only_ik",
        "position_priority_groups": [
            "left_arm_position_priority",
            "right_arm_position_priority",
        ],
        "orientation_weight": 0.0,
        "cartesian_backoff_fractions": [1.0, 0.75, 0.5, 0.25, 0.125],
        "minimum_cartesian_progress_m": 0.001,
        "maximum_orientation_relaxation_rad": 0.04,
        "ompl_seed_reset_per_request": True,
        "terminal_goal_normalization_limit_rad": 1e-5,
        "terminal_goal_normalization_validation": (
            "moveit_planning_scene_is_path_valid"
        ),
        "global_planner": "official_moveit2_ompl_rrtconnect",
        "feedback_replans_original_target": True,
        "single_axis": (
            "official_moveit_position_only_lma_waypoint_then_ompl"
        ),
        "affects_final_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_seed_or_label_boundaries": False,
    }
    if position_priority_waypoint != expected_position_priority_waypoint:
        raise ValueError("Official MoveIt position-priority evidence differs.")
    position_source_plan = _repository_path(
        str(position_priority_waypoint["source_plan"])
    )
    position_source_audit = _repository_path(
        str(position_priority_waypoint["source_audit"])
    )
    position_preregistration = _repository_path(
        str(position_priority_waypoint["preregistration"])
    )
    position_runtime_audit_path = _repository_path(
        str(position_priority_waypoint["runtime_audit"])
    )
    position_repeat_path = _repository_path(
        str(position_priority_waypoint["repeat_report"])
    )
    if (
        file_sha256(position_source_plan)
        != position_priority_waypoint["source_plan_sha256"]
        or file_sha256(position_source_audit)
        != position_priority_waypoint["source_audit_sha256"]
        or file_sha256(position_preregistration)
        != position_priority_waypoint["preregistration_sha256"]
        or file_sha256(position_runtime_audit_path)
        != position_priority_waypoint["runtime_audit_sha256"]
        or file_sha256(position_repeat_path)
        != position_priority_waypoint["repeat_report_sha256"]
    ):
        raise ValueError("Official MoveIt position-priority evidence hash differs.")
    position_audit = _mapping(
        json.loads(position_source_audit.read_text(encoding="utf-8")),
        "position-priority source audit",
    )
    position_failure = _mapping(
        position_audit.get("failure_boundary"),
        "position-priority source failure",
    )
    position_runtime_audit = _mapping(
        json.loads(position_runtime_audit_path.read_text(encoding="utf-8")),
        "position-priority runtime audit",
    )
    position_runtime_identity = _mapping(
        position_runtime_audit.get("identity"),
        "position-priority runtime identity",
    )
    position_runtime_repeat = _mapping(
        position_runtime_audit.get("repeat_proof"),
        "position-priority runtime repeat proof",
    )
    position_repeat = _mapping(
        json.loads(position_repeat_path.read_text(encoding="utf-8")),
        "position-priority repeat report",
    )
    position_repeat_result = _mapping(
        position_repeat.get("repeat_result"),
        "position-priority repeat result",
    )
    if (
        position_audit.get("audit_id")
        != "m2-smolvla-athena-plan043-local-exact-audit-2026-08-16"
        or position_failure.get("solver_failure") != "bimanual_lma_ik_failed"
        or position_failure.get("deterministic_ik_attempts_used") != 256
        or position_failure.get("pre_step_minimum_margin_joint")
        != "right_wrist_rotate"
        or position_runtime_audit.get("status") != "passed_before_exact"
        or position_runtime_identity.get("ik_task_modes")
        != ["full_pose", "position_priority"]
        or position_runtime_identity.get("position_priority_groups")
        != position_priority_waypoint["position_priority_groups"]
        or position_runtime_identity.get("position_priority_orientation_weight")
        != 0.0
        or position_runtime_identity.get(
            "position_priority_ompl_seed_reset_per_request"
        )
        is not True
        or position_runtime_identity.get(
            "position_priority_terminal_goal_normalization_limit_rad"
        )
        != position_priority_waypoint["terminal_goal_normalization_limit_rad"]
        or position_runtime_repeat.get("report_sha256")
        != position_priority_waypoint["repeat_report_sha256"]
        or position_repeat.get("status") != "passed"
        or position_repeat_result.get("both_status_ok") is not True
        or position_repeat_result.get("goal_vectors_exactly_equal") is not True
        or position_repeat_result.get("trajectory_vectors_exactly_equal") is not True
        or position_repeat_result.get("attempt_counts_equal") is not True
        or float(
            position_repeat_result.get(
                "maximum_goal_position_error_m", math.inf
            )
        )
        > 0.001
        or float(
            position_repeat_result.get(
                "maximum_goal_orientation_error_rad", math.inf
            )
        )
        > position_priority_waypoint["maximum_orientation_relaxation_rad"]
        or float(
            position_repeat_result.get(
                "minimum_path_joint_limit_margin_rad", -math.inf
            )
        )
        < 0.04540462255477905
    ):
        raise ValueError("Official MoveIt position-priority source proof differs.")

    contact_phase_feedforward = _mapping(
        plan.get("contact_phase_feedforward"),
        "contact_phase_feedforward",
    )
    expected_contact_phase_feedforward = {
        "schema": "phase_scoped_mujoco_static_position_feedforward_v1",
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_039.yaml",
        "source_plan_sha256": (
            "a0a919d778e8af06395103c3ba8a432c2348a34ffbe556450f246f337a3844c5"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan039-local-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "463e16d744e50ff7c1d1dcf726dc2158d09a9d66a356f05e735fbe21bb157616"
        ),
        "source_exact_report_sha256": (
            "2296577282cd3e9fe20b514eabb12d4b155cc4e46d66a6a52ba6a9030e64ad5b"
        ),
        "source_observed_contact_diagnostic_sha256": (
            "c3e29a05b2f2f1b0e3af6ad9c2b4ff55fb46393ce574c52af23794f10dde1909"
        ),
        "source_commanded_contact_diagnostic_sha256": (
            "96a64e5c80b56ec351baca63235ade99586d519b471106bbef34cf1a6096e80b"
        ),
        "source_feedforward_counterfactual_sha256": (
            "8d9f4ce94feee82fd2bf23c4b6de37b3c1bdb6ec6f1356f2f82537ff0e542da7"
        ),
        "preregistration": (
            "reports/training/"
            "m2-smolvla-contact-phase-feedforward-preregistration-2026-08-15.json"
        ),
        "preregistration_sha256": (
            "eb59a07fc8a9fd1e629bc6b2491b243d6f9182da78e48226a29cdf9459211e49"
        ),
        "source_failure_step": 441,
        "source_unexpected_pair": [
            "table",
            "vx300s_left/10_left_gripper_finger",
        ],
        "backend": "mujoco_static_inverse_dynamics_affine_position_feedforward",
        "phases": ["descend", "grasp"],
        "maximum_correction_rad": 0.05,
        "joint_limit_margin_rad": 0.04540462255477905,
        "neutral_reference_tolerance_rad": 1e-9,
        "official_mujoco_actuation_source": (
            "https://mujoco.readthedocs.io/en/stable/computation/"
            "index.html#actuation-model"
        ),
        "official_mujoco_force_balance_source": (
            "https://mujoco.readthedocs.io/en/stable/computation/"
            "index.html#passive-forces"
        ),
        "affects_collision_allowlist": False,
        "affects_pose_gates": False,
        "affects_joint_limit_margins": False,
        "affects_planner_or_ik": False,
        "affects_horizon": False,
        "affects_seed_or_label_boundaries": False,
    }
    if contact_phase_feedforward != expected_contact_phase_feedforward:
        raise ValueError("Contact-phase feedforward evidence differs.")
    feedforward_source_plan = _repository_path(
        str(contact_phase_feedforward["source_plan"])
    )
    feedforward_source_audit = _repository_path(
        str(contact_phase_feedforward["source_audit"])
    )
    feedforward_preregistration = _repository_path(
        str(contact_phase_feedforward["preregistration"])
    )
    if file_sha256(feedforward_source_plan) != contact_phase_feedforward[
        "source_plan_sha256"
    ]:
        raise ValueError("Contact-phase feedforward source plan identity differs.")
    if file_sha256(feedforward_source_audit) != contact_phase_feedforward[
        "source_audit_sha256"
    ]:
        raise ValueError("Contact-phase feedforward source audit identity differs.")
    if file_sha256(feedforward_preregistration) != contact_phase_feedforward[
        "preregistration_sha256"
    ]:
        raise ValueError("Contact-phase feedforward preregistration identity differs.")
    feedforward_evidence = _mapping(
        json.loads(feedforward_source_audit.read_text(encoding="utf-8")),
        "contact-phase feedforward source audit",
    )
    feedforward_execution = _mapping(
        feedforward_evidence.get("execution"),
        "contact-phase feedforward source execution",
    )
    feedforward_failure = _mapping(
        feedforward_evidence.get("failure_boundary"),
        "contact-phase feedforward source failure",
    )
    feedforward_counterfactual = _mapping(
        feedforward_evidence.get("counterfactual"),
        "contact-phase feedforward source counterfactual",
    )
    feedforward_preregistered = _mapping(
        json.loads(feedforward_preregistration.read_text(encoding="utf-8")),
        "contact-phase feedforward preregistration",
    )
    feedforward_change = _mapping(
        feedforward_preregistered.get("registered_change"),
        "contact-phase feedforward preregistered change",
    )
    if (
        feedforward_evidence.get("audit_id")
        != "m2-smolvla-athena-plan039-local-exact-audit-2026-08-15"
        or feedforward_execution.get("plan_sha256")
        != contact_phase_feedforward["source_plan_sha256"]
        or feedforward_execution.get("exact_report_sha256")
        != contact_phase_feedforward["source_exact_report_sha256"]
        or feedforward_failure.get("step")
        != contact_phase_feedforward["source_failure_step"]
        or feedforward_failure.get("unexpected_pair")
        != contact_phase_feedforward["source_unexpected_pair"]
        or feedforward_failure.get("final_observed_contact_diagnostic_sha256")
        != contact_phase_feedforward[
            "source_observed_contact_diagnostic_sha256"
        ]
        or feedforward_failure.get("final_commanded_contact_diagnostic_sha256")
        != contact_phase_feedforward[
            "source_commanded_contact_diagnostic_sha256"
        ]
        or feedforward_counterfactual.get("diagnostic_sha256")
        != contact_phase_feedforward[
            "source_feedforward_counterfactual_sha256"
        ]
        or feedforward_change.get("phases")
        != contact_phase_feedforward["phases"]
        or feedforward_change.get("backend")
        != contact_phase_feedforward["backend"]
        or feedforward_change.get("maximum_correction_rad")
        != contact_phase_feedforward["maximum_correction_rad"]
        or feedforward_change.get("joint_limit_margin_rad")
        != contact_phase_feedforward["joint_limit_margin_rad"]
    ):
        raise ValueError("Contact-phase feedforward exact evidence differs.")

    inverse_kinematics = _mapping(
        plan.get("inverse_kinematics"),
        "inverse_kinematics",
    )
    if inverse_kinematics.get("solver_backend") == "mink_qp":
        joint_limit_margin = float(
            inverse_kinematics.get("mink_joint_limit_margin_rad", 0.0)
        )
        if not math.isfinite(joint_limit_margin) or not 0.0 <= joint_limit_margin <= 0.05:
            raise ValueError("Mink joint-limit margin must be within [0, 0.05] rad.")
        expected_versions = {
            "mink_version": "1.2.0",
            "qpsolvers_version": "4.13.0",
            "daqp_version": "0.8.7",
            "mink_solver": "daqp",
        }
        for key, expected in expected_versions.items():
            if inverse_kinematics.get(key) != expected:
                raise ValueError(f"Mink QP identity mismatch for {key}.")
        MinkAlohaIkSettings(
            integration_timestep_s=float(
                inverse_kinematics["mink_integration_timestep_s"]
            ),
            maximum_iterations=int(inverse_kinematics["mink_maximum_iterations"]),
            position_cost=float(inverse_kinematics["mink_position_cost"]),
            orientation_cost=float(inverse_kinematics["mink_orientation_cost"]),
            posture_cost=float(inverse_kinematics["mink_posture_cost"]),
            frame_lm_damping=float(inverse_kinematics["mink_frame_lm_damping"]),
            solver_damping=float(inverse_kinematics["mink_solver_damping"]),
            maximum_joint_velocity_rad_s=float(
                inverse_kinematics["mink_maximum_joint_velocity_rad_s"]
            ),
            configuration_limit_gain=float(
                inverse_kinematics["mink_configuration_limit_gain"]
            ),
            joint_limit_margin_rad=float(
                joint_limit_margin
            ),
        )
        path_planner_enabled = inverse_kinematics.get("path_planner_enabled")
        if path_planner_enabled is not True:
            if path_planner_enabled is not False:
                raise ValueError("Mink QP path-planner boundary must be explicit.")
            acceptance = _mapping(plan.get("acceptance"), "acceptance")
            if acceptance.get("hidden_test_loaded") is not False:
                raise ValueError(
                    "Geometry-teacher acceptance must keep hidden test sealed."
                )
            if acceptance.get("recovery_labels_authorized_on_pass") is not False:
                raise ValueError("This teacher plan cannot authorize recovery-label writes.")
            return
        if inverse_kinematics.get("path_planner_backend") != "moveit2_ompl":
            raise ValueError("Mink QP may only use the registered official MoveIt backend.")
        expected_fallback_phases = (
            ("approach", "orient", "lift")
            if lift_moveit_fallback is not None
            else ("approach", "orient")
        )
        if tuple(inverse_kinematics.get("path_planner_phases", ())) != (
            expected_fallback_phases
        ):
            raise ValueError(
                "MoveIt fallback phase scope differs from its registered evidence."
            )
        moveit_identity = {
            "path_planner_ros_distro": "humble",
            "path_planner_moveit_version": "2.5.9",
            "path_planner_ompl_version": "1.7.0",
            "path_planner_plugin": "ompl_interface/OMPLPlanner",
            "path_planner_id": "RRTConnect",
            "path_planner_type": "geometric::RRTConnect",
            "path_planner_kinematics_plugin": (
                "lma_kinematics_plugin/LMAKinematicsPlugin"
            ),
            "path_planner_ik_group_selection_mode": (
                full_pose_group_selection["group_selection_mode"]
            ),
            "path_planner_full_pose_groups": (
                full_pose_group_selection["full_pose_groups"]
            ),
            "path_planner_request_adapters": [official_start_recovery_adapter],
        }
        for key, expected in moveit_identity.items():
            if inverse_kinematics.get(key) != expected:
                raise ValueError(f"MoveIt path-planner identity mismatch for {key}.")
        full_pose_backoff_identity = {
            "path_planner_full_pose_cartesian_backoff_enabled": True,
            "path_planner_full_pose_cartesian_backoff_activation_phases": (
                full_pose_cartesian_backoff["activation_phases"]
            ),
            "path_planner_full_pose_cartesian_backoff_fractions": (
                full_pose_cartesian_backoff["fractions_largest_first"]
            ),
            "path_planner_full_pose_cartesian_backoff_position_interpolation": (
                full_pose_cartesian_backoff["position_interpolation"]
            ),
            "path_planner_full_pose_cartesian_backoff_orientation_interpolation": (
                full_pose_cartesian_backoff["orientation_interpolation"]
            ),
            "path_planner_full_pose_cartesian_backoff_minimum_linear_progress_m": (
                full_pose_cartesian_backoff["minimum_linear_progress_m"]
            ),
            "path_planner_full_pose_cartesian_backoff_minimum_angular_progress_rad": (
                full_pose_cartesian_backoff["minimum_angular_progress_rad"]
            ),
        }
        for key, expected in full_pose_backoff_identity.items():
            if inverse_kinematics.get(key) != expected:
                raise ValueError(
                    f"MoveIt full-pose Cartesian backoff identity mismatch for {key}."
                )
        position_priority_identity = {
            "path_planner_position_priority_enabled": True,
            "path_planner_position_priority_activation_phases": (
                position_priority_waypoint["activation_phases"]
            ),
            "path_planner_position_priority_kinematics_plugin": (
                position_priority_waypoint["solver"]
            ),
            "path_planner_position_priority_groups": (
                position_priority_waypoint["position_priority_groups"]
            ),
            "path_planner_position_priority_solver_mode": (
                position_priority_waypoint["solver_mode"]
            ),
            "path_planner_position_priority_orientation_weight": 0.0,
            "path_planner_position_priority_cartesian_backoff_fractions": (
                position_priority_waypoint["cartesian_backoff_fractions"]
            ),
            "path_planner_position_priority_minimum_cartesian_progress_m": (
                position_priority_waypoint["minimum_cartesian_progress_m"]
            ),
            "path_planner_position_priority_maximum_orientation_relaxation_rad": (
                position_priority_waypoint["maximum_orientation_relaxation_rad"]
            ),
            "path_planner_position_priority_ompl_seed_reset_per_request": (
                position_priority_waypoint["ompl_seed_reset_per_request"]
            ),
            "path_planner_position_priority_terminal_goal_normalization_limit_rad": (
                position_priority_waypoint["terminal_goal_normalization_limit_rad"]
            ),
            "path_planner_position_priority_terminal_goal_normalization_validation": (
                position_priority_waypoint[
                    "terminal_goal_normalization_validation"
                ]
            ),
        }
        for key, expected in position_priority_identity.items():
            if inverse_kinematics.get(key) != expected:
                raise ValueError(
                    f"MoveIt position-priority identity mismatch for {key}."
                )
        trust_region_identity = {
            "path_planner_active_set_trust_region_enabled": True,
            "path_planner_active_set_trust_region_activation_phase": (
                active_set_trust_region["activation_phase"]
            ),
            "path_planner_active_set_trust_region_active_arm_selection": (
                active_set_trust_region["active_arm_selection"]
            ),
            "path_planner_active_set_trust_region_coordinate_directions": (
                active_set_trust_region["coordinate_directions"]
            ),
            "path_planner_active_set_trust_region_radii_m": (
                active_set_trust_region["radii_m"]
            ),
            (
                "path_planner_active_set_trust_region_maximum_"
                "requested_position_relaxation_m"
            ): active_set_trust_region["maximum_requested_position_relaxation_m"],
            "path_planner_active_set_trust_region_minimum_margin_improvement_rad": (
                active_set_trust_region["minimum_margin_improvement_rad"]
            ),
            "path_planner_active_set_trust_region_orientation_progress_fractions": (
                active_set_trust_region["orientation_progress_fractions"]
            ),
            "path_planner_active_set_trust_region_persistent_hidden_state": False,
            "path_planner_active_set_trust_region_candidate_basis": (
                feedback_basis["candidate_basis"]
            ),
            "path_planner_active_set_trust_region_selection_policy": (
                orientation_first_selection["selection_policy"]
            ),
            "path_planner_active_set_trust_region_restoration_reference": (
                constraint_restoration["restoration_reference"]
            ),
            (
                "path_planner_active_set_trust_region_previous_"
                "orientation_target_budget_rad"
            ): orientation_target_budget["previous_orientation_target_budget_rad"],
            "path_planner_active_set_trust_region_orientation_target_budget_rad": (
                orientation_target_budget["orientation_target_budget_rad"]
            ),
        }
        for key, expected in trust_region_identity.items():
            if inverse_kinematics.get(key) != expected:
                raise ValueError(
                    f"MoveIt active-set trust-region identity mismatch for {key}."
                )
        deterministic_identity = {
            "path_planner_ik_search_mode": deterministic_moveit_ik["search_mode"],
            "path_planner_ik_candidate_selection_mode": (
                joint_margin_candidate_selection["selection_mode"]
            ),
            "path_planner_ik_seed": deterministic_moveit_ik["seed"],
            "path_planner_ik_maximum_attempts": deterministic_moveit_ik[
                "maximum_attempts"
            ],
            "path_planner_ik_solver_base_frames": deterministic_moveit_ik[
                "solver_base_frames"
            ],
            "path_planner_ik_solver_tip_frames": deterministic_moveit_ik[
                "solver_tip_frames"
            ],
        }
        for key, expected in deterministic_identity.items():
            if inverse_kinematics.get(key) != expected:
                raise ValueError(f"Deterministic MoveIt identity mismatch for {key}.")
        moveit_execution_identity = {
            "path_planner_execution_backend": "moveit_hybrid_planning",
            "path_planner_trajectory_operator": (
                "moveit_hybrid_planning/SimpleSampler"
            ),
            "path_planner_local_constraint_solver": (
                "moveit_hybrid_planning/ForwardTrajectory"
            ),
            "path_planner_execution_source_tag": "2.5.9",
            "path_planner_include_trajectory": True,
            "path_planner_replan_on_phase_change": True,
            "path_planner_replan_every_step": False,
            "path_planner_replan_on_terminal_completion": True,
            "path_planner_terminal_control_enabled": True,
            "path_planner_terminal_control_backend": (
                "mujoco_static_inverse_dynamics_affine_position_feedforward"
            ),
            "path_planner_terminal_control_handoff": (
                "final_waypoint_within_moveit_simple_sampler_l1_tolerance"
            ),
            "path_planner_terminal_control_replan_on_failure": False,
        }
        for key, expected in moveit_execution_identity.items():
            if inverse_kinematics.get(key) != expected:
                raise ValueError(f"MoveIt trajectory-execution identity mismatch for {key}.")
        if not math.isclose(
            float(
                inverse_kinematics.get(
                    "path_planner_waypoint_l1_tolerance_rad",
                    math.nan,
                )
            ),
            0.2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("MoveIt SimpleSampler waypoint tolerance differs.")
        if int(inverse_kinematics.get("path_planner_ompl_seed", -1)) != 2210:
            raise ValueError("MoveIt OMPL seed must remain fixed at 2210.")
        if not math.isclose(
            float(
                inverse_kinematics.get(
                    "path_planner_terminal_completion_goal_l1_tolerance_rad",
                    math.nan,
                )
            ),
            0.001,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("MoveIt terminal-completion tolerance differs.")
        terminal_correction = float(
            inverse_kinematics.get(
                "path_planner_terminal_control_maximum_correction_rad",
                math.nan,
            )
        )
        if (
            not math.isfinite(terminal_correction)
            or terminal_correction <= 0.0
            or terminal_correction > 0.05
        ):
            raise ValueError("MoveIt terminal-control correction bound differs.")
        if not math.isclose(
            float(
                inverse_kinematics.get(
                    "path_planner_terminal_control_neutral_reference_tolerance_rad",
                    math.nan,
                )
            ),
            1e-9,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("MoveIt terminal-control actuator identity tolerance differs.")
        if not math.isclose(
            float(
                inverse_kinematics.get(
                    "path_planner_terminal_control_joint_limit_margin_rad",
                    math.nan,
                )
            ),
            command_joint_margin,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("MoveIt terminal control changes the command joint margin.")
        for key in (
            "path_planner_collision_geometry_link_count",
            "path_planner_collision_geometry_shape_count",
        ):
            value = inverse_kinematics.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"MoveIt collision-geometry identity differs for {key}.")
        positive_moveit_settings = {
            "path_planner_allowed_planning_time_s",
            "path_planner_ik_timeout_s",
            "path_planner_response_timeout_s",
            "path_planner_maximum_joint_step_rad",
            "path_planner_position_tolerance_m",
            "path_planner_orientation_tolerance_rad",
            "path_planner_start_bound_reconciliation_tolerance_rad",
            "path_planner_physical_joint_limit_margin_rad",
            "path_planner_joint_limit_margin_rad",
            "path_planner_finger_lower_m",
            "path_planner_finger_upper_m",
            "path_planner_finger_bound_reconciliation_tolerance_m",
            "path_planner_position_priority_minimum_cartesian_progress_m",
            "path_planner_position_priority_maximum_orientation_relaxation_rad",
            "path_planner_position_priority_terminal_goal_normalization_limit_rad",
            (
                "path_planner_active_set_trust_region_maximum_"
                "requested_position_relaxation_m"
            ),
            "path_planner_active_set_trust_region_minimum_margin_improvement_rad",
        }
        for key in positive_moveit_settings:
            value = float(inverse_kinematics.get(key, math.nan))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"MoveIt setting {key} must be finite and positive.")
        if not math.isclose(
            float(inverse_kinematics["path_planner_ik_timeout_s"]),
            float(moveit_ik_budget["ik_timeout_s"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("MoveIt IK timeout differs from its registered budget.")
        if not math.isclose(
            float(inverse_kinematics["path_planner_ik_timeout_s"]),
            float(deterministic_moveit_ik["outer_timeout_s"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("MoveIt IK timeout differs from deterministic search.")
        if not math.isclose(
            float(inverse_kinematics["path_planner_allowed_planning_time_s"]),
            float(moveit_ik_budget["allowed_planning_time_s"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("MoveIt planning time differs from the frozen IK budget axis.")
        if float(inverse_kinematics["path_planner_maximum_joint_step_rad"]) > float(
            inverse_kinematics["maximum_joint_target_delta"]
        ):
            raise ValueError("MoveIt waypoint exceeds the registered joint target delta.")
        if not math.isclose(
            float(
                inverse_kinematics[
                    "path_planner_start_bound_reconciliation_tolerance_rad"
                ]
            ),
            0.00002,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("MoveIt start-bound reconciliation tolerance differs.")
        if not math.isclose(
            float(
                inverse_kinematics[
                    "path_planner_physical_joint_limit_margin_rad"
                ]
            ),
            physical_joint_margin,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("MoveIt physical joint margin differs from the guard.")
        if not math.isclose(
            float(inverse_kinematics["path_planner_joint_limit_margin_rad"]),
            joint_limit_margin,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "MoveIt joint path-constraint margin must equal the Mink margin."
            )
        if not math.isclose(
            joint_limit_margin,
            command_joint_margin,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Mink and MoveIt margins must equal the tightened command margin."
            )
        finger_lower = float(inverse_kinematics["path_planner_finger_lower_m"])
        finger_upper = float(inverse_kinematics["path_planner_finger_upper_m"])
        finger_tolerance = float(
            inverse_kinematics[
                "path_planner_finger_bound_reconciliation_tolerance_m"
            ]
        )
        if not (
            math.isclose(finger_lower, 0.021, rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(finger_upper, 0.057, rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(finger_tolerance, 0.001, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise ValueError("MoveIt official finger-bound adapter differs.")
        if float(inverse_kinematics["path_planner_position_tolerance_m"]) > float(
            inverse_kinematics["maximum_accepted_error"]
        ):
            raise ValueError("MoveIt position tolerance relaxes the frozen IK gate.")
        if (
            float(inverse_kinematics["path_planner_orientation_tolerance_rad"])
            * float(inverse_kinematics["rotation_weight"])
            > float(inverse_kinematics["maximum_accepted_error"])
        ):
            raise ValueError("MoveIt orientation tolerance relaxes the frozen IK gate.")
        if float(
            inverse_kinematics[
                "path_planner_position_priority_maximum_orientation_relaxation_rad"
            ]
        ) > float(teacher_settings.maximum_orientation_step_rad):
            raise ValueError(
                "MoveIt position-priority waypoint exceeds the teacher orientation step."
            )
        if not math.isclose(
            float(teacher_settings.maximum_orientation_step_rad),
            float(orientation_target_budget["orientation_target_budget_rad"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Expanded orientation target budget differs from teacher.")
        runtime = _mapping(
            inverse_kinematics.get("path_planner_runtime"),
            "path_planner_runtime",
        )
        expected_runtime_image = (
            "rosetta-reality-aloha-moveit2:"
            "humble-2.5.9-joint-margin-selection-006"
        )
        if runtime.get("image") != expected_runtime_image:
            raise ValueError("MoveIt runtime image tag differs.")
        image_id = runtime.get("image_id")
        if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
            raise ValueError("MoveIt runtime image ID must be immutable.")
        for key in (
            "executable_sha256",
            "urdf_sha256",
            "srdf_sha256",
            "description_manifest_sha256",
            "mesh_manifest_sha256",
            "urdf_source_manifest_sha256",
            "model_parity_report_sha256",
        ):
            value = runtime.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"MoveIt runtime {key} is not a SHA-256 digest.")
        parity_path = _repository_path(str(runtime.get("model_parity_report")))
        if file_sha256(parity_path) != runtime["model_parity_report_sha256"]:
            raise ValueError("MoveIt/Gym model-parity evidence identity differs.")
        parity = _mapping(
            json.loads(parity_path.read_text(encoding="utf-8")),
            "MoveIt/Gym model-parity evidence",
        )
        parity_identity = {
            "schema_version": 1,
            "status": "passed",
            "report_type": "aloha_moveit_gym_model_parity",
            "moveit_image": runtime["image"],
            "moveit_image_id": runtime["image_id"],
            "simulator_seed": int(exact["simulator_seed"]),
            "hidden_test_loaded": False,
            "dataset_rows_loaded": False,
        }
        for key, expected in parity_identity.items():
            if parity.get(key) != expected:
                raise ValueError(f"MoveIt/Gym model-parity field differs for {key}.")
        if parity.get("action_contract_sha256") != file_sha256(
            _repository_path(str(scope["action_contract"]))
        ):
            raise ValueError("MoveIt/Gym model-parity Action Contract differs.")
        for error_name, tolerance_name in (
            ("maximum_position_error_m", "position_tolerance_m"),
            ("maximum_orientation_error_rad", "orientation_tolerance_rad"),
        ):
            observed = float(parity.get(error_name, math.nan))
            tolerance = float(parity.get(tolerance_name, math.nan))
            if (
                not math.isfinite(observed)
                or not math.isfinite(tolerance)
                or observed < 0.0
                or tolerance <= 0.0
                or observed > tolerance
            ):
                raise ValueError(f"MoveIt/Gym model-parity tolerance failed for {error_name}.")
        acceptance = _mapping(plan.get("acceptance"), "acceptance")
        if not math.isclose(
            float(
                acceptance.get(
                    "maximum_position_priority_orientation_relaxation_rad",
                    math.nan,
                )
            ),
            float(
                position_priority_waypoint[
                    "maximum_orientation_relaxation_rad"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Position-priority acceptance orientation bound differs."
            )
        if acceptance.get("hidden_test_loaded") is not False:
            raise ValueError("Geometry-teacher acceptance must keep hidden test sealed.")
        if acceptance.get("recovery_labels_authorized_on_pass") is not False:
            raise ValueError("This teacher plan cannot authorize recovery-label writes.")
        if int(acceptance.get("maximum_commanded_margin_breach_events", -1)) != 0:
            raise ValueError("Commanded physical-margin breaches must fail closed.")
        if int(acceptance.get("maximum_observed_margin_breach_events", -1)) != 0:
            raise ValueError("Observed physical-margin breaches must fail closed.")
        if int(
            acceptance.get(
                "minimum_path_planner_terminal_control_completions",
                -1,
            )
        ) != 1:
            raise ValueError("MoveIt terminal completion must be an exact-gate event.")
        if int(acceptance.get("minimum_task_contact_exemption_events", -1)) != 1:
            raise ValueError("Registered task contact must be observed in train exact.")
        if int(
            acceptance.get("minimum_trust_region_margin_restoration_events", -1)
        ) != 1:
            raise ValueError(
                "Active-set trust-region margin restoration must be an exact-gate event."
            )
        if int(
            acceptance.get("minimum_trust_region_orientation_progress_events", -1)
        ) != 1:
            raise ValueError(
                "Active-set trust-region orientation progress must be an exact-gate event."
            )
        if int(acceptance.get("minimum_trust_region_feedback_basis_events", -1)) != 1:
            raise ValueError(
                "Feedback-aligned trust-region basis must be an exact-gate event."
            )
        if int(
            acceptance.get("minimum_trust_region_orientation_first_events", -1)
        ) != 1:
            raise ValueError(
                "Orientation-first trust-region selection must be an exact-gate event."
            )
        if int(
            acceptance.get(
                "minimum_trust_region_constraint_anchored_restoration_events",
                -1,
            )
        ) != 1:
            raise ValueError(
                "Constraint-anchored restoration must be an exact-gate event."
            )
        if int(
            acceptance.get(
                "minimum_expanded_orientation_target_budget_events",
                -1,
            )
        ) != 1:
            raise ValueError(
                "Expanded orientation target budget must be an exact-gate event."
            )
        return
    if inverse_kinematics.get("path_planner_enabled") is not True:
        raise ValueError("The registered geometric path planner must be enabled.")
    if tuple(inverse_kinematics.get("path_planner_phases", ())) != (
        "approach",
        "orient",
    ):
        raise ValueError(
            "The geometric path planner must remain restricted to approach and orient."
        )
    fractions = tuple(
        float(value)
        for value in inverse_kinematics.get(
            "path_planner_cartesian_backoff_fractions",
            (),
        )
    )
    if (
        not fractions
        or fractions[0] != 1.0
        or any(not 0.0 < value <= 1.0 for value in fractions)
        or any(first <= second for first, second in zip(fractions, fractions[1:]))
    ):
        raise ValueError(
            "Path-planner Cartesian fractions must start at one and strictly decrease."
        )
    maximum_relaxation = float(
        inverse_kinematics.get(
            "path_planner_maximum_orientation_relaxation_rad",
            math.nan,
        )
    )
    if not 0.0 < maximum_relaxation <= teacher_settings.maximum_orientation_step_rad:
        raise ValueError(
            "Path-planner orientation relaxation exceeds the teacher step bound."
        )
    minimum_progress = float(
        inverse_kinematics.get(
            "path_planner_minimum_cartesian_progress_m",
            math.nan,
        )
    )
    if not 0.0 < minimum_progress <= teacher_settings.maximum_cartesian_step_m:
        raise ValueError("Path-planner minimum progress is outside the teacher step bound.")
    maximum_position_relaxation = float(
        inverse_kinematics.get(
            "path_planner_maximum_position_relaxation_m",
            math.nan,
        )
    )
    if not 0.0 < maximum_position_relaxation <= teacher_settings.approach_tolerance_m:
        raise ValueError(
            "Path-planner position relaxation exceeds the approach tolerance."
        )
    maximum_cartesian_step = float(
        inverse_kinematics.get(
            "path_planner_maximum_cartesian_step_m",
            math.nan,
        )
    )
    if not 0.0 < maximum_cartesian_step <= teacher_settings.maximum_cartesian_step_m:
        raise ValueError("Path-planner Cartesian step exceeds the teacher step bound.")
    minimum_orientation_progress = float(
        inverse_kinematics.get(
            "path_planner_minimum_orientation_progress_rad",
            math.nan,
        )
    )
    if not 0.0 < minimum_orientation_progress <= maximum_relaxation:
        raise ValueError("Path-planner orientation progress is outside its step bound.")

    acceptance = _mapping(plan.get("acceptance"), "acceptance")
    if acceptance.get("hidden_test_loaded") is not False:
        raise ValueError("Geometry-teacher acceptance must keep hidden test sealed.")
    if acceptance.get("recovery_labels_authorized_on_pass") is not False:
        raise ValueError("This teacher plan cannot authorize recovery-label writes.")


def _robot_state(observation: dict[str, Any], dimension: int) -> Tensor:
    state = observation.get("robot_state")
    if not isinstance(state, Tensor) or state.shape != (dimension,):
        raise ValueError("Simulator observation violates the registered robot-state shape.")
    state = state.detach().to(torch.float32).cpu()
    if not bool(torch.isfinite(state).all()):
        raise ValueError("Simulator robot state contains NaN or Inf.")
    return state


def _arm_joint_margin_snapshot(
    state: Tensor,
    contract: ActionContract,
    *,
    registered_margin_rad: float,
) -> dict[str, Any]:
    """Measure physical arm-joint distance to both Action Contract limits."""

    contract.validate_tensor(state, allow_chunk=False)
    if not math.isfinite(registered_margin_rad) or registered_margin_rad < 0.0:
        raise ValueError("Registered execution joint margin must be finite and non-negative.")
    if tuple(contract.dimension_names[index] for index in ARM_ACTION_INDICES) != (
        ARM_ACTION_NAMES
    ):
        raise ValueError("Execution diagnostic received an unexpected arm ordering.")

    joints: list[dict[str, Any]] = []
    for index in ARM_ACTION_INDICES:
        dimension = contract.dimensions[index]
        if dimension.unit != "radian":
            raise ValueError(
                f"Execution diagnostic requires radian arm joint {dimension.name!r}."
            )
        value = float(state[index])
        lower_margin = value - dimension.minimum
        upper_margin = dimension.maximum - value
        if lower_margin <= upper_margin:
            nearest_bound = "lower"
            minimum_margin = lower_margin
        else:
            nearest_bound = "upper"
            minimum_margin = upper_margin
        joints.append(
            {
                "index": index,
                "name": dimension.name,
                "value_rad": value,
                "lower_rad": dimension.minimum,
                "upper_rad": dimension.maximum,
                "lower_margin_rad": lower_margin,
                "upper_margin_rad": upper_margin,
                "minimum_margin_rad": minimum_margin,
                "nearest_bound": nearest_bound,
                "inside_registered_margin": minimum_margin < registered_margin_rad,
            }
        )

    minimum_joint = min(joints, key=lambda joint: float(joint["minimum_margin_rad"]))
    minimum_margin = float(minimum_joint["minimum_margin_rad"])
    return {
        "registered_margin_rad": registered_margin_rad,
        "minimum_margin_rad": minimum_margin,
        "minimum_margin_joint": minimum_joint["name"],
        "minimum_margin_bound": minimum_joint["nearest_bound"],
        "margin_shortfall_rad": max(0.0, registered_margin_rad - minimum_margin),
        "inside_registered_margin": minimum_margin < registered_margin_rad,
        "joints": joints,
    }


def _joint_execution_diagnostic(
    pre_step_state: Tensor,
    commanded_action: Tensor,
    observed_post_step_state: Tensor,
    contract: ActionContract,
    *,
    registered_margin_rad: float,
    source: str,
) -> dict[str, Any]:
    """Compare an absolute joint command with the next simulator observation."""

    pre = _arm_joint_margin_snapshot(
        pre_step_state,
        contract,
        registered_margin_rad=registered_margin_rad,
    )
    command = _arm_joint_margin_snapshot(
        commanded_action,
        contract,
        registered_margin_rad=registered_margin_rad,
    )
    observed = _arm_joint_margin_snapshot(
        observed_post_step_state,
        contract,
        registered_margin_rad=registered_margin_rad,
    )

    tracking: list[dict[str, Any]] = []
    for index in ARM_ACTION_INDICES:
        name = contract.dimension_names[index]
        before = float(pre_step_state[index])
        commanded = float(commanded_action[index])
        after = float(observed_post_step_state[index])
        command_delta = commanded - before
        observed_delta = after - before
        tracking_error = after - commanded
        overshoot_toward_commanded_bound = 0.0
        bound: str | None = None
        if command_delta < 0.0 and after < commanded:
            overshoot_toward_commanded_bound = commanded - after
            bound = "lower"
        elif command_delta > 0.0 and after > commanded:
            overshoot_toward_commanded_bound = after - commanded
            bound = "upper"
        command_joint = command["joints"][len(tracking)]
        observed_joint = observed["joints"][len(tracking)]
        tracking.append(
            {
                "index": index,
                "name": name,
                "command_delta_rad": command_delta,
                "observed_delta_rad": observed_delta,
                "observed_minus_command_rad": tracking_error,
                "absolute_tracking_error_rad": abs(tracking_error),
                "overshoot_toward_commanded_bound_rad": (
                    overshoot_toward_commanded_bound
                ),
                "overshoot_bound": bound,
                "margin_loss_command_to_observation_rad": max(
                    0.0,
                    float(command_joint["minimum_margin_rad"])
                    - float(observed_joint["minimum_margin_rad"]),
                ),
            }
        )

    maximum_error = max(
        tracking,
        key=lambda item: float(item["absolute_tracking_error_rad"]),
    )
    maximum_overshoot = max(
        tracking,
        key=lambda item: float(item["overshoot_toward_commanded_bound_rad"]),
    )
    maximum_margin_loss = max(
        tracking,
        key=lambda item: float(item["margin_loss_command_to_observation_rad"]),
    )
    return {
        "schema": "commanded_vs_observed_joint_margin_v1",
        "source": source,
        "pre_step": pre,
        "commanded": command,
        "observed_post_step": observed,
        "maximum_absolute_tracking_error_rad": float(
            maximum_error["absolute_tracking_error_rad"]
        ),
        "maximum_absolute_tracking_error_joint": maximum_error["name"],
        "maximum_overshoot_toward_commanded_bound_rad": float(
            maximum_overshoot["overshoot_toward_commanded_bound_rad"]
        ),
        "maximum_overshoot_toward_commanded_bound_joint": maximum_overshoot["name"],
        "maximum_margin_loss_command_to_observation_rad": float(
            maximum_margin_loss["margin_loss_command_to_observation_rad"]
        ),
        "maximum_margin_loss_command_to_observation_joint": maximum_margin_loss["name"],
        "tracking": tracking,
    }


def _physics(environment: GymAlohaEnvironment) -> Any:
    unwrapped = getattr(environment.raw_environment, "unwrapped", environment.raw_environment)
    control_environment = getattr(unwrapped, "_env", None)
    physics = getattr(control_environment, "physics", None)
    if physics is None:
        raise RuntimeError("Geometry teacher requires the registered MuJoCo backend.")
    return physics


def _body_pose(physics: Any, name: str) -> GeometryPose:
    try:
        body_id = int(physics.model.name2id(name, "body"))
    except (KeyError, ValueError) as error:
        raise ValueError(f"MuJoCo body {name!r} is unavailable.") from error
    return GeometryPose(
        position=torch.as_tensor(np.asarray(physics.data.xpos[body_id]).copy()),
        quaternion=torch.as_tensor(np.asarray(physics.data.xquat[body_id]).copy()),
    )


def _site_pose(physics: Any, name: str) -> GeometryPose:
    from dm_control.mujoco.wrapper.mjbindings import mjlib

    position = np.asarray(physics.named.data.site_xpos[name]).copy()
    quaternion = np.empty(4, dtype=physics.data.qpos.dtype)
    mjlib.mju_mat2Quat(quaternion, physics.named.data.site_xmat[name])
    return GeometryPose(
        position=torch.as_tensor(position),
        quaternion=torch.as_tensor(quaternion),
    )


def _has_contact(pairs: set[frozenset[str]], first: str, second: str) -> bool:
    return frozenset({first, second}) in pairs


def _collision_classification(
    environment: GymAlohaEnvironment,
    allowed_task_contact_pairs: frozenset[frozenset[str]],
) -> tuple[int, tuple[tuple[str, str], ...]]:
    """Keep raw contacts visible while exempting only registered task pairs."""

    unexpected = 0
    exemptions: list[tuple[str, str]] = []
    for first, second in environment.contact_pairs():
        if not environment.is_unexpected_collision_pair(first, second):
            continue
        if frozenset({first, second}) in allowed_task_contact_pairs:
            exemptions.append((first, second))
        else:
            unexpected += 1
    return unexpected, tuple(exemptions)


def _current_geometry(
    environment: GymAlohaEnvironment,
    observation: dict[str, Any],
    *,
    observed_reward: float,
    allowed_task_contact_pairs: frozenset[frozenset[str]] = frozenset(),
) -> InsertionGeometry:
    physics = _physics(environment)
    contacts = {frozenset(pair) for pair in environment.contact_pairs()}
    unexpected_collision_count, _ = _collision_classification(
        environment,
        allowed_task_contact_pairs,
    )
    left_fingers = (
        "vx300s_left/10_left_gripper_finger",
        "vx300s_left/10_right_gripper_finger",
    )
    right_fingers = (
        "vx300s_right/10_left_gripper_finger",
        "vx300s_right/10_right_gripper_finger",
    )
    socket_geometries = tuple(f"socket-{index}" for index in range(1, 5))
    return InsertionGeometry(
        robot_state=_robot_state(observation, 14),
        left_eef=_site_pose(physics, LEFT_SITE),
        right_eef=_site_pose(physics, RIGHT_SITE),
        socket=_body_pose(physics, "socket"),
        peg=_body_pose(physics, "peg"),
        observed_reward=observed_reward,
        socket_grasp_contact=any(
            _has_contact(contacts, socket, finger)
            for socket in socket_geometries
            for finger in left_fingers
        ),
        peg_grasp_contact=any(
            _has_contact(contacts, "red_peg", finger) for finger in right_fingers
        ),
        socket_on_table=any(
            _has_contact(contacts, socket, "table") for socket in socket_geometries
        ),
        peg_on_table=_has_contact(contacts, "red_peg", "table"),
        peg_socket_contact=any(
            _has_contact(contacts, "red_peg", socket) for socket in socket_geometries
        ),
        pin_contact=_has_contact(contacts, "red_peg", "pin"),
        unexpected_collision_count=unexpected_collision_count,
    )


def _trajectory_rows(
    root: Path,
    episode: int,
    dataset_config: DatasetConfig,
    contract: ActionContract,
) -> list[dict[str, Any]]:
    import pyarrow.dataset as arrow_dataset

    fields_value = dataset_config.fields
    dataset = arrow_dataset.dataset(root / "data", format="parquet")
    table = dataset.to_table(
        columns=[
            fields_value.episode_index,
            fields_value.frame_index,
            fields_value.timestamp,
            fields_value.state,
            fields_value.action,
        ],
        filter=arrow_dataset.field(fields_value.episode_index) == episode,
    )
    rows = sorted(table.to_pylist(), key=lambda row: int(row[fields_value.frame_index]))
    if not rows:
        raise ValueError(f"Calibration episode {episode} contains no rows.")
    for expected_frame, row in enumerate(rows):
        if (
            int(row[fields_value.episode_index]) != episode
            or int(row[fields_value.frame_index]) != expected_frame
        ):
            raise ValueError("Calibration trajectory is not an exact contiguous episode.")
        timestamp = float(row[fields_value.timestamp])
        if not math.isclose(
            timestamp,
            expected_frame / contract.frequency_hz,
            rel_tol=0.0,
            abs_tol=1e-4,
        ):
            raise ValueError("Calibration trajectory violates the Action Contract frequency.")
        contract.validate_tensor(
            torch.as_tensor(row[fields_value.action], dtype=torch.float32),
            allow_chunk=False,
        )
    return rows


def _calibrate_from_replay(
    rows: list[dict[str, Any]],
    *,
    episode: int,
    seed: int,
    insertion_axis: Tensor,
    dataset_config: DatasetConfig,
    contract: ActionContract,
) -> CalibrationReplay:
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=len(rows))
    first_grasp_step: int | None = None
    terminal_step: int | None = None
    socket_to_left: GeometryPose | None = None
    peg_to_right: GeometryPose | None = None
    terminal_socket_to_peg: GeometryPose | None = None
    maximum_reward = 0.0
    success = False
    try:
        observation = dict(environment.reset(seed=seed))
        for step, row in enumerate(rows):
            raw_action = torch.as_tensor(
                row[dataset_config.fields.action],
                dtype=torch.float32,
            )
            action, _clip_mask = contract.clip(raw_action)
            observation_value, reward, done, info = environment.step(action)
            observation = dict(observation_value)
            maximum_reward = max(maximum_reward, float(reward))
            geometry = _current_geometry(
                environment,
                observation,
                observed_reward=float(reward),
            )
            if (
                first_grasp_step is None
                and reward >= 1.0
                and geometry.socket_grasp_contact
                and geometry.peg_grasp_contact
            ):
                first_grasp_step = step
                socket_to_left = relative_pose(geometry.socket, geometry.left_eef)
                peg_to_right = relative_pose(geometry.peg, geometry.right_eef)
            if done:
                success = bool(info.get("is_success", False))
                terminal_step = step
                terminal_socket_to_peg = relative_pose(geometry.socket, geometry.peg)
                break
    finally:
        environment.close()
    if (
        not success
        or maximum_reward != 4.0
        or first_grasp_step is None
        or terminal_step is None
        or socket_to_left is None
        or peg_to_right is None
        or terminal_socket_to_peg is None
    ):
        raise RuntimeError(
            f"Calibration episode {episode} seed {seed} is not a complete reward-four replay."
        )
    return CalibrationReplay(
        calibration=InsertionTeacherCalibration(
            socket_to_left_eef_at_grasp=socket_to_left,
            peg_to_right_eef_at_grasp=peg_to_right,
            terminal_socket_to_peg=terminal_socket_to_peg,
            insertion_axis_in_socket=insertion_axis,
            source_episode=episode,
            source_seed=seed,
            terminal_reward=maximum_reward,
            terminal_success=success,
        ),
        steps_executed=terminal_step + 1,
        first_grasp_step=first_grasp_step,
        terminal_step=terminal_step,
        maximum_reward=maximum_reward,
    )


def _expanded_robot_qpos(logical: Tensor) -> np.ndarray:
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


def _mink_ik_action(
    scratch_physics: Any,
    current_state: Tensor,
    target: InsertionTaskSpaceTarget,
    *,
    contract: ActionContract,
    settings: dict[str, Any],
    solver: MinkAlohaIkSolver,
) -> IkActionResult:
    """Solve the registered dual-arm pose through Mink's constrained QP."""

    def projected_error(action: Tensor) -> float:
        scratch_physics.data.qpos[:16] = _expanded_robot_qpos(action)
        scratch_physics.forward()
        errors: list[float] = []
        for site_name, target_pose in (
            (LEFT_SITE, target.left_eef),
            (RIGHT_SITE, target.right_eef),
        ):
            achieved = _site_pose(scratch_physics, site_name)
            position_error = float(
                torch.linalg.vector_norm(achieved.position - target_pose.position)
            )
            quaternion_dot = abs(
                float(torch.dot(achieved.quaternion, target_pose.quaternion))
            )
            quaternion_dot = min(1.0, max(-1.0, quaternion_dot))
            rotation_error = 2.0 * math.acos(quaternion_dot)
            errors.append(
                position_error + float(settings["rotation_weight"]) * rotation_error
            )
        return max(errors)

    try:
        solved = solver.solve(
            _expanded_robot_qpos(current_state),
            left_position=target.left_eef.position.numpy(),
            left_quaternion_wxyz=target.left_eef.quaternion.numpy(),
            right_position=target.right_eef.position.numpy(),
            right_quaternion_wxyz=target.right_eef.quaternion.numpy(),
        )
    except (RuntimeError, ValueError) as error:
        failure_error = projected_error(current_state)
        return IkActionResult(
            action=current_state.detach().cpu(),
            success=False,
            maximum_error=failure_error,
            maximum_projected_error=failure_error,
            joint_delta_saturations=0,
            contract_clip_fields=(),
            solver_backend="mink_qp_daqp",
            solver_iterations=0,
            solver_failure=str(error),
        )
    raw = current_state.clone()
    raw[:6] = torch.as_tensor(solved.qpos[:6], dtype=torch.float32)
    raw[7:13] = torch.as_tensor(solved.qpos[8:14], dtype=torch.float32)
    raw[6] = float(target.left_gripper)
    raw[13] = float(target.right_gripper)
    maximum_delta = float(settings["maximum_joint_target_delta"])
    joint_indices = (*range(6), *range(7, 13))
    deltas = (raw[list(joint_indices)] - current_state[list(joint_indices)]).abs()
    joint_delta_saturations = int(deltas.gt(maximum_delta).sum())
    action, clip_mask = contract.clip(raw)
    clip_fields = tuple(
        name
        for name, clipped in zip(contract.dimension_names, clip_mask.tolist())
        if clipped
    )
    weighted_errors = tuple(
        position + float(settings["rotation_weight"]) * orientation
        for position, orientation in zip(
            solved.position_errors_m,
            solved.orientation_errors_rad,
        )
    )
    maximum_error = max(weighted_errors)

    maximum_projected_error = projected_error(action)
    success = (
        math.isfinite(maximum_error)
        and maximum_error <= float(settings["maximum_accepted_error"])
        and maximum_projected_error
        <= float(settings["maximum_accepted_projected_error"])
        and joint_delta_saturations == 0
        and not clip_fields
    )
    return IkActionResult(
        action=action.detach().cpu(),
        success=success,
        maximum_error=maximum_error,
        maximum_projected_error=maximum_projected_error,
        joint_delta_saturations=joint_delta_saturations,
        contract_clip_fields=clip_fields,
        solver_backend="mink_qp_daqp",
        solver_iterations=solved.iterations,
    )


def _moveit_reference_action(
    current_state: Tensor,
    target: InsertionTaskSpaceTarget,
    *,
    contract: ActionContract,
    settings: dict[str, Any],
    initial_result: IkActionResult,
    planned: MoveItAlohaPlanResult,
    command: MoveItAlohaTrajectoryCommand,
    attempted: bool,
    terminal_control: MujocoPositionFeedforwardResult | None = None,
) -> IkActionResult:
    """Map one retained official trajectory waypoint to the Action Contract."""

    raw = current_state.clone()
    raw[:6] = torch.as_tensor(command.positions[:6], dtype=torch.float32)
    raw[7:13] = torch.as_tensor(command.positions[6:], dtype=torch.float32)
    raw[6] = float(target.left_gripper)
    raw[13] = float(target.right_gripper)
    action, clip_mask = contract.clip(raw)
    final_clip_fields = tuple(
        name
        for name, clipped in zip(contract.dimension_names, clip_mask.tolist())
        if clipped
    )
    if final_clip_fields:
        raise MoveItAlohaPlannerError(
            "MoveIt retained waypoint required additional Action Contract clipping."
        )
    maximum_delta = max(
        abs(float(action[index] - current_state[index]))
        for index in ARM_ACTION_INDICES
    )
    quantization_scale = max(
        1.0,
        *(abs(float(action[index])) for index in ARM_ACTION_INDICES),
        *(abs(float(current_state[index])) for index in ARM_ACTION_INDICES),
    )
    quantization_tolerance = float(torch.finfo(action.dtype).eps) * quantization_scale
    if (
        maximum_delta
        > float(settings["maximum_joint_target_delta"]) + quantization_tolerance
    ):
        raise MoveItAlohaPlannerError(
            "MoveIt retained waypoint exceeds the evaluator joint target delta."
        )
    position_priority = planned.ik_task_mode == "position_priority"
    goal_error = (
        planned.maximum_goal_position_error_m
        if position_priority
        else planned.maximum_goal_weighted_error
    )
    success = (
        planned.maximum_goal_position_error_m
        <= float(settings["path_planner_position_tolerance_m"])
        and planned.maximum_goal_orientation_error_rad
        <= float(settings["path_planner_position_priority_maximum_orientation_relaxation_rad"])
        if position_priority
        else goal_error <= float(settings["maximum_accepted_error"])
        and goal_error <= float(settings["maximum_accepted_projected_error"])
    )
    recovery = planned.start_state_path_constraint_recovery
    return IkActionResult(
        action=action.detach().cpu(),
        success=success,
        maximum_error=goal_error,
        maximum_projected_error=goal_error,
        joint_delta_saturations=0,
        contract_clip_fields=initial_result.contract_clip_fields,
        path_planner_attempted=attempted,
        path_planner_used=success,
        path_planner_mode=(
            (
                "moveit2_fix_start_state_path_constraints+ompl_rrtconnect+"
                "simple_sampler+forward_trajectory"
            )
            if recovery
            else "moveit2_ompl_rrtconnect+simple_sampler+forward_trajectory"
        )
        + ("+official_lma_position_only_ik" if position_priority else "")
        + (
            "+mujoco_static_inverse_dynamics_position_feedforward"
            if command.terminal_control_active
            else ""
        ),
        path_planner_fraction=command.interpolation,
        path_planner_orientation_relaxation_rad=(
            planned.maximum_goal_orientation_error_rad
            if position_priority
            else 0.0
        ),
        path_planner_initial_projected_error=(
            initial_result.maximum_projected_error if attempted else None
        ),
        path_planner_planning_time_s=(planned.planning_time_s if attempted else None),
        path_planner_waypoint_count=planned.waypoint_count,
        path_planner_path_length_rad=planned.path_length_rad,
        path_planner_goal_position_error_m=planned.maximum_goal_position_error_m,
        path_planner_goal_orientation_error_rad=(
            planned.maximum_goal_orientation_error_rad
        ),
        path_planner_goal_weighted_error=planned.maximum_goal_weighted_error,
        path_planner_ik_search_mode=planned.ik_search_mode,
        path_planner_ik_candidate_selection_mode=(
            planned.ik_candidate_selection_mode
        ),
        path_planner_ik_seed=planned.ik_seed,
        path_planner_ik_maximum_attempts=planned.ik_maximum_attempts,
        path_planner_ik_attempts_used=planned.ik_attempts_used,
        path_planner_valid_ik_candidate_count=(
            planned.valid_ik_candidate_count
        ),
        path_planner_selected_ik_attempt=planned.selected_ik_attempt,
        path_planner_selected_ik_minimum_joint_limit_margin_rad=(
            planned.selected_ik_minimum_joint_limit_margin_rad
        ),
        path_planner_selected_ik_maximum_start_delta_rad=(
            planned.selected_ik_maximum_start_delta_rad
        ),
        path_planner_ik_outer_timeout_s=planned.ik_outer_timeout_s,
        path_planner_joint_limit_margin_rad=planned.joint_limit_margin_rad,
        path_planner_physical_joint_limit_margin_rad=(
            planned.physical_joint_limit_margin_rad
        ),
        path_planner_start_state_path_constraint_recovery=recovery,
        path_planner_adapter_prefix_waypoint_count=(
            planned.adapter_prefix_waypoint_count
        ),
        path_planner_minimum_recovery_progress_rad=(
            planned.minimum_recovery_progress_rad
        ),
        path_planner_minimum_start_joint_limit_margin_rad=(
            planned.minimum_start_joint_limit_margin_rad
        ),
        path_planner_minimum_goal_joint_limit_margin_rad=(
            planned.minimum_goal_joint_limit_margin_rad
        ),
        path_planner_minimum_path_joint_limit_margin_rad=(
            planned.minimum_path_joint_limit_margin_rad
        ),
        path_planner_minimum_constrained_path_joint_limit_margin_rad=(
            planned.minimum_constrained_path_joint_limit_margin_rad
        ),
        path_planner_minimum_adapter_prefix_physical_joint_limit_margin_rad=(
            planned.minimum_adapter_prefix_physical_joint_limit_margin_rad
        ),
        path_planner_minimum_next_joint_limit_margin_rad=(
            planned.minimum_next_joint_limit_margin_rad
        ),
        path_planner_start_bound_reconciliations=planned.start_bound_reconciliations,
        path_planner_maximum_start_bound_reconciliation_rad=(
            planned.maximum_start_bound_reconciliation_rad
        ),
        path_planner_reference_reused=command.reference_reused,
        path_planner_reference_waypoint_index=command.waypoint_index,
        path_planner_reference_waypoint_advanced=command.waypoint_advanced,
        path_planner_reference_waypoint_l1_distance_rad=(
            command.waypoint_l1_distance_rad
        ),
        path_planner_terminal_control_active=command.terminal_control_active,
        path_planner_terminal_control_activated=command.terminal_control_activated,
        path_planner_terminal_control_maximum_correction_rad=(
            terminal_control.maximum_correction_rad
            if terminal_control is not None
            else None
        ),
        path_planner_terminal_control_minimum_command_margin_rad=(
            terminal_control.minimum_command_joint_limit_margin_rad
            if terminal_control is not None
            else None
        ),
        solver_backend=(
            "moveit2_fix_start_state_path_constraints+ompl_lma_rrtconnect+"
            "simple_sampler+forward_trajectory"
            if recovery
            else "moveit2_ompl_lma_rrtconnect+simple_sampler+forward_trajectory"
        )
        + ("+official_lma_position_only_ik" if position_priority else "")
        + (
            "+mujoco_static_inverse_dynamics_position_feedforward"
            if command.terminal_control_active
            else ""
        ),
    )


def _moveit_path_action(
    current_state: Tensor,
    target: InsertionTaskSpaceTarget,
    *,
    scratch_physics: Any | None = None,
    contract: ActionContract,
    settings: dict[str, Any],
    initial_result: IkActionResult,
    planner: MoveItAlohaPlanner,
    executor: MoveItAlohaTrajectoryExecutor,
) -> IkActionResult:
    """Use an official collision-checked OMPL path after local Mink stalls."""

    arm_indices = (*range(6), *range(7, 13))
    start = [float(current_state[index]) for index in arm_indices]
    expanded = _expanded_robot_qpos(current_state)
    try:
        finger_lower = float(settings["path_planner_finger_lower_m"])
        finger_upper = float(settings["path_planner_finger_upper_m"])
        raw_fingers = (float(expanded[6]), float(expanded[14]))
        fingers = tuple(
            min(finger_upper, max(finger_lower, value)) for value in raw_fingers
        )
        maximum_finger_reconciliation = max(
            abs(reconciled - raw)
            for reconciled, raw in zip(fingers, raw_fingers)
        )
        if maximum_finger_reconciliation > float(
            settings["path_planner_finger_bound_reconciliation_tolerance_m"]
        ) + 1e-12:
            raise MoveItAlohaPlannerError(
                "Gym gripper state exceeds the registered official finger-bound adapter."
            )
        def request_plan(
            planning_target: InsertionTaskSpaceTarget,
            *,
            ik_task_mode: str,
        ) -> MoveItAlohaPlanResult:
            return planner.plan(
            start=start,
            finger_positions=list(fingers),
            left_position=planning_target.left_eef.position.tolist(),
            left_quaternion_wxyz=planning_target.left_eef.quaternion.tolist(),
            right_position=planning_target.right_eef.position.tolist(),
            right_quaternion_wxyz=planning_target.right_eef.quaternion.tolist(),
            allowed_planning_time_s=float(
                settings["path_planner_allowed_planning_time_s"]
            ),
            ik_timeout_s=float(settings["path_planner_ik_timeout_s"]),
            ik_search_mode=str(settings["path_planner_ik_search_mode"]),
            ik_seed=int(settings["path_planner_ik_seed"]),
            ik_maximum_attempts=int(
                settings["path_planner_ik_maximum_attempts"]
            ),
            maximum_joint_step_rad=float(
                settings["path_planner_maximum_joint_step_rad"]
            ),
            position_tolerance_m=float(
                settings["path_planner_position_tolerance_m"]
            ),
            orientation_tolerance_rad=float(
                settings["path_planner_orientation_tolerance_rad"]
            ),
            rotation_weight=float(settings["rotation_weight"]),
            maximum_accepted_error=float(settings["maximum_accepted_error"]),
            maximum_accepted_projected_error=float(
                settings["maximum_accepted_projected_error"]
            ),
            start_bound_reconciliation_tolerance_rad=float(
                settings["path_planner_start_bound_reconciliation_tolerance_rad"]
            ),
            physical_joint_limit_margin_rad=float(
                settings["path_planner_physical_joint_limit_margin_rad"]
            ),
            joint_limit_margin_rad=float(
                settings["path_planner_joint_limit_margin_rad"]
            ),
            ik_task_mode=ik_task_mode,
            maximum_orientation_relaxation_rad=float(
                settings[
                    "path_planner_position_priority_maximum_orientation_relaxation_rad"
                ]
            ),
            include_trajectory=bool(settings["path_planner_include_trajectory"]),
        )
        planned = request_plan(target, ik_task_mode="full_pose")
    except MoveItAlohaPlanningError as error:
        full_pose_backoff_enabled = (
            error.reason == "bimanual_lma_ik_failed"
            and target.phase.value
            in settings.get(
                "path_planner_full_pose_cartesian_backoff_activation_phases",
                (),
            )
            and settings.get("path_planner_full_pose_cartesian_backoff_enabled")
            is True
        )
        position_priority_enabled = (
            error.reason == "bimanual_lma_ik_failed"
            and target.phase.value == "approach"
            and settings.get("path_planner_position_priority_enabled") is True
        )
        if full_pose_backoff_enabled or position_priority_enabled:
            if scratch_physics is None:
                raise MoveItAlohaPlannerError(
                    "MoveIt Cartesian fallback requires MuJoCo kinematics."
                )
            scratch_physics.data.qpos[:16] = _expanded_robot_qpos(current_state)
            scratch_physics.forward()
            current_poses = (
                _site_pose(scratch_physics, LEFT_SITE),
                _site_pose(scratch_physics, RIGHT_SITE),
            )
            requested_poses = (target.left_eef, target.right_eef)

        if full_pose_backoff_enabled:
            minimum_linear_progress = float(
                settings[
                    "path_planner_full_pose_cartesian_backoff_minimum_linear_progress_m"
                ]
            )
            minimum_angular_progress = float(
                settings[
                    "path_planner_full_pose_cartesian_backoff_minimum_angular_progress_rad"
                ]
            )
            for fraction_value in settings[
                "path_planner_full_pose_cartesian_backoff_fractions"
            ]:
                fraction = float(fraction_value)
                candidate_poses = tuple(
                    _full_pose_cartesian_waypoint(current, requested, fraction)
                    for current, requested in zip(current_poses, requested_poses)
                )
                linear_progress = max(
                    float(
                        torch.linalg.vector_norm(
                            candidate.position - current.position
                        )
                    )
                    for current, candidate in zip(current_poses, candidate_poses)
                )
                angular_progress = max(
                    _quaternion_distance(current.quaternion, candidate.quaternion)
                    for current, candidate in zip(current_poses, candidate_poses)
                )
                if (
                    linear_progress < minimum_linear_progress
                    and angular_progress < minimum_angular_progress
                ):
                    continue
                candidate_target = replace(
                    target,
                    left_eef=candidate_poses[0],
                    right_eef=candidate_poses[1],
                )
                try:
                    planned = request_plan(
                        candidate_target,
                        ik_task_mode="full_pose",
                    )
                except MoveItAlohaPlanningError as backoff_error:
                    if backoff_error.reason not in {
                        "bimanual_lma_ik_failed",
                        "ompl_planning_failed",
                    }:
                        raise
                    continue
                executor.install(planned, phase=target.phase.value)
                command = executor.command(start, phase=target.phase.value)
                result = _moveit_reference_action(
                    current_state,
                    target,
                    contract=contract,
                    settings=settings,
                    initial_result=initial_result,
                    planned=planned,
                    command=command,
                    attempted=True,
                )
                return replace(
                    result,
                    path_planner_fraction=fraction,
                    path_planner_mode=(
                        f"{result.path_planner_mode}+full_pose_cartesian_backoff"
                    ),
                )

        trust_region_enabled = (
            error.reason == "bimanual_lma_ik_failed"
            and target.phase.value == "orient"
            and settings.get("path_planner_active_set_trust_region_enabled") is True
        )
        if trust_region_enabled:
            if scratch_physics is None:
                raise MoveItAlohaPlannerError(
                    "MoveIt active-set trust region requires MuJoCo kinematics."
                )
            scratch_physics.data.qpos[:16] = _expanded_robot_qpos(current_state)
            scratch_physics.forward()
            current_poses = (
                _site_pose(scratch_physics, LEFT_SITE),
                _site_pose(scratch_physics, RIGHT_SITE),
            )
            requested_poses = (target.left_eef, target.right_eef)
            current_margin = _arm_joint_margin_snapshot(
                current_state,
                contract,
                registered_margin_rad=float(
                    settings["path_planner_joint_limit_margin_rad"]
                ),
            )
            minimum_joint = str(current_margin["minimum_margin_joint"])
            if minimum_joint.startswith("left_"):
                active_arm_index = 0
                active_arm = "left"
            elif minimum_joint.startswith("right_"):
                active_arm_index = 1
                active_arm = "right"
            else:
                raise MoveItAlohaPlannerError(
                    "Active-set trust region could not map the minimum-margin joint."
                )
            passive_arm_index = 1 - active_arm_index
            orientation_target_rad = _quaternion_distance(
                current_poses[active_arm_index].quaternion,
                requested_poses[active_arm_index].quaternion,
            )
            maximum_relaxation_m = float(
                settings[
                    "path_planner_active_set_trust_region_maximum_"
                    "requested_position_relaxation_m"
                ]
            )
            passive_relaxation_m = float(
                torch.linalg.vector_norm(
                    current_poses[passive_arm_index].position
                    - requested_poses[passive_arm_index].position
                )
            )

            def trust_region_candidates(
                orientation_fraction: float,
            ) -> tuple[
                list[
                    tuple[
                        MoveItAlohaPlanResult,
                        float,
                        tuple[float, float, float],
                        float,
                    ]
                ],
                int,
            ]:
                if not 0.0 <= orientation_fraction <= 1.0:
                    raise MoveItAlohaPlannerError(
                        "Trust-region orientation fraction is outside [0, 1]."
                    )
                active_current = current_poses[active_arm_index]
                active_requested = requested_poses[active_arm_index]
                if orientation_fraction == 0.0:
                    active_quaternion = active_current.quaternion
                else:
                    active_quaternion, _ = _bounded_orientation_waypoint(
                        active_current.quaternion,
                        active_requested.quaternion,
                        _quaternion_distance(
                            active_current.quaternion,
                            active_requested.quaternion,
                        )
                        * orientation_fraction,
                    )
                successful: list[
                    tuple[
                        MoveItAlohaPlanResult,
                        float,
                        tuple[float, float, float],
                        float,
                    ]
                ] = []
                evaluated = 0
                trust_region_basis = settings.get(
                    "path_planner_active_set_trust_region_candidate_basis"
                )
                if trust_region_basis == "feedback_aligned_orthonormal_v1":
                    try:
                        directions = _feedback_aligned_orthonormal_basis(
                            active_current.position,
                            active_requested.position,
                        )
                    except ValueError:
                        return [], 0
                else:
                    directions = tuple(
                        tuple(float(component) for component in direction)
                        for direction in settings[
                            "path_planner_active_set_trust_region_coordinate_directions"
                        ]
                    )
                for radius_value in settings[
                    "path_planner_active_set_trust_region_radii_m"
                ]:
                    radius_m = float(radius_value)
                    candidate_directions = (
                        ((0.0, 0.0, 0.0),) if radius_m == 0.0 else directions
                    )
                    for direction in candidate_directions:
                        direction_tensor = torch.tensor(
                            direction,
                            dtype=active_current.position.dtype,
                        )
                        active_position = (
                            active_current.position + direction_tensor * radius_m
                        )
                        active_relaxation_m = float(
                            torch.linalg.vector_norm(
                                active_position - active_requested.position
                            )
                        )
                        requested_relaxation_m = max(
                            passive_relaxation_m,
                            active_relaxation_m,
                        )
                        if requested_relaxation_m > maximum_relaxation_m + 1e-12:
                            continue
                        candidate_poses = [
                            GeometryPose(
                                position=current.position,
                                quaternion=requested.quaternion,
                            )
                            for current, requested in zip(
                                current_poses,
                                requested_poses,
                            )
                        ]
                        candidate_poses[active_arm_index] = GeometryPose(
                            position=active_position,
                            quaternion=active_quaternion,
                        )
                        candidate_target = replace(
                            target,
                            left_eef=candidate_poses[0],
                            right_eef=candidate_poses[1],
                        )
                        evaluated += 1
                        try:
                            candidate_plan = request_plan(
                                candidate_target,
                                ik_task_mode="full_pose",
                            )
                        except MoveItAlohaPlanningError as trust_error:
                            if trust_error.reason not in {
                                "bimanual_lma_ik_failed",
                                "ompl_planning_failed",
                            }:
                                raise
                            continue
                        successful.append(
                            (
                                candidate_plan,
                                radius_m,
                                direction,
                                requested_relaxation_m,
                            )
                        )
                return successful, evaluated

            selected: tuple[
                MoveItAlohaPlanResult,
                float,
                tuple[float, float, float],
                float,
            ] | None = None
            selected_mode: str | None = None
            selected_fraction = 0.0
            selected_valid_candidate_count = 0
            candidates_evaluated = 0
            selection_policy = settings.get(
                "path_planner_active_set_trust_region_selection_policy"
            )
            if selection_policy == "orientation_progress_first_v1":
                for fraction_value in settings[
                    "path_planner_active_set_trust_region_"
                    "orientation_progress_fractions"
                ]:
                    orientation_fraction = float(fraction_value)
                    progress_candidates, evaluated = trust_region_candidates(
                        orientation_fraction
                    )
                    candidates_evaluated += evaluated
                    if not progress_candidates:
                        continue
                    selected = min(
                        progress_candidates,
                        key=lambda candidate: (
                            -candidate[0].minimum_goal_joint_limit_margin_rad,
                            candidate[1],
                            candidate[2],
                        ),
                    )
                    selected_mode = "orientation_progress"
                    selected_fraction = orientation_fraction
                    selected_valid_candidate_count = len(progress_candidates)
                    break
            if selected is None:
                margin_candidates, evaluated = trust_region_candidates(0.0)
                candidates_evaluated += evaluated
                minimum_margin_improvement_rad = float(
                    settings[
                        "path_planner_active_set_trust_region_"
                        "minimum_margin_improvement_rad"
                    ]
                )
                restoration_reference = settings.get(
                    "path_planner_active_set_trust_region_restoration_reference"
                )
                if restoration_reference == "command_margin_boundary":
                    restoration_threshold_rad = (
                        float(settings["path_planner_joint_limit_margin_rad"])
                        + minimum_margin_improvement_rad
                    )
                else:
                    restoration_threshold_rad = (
                        float(current_margin["minimum_margin_rad"])
                        + minimum_margin_improvement_rad
                    )
                improving_candidates = [
                    candidate
                    for candidate in margin_candidates
                    if candidate[0].minimum_goal_joint_limit_margin_rad
                    >= restoration_threshold_rad - 1e-12
                ]
            else:
                improving_candidates = []
            if selected is None and improving_candidates:
                selected = min(
                    improving_candidates,
                    key=lambda candidate: (
                        -candidate[0].minimum_goal_joint_limit_margin_rad,
                        candidate[1],
                        candidate[2],
                    ),
                )
                selected_mode = "margin_restoration"
                selected_valid_candidate_count = len(improving_candidates)
            elif selected is None and selection_policy != "orientation_progress_first_v1":
                for fraction_value in settings[
                    "path_planner_active_set_trust_region_"
                    "orientation_progress_fractions"
                ]:
                    orientation_fraction = float(fraction_value)
                    progress_candidates, evaluated = trust_region_candidates(
                        orientation_fraction
                    )
                    candidates_evaluated += evaluated
                    if not progress_candidates:
                        continue
                    selected = min(
                        progress_candidates,
                        key=lambda candidate: (
                            -candidate[0].minimum_goal_joint_limit_margin_rad,
                            candidate[1],
                            candidate[2],
                        ),
                    )
                    selected_mode = "orientation_progress"
                    selected_fraction = orientation_fraction
                    selected_valid_candidate_count = len(progress_candidates)
                    break
            if selected is not None and selected_mode is not None:
                planned, radius_m, direction, requested_relaxation_m = selected
                executor.install(planned, phase=target.phase.value)
                command = executor.command(start, phase=target.phase.value)
                result = _moveit_reference_action(
                    current_state,
                    target,
                    contract=contract,
                    settings=settings,
                    initial_result=initial_result,
                    planned=planned,
                    command=command,
                    attempted=True,
                )
                return replace(
                    result,
                    path_planner_mode=(
                        f"{result.path_planner_mode}+active_set_cartesian_trust_region"
                    ),
                    path_planner_fraction=selected_fraction,
                    path_planner_trust_region_mode=selected_mode,
                    path_planner_trust_region_basis=settings.get(
                        "path_planner_active_set_trust_region_candidate_basis",
                        "fixed_world_coordinate_basis",
                    ),
                    path_planner_trust_region_selection_policy=selection_policy,
                    path_planner_trust_region_restoration_reference=settings.get(
                        "path_planner_active_set_trust_region_restoration_reference",
                        "current_margin",
                    ),
                    path_planner_trust_region_active_arm=active_arm,
                    path_planner_trust_region_radius_m=radius_m,
                    path_planner_trust_region_direction=direction,
                    path_planner_trust_region_orientation_fraction=selected_fraction,
                    path_planner_trust_region_orientation_target_rad=(
                        orientation_target_rad
                    ),
                    path_planner_trust_region_margin_improvement_rad=(
                        planned.minimum_goal_joint_limit_margin_rad
                        - float(current_margin["minimum_margin_rad"])
                    ),
                    path_planner_trust_region_requested_position_relaxation_m=(
                        requested_relaxation_m
                    ),
                    path_planner_trust_region_candidates_evaluated=(
                        candidates_evaluated
                    ),
                    path_planner_trust_region_valid_candidates=(
                        selected_valid_candidate_count
                    ),
                )

        if position_priority_enabled:
            minimum_progress = float(
                settings["path_planner_position_priority_minimum_cartesian_progress_m"]
            )
            for fraction_value in settings[
                "path_planner_position_priority_cartesian_backoff_fractions"
            ]:
                fraction = float(fraction_value)
                candidate_poses = tuple(
                    _cartesian_waypoint(
                        current,
                        requested,
                        fraction,
                        requested.quaternion,
                    )
                    for current, requested in zip(current_poses, requested_poses)
                )
                progress = max(
                    float(
                        torch.linalg.vector_norm(
                            candidate.position - current.position
                        )
                    )
                    for current, candidate in zip(current_poses, candidate_poses)
                )
                if progress < minimum_progress:
                    continue
                candidate_target = replace(
                    target,
                    left_eef=candidate_poses[0],
                    right_eef=candidate_poses[1],
                )
                try:
                    planned = request_plan(
                        candidate_target,
                        ik_task_mode="position_priority",
                    )
                except MoveItAlohaPlanningError as position_error:
                    if position_error.reason not in {
                        "bimanual_position_priority_lma_ik_failed",
                        "ompl_planning_failed",
                    }:
                        raise
                    continue
                executor.install(planned, phase=target.phase.value)
                command = executor.command(start, phase=target.phase.value)
                result = _moveit_reference_action(
                    current_state,
                    target,
                    contract=contract,
                    settings=settings,
                    initial_result=initial_result,
                    planned=planned,
                    command=command,
                    attempted=True,
                )
                return replace(
                    result,
                    path_planner_fraction=fraction,
                    path_planner_mode=(
                        f"{result.path_planner_mode}+cartesian_backoff"
                    ),
                )
        planning_error = error
        tolerance = float(
            settings["path_planner_start_bound_reconciliation_tolerance_rad"]
        )
        reconciliations = error.response.get("start_bound_reconciliations", [])
        violations = error.response.get("start_bound_violations", [])
        if not isinstance(reconciliations, list) or not isinstance(violations, list):
            raise MoveItAlohaPlannerError(
                "MoveIt start-bound failure evidence is not a list."
            ) from error

        def _joint_deltas(
            entries: list[Any],
            *,
            maximum_name: str,
        ) -> tuple[tuple[str, ...], float]:
            names: list[str] = []
            deltas: list[float] = []
            for item in entries:
                if not isinstance(item, dict):
                    raise MoveItAlohaPlannerError(
                        "MoveIt start-bound failure entry is not an object."
                    ) from planning_error
                name = item.get("joint_name")
                delta = float(item.get("delta_rad", math.nan))
                if (
                    not isinstance(name, str)
                    or name not in EXPECTED_JOINT_NAMES
                    or name in names
                    or not math.isfinite(delta)
                    or delta <= 0.0
                ):
                    raise MoveItAlohaPlannerError(
                        "MoveIt start-bound failure evidence is invalid."
                    ) from planning_error
                names.append(name)
                deltas.append(delta)
            maximum = float(planning_error.response.get(maximum_name, 0.0))
            if not math.isclose(
                maximum,
                max(deltas, default=0.0),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise MoveItAlohaPlannerError(
                    "MoveIt start-bound failure summary differs."
                ) from planning_error
            return tuple(names), maximum

        reconciliation_names, maximum_reconciliation = _joint_deltas(
            reconciliations,
            maximum_name="maximum_start_bound_reconciliation_rad",
        )
        violation_names, maximum_violation = _joint_deltas(
            violations,
            maximum_name="maximum_start_bound_violation_rad",
        )
        if maximum_reconciliation > tolerance + 1e-12:
            raise MoveItAlohaPlannerError(
                "MoveIt accepted a start-bound reconciliation beyond tolerance."
            ) from error
        return replace(
            initial_result,
            path_planner_attempted=True,
            path_planner_mode=(
                "moveit2_fix_start_state_path_constraints+ompl_rrtconnect"
            ),
            path_planner_initial_projected_error=(
                initial_result.maximum_projected_error
            ),
            solver_backend=(
                "moveit2_fix_start_state_path_constraints+ompl_lma_rrtconnect"
            ),
            solver_failure=error.reason,
            path_planner_start_bound_reconciliations=reconciliation_names,
            path_planner_maximum_start_bound_reconciliation_rad=(
                maximum_reconciliation
            ),
            path_planner_start_bound_violations=violation_names,
            path_planner_maximum_start_bound_violation_rad=maximum_violation,
            path_planner_joint_limit_margin_rad=float(
                settings["path_planner_joint_limit_margin_rad"]
            ),
            path_planner_physical_joint_limit_margin_rad=float(
                settings["path_planner_physical_joint_limit_margin_rad"]
            ),
            path_planner_ik_search_mode=(
                str(error.response["ik_search_mode"])
                if "ik_search_mode" in error.response
                else None
            ),
            path_planner_ik_candidate_selection_mode=(
                str(error.response["ik_candidate_selection_mode"])
                if "ik_candidate_selection_mode" in error.response
                else None
            ),
            path_planner_ik_seed=(
                int(error.response["ik_seed"])
                if "ik_seed" in error.response
                else None
            ),
            path_planner_ik_maximum_attempts=(
                int(error.response["ik_maximum_attempts"])
                if "ik_maximum_attempts" in error.response
                else None
            ),
            path_planner_ik_attempts_used=(
                int(error.response["ik_attempts_used"])
                if "ik_attempts_used" in error.response
                else None
            ),
            path_planner_valid_ik_candidate_count=(
                int(error.response["valid_ik_candidate_count"])
                if "valid_ik_candidate_count" in error.response
                else None
            ),
            path_planner_ik_outer_timeout_s=(
                float(error.response["ik_outer_timeout_s"])
                if "ik_outer_timeout_s" in error.response
                else None
            ),
            path_planner_minimum_start_joint_limit_margin_rad=(
                float(error.response["minimum_start_joint_limit_margin_rad"])
                if "minimum_start_joint_limit_margin_rad" in error.response
                else None
            ),
        )

    executor.install(planned, phase=target.phase.value)
    command = executor.command(start, phase=target.phase.value)
    return _moveit_reference_action(
        current_state,
        target,
        contract=contract,
        settings=settings,
        initial_result=initial_result,
        planned=planned,
        command=command,
        attempted=True,
    )


def _activate_moveit_terminal_control(
    scratch_physics: Any,
    current_state: Tensor,
    target: InsertionTaskSpaceTarget,
    *,
    contract: ActionContract,
    settings: dict[str, Any],
    planned: MoveItAlohaPlanResult,
    executor: MoveItAlohaTrajectoryExecutor,
) -> MujocoPositionFeedforwardResult:
    """Latch a static inverse-dynamics reference at the accepted final waypoint."""

    desired_action = current_state.detach().clone()
    desired_action[list(ARM_ACTION_INDICES)] = torch.as_tensor(
        planned.trajectory[-1],
        dtype=desired_action.dtype,
    )
    desired_action[6] = float(target.left_gripper)
    desired_action[13] = float(target.right_gripper)
    arm_dimensions = tuple(contract.dimensions[index] for index in ARM_ACTION_INDICES)
    desired_robot_qpos = _expanded_robot_qpos(desired_action)
    finger_lower = float(settings["path_planner_finger_lower_m"])
    finger_upper = float(settings["path_planner_finger_upper_m"])
    desired_robot_qpos[6] = min(
        finger_upper,
        max(finger_lower, desired_robot_qpos[6]),
    )
    desired_robot_qpos[7] = -desired_robot_qpos[6]
    desired_robot_qpos[14] = min(
        finger_upper,
        max(finger_lower, desired_robot_qpos[14]),
    )
    desired_robot_qpos[15] = -desired_robot_qpos[14]
    feedforward = static_position_feedforward(
        scratch_physics,
        desired_robot_qpos=desired_robot_qpos,
        arm_joint_names=(*LEFT_JOINTS, *RIGHT_JOINTS),
        joint_lower_rad=[dimension.minimum for dimension in arm_dimensions],
        joint_upper_rad=[dimension.maximum for dimension in arm_dimensions],
        joint_limit_margin_rad=float(
            settings["path_planner_terminal_control_joint_limit_margin_rad"]
        ),
        maximum_correction_rad=float(
            settings["path_planner_terminal_control_maximum_correction_rad"]
        ),
        neutral_reference_tolerance_rad=float(
            settings[
                "path_planner_terminal_control_neutral_reference_tolerance_rad"
            ]
        ),
    )
    executor.activate_terminal_control(
        feedforward.positions,
        phase=target.phase.value,
    )
    return feedforward


def _contact_phase_feedforward_action(
    scratch_physics: Any,
    desired_action: Tensor,
    *,
    contract: ActionContract,
    ik_settings: dict[str, Any],
    feedforward_settings: dict[str, Any],
) -> tuple[Tensor, MujocoPositionFeedforwardResult]:
    """Wrap a successful near-table command with bounded static feedforward."""

    desired_robot_qpos = _expanded_robot_qpos(desired_action)
    finger_lower = float(ik_settings["path_planner_finger_lower_m"])
    finger_upper = float(ik_settings["path_planner_finger_upper_m"])
    desired_robot_qpos[6] = min(
        finger_upper,
        max(finger_lower, desired_robot_qpos[6]),
    )
    desired_robot_qpos[7] = -desired_robot_qpos[6]
    desired_robot_qpos[14] = min(
        finger_upper,
        max(finger_lower, desired_robot_qpos[14]),
    )
    desired_robot_qpos[15] = -desired_robot_qpos[14]
    arm_dimensions = tuple(
        contract.dimensions[index] for index in ARM_ACTION_INDICES
    )
    feedforward = static_position_feedforward(
        scratch_physics,
        desired_robot_qpos=desired_robot_qpos,
        arm_joint_names=(*LEFT_JOINTS, *RIGHT_JOINTS),
        joint_lower_rad=[dimension.minimum for dimension in arm_dimensions],
        joint_upper_rad=[dimension.maximum for dimension in arm_dimensions],
        joint_limit_margin_rad=float(
            feedforward_settings["joint_limit_margin_rad"]
        ),
        maximum_correction_rad=float(
            feedforward_settings["maximum_correction_rad"]
        ),
        neutral_reference_tolerance_rad=float(
            feedforward_settings["neutral_reference_tolerance_rad"]
        ),
    )
    compensated = desired_action.detach().clone()
    compensated[list(ARM_ACTION_INDICES)] = torch.as_tensor(
        feedforward.positions,
        dtype=compensated.dtype,
    )
    return compensated, feedforward


def _ik_action(
    scratch_physics: Any,
    current_state: Tensor,
    target: InsertionTaskSpaceTarget,
    *,
    contract: ActionContract,
    settings: dict[str, Any],
    _allow_path_planner: bool = True,
    _path_planner_anchor: tuple[GeometryPose, GeometryPose] | None = None,
    _mink_solver: MinkAlohaIkSolver | None = None,
    _moveit_planner: MoveItAlohaPlanner | None = None,
    _moveit_executor: MoveItAlohaTrajectoryExecutor | None = None,
) -> IkActionResult:
    if settings.get("solver_backend") == "mink_qp":
        if _mink_solver is None:
            raise RuntimeError("Mink QP backend requires an initialized solver.")
        terminal_control_completed = False
        if _moveit_executor is not None:
            phase = target.phase.value
            if (
                _moveit_executor.plan_result is not None
                and not _moveit_executor.active_for(phase)
            ):
                _moveit_executor.reset()
            if (
                settings.get("path_planner_replan_on_terminal_completion") is True
                and _moveit_executor.terminal_control_active_for(phase)
            ):
                current = [
                    float(current_state[index]) for index in ARM_ACTION_INDICES
                ]
                terminal_control_completed = (
                    _moveit_executor.complete_terminal_control(
                        current,
                        phase=phase,
                        goal_l1_tolerance_rad=float(
                            settings[
                                "path_planner_terminal_completion_goal_l1_"
                                "tolerance_rad"
                            ]
                        ),
                    )
                )
            if _moveit_executor.active_for(phase):
                planned = _moveit_executor.plan_result
                if planned is None:
                    raise RuntimeError("MoveIt executor lost its retained plan.")
                current = [
                    float(current_state[index]) for index in ARM_ACTION_INDICES
                ]
                command = _moveit_executor.command(current, phase=phase)
                terminal_control: MujocoPositionFeedforwardResult | None = None
                if (
                    settings.get("path_planner_terminal_control_enabled") is True
                    and command.terminal_handoff_ready
                ):
                    initial_result = IkActionResult(
                        action=current_state.detach().cpu(),
                        success=False,
                        maximum_error=planned.maximum_goal_weighted_error,
                        maximum_projected_error=planned.maximum_goal_weighted_error,
                        joint_delta_saturations=0,
                        contract_clip_fields=(),
                        solver_backend="moveit2_retained_reference_trajectory",
                    )
                    try:
                        terminal_control = _activate_moveit_terminal_control(
                            scratch_physics,
                            current_state,
                            target,
                            contract=contract,
                            settings=settings,
                            planned=planned,
                            executor=_moveit_executor,
                        )
                    except ValueError as error:
                        failed = _moveit_reference_action(
                            current_state,
                            target,
                            contract=contract,
                            settings=settings,
                            initial_result=initial_result,
                            planned=planned,
                            command=command,
                            attempted=False,
                        )
                        return replace(
                            failed,
                            success=False,
                            solver_failure=(
                                "MuJoCo terminal position feedforward failed: "
                                f"{error}"
                            ),
                        )
                    command = _moveit_executor.command(current, phase=phase)
                return _moveit_reference_action(
                    current_state,
                    target,
                    contract=contract,
                    settings=settings,
                    initial_result=IkActionResult(
                        action=current_state.detach().cpu(),
                        success=False,
                        maximum_error=planned.maximum_goal_weighted_error,
                        maximum_projected_error=planned.maximum_goal_weighted_error,
                        joint_delta_saturations=0,
                        contract_clip_fields=(),
                        solver_backend="moveit2_retained_reference_trajectory",
                    ),
                    planned=planned,
                    command=command,
                    attempted=False,
                    terminal_control=terminal_control,
                )
        mink_result = _mink_ik_action(
            scratch_physics,
            current_state,
            target,
            contract=contract,
            settings=settings,
            solver=_mink_solver,
        )
        if terminal_control_completed:
            mink_result = replace(
                mink_result,
                path_planner_terminal_control_completed=True,
            )
        if (
            not mink_result.success
            and target.phase
            in (InsertionTeacherPhase.LIFT, InsertionTeacherPhase.COARSE_ALIGN, InsertionTeacherPhase.INSERT)
        ):
            scratch_physics.data.qpos[:16] = _expanded_robot_qpos(current_state)
            scratch_physics.forward()
            current_poses = (
                _site_pose(scratch_physics, LEFT_SITE),
                _site_pose(scratch_physics, RIGHT_SITE),
            )
            requested_poses = (target.left_eef, target.right_eef)
            for fraction_value in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                candidate_poses = tuple(
                    _full_pose_cartesian_waypoint(current, requested, fraction_value)
                    for current, requested in zip(current_poses, requested_poses)
                )
                candidate_target = replace(
                    target,
                    left_eef=candidate_poses[0],
                    right_eef=candidate_poses[1],
                )
                backoff_result = _mink_ik_action(
                    scratch_physics,
                    current_state,
                    candidate_target,
                    contract=contract,
                    settings=settings,
                    solver=_mink_solver,
                )
                if backoff_result.success:
                    return replace(
                        backoff_result,
                        solver_backend="mink_qp_daqp+lift_cartesian_backoff",
                        path_planner_fraction=fraction_value,
                    )
        if (
            mink_result.success
            or not _allow_path_planner
            or settings.get("path_planner_enabled") is not True
            or target.phase.value not in settings.get("path_planner_phases", ())
        ):
            return mink_result
        if settings.get("path_planner_backend") != "moveit2_ompl":
            raise RuntimeError("Mink QP fallback is not the registered MoveIt backend.")
        if _moveit_planner is None:
            raise RuntimeError("MoveIt fallback requires an initialized sidecar.")
        if _moveit_executor is None:
            raise RuntimeError("MoveIt fallback requires a retained-path executor.")
        path_result = _moveit_path_action(
            current_state,
            target,
            scratch_physics=scratch_physics,
            contract=contract,
            settings=settings,
            initial_result=mink_result,
            planner=_moveit_planner,
            executor=_moveit_executor,
        )
        if terminal_control_completed:
            path_result = replace(
                path_result,
                path_planner_terminal_control_completed=True,
            )
        return path_result

    from dm_control.utils.inverse_kinematics import qpos_from_site_pose

    scratch_physics.data.qpos[:16] = _expanded_robot_qpos(current_state)
    scratch_physics.forward()
    arm_specs = (
        (LEFT_SITE, LEFT_JOINTS, target.left_eef, tuple(range(6)), slice(0, 6)),
        (
            RIGHT_SITE,
            RIGHT_JOINTS,
            target.right_eef,
            tuple(range(7, 13)),
            slice(8, 14),
        )
    )

    def solve(spec: tuple[Any, ...], joint_names: tuple[str, ...]) -> Any:
        site_name, _, pose, _, _ = spec
        return qpos_from_site_pose(
            scratch_physics,
            site_name,
            target_pos=pose.position.numpy(),
            target_quat=pose.quaternion.numpy(),
            joint_names=joint_names,
            tol=float(settings["tolerance"]),
            rot_weight=float(settings["rotation_weight"]),
            regularization_threshold=float(settings["regularization_threshold"]),
            regularization_strength=float(settings["regularization_strength"]),
            max_update_norm=float(settings["maximum_update_norm"]),
            max_steps=int(settings["maximum_steps"]),
            inplace=False,
        )

    results = [solve(spec, spec[1]) for spec in arm_specs]
    raw = current_state.clone()
    raw[:6] = torch.as_tensor(results[0].qpos[:6], dtype=torch.float32)
    raw[7:13] = torch.as_tensor(results[1].qpos[8:14], dtype=torch.float32)
    raw[6] = float(target.left_gripper)
    raw[13] = float(target.right_gripper)

    maximum_delta = float(settings["maximum_joint_target_delta"])
    joint_delta_saturations = 0
    joint_index_groups = (tuple(range(6)), tuple(range(7, 13)))
    joint_indices = (*joint_index_groups[0], *joint_index_groups[1])

    def bound_joint_delta(candidate: Tensor) -> Tensor:
        nonlocal joint_delta_saturations
        bounded = candidate.clone()
        for indices in joint_index_groups:
            delta = bounded[list(indices)] - current_state[list(indices)]
            joint_delta_saturations += int(delta.abs().gt(maximum_delta).sum())
            bounded[list(indices)] = current_state[list(indices)] + delta.clamp(
                min=-maximum_delta,
                max=maximum_delta,
            )
        return bounded

    raw = bound_joint_delta(raw)
    action, clip_mask = contract.clip(raw)
    projection_mask = clip_mask.clone()

    # dm_control's generic pose solver explicitly ignores joint limits.  When
    # its unconstrained solution crosses the registered Action Contract, use
    # the clipped joints as an active set and re-solve with those DOFs fixed at
    # their physical bounds.  The final task-space residual remains subject to
    # the preregistered projected-pose threshold below.
    for _ in range(int(settings["joint_limit_active_set_retries"])):
        clipped_joints = bool(clip_mask[list(joint_indices)].any())
        if not clipped_joints:
            break
        scratch_physics.data.qpos[:16] = _expanded_robot_qpos(action)
        scratch_physics.forward()
        raw = action.clone()
        next_results = list(results)
        for arm_index, spec in enumerate(arm_specs):
            _, joint_names, _, logical_indices, qpos_indices = spec
            active_local = {
                local_index
                for local_index, logical_index in enumerate(logical_indices)
                if bool(projection_mask[logical_index])
            }
            if not active_local:
                continue
            remaining = tuple(
                name
                for local_index, name in enumerate(joint_names)
                if local_index not in active_local
            )
            if not remaining:
                continue
            result = solve(spec, remaining)
            next_results[arm_index] = result
            solved = torch.as_tensor(result.qpos[qpos_indices], dtype=torch.float32)
            for local_index, logical_index in enumerate(logical_indices):
                if local_index not in active_local:
                    raw[logical_index] = solved[local_index]
        raw[6] = float(target.left_gripper)
        raw[13] = float(target.right_gripper)
        raw = bound_joint_delta(raw)
        action, clip_mask = contract.clip(raw)
        projection_mask.logical_or_(clip_mask)
        results = next_results

    maximum_error = max(float(result.err_norm) for result in results)
    success = math.isfinite(maximum_error) and maximum_error <= float(
        settings["maximum_accepted_error"]
    )
    clip_fields = tuple(
        name
        for name, clipped in zip(contract.dimension_names, projection_mask.tolist())
        if clipped
    )
    scratch_physics.data.qpos[:16] = _expanded_robot_qpos(action)
    scratch_physics.forward()
    projected_errors: list[float] = []
    for site_name, target_pose in (
        (LEFT_SITE, target.left_eef),
        (RIGHT_SITE, target.right_eef),
    ):
        achieved = _site_pose(scratch_physics, site_name)
        position_error = float(
            torch.linalg.vector_norm(achieved.position - target_pose.position)
        )
        quaternion_dot = abs(float(torch.dot(achieved.quaternion, target_pose.quaternion)))
        quaternion_dot = min(1.0, max(-1.0, quaternion_dot))
        rotation_error = 2.0 * math.acos(quaternion_dot)
        projected_errors.append(
            position_error + float(settings["rotation_weight"]) * rotation_error
        )
    maximum_projected_error = max(projected_errors)
    success = success and maximum_projected_error <= float(
        settings["maximum_accepted_projected_error"]
    )
    result = IkActionResult(
        action=action.detach().cpu(),
        success=success,
        maximum_error=maximum_error,
        maximum_projected_error=maximum_projected_error,
        joint_delta_saturations=joint_delta_saturations,
        contract_clip_fields=clip_fields,
    )
    if (
        result.success
        or not _allow_path_planner
        or settings.get("path_planner_enabled") is not True
        or target.phase.value not in settings.get("path_planner_phases", ())
        or not result.contract_clip_fields
    ):
        return result
    if target.phase.value == "orient":
        if _path_planner_anchor is None:
            return result
        return _orientation_priority_path_action(
            scratch_physics,
            current_state,
            target,
            contract=contract,
            settings=settings,
            initial_result=result,
            anchor_poses=_path_planner_anchor,
        )
    return _position_priority_path_action(
        scratch_physics,
        current_state,
        target,
        contract=contract,
        settings=settings,
        initial_result=result,
    )


def _position_priority_path_action(
    scratch_physics: Any,
    current_state: Tensor,
    target: InsertionTaskSpaceTarget,
    *,
    contract: ActionContract,
    settings: dict[str, Any],
    initial_result: IkActionResult,
) -> IkActionResult:
    """Search precise joint-feasible approach waypoints without relaxing gates."""

    from dm_control.utils.inverse_kinematics import qpos_from_site_pose

    scratch_physics.data.qpos[:16] = _expanded_robot_qpos(current_state)
    scratch_physics.forward()
    current_poses = (
        _site_pose(scratch_physics, LEFT_SITE),
        _site_pose(scratch_physics, RIGHT_SITE),
    )
    arm_specs = (
        (LEFT_SITE, LEFT_JOINTS, tuple(range(6)), slice(0, 6)),
        (RIGHT_SITE, RIGHT_JOINTS, tuple(range(7, 13)), slice(8, 14)),
    )
    requested_poses = (target.left_eef, target.right_eef)
    projected_names = set(initial_result.contract_clip_fields)
    maximum_relaxation = float(
        settings["path_planner_maximum_orientation_relaxation_rad"]
    )
    minimum_progress = float(settings["path_planner_minimum_cartesian_progress_m"])

    for fraction_value in settings["path_planner_cartesian_backoff_fractions"]:
        fraction = float(fraction_value)
        candidate_poses = tuple(
            _cartesian_waypoint(
                current,
                requested,
                fraction,
                requested.quaternion,
            )
            for current, requested in zip(current_poses, requested_poses)
        )
        progress = max(
            float(torch.linalg.vector_norm(candidate.position - current.position))
            for current, candidate in zip(current_poses, candidate_poses)
        )
        if progress < minimum_progress:
            continue

        position_action = initial_result.action.clone()
        for arm_index, spec in enumerate(arm_specs):
            site_name, joint_names, logical_indices, qpos_indices = spec
            active_local = {
                local_index
                for local_index, logical_index in enumerate(logical_indices)
                if contract.dimension_names[logical_index] in projected_names
            }
            if not active_local:
                continue
            remaining = tuple(
                name
                for local_index, name in enumerate(joint_names)
                if local_index not in active_local
            )
            if not remaining:
                continue
            scratch_physics.data.qpos[:16] = _expanded_robot_qpos(position_action)
            scratch_physics.forward()
            position_result = qpos_from_site_pose(
                scratch_physics,
                site_name,
                target_pos=candidate_poses[arm_index].position.numpy(),
                target_quat=None,
                joint_names=remaining,
                tol=float(settings["tolerance"]),
                rot_weight=float(settings["rotation_weight"]),
                regularization_threshold=float(settings["regularization_threshold"]),
                regularization_strength=float(settings["regularization_strength"]),
                max_update_norm=float(settings["maximum_update_norm"]),
                max_steps=int(settings["maximum_steps"]),
                inplace=False,
            )
            solved = torch.as_tensor(
                position_result.qpos[qpos_indices],
                dtype=torch.float32,
            )
            for local_index, logical_index in enumerate(logical_indices):
                if local_index not in active_local:
                    position_action[logical_index] = solved[local_index]
            position_action, _ = contract.clip(position_action)

        scratch_physics.data.qpos[:16] = _expanded_robot_qpos(position_action)
        scratch_physics.forward()
        achieved_poses = (
            _site_pose(scratch_physics, LEFT_SITE),
            _site_pose(scratch_physics, RIGHT_SITE),
        )
        planned_poses: list[GeometryPose] = []
        relaxations: list[float] = []
        for requested, candidate, achieved, spec in zip(
            requested_poses,
            candidate_poses,
            achieved_poses,
            arm_specs,
        ):
            _, _, logical_indices, _ = spec
            arm_is_limited = any(
                contract.dimension_names[index] in projected_names
                for index in logical_indices
            )
            if arm_is_limited:
                quaternion, relaxation = _bounded_orientation_waypoint(
                    requested.quaternion,
                    achieved.quaternion,
                    maximum_relaxation,
                )
            else:
                quaternion = requested.quaternion
                relaxation = 0.0
            planned_poses.append(
                GeometryPose(position=candidate.position, quaternion=quaternion)
            )
            relaxations.append(relaxation)

        planned_target = replace(
            target,
            left_eef=planned_poses[0],
            right_eef=planned_poses[1],
        )
        planned_result = _ik_action(
            scratch_physics,
            current_state,
            planned_target,
            contract=contract,
            settings=settings,
            _allow_path_planner=False,
        )
        if planned_result.success:
            return replace(
                planned_result,
                contract_clip_fields=tuple(
                    sorted(
                        set(initial_result.contract_clip_fields)
                        | set(planned_result.contract_clip_fields)
                    )
                ),
                path_planner_attempted=True,
                path_planner_used=True,
                path_planner_mode="position_priority_orientation_relaxation",
                path_planner_fraction=fraction,
                path_planner_orientation_relaxation_rad=max(relaxations),
                path_planner_initial_projected_error=(
                    initial_result.maximum_projected_error
                ),
            )

    return replace(
        initial_result,
        path_planner_attempted=True,
        path_planner_initial_projected_error=initial_result.maximum_projected_error,
    )


def _orientation_priority_path_action(
    scratch_physics: Any,
    current_state: Tensor,
    target: InsertionTaskSpaceTarget,
    *,
    contract: ActionContract,
    settings: dict[str, Any],
    initial_result: IkActionResult,
    anchor_poses: tuple[GeometryPose, GeometryPose],
) -> IkActionResult:
    """Recover orientation on the feasible manifold while staying near approach."""

    from dm_control.utils.inverse_kinematics import qpos_from_site_pose

    scratch_physics.data.qpos[:16] = _expanded_robot_qpos(current_state)
    scratch_physics.forward()
    current_poses = (
        _site_pose(scratch_physics, LEFT_SITE),
        _site_pose(scratch_physics, RIGHT_SITE),
    )
    requested_poses = (target.left_eef, target.right_eef)
    arm_specs = (
        (LEFT_SITE, LEFT_JOINTS, tuple(range(6)), slice(0, 6)),
        (RIGHT_SITE, RIGHT_JOINTS, tuple(range(7, 13)), slice(8, 14)),
    )
    projected_names = set(initial_result.contract_clip_fields)
    maximum_position_relaxation = float(
        settings["path_planner_maximum_position_relaxation_m"]
    )
    maximum_cartesian_step = float(settings["path_planner_maximum_cartesian_step_m"])
    minimum_orientation_progress = float(
        settings["path_planner_minimum_orientation_progress_rad"]
    )

    for fraction_value in settings["path_planner_cartesian_backoff_fractions"]:
        fraction = float(fraction_value)
        candidate_quaternions: list[Tensor] = []
        orientation_progress: list[float] = []
        for current, requested in zip(current_poses, requested_poses):
            total = _quaternion_distance(current.quaternion, requested.quaternion)
            quaternion, progress = _bounded_orientation_waypoint(
                current.quaternion,
                requested.quaternion,
                total * fraction,
            )
            candidate_quaternions.append(quaternion)
            orientation_progress.append(progress)
        if max(orientation_progress) < minimum_orientation_progress:
            continue

        orientation_action = initial_result.action.clone()
        for arm_index, spec in enumerate(arm_specs):
            site_name, joint_names, logical_indices, qpos_indices = spec
            active_local = {
                local_index
                for local_index, logical_index in enumerate(logical_indices)
                if contract.dimension_names[logical_index] in projected_names
            }
            if not active_local:
                continue
            remaining = tuple(
                name
                for local_index, name in enumerate(joint_names)
                if local_index not in active_local
            )
            if not remaining:
                continue
            scratch_physics.data.qpos[:16] = _expanded_robot_qpos(orientation_action)
            scratch_physics.forward()
            orientation_result = qpos_from_site_pose(
                scratch_physics,
                site_name,
                target_pos=anchor_poses[arm_index].position.numpy(),
                target_quat=candidate_quaternions[arm_index].numpy(),
                joint_names=remaining,
                tol=float(settings["tolerance"]),
                rot_weight=float(
                    settings["path_planner_orientation_rotation_weight"]
                ),
                regularization_threshold=float(settings["regularization_threshold"]),
                regularization_strength=float(settings["regularization_strength"]),
                max_update_norm=float(settings["maximum_update_norm"]),
                max_steps=int(settings["maximum_steps"]),
                inplace=False,
            )
            solved = torch.as_tensor(
                orientation_result.qpos[qpos_indices],
                dtype=torch.float32,
            )
            for local_index, logical_index in enumerate(logical_indices):
                if local_index not in active_local:
                    orientation_action[logical_index] = solved[local_index]
            orientation_action, _ = contract.clip(orientation_action)

        scratch_physics.data.qpos[:16] = _expanded_robot_qpos(orientation_action)
        scratch_physics.forward()
        achieved_poses = (
            _site_pose(scratch_physics, LEFT_SITE),
            _site_pose(scratch_physics, RIGHT_SITE),
        )
        planned_poses: list[GeometryPose] = []
        feasible = True
        for current, achieved, quaternion, anchor, spec in zip(
            current_poses,
            achieved_poses,
            candidate_quaternions,
            anchor_poses,
            arm_specs,
        ):
            _, _, logical_indices, _ = spec
            arm_is_limited = any(
                contract.dimension_names[index] in projected_names
                for index in logical_indices
            )
            if arm_is_limited:
                anchor_drift = float(
                    torch.linalg.vector_norm(achieved.position - anchor.position)
                )
                current_step = float(
                    torch.linalg.vector_norm(achieved.position - current.position)
                )
                if (
                    anchor_drift > maximum_position_relaxation
                    or current_step > maximum_cartesian_step
                ):
                    feasible = False
                    break
                position = achieved.position
            else:
                position = requested.position
            planned_poses.append(GeometryPose(position=position, quaternion=quaternion))
        if not feasible:
            continue

        planned_target = replace(
            target,
            left_eef=planned_poses[0],
            right_eef=planned_poses[1],
        )
        planned_result = _ik_action(
            scratch_physics,
            current_state,
            planned_target,
            contract=contract,
            settings=settings,
            _allow_path_planner=False,
        )
        if planned_result.success:
            return replace(
                planned_result,
                contract_clip_fields=tuple(
                    sorted(
                        set(initial_result.contract_clip_fields)
                        | set(planned_result.contract_clip_fields)
                    )
                ),
                path_planner_attempted=True,
                path_planner_used=True,
                path_planner_mode="orientation_priority_position_relaxation",
                path_planner_fraction=fraction,
                path_planner_orientation_relaxation_rad=max(orientation_progress),
                path_planner_initial_projected_error=(
                    initial_result.maximum_projected_error
                ),
            )

    return replace(
        initial_result,
        path_planner_attempted=True,
        path_planner_mode="orientation_priority_position_relaxation",
        path_planner_initial_projected_error=initial_result.maximum_projected_error,
    )


def _pose_payload(value: GeometryPose) -> dict[str, list[float]]:
    return {
        "position": value.position.tolist(),
        "quaternion": value.quaternion.tolist(),
    }


def _geometry_payload(value: InsertionGeometry) -> dict[str, Any]:
    return {
        "socket": _pose_payload(value.socket),
        "peg": _pose_payload(value.peg),
        "left_eef": _pose_payload(value.left_eef),
        "right_eef": _pose_payload(value.right_eef),
        "socket_grasp_contact": value.socket_grasp_contact,
        "peg_grasp_contact": value.peg_grasp_contact,
        "socket_on_table": value.socket_on_table,
        "peg_on_table": value.peg_on_table,
        "peg_socket_contact": value.peg_socket_contact,
        "pin_contact": value.pin_contact,
        "unexpected_collision_count": value.unexpected_collision_count,
    }


def _evaluate_seed(
    *,
    seed: int,
    calibration: InsertionTeacherCalibration,
    contract: ActionContract,
    plan: dict[str, Any],
    moveit_planner: MoveItAlohaPlanner | None = None,
) -> dict[str, Any]:
    evaluation = _mapping(plan["evaluation"], "evaluation")
    teacher_settings = InsertionTeacherSettings(
        **_mapping(plan["teacher"], "teacher")
    )
    ik_settings = _mapping(plan["inverse_kinematics"], "inverse_kinematics")
    execution_diagnostic_settings = _mapping(
        plan["execution_diagnostics"],
        "execution_diagnostics",
    )
    execution_guard_settings = _mapping(
        plan["execution_guard"],
        "execution_guard",
    )
    task_contact_policy = _mapping(
        plan["task_contact_policy"],
        "task_contact_policy",
    )
    lift_contact_exemption = plan.get("lift_contact_exemption")
    lift_contact_exemption_pairs = frozenset()
    if lift_contact_exemption is not None:
        lift_contact_exemption = _mapping(
            lift_contact_exemption,
            "lift_contact_exemption",
        )
        lift_contact_exemption_phases = frozenset(
            str(value) for value in lift_contact_exemption["phases"]
        )
        lift_contact_exemption_pairs = frozenset(
            frozenset(str(name) for name in pair)
            for pair in lift_contact_exemption[
                "allowed_unordered_geom_pairs"
            ]
        )
    else:
        lift_contact_exemption_phases = frozenset()
    contact_phase_feedforward_settings = _mapping(
        plan["contact_phase_feedforward"],
        "contact_phase_feedforward",
    )
    contact_phase_feedforward_phases = frozenset(
        str(value) for value in contact_phase_feedforward_settings["phases"]
    )
    task_contact_phases = frozenset(str(value) for value in task_contact_policy["phases"])
    task_contact_phases = task_contact_phases | lift_contact_exemption_phases
    task_contact_pairs = frozenset(
        frozenset(str(name) for name in pair)
        for pair in task_contact_policy["allowed_unordered_geom_pairs"]
    )
    task_contact_pairs = task_contact_pairs | lift_contact_exemption_pairs
    execution_joint_margin = float(
        execution_diagnostic_settings["joint_limit_margin_rad"]
    )
    maximum_steps = int(evaluation["maximum_steps"])
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    scratch = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    mink_solver: MinkAlohaIkSolver | None = None
    moveit_executor: MoveItAlohaTrajectoryExecutor | None = None
    if ik_settings.get("solver_backend") == "mink_qp":
        mink_solver = MinkAlohaIkSolver.from_gym_aloha(
            left_site=LEFT_SITE,
            right_site=RIGHT_SITE,
            settings=MinkAlohaIkSettings(
                integration_timestep_s=float(
                    ik_settings["mink_integration_timestep_s"]
                ),
                maximum_iterations=int(ik_settings["mink_maximum_iterations"]),
                position_cost=float(ik_settings["mink_position_cost"]),
                orientation_cost=float(ik_settings["mink_orientation_cost"]),
                posture_cost=float(ik_settings["mink_posture_cost"]),
                frame_lm_damping=float(ik_settings["mink_frame_lm_damping"]),
                solver_damping=float(ik_settings["mink_solver_damping"]),
                maximum_joint_velocity_rad_s=float(
                    ik_settings["mink_maximum_joint_velocity_rad_s"]
                ),
                configuration_limit_gain=float(
                    ik_settings["mink_configuration_limit_gain"]
                ),
                joint_limit_margin_rad=float(
                    ik_settings.get("mink_joint_limit_margin_rad", 0.0)
                ),
            ),
        )
    if moveit_planner is not None:
        moveit_executor = MoveItAlohaTrajectoryExecutor(
            waypoint_l1_tolerance_rad=float(
                ik_settings["path_planner_waypoint_l1_tolerance_rad"]
            ),
            maximum_joint_step_rad=float(
                ik_settings["maximum_joint_target_delta"]
            ),
        )
    teacher = ObjectGeometryInsertionTeacher(calibration, teacher_settings)
    trace: list[dict[str, Any]] = []
    execution_trace: list[dict[str, Any]] = []
    phase_visits: dict[str, int] = {}
    maximum_reward = 0.0
    last_reward = 0.0
    success = False
    teacher_failure: str | None = None
    ik_failures = 0
    orientation_anchor: tuple[GeometryPose, GeometryPose] | None = None
    maximum_ik_error = 0.0
    maximum_projected_ik_error = 0.0
    joint_delta_saturations = 0
    joint_limit_projection_events = 0
    path_planner_attempts = 0
    path_planner_waypoints = 0
    path_planner_trust_region_margin_restoration_events = 0
    path_planner_trust_region_orientation_progress_events = 0
    path_planner_trust_region_feedback_basis_events = 0
    path_planner_trust_region_orientation_first_events = 0
    path_planner_trust_region_constraint_anchored_restoration_events = 0
    path_planner_trust_region_expanded_orientation_target_budget_events = 0
    lift_feedback_anchor_commands = 0
    path_planner_trust_region_candidates_evaluated = 0
    path_planner_trust_region_valid_candidates = 0
    maximum_path_planner_trust_region_requested_position_relaxation_m = 0.0
    maximum_path_planner_trust_region_orientation_target_rad = 0.0
    minimum_path_planner_trust_region_margin_improvement_rad = math.inf
    path_planner_total_planning_time_s = 0.0
    maximum_path_planner_waypoint_count = 0
    maximum_path_planner_path_length_rad = 0.0
    maximum_path_planner_goal_position_error_m = 0.0
    maximum_path_planner_goal_orientation_error_rad = 0.0
    maximum_path_planner_goal_weighted_error = 0.0
    maximum_path_planner_ik_attempts_used = 0
    maximum_path_planner_orientation_relaxation_rad = 0.0
    minimum_path_planner_path_joint_limit_margin_rad = math.inf
    minimum_path_planner_constrained_path_joint_limit_margin_rad = math.inf
    minimum_path_planner_adapter_prefix_physical_joint_limit_margin_rad = math.inf
    minimum_path_planner_next_joint_limit_margin_rad = math.inf
    path_planner_start_state_recovery_events = 0
    maximum_path_planner_adapter_prefix_waypoint_count = 0
    minimum_path_planner_recovery_progress_rad = math.inf
    path_planner_start_bound_reconciliation_events = 0
    maximum_path_planner_start_bound_reconciliation_rad = 0.0
    path_planner_start_bound_violation_events = 0
    maximum_path_planner_start_bound_violation_rad = 0.0
    path_planner_reference_reuse_commands = 0
    path_planner_reference_waypoint_advancements = 0
    maximum_path_planner_reference_waypoint_index = 0
    path_planner_terminal_control_commands = 0
    path_planner_terminal_control_activations = 0
    path_planner_terminal_control_completions = 0
    task_contact_exemption_events = 0
    task_contact_exemption_steps = 0
    lift_contact_exemption_events = 0
    lift_gripper_bar_contact_exemption_events = 0
    lift_moveit_fallback_events = 0
    contact_phase_feedforward_commands = 0
    contact_phase_feedforward_failures = 0
    contact_phase_feedforward_failure: str | None = None
    maximum_contact_phase_feedforward_correction_rad = 0.0
    minimum_contact_phase_feedforward_command_margin_rad = math.inf
    maximum_path_planner_terminal_control_correction_rad = 0.0
    minimum_path_planner_terminal_control_command_margin_rad = math.inf
    maximum_solver_iterations = 0
    contract_clip_fields: set[str] = set()
    adapter_clip_failures = 0
    minimum_pre_step_joint_limit_margin_rad = math.inf
    minimum_commanded_joint_limit_margin_rad = math.inf
    minimum_observed_post_step_joint_limit_margin_rad = math.inf
    maximum_command_to_observed_joint_error_rad = 0.0
    maximum_tracking_overshoot_toward_limit_rad = 0.0
    maximum_margin_loss_command_to_observation_rad = 0.0
    commanded_margin_breach_events = 0
    observed_margin_breach_events = 0
    first_commanded_margin_breach_step: int | None = None
    first_observed_margin_breach_step: int | None = None
    previous_phase: str | None = None
    try:
        observation = dict(environment.reset(seed=seed))
        scratch.reset(seed=0)
        initial_geometry = _current_geometry(
            environment,
            observation,
            observed_reward=0.0,
        )
        for step in range(maximum_steps):
            active_task_contact_pairs = (
                task_contact_pairs
                if teacher.phase.value in task_contact_phases
                else frozenset()
            )
            _, task_contact_exemptions = _collision_classification(
                environment,
                active_task_contact_pairs,
            )
            task_contact_exemption_events += len(task_contact_exemptions)
            task_contact_exemption_steps += int(bool(task_contact_exemptions))
            lift_contact_exemption_events += int(
                teacher.phase is InsertionTeacherPhase.LIFT
                and bool(task_contact_exemptions)
            )
            lift_gripper_bar_contact_exemption_events += int(
                teacher.phase is InsertionTeacherPhase.LIFT
                and any(
                    frozenset(pair)
                    == frozenset(
                        {"table", "vx300s_right/9_gripper_bar"}
                    )
                    for pair in task_contact_exemptions
                )
            )
            geometry = _current_geometry(
                environment,
                observation,
                observed_reward=last_reward,
                allowed_task_contact_pairs=active_task_contact_pairs,
            )
            try:
                target = teacher.decide(geometry)
            except GeometryTeacherError as error:
                teacher_failure = str(error)
                trace.append(
                    {
                        "step": step,
                        "event": "geometry_teacher_failure",
                        "error": teacher_failure,
                        "geometry": _geometry_payload(geometry),
                    }
                )
                break
            phase = target.phase.value
            phase_visits[phase] = phase_visits.get(phase, 0) + 1
            lift_feedback_anchor_commands += int(target.lift_feedback_anchor)
            if (
                target.phase is InsertionTeacherPhase.ORIENT
                and orientation_anchor is None
            ):
                orientation_anchor = (target.left_eef, target.right_eef)
            ik = _ik_action(
                _physics(scratch),
                geometry.robot_state,
                target,
                contract=contract,
                settings=ik_settings,
                _path_planner_anchor=orientation_anchor,
                _mink_solver=mink_solver,
                _moveit_planner=moveit_planner,
                _moveit_executor=moveit_executor,
            )
            maximum_solver_iterations = max(
                maximum_solver_iterations,
                ik.solver_iterations or 0,
            )
            maximum_ik_error = max(maximum_ik_error, ik.maximum_error)
            maximum_projected_ik_error = max(
                maximum_projected_ik_error,
                ik.maximum_projected_error,
            )
            joint_delta_saturations += ik.joint_delta_saturations
            contract_clip_fields.update(ik.contract_clip_fields)
            joint_limit_projection_events += int(bool(ik.contract_clip_fields))
            path_planner_attempts += int(ik.path_planner_attempted)
            path_planner_waypoints += int(ik.path_planner_used)
            lift_moveit_fallback_events += int(
                target.phase is InsertionTeacherPhase.LIFT
                and ik.path_planner_attempted
            )
            path_planner_trust_region_margin_restoration_events += int(
                ik.path_planner_trust_region_mode == "margin_restoration"
            )
            path_planner_trust_region_orientation_progress_events += int(
                ik.path_planner_trust_region_mode == "orientation_progress"
            )
            path_planner_trust_region_feedback_basis_events += int(
                ik.path_planner_trust_region_basis
                == "feedback_aligned_orthonormal_v1"
            )
            path_planner_trust_region_orientation_first_events += int(
                ik.path_planner_trust_region_selection_policy
                == "orientation_progress_first_v1"
                and ik.path_planner_trust_region_mode == "orientation_progress"
            )
            path_planner_trust_region_constraint_anchored_restoration_events += int(
                ik.path_planner_trust_region_restoration_reference
                == "command_margin_boundary"
                and ik.path_planner_trust_region_mode == "margin_restoration"
            )
            if ik.path_planner_trust_region_orientation_target_rad is not None:
                maximum_path_planner_trust_region_orientation_target_rad = max(
                    maximum_path_planner_trust_region_orientation_target_rad,
                    ik.path_planner_trust_region_orientation_target_rad,
                )
                path_planner_trust_region_expanded_orientation_target_budget_events += int(
                    ik.path_planner_trust_region_mode == "orientation_progress"
                    and ik.path_planner_trust_region_orientation_target_rad
                    > float(
                        ik_settings[
                            "path_planner_active_set_trust_region_previous_"
                            "orientation_target_budget_rad"
                        ]
                    )
                    + 1e-12
                )
            path_planner_trust_region_candidates_evaluated += (
                ik.path_planner_trust_region_candidates_evaluated
            )
            path_planner_trust_region_valid_candidates += (
                ik.path_planner_trust_region_valid_candidates
            )
            maximum_path_planner_trust_region_requested_position_relaxation_m = max(
                maximum_path_planner_trust_region_requested_position_relaxation_m,
                ik.path_planner_trust_region_requested_position_relaxation_m or 0.0,
            )
            if ik.path_planner_trust_region_margin_improvement_rad is not None:
                minimum_path_planner_trust_region_margin_improvement_rad = min(
                    minimum_path_planner_trust_region_margin_improvement_rad,
                    ik.path_planner_trust_region_margin_improvement_rad,
                )
            path_planner_total_planning_time_s += ik.path_planner_planning_time_s or 0.0
            maximum_path_planner_waypoint_count = max(
                maximum_path_planner_waypoint_count,
                ik.path_planner_waypoint_count or 0,
            )
            maximum_path_planner_path_length_rad = max(
                maximum_path_planner_path_length_rad,
                ik.path_planner_path_length_rad or 0.0,
            )
            maximum_path_planner_goal_position_error_m = max(
                maximum_path_planner_goal_position_error_m,
                ik.path_planner_goal_position_error_m or 0.0,
            )
            maximum_path_planner_goal_orientation_error_rad = max(
                maximum_path_planner_goal_orientation_error_rad,
                ik.path_planner_goal_orientation_error_rad or 0.0,
            )
            maximum_path_planner_goal_weighted_error = max(
                maximum_path_planner_goal_weighted_error,
                ik.path_planner_goal_weighted_error or 0.0,
            )
            maximum_path_planner_ik_attempts_used = max(
                maximum_path_planner_ik_attempts_used,
                ik.path_planner_ik_attempts_used or 0,
            )
            maximum_path_planner_orientation_relaxation_rad = max(
                maximum_path_planner_orientation_relaxation_rad,
                ik.path_planner_orientation_relaxation_rad,
            )
            if ik.path_planner_minimum_path_joint_limit_margin_rad is not None:
                minimum_path_planner_path_joint_limit_margin_rad = min(
                    minimum_path_planner_path_joint_limit_margin_rad,
                    ik.path_planner_minimum_path_joint_limit_margin_rad,
                )
            if (
                ik.path_planner_minimum_constrained_path_joint_limit_margin_rad
                is not None
            ):
                minimum_path_planner_constrained_path_joint_limit_margin_rad = min(
                    minimum_path_planner_constrained_path_joint_limit_margin_rad,
                    ik.path_planner_minimum_constrained_path_joint_limit_margin_rad,
                )
            if ik.path_planner_minimum_next_joint_limit_margin_rad is not None:
                minimum_path_planner_next_joint_limit_margin_rad = min(
                    minimum_path_planner_next_joint_limit_margin_rad,
                    ik.path_planner_minimum_next_joint_limit_margin_rad,
                )
            if (
                ik.path_planner_start_state_path_constraint_recovery
                and ik.path_planner_attempted
            ):
                path_planner_start_state_recovery_events += 1
                maximum_path_planner_adapter_prefix_waypoint_count = max(
                    maximum_path_planner_adapter_prefix_waypoint_count,
                    ik.path_planner_adapter_prefix_waypoint_count,
                )
                if ik.path_planner_minimum_recovery_progress_rad is not None:
                    minimum_path_planner_recovery_progress_rad = min(
                        minimum_path_planner_recovery_progress_rad,
                        ik.path_planner_minimum_recovery_progress_rad,
                    )
                if (
                    ik.path_planner_minimum_adapter_prefix_physical_joint_limit_margin_rad
                    is not None
                ):
                    minimum_path_planner_adapter_prefix_physical_joint_limit_margin_rad = (
                        min(
                            minimum_path_planner_adapter_prefix_physical_joint_limit_margin_rad,
                            ik.path_planner_minimum_adapter_prefix_physical_joint_limit_margin_rad,
                        )
                    )
            if ik.path_planner_attempted:
                path_planner_start_bound_reconciliation_events += len(
                    ik.path_planner_start_bound_reconciliations
                )
                maximum_path_planner_start_bound_reconciliation_rad = max(
                    maximum_path_planner_start_bound_reconciliation_rad,
                    ik.path_planner_maximum_start_bound_reconciliation_rad,
                )
                path_planner_start_bound_violation_events += len(
                    ik.path_planner_start_bound_violations
                )
                maximum_path_planner_start_bound_violation_rad = max(
                    maximum_path_planner_start_bound_violation_rad,
                    ik.path_planner_maximum_start_bound_violation_rad,
                )
            path_planner_reference_reuse_commands += int(
                ik.path_planner_reference_reused
            )
            path_planner_reference_waypoint_advancements += int(
                ik.path_planner_reference_waypoint_advanced
            )
            maximum_path_planner_reference_waypoint_index = max(
                maximum_path_planner_reference_waypoint_index,
                ik.path_planner_reference_waypoint_index or 0,
            )
            path_planner_terminal_control_commands += int(
                ik.path_planner_terminal_control_active
            )
            path_planner_terminal_control_activations += int(
                ik.path_planner_terminal_control_activated
            )
            path_planner_terminal_control_completions += int(
                ik.path_planner_terminal_control_completed
            )
            maximum_path_planner_terminal_control_correction_rad = max(
                maximum_path_planner_terminal_control_correction_rad,
                ik.path_planner_terminal_control_maximum_correction_rad or 0.0,
            )
            if (
                ik.path_planner_terminal_control_minimum_command_margin_rad
                is not None
            ):
                minimum_path_planner_terminal_control_command_margin_rad = min(
                    minimum_path_planner_terminal_control_command_margin_rad,
                    ik.path_planner_terminal_control_minimum_command_margin_rad,
                )
            if not ik.success:
                ik_failures += 1
                pre_step_margin = _arm_joint_margin_snapshot(
                    geometry.robot_state,
                    contract,
                    registered_margin_rad=execution_joint_margin,
                )
                minimum_pre_step_joint_limit_margin_rad = min(
                    minimum_pre_step_joint_limit_margin_rad,
                    float(pre_step_margin["minimum_margin_rad"]),
                )
                trace.append(
                    {
                        "step": step,
                        "event": "inverse_kinematics_failure",
                        "phase": phase,
                        "ik_error": ik.maximum_error,
                        "projected_ik_error": ik.maximum_projected_error,
                        "contract_clip_fields": list(ik.contract_clip_fields),
                        "path_planner_attempted": ik.path_planner_attempted,
                        "path_planner_used": ik.path_planner_used,
                        "path_planner_mode": ik.path_planner_mode,
                        "path_planner_trust_region_mode": (
                            ik.path_planner_trust_region_mode
                        ),
                        "path_planner_trust_region_basis": (
                            ik.path_planner_trust_region_basis
                        ),
                        "path_planner_trust_region_selection_policy": (
                            ik.path_planner_trust_region_selection_policy
                        ),
                        "path_planner_trust_region_restoration_reference": (
                            ik.path_planner_trust_region_restoration_reference
                        ),
                        "path_planner_trust_region_active_arm": (
                            ik.path_planner_trust_region_active_arm
                        ),
                        "path_planner_trust_region_radius_m": (
                            ik.path_planner_trust_region_radius_m
                        ),
                        "path_planner_trust_region_direction": (
                            ik.path_planner_trust_region_direction
                        ),
                        "path_planner_trust_region_orientation_fraction": (
                            ik.path_planner_trust_region_orientation_fraction
                        ),
                        "path_planner_trust_region_orientation_target_rad": (
                            ik.path_planner_trust_region_orientation_target_rad
                        ),
                        "path_planner_trust_region_margin_improvement_rad": (
                            ik.path_planner_trust_region_margin_improvement_rad
                        ),
                        "path_planner_trust_region_requested_position_relaxation_m": (
                            ik.path_planner_trust_region_requested_position_relaxation_m
                        ),
                        "path_planner_initial_projected_error": (
                            ik.path_planner_initial_projected_error
                        ),
                        "path_planner_planning_time_s": (
                            ik.path_planner_planning_time_s
                        ),
                        "path_planner_waypoint_count": (
                            ik.path_planner_waypoint_count
                        ),
                        "path_planner_path_length_rad": (
                            ik.path_planner_path_length_rad
                        ),
                        "path_planner_goal_position_error_m": (
                            ik.path_planner_goal_position_error_m
                        ),
                        "path_planner_goal_orientation_error_rad": (
                            ik.path_planner_goal_orientation_error_rad
                        ),
                        "path_planner_goal_weighted_error": (
                            ik.path_planner_goal_weighted_error
                        ),
                        "path_planner_ik_search_mode": (
                            ik.path_planner_ik_search_mode
                        ),
                        "path_planner_ik_candidate_selection_mode": (
                            ik.path_planner_ik_candidate_selection_mode
                        ),
                        "path_planner_ik_seed": ik.path_planner_ik_seed,
                        "path_planner_ik_maximum_attempts": (
                            ik.path_planner_ik_maximum_attempts
                        ),
                        "path_planner_ik_attempts_used": (
                            ik.path_planner_ik_attempts_used
                        ),
                        "path_planner_valid_ik_candidate_count": (
                            ik.path_planner_valid_ik_candidate_count
                        ),
                        "path_planner_selected_ik_attempt": (
                            ik.path_planner_selected_ik_attempt
                        ),
                        "path_planner_selected_ik_minimum_joint_limit_margin_rad": (
                            ik.path_planner_selected_ik_minimum_joint_limit_margin_rad
                        ),
                        "path_planner_selected_ik_maximum_start_delta_rad": (
                            ik.path_planner_selected_ik_maximum_start_delta_rad
                        ),
                        "path_planner_ik_outer_timeout_s": (
                            ik.path_planner_ik_outer_timeout_s
                        ),
                        "path_planner_joint_limit_margin_rad": (
                            ik.path_planner_joint_limit_margin_rad
                        ),
                        "path_planner_physical_joint_limit_margin_rad": (
                            ik.path_planner_physical_joint_limit_margin_rad
                        ),
                        "path_planner_start_state_path_constraint_recovery": (
                            ik.path_planner_start_state_path_constraint_recovery
                        ),
                        "path_planner_adapter_prefix_waypoint_count": (
                            ik.path_planner_adapter_prefix_waypoint_count
                        ),
                        "path_planner_minimum_recovery_progress_rad": (
                            ik.path_planner_minimum_recovery_progress_rad
                        ),
                        "path_planner_minimum_start_joint_limit_margin_rad": (
                            ik.path_planner_minimum_start_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_goal_joint_limit_margin_rad": (
                            ik.path_planner_minimum_goal_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_path_joint_limit_margin_rad": (
                            ik.path_planner_minimum_path_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_constrained_path_joint_limit_margin_rad": (
                            ik.path_planner_minimum_constrained_path_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_adapter_prefix_physical_joint_limit_margin_rad": (
                            ik.path_planner_minimum_adapter_prefix_physical_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_next_joint_limit_margin_rad": (
                            ik.path_planner_minimum_next_joint_limit_margin_rad
                        ),
                        "path_planner_start_bound_reconciliations": list(
                            ik.path_planner_start_bound_reconciliations
                        ),
                        "path_planner_maximum_start_bound_reconciliation_rad": (
                            ik.path_planner_maximum_start_bound_reconciliation_rad
                        ),
                        "path_planner_start_bound_violations": list(
                            ik.path_planner_start_bound_violations
                        ),
                        "path_planner_maximum_start_bound_violation_rad": (
                            ik.path_planner_maximum_start_bound_violation_rad
                        ),
                        "path_planner_reference_reused": (
                            ik.path_planner_reference_reused
                        ),
                        "path_planner_reference_waypoint_index": (
                            ik.path_planner_reference_waypoint_index
                        ),
                        "path_planner_reference_waypoint_advanced": (
                            ik.path_planner_reference_waypoint_advanced
                        ),
                        "path_planner_reference_waypoint_l1_distance_rad": (
                            ik.path_planner_reference_waypoint_l1_distance_rad
                        ),
                        "path_planner_terminal_control_active": (
                            ik.path_planner_terminal_control_active
                        ),
                        "path_planner_terminal_control_activated": (
                            ik.path_planner_terminal_control_activated
                        ),
                        "path_planner_terminal_control_completed": (
                            ik.path_planner_terminal_control_completed
                        ),
                        "path_planner_terminal_control_maximum_correction_rad": (
                            ik.path_planner_terminal_control_maximum_correction_rad
                        ),
                        "path_planner_terminal_control_minimum_command_margin_rad": (
                            ik.path_planner_terminal_control_minimum_command_margin_rad
                        ),
                        "solver_backend": ik.solver_backend,
                        "solver_iterations": ik.solver_iterations,
                        "solver_failure": ik.solver_failure,
                        "pre_step_joint_margin": pre_step_margin,
                    }
                )
                execution_trace.append(
                    {
                        "step": step,
                        "phase": phase,
                        "event": "command_not_executed",
                        "source": ik.solver_backend,
                        "solver_failure": ik.solver_failure,
                        "pre_step": pre_step_margin,
                    }
                )
                break
            executed_action = ik.action
            contact_feedforward: MujocoPositionFeedforwardResult | None = None
            if phase in contact_phase_feedforward_phases:
                try:
                    executed_action, contact_feedforward = (
                        _contact_phase_feedforward_action(
                            _physics(scratch),
                            ik.action,
                            contract=contract,
                            ik_settings=ik_settings,
                            feedforward_settings=contact_phase_feedforward_settings,
                        )
                    )
                except ValueError as error:
                    if phase == "coarse_align":
                        executed_action = ik.action
                        contact_feedforward = None
                        continue
                    contact_phase_feedforward_failures += 1
                    contact_phase_feedforward_failure = str(error)
                    pre_step_margin = _arm_joint_margin_snapshot(
                        geometry.robot_state,
                        contract,
                        registered_margin_rad=execution_joint_margin,
                    )
                    minimum_pre_step_joint_limit_margin_rad = min(
                        minimum_pre_step_joint_limit_margin_rad,
                        float(pre_step_margin["minimum_margin_rad"]),
                    )
                    trace.append(
                        {
                            "step": step,
                            "event": "contact_phase_feedforward_failure",
                            "phase": phase,
                            "error": contact_phase_feedforward_failure,
                            "pre_step_joint_margin": pre_step_margin,
                        }
                    )
                    execution_trace.append(
                        {
                            "step": step,
                            "phase": phase,
                            "event": "command_not_executed",
                            "source": contact_phase_feedforward_settings["backend"],
                            "solver_failure": contact_phase_feedforward_failure,
                            "pre_step": pre_step_margin,
                        }
                    )
                    break
                contact_phase_feedforward_commands += 1
                maximum_contact_phase_feedforward_correction_rad = max(
                    maximum_contact_phase_feedforward_correction_rad,
                    contact_feedforward.maximum_correction_rad,
                )
                minimum_contact_phase_feedforward_command_margin_rad = min(
                    minimum_contact_phase_feedforward_command_margin_rad,
                    contact_feedforward.minimum_command_joint_limit_margin_rad,
                )
            pre_step_state = geometry.robot_state.detach().clone()
            applied_action, adapter_clip_mask = contract.clip(executed_action)
            observation_value, reward, done, info = environment.step(executed_action)
            observation = dict(observation_value)
            adapter_clip_failures += int(bool(environment.last_clip_mask.any()))
            observed_post_step_state = _robot_state(observation, contract.dimension)
            execution_diagnostic = _joint_execution_diagnostic(
                pre_step_state,
                applied_action,
                observed_post_step_state,
                contract,
                registered_margin_rad=execution_joint_margin,
                source=(
                    ik.solver_backend
                    + (
                        "+mujoco_static_inverse_dynamics_position_feedforward"
                        if contact_feedforward is not None
                        else ""
                    )
                ),
            )
            execution_diagnostic.update(
                {
                    "step": step,
                    "phase": phase,
                    "adapter_clip_fields": [
                        name
                        for name, clipped in zip(
                            contract.dimension_names,
                            adapter_clip_mask.tolist(),
                        )
                        if clipped
                    ],
                    "contact_phase_feedforward_active": (
                        contact_feedforward is not None
                    ),
                    "contact_phase_feedforward_maximum_correction_rad": (
                        None
                        if contact_feedforward is None
                        else contact_feedforward.maximum_correction_rad
                    ),
                    "contact_phase_feedforward_minimum_command_margin_rad": (
                        None
                        if contact_feedforward is None
                        else contact_feedforward.minimum_command_joint_limit_margin_rad
                    ),
                }
            )
            execution_trace.append(execution_diagnostic)
            pre_step_margin = float(
                execution_diagnostic["pre_step"]["minimum_margin_rad"]
            )
            commanded_margin = float(
                execution_diagnostic["commanded"]["minimum_margin_rad"]
            )
            observed_margin = float(
                execution_diagnostic["observed_post_step"]["minimum_margin_rad"]
            )
            minimum_pre_step_joint_limit_margin_rad = min(
                minimum_pre_step_joint_limit_margin_rad,
                pre_step_margin,
            )
            minimum_commanded_joint_limit_margin_rad = min(
                minimum_commanded_joint_limit_margin_rad,
                commanded_margin,
            )
            minimum_observed_post_step_joint_limit_margin_rad = min(
                minimum_observed_post_step_joint_limit_margin_rad,
                observed_margin,
            )
            maximum_command_to_observed_joint_error_rad = max(
                maximum_command_to_observed_joint_error_rad,
                float(execution_diagnostic["maximum_absolute_tracking_error_rad"]),
            )
            maximum_tracking_overshoot_toward_limit_rad = max(
                maximum_tracking_overshoot_toward_limit_rad,
                float(
                    execution_diagnostic[
                        "maximum_overshoot_toward_commanded_bound_rad"
                    ]
                ),
            )
            maximum_margin_loss_command_to_observation_rad = max(
                maximum_margin_loss_command_to_observation_rad,
                float(
                    execution_diagnostic[
                        "maximum_margin_loss_command_to_observation_rad"
                    ]
                ),
            )
            if bool(execution_diagnostic["commanded"]["inside_registered_margin"]):
                commanded_margin_breach_events += 1
                if first_commanded_margin_breach_step is None:
                    first_commanded_margin_breach_step = step
            if bool(
                execution_diagnostic["observed_post_step"]["inside_registered_margin"]
            ):
                observed_margin_breach_events += 1
                if first_observed_margin_breach_step is None:
                    first_observed_margin_breach_step = step
            last_reward = float(reward)
            maximum_reward = max(maximum_reward, last_reward)
            phase_changed = phase != previous_phase
            if (
                phase_changed
                or step < 5
                or step % 25 == 0
                or reward != 0.0
                or done
                or ik.path_planner_attempted
                or ik.path_planner_reference_waypoint_advanced
                or ik.path_planner_terminal_control_activated
                or ik.path_planner_terminal_control_completed
                or task_contact_exemptions
            ):
                trace.append(
                    {
                        "step": step,
                        "phase": phase,
                        "phase_changed": phase_changed,
                        "target_position_error_m": target.maximum_position_error_m,
                        "best_observed_reward": target.best_observed_reward,
                        "reward_after_step": last_reward,
                        "ik_error": ik.maximum_error,
                        "projected_ik_error": ik.maximum_projected_error,
                        "joint_delta_saturations": ik.joint_delta_saturations,
                        "joint_limit_projection_fields": list(ik.contract_clip_fields),
                        "path_planner_attempted": ik.path_planner_attempted,
                        "path_planner_used": ik.path_planner_used,
                        "path_planner_mode": ik.path_planner_mode,
                        "path_planner_trust_region_mode": (
                            ik.path_planner_trust_region_mode
                        ),
                        "path_planner_trust_region_basis": (
                            ik.path_planner_trust_region_basis
                        ),
                        "path_planner_trust_region_selection_policy": (
                            ik.path_planner_trust_region_selection_policy
                        ),
                        "path_planner_trust_region_restoration_reference": (
                            ik.path_planner_trust_region_restoration_reference
                        ),
                        "path_planner_trust_region_active_arm": (
                            ik.path_planner_trust_region_active_arm
                        ),
                        "path_planner_trust_region_radius_m": (
                            ik.path_planner_trust_region_radius_m
                        ),
                        "path_planner_trust_region_direction": (
                            ik.path_planner_trust_region_direction
                        ),
                        "path_planner_trust_region_orientation_fraction": (
                            ik.path_planner_trust_region_orientation_fraction
                        ),
                        "path_planner_trust_region_orientation_target_rad": (
                            ik.path_planner_trust_region_orientation_target_rad
                        ),
                        "path_planner_trust_region_margin_improvement_rad": (
                            ik.path_planner_trust_region_margin_improvement_rad
                        ),
                        "path_planner_trust_region_requested_position_relaxation_m": (
                            ik.path_planner_trust_region_requested_position_relaxation_m
                        ),
                        "path_planner_fraction": ik.path_planner_fraction,
                        "path_planner_orientation_relaxation_rad": (
                            ik.path_planner_orientation_relaxation_rad
                        ),
                        "path_planner_initial_projected_error": (
                            ik.path_planner_initial_projected_error
                        ),
                        "path_planner_planning_time_s": (
                            ik.path_planner_planning_time_s
                        ),
                        "path_planner_waypoint_count": (
                            ik.path_planner_waypoint_count
                        ),
                        "path_planner_path_length_rad": (
                            ik.path_planner_path_length_rad
                        ),
                        "path_planner_goal_position_error_m": (
                            ik.path_planner_goal_position_error_m
                        ),
                        "path_planner_goal_orientation_error_rad": (
                            ik.path_planner_goal_orientation_error_rad
                        ),
                        "path_planner_goal_weighted_error": (
                            ik.path_planner_goal_weighted_error
                        ),
                        "path_planner_ik_search_mode": (
                            ik.path_planner_ik_search_mode
                        ),
                        "path_planner_ik_candidate_selection_mode": (
                            ik.path_planner_ik_candidate_selection_mode
                        ),
                        "path_planner_ik_seed": ik.path_planner_ik_seed,
                        "path_planner_ik_maximum_attempts": (
                            ik.path_planner_ik_maximum_attempts
                        ),
                        "path_planner_ik_attempts_used": (
                            ik.path_planner_ik_attempts_used
                        ),
                        "path_planner_valid_ik_candidate_count": (
                            ik.path_planner_valid_ik_candidate_count
                        ),
                        "path_planner_selected_ik_attempt": (
                            ik.path_planner_selected_ik_attempt
                        ),
                        "path_planner_selected_ik_minimum_joint_limit_margin_rad": (
                            ik.path_planner_selected_ik_minimum_joint_limit_margin_rad
                        ),
                        "path_planner_selected_ik_maximum_start_delta_rad": (
                            ik.path_planner_selected_ik_maximum_start_delta_rad
                        ),
                        "path_planner_ik_outer_timeout_s": (
                            ik.path_planner_ik_outer_timeout_s
                        ),
                        "path_planner_joint_limit_margin_rad": (
                            ik.path_planner_joint_limit_margin_rad
                        ),
                        "path_planner_physical_joint_limit_margin_rad": (
                            ik.path_planner_physical_joint_limit_margin_rad
                        ),
                        "path_planner_start_state_path_constraint_recovery": (
                            ik.path_planner_start_state_path_constraint_recovery
                        ),
                        "path_planner_adapter_prefix_waypoint_count": (
                            ik.path_planner_adapter_prefix_waypoint_count
                        ),
                        "path_planner_minimum_recovery_progress_rad": (
                            ik.path_planner_minimum_recovery_progress_rad
                        ),
                        "path_planner_minimum_start_joint_limit_margin_rad": (
                            ik.path_planner_minimum_start_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_goal_joint_limit_margin_rad": (
                            ik.path_planner_minimum_goal_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_path_joint_limit_margin_rad": (
                            ik.path_planner_minimum_path_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_constrained_path_joint_limit_margin_rad": (
                            ik.path_planner_minimum_constrained_path_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_adapter_prefix_physical_joint_limit_margin_rad": (
                            ik.path_planner_minimum_adapter_prefix_physical_joint_limit_margin_rad
                        ),
                        "path_planner_minimum_next_joint_limit_margin_rad": (
                            ik.path_planner_minimum_next_joint_limit_margin_rad
                        ),
                        "path_planner_start_bound_reconciliations": list(
                            ik.path_planner_start_bound_reconciliations
                        ),
                        "path_planner_maximum_start_bound_reconciliation_rad": (
                            ik.path_planner_maximum_start_bound_reconciliation_rad
                        ),
                        "path_planner_start_bound_violations": list(
                            ik.path_planner_start_bound_violations
                        ),
                        "path_planner_maximum_start_bound_violation_rad": (
                            ik.path_planner_maximum_start_bound_violation_rad
                        ),
                        "path_planner_reference_reused": (
                            ik.path_planner_reference_reused
                        ),
                        "path_planner_reference_waypoint_index": (
                            ik.path_planner_reference_waypoint_index
                        ),
                        "path_planner_reference_waypoint_advanced": (
                            ik.path_planner_reference_waypoint_advanced
                        ),
                        "path_planner_reference_waypoint_l1_distance_rad": (
                            ik.path_planner_reference_waypoint_l1_distance_rad
                        ),
                        "path_planner_terminal_control_active": (
                            ik.path_planner_terminal_control_active
                        ),
                        "path_planner_terminal_control_activated": (
                            ik.path_planner_terminal_control_activated
                        ),
                        "path_planner_terminal_control_completed": (
                            ik.path_planner_terminal_control_completed
                        ),
                        "task_contact_exemptions": [
                            list(pair) for pair in task_contact_exemptions
                        ],
                        "contact_phase_feedforward_active": (
                            contact_feedforward is not None
                        ),
                        "contact_phase_feedforward_maximum_correction_rad": (
                            None
                            if contact_feedforward is None
                            else contact_feedforward.maximum_correction_rad
                        ),
                        "contact_phase_feedforward_minimum_command_margin_rad": (
                            None
                            if contact_feedforward is None
                            else contact_feedforward.minimum_command_joint_limit_margin_rad
                        ),
                        "path_planner_terminal_control_maximum_correction_rad": (
                            ik.path_planner_terminal_control_maximum_correction_rad
                        ),
                        "path_planner_terminal_control_minimum_command_margin_rad": (
                            ik.path_planner_terminal_control_minimum_command_margin_rad
                        ),
                        "solver_backend": ik.solver_backend,
                        "solver_iterations": ik.solver_iterations,
                        "solver_failure": ik.solver_failure,
                        "geometry_before_step": _geometry_payload(geometry),
                    }
                )
            previous_phase = phase
            if done:
                success = bool(info.get("is_success", False))
                break
        final_geometry = _current_geometry(
            environment,
            observation,
            observed_reward=last_reward,
            allowed_task_contact_pairs=(
                task_contact_pairs
                if teacher.phase.value in task_contact_phases
                else frozenset()
            ),
        )
        return {
            "seed": seed,
            "status": "passed" if success else "failed",
            "success": success,
            "maximum_reward": maximum_reward,
            "steps_executed": step + 1,
            "final_phase": teacher.phase.value,
            "phase_visits": phase_visits,
            "teacher_failure": teacher_failure,
            "inverse_kinematics_failures": ik_failures,
            "maximum_inverse_kinematics_error": maximum_ik_error,
            "maximum_projected_inverse_kinematics_error": maximum_projected_ik_error,
            "joint_delta_saturations": joint_delta_saturations,
            "joint_limit_projection_events": joint_limit_projection_events,
            "path_planner_attempts": path_planner_attempts,
            "path_planner_waypoints": path_planner_waypoints,
            "path_planner_trust_region_margin_restoration_events": (
                path_planner_trust_region_margin_restoration_events
            ),
            "path_planner_trust_region_orientation_progress_events": (
                path_planner_trust_region_orientation_progress_events
            ),
            "path_planner_trust_region_feedback_basis_events": (
                path_planner_trust_region_feedback_basis_events
            ),
            "path_planner_trust_region_orientation_first_events": (
                path_planner_trust_region_orientation_first_events
            ),
            "path_planner_trust_region_constraint_anchored_restoration_events": (
                path_planner_trust_region_constraint_anchored_restoration_events
            ),
            "path_planner_trust_region_expanded_orientation_target_budget_events": (
                path_planner_trust_region_expanded_orientation_target_budget_events
            ),
            "lift_feedback_anchor_commands": lift_feedback_anchor_commands,
            "path_planner_trust_region_candidates_evaluated": (
                path_planner_trust_region_candidates_evaluated
            ),
            "path_planner_trust_region_valid_candidates": (
                path_planner_trust_region_valid_candidates
            ),
            "maximum_path_planner_trust_region_requested_position_relaxation_m": (
                maximum_path_planner_trust_region_requested_position_relaxation_m
            ),
            "maximum_path_planner_trust_region_orientation_target_rad": (
                maximum_path_planner_trust_region_orientation_target_rad
            ),
            "minimum_path_planner_trust_region_margin_improvement_rad": (
                None
                if math.isinf(minimum_path_planner_trust_region_margin_improvement_rad)
                else minimum_path_planner_trust_region_margin_improvement_rad
            ),
            "path_planner_total_planning_time_s": (
                path_planner_total_planning_time_s
            ),
            "maximum_path_planner_waypoint_count": (
                maximum_path_planner_waypoint_count
            ),
            "maximum_path_planner_path_length_rad": (
                maximum_path_planner_path_length_rad
            ),
            "maximum_path_planner_goal_position_error_m": (
                maximum_path_planner_goal_position_error_m
            ),
            "maximum_path_planner_goal_orientation_error_rad": (
                maximum_path_planner_goal_orientation_error_rad
            ),
            "maximum_path_planner_goal_weighted_error": (
                maximum_path_planner_goal_weighted_error
            ),
            "maximum_path_planner_ik_attempts_used": (
                maximum_path_planner_ik_attempts_used
            ),
            "maximum_path_planner_orientation_relaxation_rad": (
                maximum_path_planner_orientation_relaxation_rad
            ),
            "path_planner_joint_limit_margin_rad": float(
                ik_settings.get("path_planner_joint_limit_margin_rad", 0.0)
            ),
            "path_planner_physical_joint_limit_margin_rad": float(
                ik_settings.get("path_planner_physical_joint_limit_margin_rad", 0.0)
            ),
            "path_planner_start_state_recovery_events": (
                path_planner_start_state_recovery_events
            ),
            "maximum_path_planner_adapter_prefix_waypoint_count": (
                maximum_path_planner_adapter_prefix_waypoint_count
            ),
            "minimum_path_planner_recovery_progress_rad": (
                None
                if math.isinf(minimum_path_planner_recovery_progress_rad)
                else minimum_path_planner_recovery_progress_rad
            ),
            "minimum_path_planner_path_joint_limit_margin_rad": (
                None
                if math.isinf(minimum_path_planner_path_joint_limit_margin_rad)
                else minimum_path_planner_path_joint_limit_margin_rad
            ),
            "minimum_path_planner_next_joint_limit_margin_rad": (
                None
                if math.isinf(minimum_path_planner_next_joint_limit_margin_rad)
                else minimum_path_planner_next_joint_limit_margin_rad
            ),
            "minimum_path_planner_constrained_path_joint_limit_margin_rad": (
                None
                if math.isinf(
                    minimum_path_planner_constrained_path_joint_limit_margin_rad
                )
                else minimum_path_planner_constrained_path_joint_limit_margin_rad
            ),
            "minimum_path_planner_adapter_prefix_physical_joint_limit_margin_rad": (
                None
                if math.isinf(
                    minimum_path_planner_adapter_prefix_physical_joint_limit_margin_rad
                )
                else minimum_path_planner_adapter_prefix_physical_joint_limit_margin_rad
            ),
            "path_planner_start_bound_reconciliation_events": (
                path_planner_start_bound_reconciliation_events
            ),
            "maximum_path_planner_start_bound_reconciliation_rad": (
                maximum_path_planner_start_bound_reconciliation_rad
            ),
            "path_planner_start_bound_violation_events": (
                path_planner_start_bound_violation_events
            ),
            "maximum_path_planner_start_bound_violation_rad": (
                maximum_path_planner_start_bound_violation_rad
            ),
            "path_planner_reference_reuse_commands": (
                path_planner_reference_reuse_commands
            ),
            "path_planner_reference_waypoint_advancements": (
                path_planner_reference_waypoint_advancements
            ),
            "maximum_path_planner_reference_waypoint_index": (
                maximum_path_planner_reference_waypoint_index
            ),
            "path_planner_terminal_control_commands": (
                path_planner_terminal_control_commands
            ),
            "path_planner_terminal_control_activations": (
                path_planner_terminal_control_activations
            ),
            "path_planner_terminal_control_completions": (
                path_planner_terminal_control_completions
            ),
            "task_contact_exemption_events": task_contact_exemption_events,
            "task_contact_exemption_steps": task_contact_exemption_steps,
            "lift_contact_exemption_events": lift_contact_exemption_events,
            "lift_gripper_bar_contact_exemption_events": (
                lift_gripper_bar_contact_exemption_events
            ),
            "lift_moveit_fallback_events": lift_moveit_fallback_events,
            "contact_phase_feedforward_commands": (
                contact_phase_feedforward_commands
            ),
            "contact_phase_feedforward_failures": (
                contact_phase_feedforward_failures
            ),
            "contact_phase_feedforward_failure": contact_phase_feedforward_failure,
            "maximum_contact_phase_feedforward_correction_rad": (
                maximum_contact_phase_feedforward_correction_rad
            ),
            "minimum_contact_phase_feedforward_command_margin_rad": (
                None
                if math.isinf(
                    minimum_contact_phase_feedforward_command_margin_rad
                )
                else minimum_contact_phase_feedforward_command_margin_rad
            ),
            "maximum_path_planner_terminal_control_correction_rad": (
                maximum_path_planner_terminal_control_correction_rad
            ),
            "minimum_path_planner_terminal_control_command_margin_rad": (
                None
                if math.isinf(
                    minimum_path_planner_terminal_control_command_margin_rad
                )
                else minimum_path_planner_terminal_control_command_margin_rad
            ),
            "solver_backend": (
                "mink_qp_daqp+moveit2_fix_start_state_path_constraints+"
                "ompl_lma_rrtconnect"
                + (
                    "+mujoco_static_inverse_dynamics_position_feedforward"
                    if ik_settings.get("path_planner_terminal_control_enabled") is True
                    else ""
                )
                if moveit_planner is not None
                else "mink_qp_daqp"
                if mink_solver is not None
                else "dm_control_legacy"
            ),
            "maximum_solver_iterations": maximum_solver_iterations,
            "contract_clip_fields": sorted(contract_clip_fields),
            "adapter_clip_failures": adapter_clip_failures,
            "execution_diagnostic_schema": execution_diagnostic_settings["schema"],
            "execution_diagnostic_joint_limit_margin_rad": execution_joint_margin,
            "execution_guard_schema": execution_guard_settings["schema"],
            "execution_guard_tracking_reserve_rad": float(
                execution_guard_settings["tracking_reserve_rad"]
            ),
            "execution_guard_command_joint_limit_margin_rad": float(
                execution_guard_settings["command_joint_limit_margin_rad"]
            ),
            "minimum_pre_step_joint_limit_margin_rad": (
                None
                if math.isinf(minimum_pre_step_joint_limit_margin_rad)
                else minimum_pre_step_joint_limit_margin_rad
            ),
            "minimum_commanded_joint_limit_margin_rad": (
                None
                if math.isinf(minimum_commanded_joint_limit_margin_rad)
                else minimum_commanded_joint_limit_margin_rad
            ),
            "minimum_observed_post_step_joint_limit_margin_rad": (
                None
                if math.isinf(minimum_observed_post_step_joint_limit_margin_rad)
                else minimum_observed_post_step_joint_limit_margin_rad
            ),
            "maximum_command_to_observed_joint_error_rad": (
                maximum_command_to_observed_joint_error_rad
            ),
            "maximum_tracking_overshoot_toward_limit_rad": (
                maximum_tracking_overshoot_toward_limit_rad
            ),
            "maximum_margin_loss_command_to_observation_rad": (
                maximum_margin_loss_command_to_observation_rad
            ),
            "commanded_margin_breach_events": commanded_margin_breach_events,
            "observed_margin_breach_events": observed_margin_breach_events,
            "first_commanded_margin_breach_step": first_commanded_margin_breach_step,
            "first_observed_margin_breach_step": first_observed_margin_breach_step,
            "orientation_anchor": (
                None
                if orientation_anchor is None
                else [_pose_payload(pose) for pose in orientation_anchor]
            ),
            "initial_geometry": _geometry_payload(initial_geometry),
            "final_geometry": _geometry_payload(final_geometry),
            "trace": trace,
            "execution_trace": execution_trace,
        }
    finally:
        environment.close()
        scratch.close()


def _calibration_payload(value: CalibrationReplay) -> dict[str, Any]:
    calibration = value.calibration
    return {
        "source_episode": calibration.source_episode,
        "source_seed": calibration.source_seed,
        "steps_executed": value.steps_executed,
        "first_grasp_step": value.first_grasp_step,
        "terminal_step": value.terminal_step,
        "maximum_reward": value.maximum_reward,
        "socket_to_left_eef_at_grasp": _pose_payload(
            calibration.socket_to_left_eef_at_grasp
        ),
        "peg_to_right_eef_at_grasp": _pose_payload(
            calibration.peg_to_right_eef_at_grasp
        ),
        "terminal_socket_to_peg": _pose_payload(
            calibration.terminal_socket_to_peg
        ),
        "insertion_axis_in_socket": calibration.insertion_axis_in_socket.tolist(),
    }


def _stage_report_path(plan_path: Path, plan: dict[str, Any], stage: str) -> Path:
    output = _mapping(plan["output"], "output")
    if output.get("reports_are_scoped_by_plan_sha256_and_stage") is not True:
        raise ValueError("Geometry-teacher reports must be scoped by plan and stage.")
    return (
        _run_root()
        / str(output["run_directory"])
        / file_sha256(plan_path)[:16]
        / f"{stage}.json"
    )


def _start_moveit_planner(
    plan_path: Path,
    plan: dict[str, Any],
    stage: str,
) -> tuple[MoveItAlohaPlanner, dict[str, Any]]:
    inverse_kinematics = _mapping(plan["inverse_kinematics"], "inverse_kinematics")
    runtime = _mapping(
        inverse_kinematics["path_planner_runtime"],
        "path_planner_runtime",
    )
    source_image = os.environ.get("ROSETTA_ALOHA_MOVEIT_SOURCE_IMAGE")
    source_image_id = os.environ.get("ROSETTA_ALOHA_MOVEIT_SOURCE_IMAGE_ID")
    if source_image != runtime["image"] or source_image_id != runtime["image_id"]:
        raise OSError("MoveIt source image identity differs from the registered plan.")
    manifest_raw = os.environ.get("ROSETTA_ALOHA_MOVEIT_DESCRIPTION_MANIFEST")
    if not manifest_raw:
        raise OSError(
            "ROSETTA_ALOHA_MOVEIT_DESCRIPTION_MANIFEST must identify the pinned manifest."
        )
    manifest_path = Path(manifest_raw)
    resource_manifest_variables = {
        "mesh_manifest_sha256": "ROSETTA_ALOHA_MOVEIT_MESH_MANIFEST",
        "urdf_source_manifest_sha256": (
            "ROSETTA_ALOHA_MOVEIT_URDF_SOURCE_MANIFEST"
        ),
    }
    resource_manifests: dict[str, Path] = {}
    for artifact_name, variable in resource_manifest_variables.items():
        raw = os.environ.get(variable)
        if not raw:
            raise OSError(f"{variable} must identify a pinned resource manifest.")
        resource_manifests[artifact_name] = Path(raw)

    def _verify_resource_manifest(path: Path, root: Path) -> int:
        entries = 0
        resolved_root = root.resolve(strict=True)
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"MoveIt resource manifest line is invalid: {path}.")
            expected, raw_resource = fields
            if (
                len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
            ):
                raise ValueError(f"MoveIt resource digest is invalid: {path}.")
            resource = Path(raw_resource.lstrip(" *")).resolve(strict=True)
            try:
                resource.relative_to(resolved_root)
            except ValueError as error:
                raise ValueError(
                    f"MoveIt resource exits its registered root: {resource}."
                ) from error
            if file_sha256(resource) != expected:
                raise ValueError(f"MoveIt resource identity differs: {resource}.")
            entries += 1
        if entries == 0:
            raise ValueError(f"MoveIt resource manifest is empty: {path}.")
        return entries

    resource_manifest_entries = {
        "mesh": _verify_resource_manifest(
            resource_manifests["mesh_manifest_sha256"],
            Path("/opt/ros/humble/share/interbotix_xsarm_descriptions/meshes"),
        ),
        "urdf_source": _verify_resource_manifest(
            resource_manifests["urdf_source_manifest_sha256"],
            Path("/opt/ros/humble/share/interbotix_xsarm_descriptions/urdf"),
        ),
    }
    stderr_log = _stage_report_path(plan_path, plan, stage).with_suffix(
        ".moveit.stderr.log"
    )
    settings = MoveItAlohaPlannerSettings.from_environment(
        stderr_log=stderr_log,
        ompl_seed=int(inverse_kinematics["path_planner_ompl_seed"]),
        response_timeout_s=float(
            inverse_kinematics["path_planner_response_timeout_s"]
        ),
    )
    artifacts = {
        "executable_sha256": file_sha256(settings.executable),
        "urdf_sha256": file_sha256(settings.urdf),
        "srdf_sha256": file_sha256(settings.srdf),
        "description_manifest_sha256": file_sha256(manifest_path),
        **{
            name: file_sha256(path)
            for name, path in resource_manifests.items()
        },
    }
    for name, observed in artifacts.items():
        if observed != runtime[name]:
            raise ValueError(f"MoveIt runtime artifact identity differs for {name}.")
    planner = MoveItAlohaPlanner(settings)
    identity = dict(planner.identity)
    identity.pop("request_id", None)
    for plan_name, identity_name in (
        (
            "path_planner_ik_group_selection_mode",
            "ik_group_selection_mode",
        ),
        (
            "path_planner_full_pose_groups",
            "full_pose_groups",
        ),
        (
            "path_planner_collision_geometry_link_count",
            "collision_geometry_link_count",
        ),
        (
            "path_planner_collision_geometry_shape_count",
            "collision_geometry_shape_count",
        ),
        (
            "path_planner_position_priority_groups",
            "position_priority_groups",
        ),
        (
            "path_planner_position_priority_orientation_weight",
            "position_priority_orientation_weight",
        ),
        (
            "path_planner_position_priority_ompl_seed_reset_per_request",
            "position_priority_ompl_seed_reset_per_request",
        ),
        (
            "path_planner_position_priority_terminal_goal_normalization_limit_rad",
            "position_priority_terminal_goal_normalization_limit_rad",
        ),
        (
            "path_planner_ik_candidate_selection_mode",
            "ik_candidate_selection_mode",
        ),
    ):
        if identity.get(identity_name) != inverse_kinematics[plan_name]:
            planner.close()
            raise ValueError(f"MoveIt runtime identity differs for {identity_name}.")
    if identity.get("planning_request_adapters") != inverse_kinematics.get(
        "path_planner_request_adapters"
    ):
        planner.close()
        raise ValueError("MoveIt runtime planning-request adapter identity differs.")
    return planner, {
        "source_image": source_image,
        "source_image_id": source_image_id,
        "nested_docker_used": False,
        "artifacts": artifacts,
        "resource_manifest_entries": resource_manifest_entries,
        "identity": identity,
        "stderr_log": stderr_log.name,
    }


def _require_prior_stage(plan_path: Path, plan: dict[str, Any], stage: str) -> None:
    path = _stage_report_path(plan_path, plan, stage)
    if not path.is_file():
        raise FileNotFoundError(
            f"Geometry-teacher {stage} evidence must pass before this stage: {path}."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("plan_sha256") != file_sha256(plan_path)
        or report.get("stage") != stage
        or report.get("status") != "diagnostic_passed"
    ):
        raise ValueError(f"Geometry-teacher {stage} evidence is not accepted.")


def _main(plan_path: Path, *, stage: str) -> int:
    plan = _load_plan(plan_path)
    _validate_plan_boundaries(plan)
    if stage == "tuning":
        _require_prior_stage(plan_path, plan, "exact")
    elif stage == "full":
        _require_prior_stage(plan_path, plan, "exact")
        _require_prior_stage(plan_path, plan, "tuning")

    scope = _mapping(plan["scope"], "scope")
    dataset_config_path = _repository_path(str(scope["dataset_config"]))
    action_contract_path = _repository_path(str(scope["action_contract"]))
    dataset_config = load_dataset_config(dataset_config_path)
    contract = load_action_contract(action_contract_path)
    if dataset_config.revision != scope["dataset_revision"]:
        raise ValueError("Geometry-teacher dataset revision differs from its config.")
    dataset_root, dataset_manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    if dataset_manifest.resolved_revision != scope["dataset_revision"]:
        raise ValueError("Prepared cache revision differs from the geometry-teacher plan.")

    calibration_plan = _mapping(plan["calibration"], "calibration")
    calibration_episode = int(calibration_plan["source_episode"])
    calibration_seed = int(calibration_plan["simulator_seed"])
    rows = _trajectory_rows(dataset_root, calibration_episode, dataset_config, contract)
    calibration = _calibrate_from_replay(
        rows,
        episode=calibration_episode,
        seed=calibration_seed,
        insertion_axis=torch.as_tensor(
            calibration_plan["insertion_axis_in_socket"],
            dtype=torch.float32,
        ),
        dataset_config=dataset_config,
        contract=contract,
    )
    print(
        "calibration "
        f"episode={calibration_episode} seed={calibration_seed} "
        f"reward=4 steps={calibration.steps_executed}",
        flush=True,
    )

    evaluation = _mapping(plan["evaluation"], "evaluation")
    exact_reports: list[dict[str, Any]] = []
    tuning_reports: list[dict[str, Any]] = []
    development_reports: list[dict[str, Any]] = []
    inverse_kinematics = _mapping(plan["inverse_kinematics"], "inverse_kinematics")
    moveit_planner: MoveItAlohaPlanner | None = None
    moveit_runtime: dict[str, Any] | None = None
    if inverse_kinematics.get("path_planner_backend") == "moveit2_ompl":
        moveit_planner, moveit_runtime = _start_moveit_planner(
            plan_path,
            plan,
            stage,
        )
    try:
        if stage == "exact":
            exact = _mapping(evaluation["exact_control"], "exact_control")
            seed = int(exact["simulator_seed"])
            print(f"evaluating geometry-teacher exact seed={seed}", flush=True)
            exact_reports.append(
                _evaluate_seed(
                    seed=seed,
                    calibration=calibration.calibration,
                    contract=contract,
                    plan=plan,
                    moveit_planner=moveit_planner,
                )
            )
        elif stage == "tuning":
            for seed_value in evaluation["tuning_simulator_seeds"]:
                seed = int(seed_value)
                print(f"evaluating geometry-teacher tuning seed={seed}", flush=True)
                tuning_reports.append(
                    _evaluate_seed(
                        seed=seed,
                        calibration=calibration.calibration,
                        contract=contract,
                        plan=plan,
                        moveit_planner=moveit_planner,
                    )
                )
        else:
            for seed_value in evaluation["development_simulator_seeds"]:
                seed = int(seed_value)
                print(f"evaluating geometry-teacher development seed={seed}", flush=True)
                development_reports.append(
                    _evaluate_seed(
                        seed=seed,
                        calibration=calibration.calibration,
                        contract=contract,
                        plan=plan,
                        moveit_planner=moveit_planner,
                    )
                )
    finally:
        if moveit_planner is not None:
            moveit_planner.close()
    if moveit_runtime is not None:
        moveit_stderr = _stage_report_path(plan_path, plan, stage).with_suffix(
            ".moveit.stderr.log"
        )
        moveit_runtime["stderr_log_sha256"] = file_sha256(moveit_stderr)

    reports = [*exact_reports, *tuning_reports, *development_reports]
    acceptance = _mapping(plan["acceptance"], "acceptance")
    successes = sum(int(report["success"]) for report in reports)
    teacher_failures = sum(int(report["teacher_failure"] is not None) for report in reports)
    ik_failures = sum(int(report["inverse_kinematics_failures"]) for report in reports)
    adapter_clip_failures = sum(int(report["adapter_clip_failures"]) for report in reports)
    joint_limit_projection_events = sum(
        int(report["joint_limit_projection_events"]) for report in reports
    )
    path_planner_attempts = sum(int(report["path_planner_attempts"]) for report in reports)
    path_planner_waypoints = sum(
        int(report["path_planner_waypoints"]) for report in reports
    )
    path_planner_trust_region_margin_restoration_events = sum(
        int(report["path_planner_trust_region_margin_restoration_events"])
        for report in reports
    )
    path_planner_trust_region_orientation_progress_events = sum(
        int(report["path_planner_trust_region_orientation_progress_events"])
        for report in reports
    )
    path_planner_trust_region_feedback_basis_events = sum(
        int(report["path_planner_trust_region_feedback_basis_events"])
        for report in reports
    )
    path_planner_trust_region_orientation_first_events = sum(
        int(report["path_planner_trust_region_orientation_first_events"])
        for report in reports
    )
    path_planner_trust_region_constraint_anchored_restoration_events = sum(
        int(
            report[
                "path_planner_trust_region_constraint_anchored_restoration_events"
            ]
        )
        for report in reports
    )
    path_planner_trust_region_expanded_orientation_target_budget_events = sum(
        int(
            report[
                "path_planner_trust_region_expanded_orientation_target_budget_events"
            ]
        )
        for report in reports
    )
    lift_feedback_anchor_commands = sum(
        int(report["lift_feedback_anchor_commands"]) for report in reports
    )
    maximum_path_planner_trust_region_orientation_target_rad = max(
        (
            float(
                report[
                    "maximum_path_planner_trust_region_orientation_target_rad"
                ]
            )
            for report in reports
        ),
        default=0.0,
    )
    maximum_path_planner_ik_attempts_used = max(
        (
            int(report["maximum_path_planner_ik_attempts_used"])
            for report in reports
        ),
        default=0,
    )
    path_planner_start_state_recovery_events = sum(
        int(report["path_planner_start_state_recovery_events"])
        for report in reports
    )
    path_planner_reference_reuse_commands = sum(
        int(report["path_planner_reference_reuse_commands"])
        for report in reports
    )
    path_planner_reference_waypoint_advancements = sum(
        int(report["path_planner_reference_waypoint_advancements"])
        for report in reports
    )
    path_planner_terminal_control_commands = sum(
        int(report["path_planner_terminal_control_commands"])
        for report in reports
    )
    path_planner_terminal_control_activations = sum(
        int(report["path_planner_terminal_control_activations"])
        for report in reports
    )
    path_planner_terminal_control_completions = sum(
        int(report["path_planner_terminal_control_completions"])
        for report in reports
    )
    task_contact_exemption_events = sum(
        int(report["task_contact_exemption_events"])
        for report in reports
    )
    task_contact_exemption_steps = sum(
        int(report["task_contact_exemption_steps"])
        for report in reports
    )
    lift_contact_exemption_events = sum(
        int(report["lift_contact_exemption_events"])
        for report in reports
    )
    lift_gripper_bar_contact_exemption_events = sum(
        int(report["lift_gripper_bar_contact_exemption_events"])
        for report in reports
    )
    lift_moveit_fallback_events = sum(
        int(report["lift_moveit_fallback_events"])
        for report in reports
    )
    contact_phase_feedforward_commands = sum(
        int(report["contact_phase_feedforward_commands"])
        for report in reports
    )
    contact_phase_feedforward_failures = sum(
        int(report["contact_phase_feedforward_failures"])
        for report in reports
    )
    commanded_margin_breach_events = sum(
        int(report["commanded_margin_breach_events"]) for report in reports
    )
    observed_margin_breach_events = sum(
        int(report["observed_margin_breach_events"]) for report in reports
    )
    maximum_position_priority_orientation_relaxation_rad = max(
        (
            float(report["maximum_path_planner_orientation_relaxation_rad"])
            for report in reports
        ),
        default=0.0,
    )
    if stage == "exact":
        required_successes = int(acceptance["exact_control_successes_required"])
    elif stage == "tuning":
        required_successes = int(acceptance["tuning_successes_required"])
    else:
        required_successes = int(acceptance["development_successes_required"])
    criteria = {
        "required_successes": successes >= required_successes,
        "geometry_teacher_failures": teacher_failures
        <= int(acceptance["maximum_geometry_teacher_failures"]),
        "inverse_kinematics_failures": ik_failures
        <= int(acceptance["maximum_inverse_kinematics_failures"]),
        "adapter_clip_failures": adapter_clip_failures
        <= int(acceptance["maximum_adapter_clip_failures"]),
        "commanded_margin_breach_events": commanded_margin_breach_events
        <= int(acceptance["maximum_commanded_margin_breach_events"]),
        "observed_margin_breach_events": observed_margin_breach_events
        <= int(acceptance["maximum_observed_margin_breach_events"]),
        "position_priority_orientation_relaxation": (
            maximum_position_priority_orientation_relaxation_rad
            <= float(
                acceptance[
                    "maximum_position_priority_orientation_relaxation_rad"
                ]
            )
            + 1e-12
        ),
        "hidden_test_loaded": acceptance.get("hidden_test_loaded") is False,
        "recovery_label_write_disabled": acceptance.get(
            "recovery_labels_authorized_on_pass"
        )
        is False,
    }
    if "minimum_path_planner_terminal_control_completions" in acceptance:
        criteria["path_planner_terminal_control_completions"] = (
            path_planner_terminal_control_completions
            >= int(acceptance["minimum_path_planner_terminal_control_completions"])
        )
    if "minimum_trust_region_margin_restoration_events" in acceptance:
        criteria["trust_region_margin_restoration_events"] = (
            path_planner_trust_region_margin_restoration_events
            >= int(acceptance["minimum_trust_region_margin_restoration_events"])
        )
    if "minimum_trust_region_orientation_progress_events" in acceptance:
        criteria["trust_region_orientation_progress_events"] = (
            path_planner_trust_region_orientation_progress_events
            >= int(acceptance["minimum_trust_region_orientation_progress_events"])
        )
    if "minimum_trust_region_feedback_basis_events" in acceptance:
        criteria["trust_region_feedback_basis_events"] = (
            path_planner_trust_region_feedback_basis_events
            >= int(acceptance["minimum_trust_region_feedback_basis_events"])
        )
    if "minimum_trust_region_orientation_first_events" in acceptance:
        criteria["trust_region_orientation_first_events"] = (
            path_planner_trust_region_orientation_first_events
            >= int(acceptance["minimum_trust_region_orientation_first_events"])
        )
    if "minimum_trust_region_constraint_anchored_restoration_events" in acceptance:
        criteria["trust_region_constraint_anchored_restoration_events"] = (
            path_planner_trust_region_constraint_anchored_restoration_events
            >= int(
                acceptance[
                    "minimum_trust_region_constraint_anchored_restoration_events"
                ]
            )
        )
    if "minimum_expanded_orientation_target_budget_events" in acceptance:
        criteria["expanded_orientation_target_budget_events"] = (
            path_planner_trust_region_expanded_orientation_target_budget_events
            >= int(acceptance["minimum_expanded_orientation_target_budget_events"])
        )
    if "minimum_lift_feedback_anchor_commands" in acceptance:
        criteria["lift_feedback_anchor_commands"] = (
            lift_feedback_anchor_commands
            >= int(acceptance["minimum_lift_feedback_anchor_commands"])
        )
    if "minimum_task_contact_exemption_events" in acceptance:
        criteria["task_contact_exemption_events"] = (
            task_contact_exemption_events
            >= int(acceptance["minimum_task_contact_exemption_events"])
        )
    if "minimum_lift_contact_exemption_events" in acceptance:
        criteria["lift_contact_exemption_events"] = (
            lift_contact_exemption_events
            >= int(acceptance["minimum_lift_contact_exemption_events"])
        )
    if "minimum_lift_gripper_bar_contact_exemption_events" in acceptance:
        criteria["lift_gripper_bar_contact_exemption_events"] = (
            lift_gripper_bar_contact_exemption_events
            >= int(
                acceptance[
                    "minimum_lift_gripper_bar_contact_exemption_events"
                ]
            )
        )
    if "minimum_lift_moveit_fallback_events" in acceptance:
        criteria["lift_moveit_fallback_events"] = (
            lift_moveit_fallback_events
            >= int(acceptance["minimum_lift_moveit_fallback_events"])
        )
    if "minimum_contact_phase_feedforward_commands" in acceptance:
        criteria["contact_phase_feedforward_commands"] = (
            contact_phase_feedforward_commands
            >= int(acceptance["minimum_contact_phase_feedforward_commands"])
        )
    if "maximum_contact_phase_feedforward_failures" in acceptance:
        criteria["contact_phase_feedforward_failures"] = (
            contact_phase_feedforward_failures
            <= int(acceptance["maximum_contact_phase_feedforward_failures"])
        )
    stage_passed = bool(reports) and all(criteria.values())
    report = {
        "schema_version": 1,
        "report_id": f"{plan['plan_id']}-{stable_hash(reports)[:16]}",
        "status": "diagnostic_passed" if stage_passed else "failed",
        "diagnostic": "object_geometry_event_feedback_teacher_evaluation",
        "stage": stage,
        "plan_id": plan["plan_id"],
        "plan_sha256": file_sha256(plan_path),
        "dataset_revision": dataset_manifest.resolved_revision,
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "action_contract_sha256": file_sha256(action_contract_path),
        "calibration": _calibration_payload(calibration),
        "exact": exact_reports,
        "tuning": tuning_reports,
        "development": development_reports,
        "acceptance": {
            "criteria": criteria,
            "successes": successes,
            "required_successes": required_successes,
            "geometry_teacher_failures": teacher_failures,
            "inverse_kinematics_failures": ik_failures,
            "adapter_clip_failures": adapter_clip_failures,
            "joint_limit_projection_events": joint_limit_projection_events,
            "path_planner_attempts": path_planner_attempts,
            "path_planner_waypoints": path_planner_waypoints,
            "path_planner_trust_region_margin_restoration_events": (
                path_planner_trust_region_margin_restoration_events
            ),
            "path_planner_trust_region_orientation_progress_events": (
                path_planner_trust_region_orientation_progress_events
            ),
            "path_planner_trust_region_feedback_basis_events": (
                path_planner_trust_region_feedback_basis_events
            ),
            "path_planner_trust_region_orientation_first_events": (
                path_planner_trust_region_orientation_first_events
            ),
            "path_planner_trust_region_constraint_anchored_restoration_events": (
                path_planner_trust_region_constraint_anchored_restoration_events
            ),
            "path_planner_trust_region_expanded_orientation_target_budget_events": (
                path_planner_trust_region_expanded_orientation_target_budget_events
            ),
            "lift_feedback_anchor_commands": (
                lift_feedback_anchor_commands
            ),
            "maximum_path_planner_trust_region_orientation_target_rad": (
                maximum_path_planner_trust_region_orientation_target_rad
            ),
            "maximum_path_planner_ik_attempts_used": (
                maximum_path_planner_ik_attempts_used
            ),
            "path_planner_start_state_recovery_events": (
                path_planner_start_state_recovery_events
            ),
            "path_planner_reference_reuse_commands": (
                path_planner_reference_reuse_commands
            ),
            "path_planner_reference_waypoint_advancements": (
                path_planner_reference_waypoint_advancements
            ),
            "path_planner_terminal_control_commands": (
                path_planner_terminal_control_commands
            ),
            "path_planner_terminal_control_activations": (
                path_planner_terminal_control_activations
            ),
            "path_planner_terminal_control_completions": (
                path_planner_terminal_control_completions
            ),
            "task_contact_exemption_events": task_contact_exemption_events,
            "task_contact_exemption_steps": task_contact_exemption_steps,
            "lift_contact_exemption_events": lift_contact_exemption_events,
            "lift_gripper_bar_contact_exemption_events": (
                lift_gripper_bar_contact_exemption_events
            ),
            "lift_moveit_fallback_events": lift_moveit_fallback_events,
            "contact_phase_feedforward_commands": (
                contact_phase_feedforward_commands
            ),
            "contact_phase_feedforward_failures": (
                contact_phase_feedforward_failures
            ),
            "commanded_margin_breach_events": commanded_margin_breach_events,
            "observed_margin_breach_events": observed_margin_breach_events,
            "maximum_position_priority_orientation_relaxation_rad": (
                maximum_position_priority_orientation_relaxation_rad
            ),
        },
        "provenance": {
            "label_type": "object_geometry_event_feedback_teacher_action",
            "state_conditioned": True,
            "object_geometry_conditioned": True,
            "environment_step_input": False,
            "timestamp_input": False,
            "source_timeline_actions_used_at_runtime": False,
            "recovery_labels_written": False,
            "hidden_test_loaded": False,
            "validation_episodes_loaded": False,
            "policy_gate4_seeds_executed": False,
            "collection_seeds_executed": False,
            "task_contact_policy_schema": plan["task_contact_policy"]["schema"],
            "task_contact_allowed_phases": list(
                plan["task_contact_policy"]["phases"]
            ),
            "task_contact_allowed_unordered_geom_pairs": list(
                plan["task_contact_policy"]["allowed_unordered_geom_pairs"]
            ),
            "execution_guard_schema": plan["execution_guard"]["schema"],
            "execution_guard_strategy": plan["execution_guard"]["strategy"],
            "execution_guard_source_exact_report_sha256": plan[
                "execution_guard"
            ]["source_exact_report_sha256"],
            "physical_joint_limit_margin_rad": float(
                plan["execution_guard"]["physical_joint_limit_margin_rad"]
            ),
            "execution_tracking_reserve_rad": float(
                plan["execution_guard"]["tracking_reserve_rad"]
            ),
            "command_joint_limit_margin_rad": float(
                plan["execution_guard"]["command_joint_limit_margin_rad"]
            ),
            "mink_joint_limit_margin_rad": float(
                inverse_kinematics.get("mink_joint_limit_margin_rad", 0.0)
            ),
            "moveit_joint_path_constraint_margin_rad": float(
                inverse_kinematics.get(
                    "path_planner_joint_limit_margin_rad",
                    0.0,
                )
            ),
            "moveit_physical_joint_limit_margin_rad": float(
                inverse_kinematics.get(
                    "path_planner_physical_joint_limit_margin_rad",
                    0.0,
                )
            ),
            "moveit_trust_region_orientation_target_budget_rad": float(
                inverse_kinematics[
                    "path_planner_active_set_trust_region_"
                    "orientation_target_budget_rad"
                ]
            ),
            "moveit_planning_request_adapters": list(
                inverse_kinematics.get("path_planner_request_adapters", ())
            ),
            "moveit_ik_search_mode": inverse_kinematics.get(
                "path_planner_ik_search_mode"
            ),
            "moveit_ik_candidate_selection_mode": inverse_kinematics.get(
                "path_planner_ik_candidate_selection_mode"
            ),
            "moveit_ik_seed": int(
                inverse_kinematics.get("path_planner_ik_seed", -1)
            ),
            "moveit_ik_maximum_attempts": int(
                inverse_kinematics.get("path_planner_ik_maximum_attempts", -1)
            ),
            "moveit_ik_solver_base_frames": list(
                inverse_kinematics.get("path_planner_ik_solver_base_frames", ())
            ),
            "moveit_ik_solver_tip_frames": list(
                inverse_kinematics.get("path_planner_ik_solver_tip_frames", ())
            ),
            "moveit_position_priority_enabled": inverse_kinematics.get(
                "path_planner_position_priority_enabled"
            ),
            "moveit_position_priority_activation_phases": list(
                inverse_kinematics.get(
                    "path_planner_position_priority_activation_phases", ()
                )
            ),
            "moveit_position_priority_groups": list(
                inverse_kinematics.get(
                    "path_planner_position_priority_groups", ()
                )
            ),
            "moveit_position_priority_solver_mode": inverse_kinematics.get(
                "path_planner_position_priority_solver_mode"
            ),
            "moveit_position_priority_orientation_weight": float(
                inverse_kinematics.get(
                    "path_planner_position_priority_orientation_weight",
                    math.nan,
                )
            ),
            "moveit_full_pose_cartesian_backoff_enabled": (
                inverse_kinematics.get(
                    "path_planner_full_pose_cartesian_backoff_enabled"
                )
            ),
            "moveit_full_pose_cartesian_backoff_activation_phases": (
                inverse_kinematics.get(
                    "path_planner_full_pose_cartesian_backoff_activation_phases"
                )
            ),
            "moveit_full_pose_cartesian_backoff_fractions": (
                inverse_kinematics.get(
                    "path_planner_full_pose_cartesian_backoff_fractions"
                )
            ),
            "moveit_full_pose_cartesian_backoff_minimum_linear_progress_m": float(
                inverse_kinematics.get(
                    "path_planner_full_pose_cartesian_backoff_minimum_linear_progress_m",
                    math.nan,
                )
            ),
            "moveit_full_pose_cartesian_backoff_minimum_angular_progress_rad": float(
                inverse_kinematics.get(
                    "path_planner_full_pose_cartesian_backoff_minimum_angular_progress_rad",
                    math.nan,
                )
            ),
            "moveit_position_priority_maximum_orientation_relaxation_rad": (
                float(
                    inverse_kinematics.get(
                        "path_planner_position_priority_maximum_orientation_relaxation_rad",
                        math.nan,
                    )
                )
            ),
            "moveit_position_priority_ompl_seed_reset_per_request": (
                inverse_kinematics.get(
                    "path_planner_position_priority_ompl_seed_reset_per_request"
                )
            ),
            "moveit_position_priority_terminal_goal_normalization_limit_rad": (
                float(
                    inverse_kinematics.get(
                        "path_planner_position_priority_terminal_goal_normalization_limit_rad",
                        math.nan,
                    )
                )
            ),
            "moveit_ik_group_selection_mode": inverse_kinematics.get(
                "path_planner_ik_group_selection_mode"
            ),
            "moveit_full_pose_groups": inverse_kinematics.get(
                "path_planner_full_pose_groups"
            ),
            "moveit_finger_bound_reconciliation_tolerance_m": float(
                inverse_kinematics.get(
                    "path_planner_finger_bound_reconciliation_tolerance_m",
                    0.0,
                )
            ),
            "moveit_global_path_planner_initialized": moveit_runtime is not None,
            "moveit_trajectory_execution_backend": inverse_kinematics.get(
                "path_planner_execution_backend"
            ),
            "moveit_trajectory_operator": inverse_kinematics.get(
                "path_planner_trajectory_operator"
            ),
            "moveit_local_constraint_solver": inverse_kinematics.get(
                "path_planner_local_constraint_solver"
            ),
            "moveit_waypoint_l1_tolerance_rad": float(
                inverse_kinematics.get(
                    "path_planner_waypoint_l1_tolerance_rad",
                    0.0,
                )
            ),
            "moveit_terminal_control_backend": inverse_kinematics.get(
                "path_planner_terminal_control_backend"
            ),
            "moveit_terminal_control_handoff": inverse_kinematics.get(
                "path_planner_terminal_control_handoff"
            ),
            "moveit_replan_on_terminal_completion": inverse_kinematics.get(
                "path_planner_replan_on_terminal_completion"
            ),
            "moveit_terminal_completion_goal_l1_tolerance_rad": float(
                inverse_kinematics.get(
                    "path_planner_terminal_completion_goal_l1_tolerance_rad",
                    0.0,
                )
            ),
            "moveit_terminal_control_maximum_correction_rad": float(
                inverse_kinematics.get(
                    "path_planner_terminal_control_maximum_correction_rad",
                    0.0,
                )
            ),
            "moveit_terminal_control_source_exact_report_sha256": plan[
                "terminal_control"
            ]["source_exact_report_sha256"],
            "planner_implementation": (
                "upstream_moveit2_hybrid_planning_retained_reference_plus_"
                "fix_start_state_path_constraints_plus_ompl_plus_official_"
                "explicit_full_pose_groups_plus_full_pose_cartesian_backoff_"
                "plus_lma_position_only_ik_waypoint_plus_deterministic_terminal_"
                "goal_normalization_plus_official_"
                "mujoco_static_position_feedforward_plus_official_terminal_"
                "completion_reset"
                if moveit_runtime is not None
                else "none_or_legacy_diagnostic"
            ),
        },
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
    }
    if moveit_runtime is not None:
        report["moveit_runtime"] = moveit_runtime
    json.dumps(report, allow_nan=False)
    destination = _stage_report_path(plan_path, plan, stage)
    create_json(destination, report)
    print(json.dumps({"status": report["status"], **report["acceptance"]}, indent=2))
    print(f"Report: {destination}")
    return 0 if stage_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--stage", choices=("exact", "tuning", "full"), default="exact")
    args = parser.parse_args()
    return _main(args.plan.resolve(), stage=args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
