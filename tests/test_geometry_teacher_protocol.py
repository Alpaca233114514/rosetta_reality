"""Preregistered object-geometry teacher protocol tests."""

import inspect
import math
from pathlib import Path

import pytest
import torch

from rosetta_reality.sim import load_action_contract
from rosetta_reality.sim.geometry_teacher import GeometryPose, ObjectGeometryInsertionTeacher
from scripts.evaluate_aloha_geometry_teacher import (
    _collision_classification,
    _feedback_aligned_orthonormal_basis,
    _full_pose_cartesian_waypoint,
    _joint_execution_diagnostic,
    _load_plan,
    _quaternion_distance,
    _validate_plan_boundaries,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPOSITORY_ROOT / "configs/sim/aloha_insertion_geometry_teacher_058.yaml"


def test_geometry_teacher_plan_is_hash_bound_and_sealed() -> None:
    plan = _load_plan(PLAN)

    _validate_plan_boundaries(plan)

    assert plan["scope"]["hidden_test_loaded"] is False


def test_geometry_teacher_plan_requires_output_contract_before_calibration() -> None:
    plan = _load_plan(PLAN)
    del plan["output"]

    with pytest.raises(ValueError, match="output must be a mapping"):
        _validate_plan_boundaries(plan)
    assert plan["acceptance"]["recovery_labels_authorized_on_pass"] is False
    assert plan["calibration"]["source_timeline_actions_used_at_runtime"] is False
    assert plan["inverse_kinematics"]["solver_backend"] == "mink_qp"
    assert plan["inverse_kinematics"]["mink_version"] == "1.2.0"
    assert plan["inverse_kinematics"]["mink_solver"] == "daqp"
    assert plan["inverse_kinematics"]["path_planner_enabled"] is True
    assert plan["inverse_kinematics"]["path_planner_backend"] == "moveit2_ompl"
    assert plan["inverse_kinematics"]["path_planner_id"] == "RRTConnect"
    assert plan["inverse_kinematics"]["path_planner_type"] == (
        "geometric::RRTConnect"
    )
    assert plan["inverse_kinematics"]["path_planner_kinematics_plugin"] == (
        "lma_kinematics_plugin/LMAKinematicsPlugin"
    )
    assert plan["full_pose_group_selection"]["group_selection_mode"] == (
        "explicit_registered_groups_v1"
    )
    assert plan["full_pose_group_selection"]["full_pose_groups"] == [
        "left_arm",
        "right_arm",
    ]
    assert plan["inverse_kinematics"]["path_planner_ik_group_selection_mode"] == (
        "explicit_registered_groups_v1"
    )
    assert plan["inverse_kinematics"]["path_planner_full_pose_groups"] == [
        "left_arm",
        "right_arm",
    ]
    assert plan["full_pose_cartesian_backoff"]["activation_phases"] == [
        "approach",
        "orient",
    ]
    assert plan["inverse_kinematics"][
        "path_planner_full_pose_cartesian_backoff_fractions"
    ] == [0.75, 0.5, 0.25, 0.125, 0.1, 0.05]
    assert plan["joint_margin_candidate_selection"]["selection_mode"] == (
        "deterministic_maximum_minimum_joint_limit_margin_v1"
    )
    assert plan["joint_margin_candidate_selection"]["validity_filter"] == [
        "satisfiesBounds",
        "registered_joint_path_constraints",
        "self_collision_free",
        "registered_task_space_tolerance",
    ]
    assert plan["inverse_kinematics"][
        "path_planner_ik_candidate_selection_mode"
    ] == "deterministic_maximum_minimum_joint_limit_margin_v1"
    assert plan["active_set_cartesian_trust_region"]["activation_phase"] == "orient"
    assert plan["active_set_cartesian_trust_region"]["coordinate_directions"] == [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    assert plan["active_set_cartesian_trust_region"]["radii_m"] == [
        0.0,
        0.0015,
        0.003,
        0.006,
        0.009,
    ]
    assert plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_enabled"
    ] is True
    assert plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_orientation_progress_fractions"
    ] == [0.5, 0.25, 0.125, 0.05]
    assert plan["feedback_aligned_trust_region_basis"]["candidate_basis"] == (
        "feedback_aligned_orthonormal_v1"
    )
    assert plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_candidate_basis"
    ] == "feedback_aligned_orthonormal_v1"
    assert plan["orientation_first_trust_region_selection"]["selection_policy"] == (
        "orientation_progress_first_v1"
    )
    assert plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_selection_policy"
    ] == "orientation_progress_first_v1"
    assert plan["constraint_anchored_restoration"]["restoration_reference"] == (
        "command_margin_boundary"
    )
    assert plan["constraint_anchored_restoration"][
        "restoration_buffer_rad"
    ] == pytest.approx(0.001)
    assert plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_restoration_reference"
    ] == "command_margin_boundary"
    assert plan["expanded_orientation_target_budget"][
        "previous_orientation_target_budget_rad"
    ] == pytest.approx(0.04)
    assert plan["expanded_orientation_target_budget"][
        "orientation_target_budget_rad"
    ] == pytest.approx(0.2)
    assert plan["teacher"]["maximum_orientation_step_rad"] == pytest.approx(0.2)
    assert plan["teacher"]["lift_feedback_step_m"] == pytest.approx(0.006)
    assert plan["lift_grasp_feedback"]["step_m"] == pytest.approx(0.006)
    assert plan["lift_grasp_feedback"]["single_axis"] == (
        "fixed_terminal_lift_target_replaced_by_feedback_anchored_increment"
    )
    assert plan["acceptance"]["minimum_lift_feedback_anchor_commands"] == 1
    assert plan["lift_contact_exemption"]["phases"] == ["lift"]
    assert plan["lift_contact_exemption"]["allowed_unordered_geom_pairs"] == [
        ["table", "vx300s_right/10_right_gripper_finger"],
        ["table", "vx300s_right/9_gripper_bar"],
    ]
    assert plan["acceptance"]["minimum_lift_contact_exemption_events"] == 1
    assert (
        plan["acceptance"]["minimum_lift_gripper_bar_contact_exemption_events"]
        == 1
    )
    assert plan["inverse_kinematics"]["path_planner_phases"] == [
        "approach",
        "orient",
        "lift",
    ]
    assert plan["lift_moveit_fallback"]["activation_phases"] == ["lift"]
    assert plan["lift_moveit_fallback"]["activation"] == (
        "only_after_unchanged_mink_qp_failure"
    )
    assert plan["acceptance"]["minimum_lift_moveit_fallback_events"] == 1
    assert plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_orientation_target_budget_rad"
    ] == pytest.approx(0.2)
    assert plan["inverse_kinematics"]["path_planner_request_adapters"] == [
        "default_planner_request_adapters/FixStartStatePathConstraints"
    ]
    assert plan["inverse_kinematics"]["path_planner_execution_backend"] == (
        "moveit_hybrid_planning"
    )
    assert plan["inverse_kinematics"]["path_planner_trajectory_operator"] == (
        "moveit_hybrid_planning/SimpleSampler"
    )
    assert plan["inverse_kinematics"]["path_planner_local_constraint_solver"] == (
        "moveit_hybrid_planning/ForwardTrajectory"
    )
    assert plan["inverse_kinematics"]["path_planner_include_trajectory"] is True
    assert plan["inverse_kinematics"][
        "path_planner_waypoint_l1_tolerance_rad"
    ] == pytest.approx(0.2)
    assert plan["inverse_kinematics"]["path_planner_replan_on_phase_change"] is True
    assert plan["inverse_kinematics"]["path_planner_replan_every_step"] is False
    assert (
        plan["inverse_kinematics"]["path_planner_replan_on_terminal_completion"]
        is True
    )
    assert plan["inverse_kinematics"]["path_planner_terminal_control_enabled"] is True
    assert plan["inverse_kinematics"]["path_planner_terminal_control_backend"] == (
        "mujoco_static_inverse_dynamics_affine_position_feedforward"
    )
    assert plan["inverse_kinematics"]["path_planner_terminal_control_handoff"] == (
        "final_waypoint_within_moveit_simple_sampler_l1_tolerance"
    )
    assert (
        plan["inverse_kinematics"][
            "path_planner_terminal_control_replan_on_failure"
        ]
        is False
    )
    assert plan["inverse_kinematics"][
        "path_planner_terminal_control_maximum_correction_rad"
    ] == pytest.approx(0.05)
    assert plan["inverse_kinematics"][
        "path_planner_terminal_control_joint_limit_margin_rad"
    ] == pytest.approx(0.04540462255477905)
    assert plan["inverse_kinematics"][
        "path_planner_terminal_completion_goal_l1_tolerance_rad"
    ] == pytest.approx(0.001)
    assert plan["inverse_kinematics"]["path_planner_phases"] == [
        "approach",
        "orient",
        "lift",
    ]
    assert plan["inverse_kinematics"]["path_planner_orientation_tolerance_rad"] == (
        0.003
    )
    assert plan["inverse_kinematics"][
        "path_planner_start_bound_reconciliation_tolerance_rad"
    ] == pytest.approx(0.00002)
    assert plan["inverse_kinematics"][
        "path_planner_collision_geometry_link_count"
    ] == 22
    assert plan["inverse_kinematics"][
        "path_planner_collision_geometry_shape_count"
    ] == 22
    assert plan["inverse_kinematics"]["path_planner_finger_lower_m"] == 0.021
    assert plan["inverse_kinematics"]["path_planner_finger_upper_m"] == 0.057
    assert plan["inverse_kinematics"][
        "path_planner_finger_bound_reconciliation_tolerance_m"
    ] == 0.001
    assert plan["inverse_kinematics"][
        "path_planner_physical_joint_limit_margin_rad"
    ] == pytest.approx(0.01)
    assert plan["inverse_kinematics"][
        "path_planner_joint_limit_margin_rad"
    ] == pytest.approx(0.04540462255477905)
    assert plan["inverse_kinematics"]["maximum_accepted_error"] == 0.001
    assert plan["inverse_kinematics"]["maximum_accepted_projected_error"] == 0.003
    assert plan["inverse_kinematics"]["mink_maximum_iterations"] == 15
    assert plan["inverse_kinematics"]["mink_joint_limit_margin_rad"] == pytest.approx(
        0.04540462255477905
    )
    assert plan["execution_diagnostics"] == {
        "schema": "commanded_vs_observed_joint_margin_v1",
        "joint_limit_margin_rad": 0.01,
        "arm_dimension_names": [
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
        ],
        "record_every_executed_step": True,
        "record_full_joint_vectors": True,
        "record_per_joint_margins": True,
        "affects_action_selection": False,
    }
    assert plan["execution_guard"] == {
        "schema": (
            "robust_joint_limit_constraint_tightening_with_"
            "official_start_recovery_safe_set_v3"
        ),
        "strategy": (
            "static_uniform_constraint_tightening_plus_official_"
            "start_state_path_constraint_adapter"
        ),
        "source_plan": "configs/sim/aloha_insertion_geometry_teacher_029.yaml",
        "source_plan_sha256": (
            "4fc4bff41966022591057bf19c9246b5f511ef3009fde66b1e87a59aca70cde0"
        ),
        "source_remote_static_attempt": "athena-plan029-static-001",
        "source_remote_direct_smoke_results_sha256": (
            "88af2663fa8bb19f52b1afd4710dbdf7697e0be3ec3549607c59ca031bd0fa13"
        ),
        "source_remote_execution_log_sha256": (
            "434871e6042b1d39b9c6c156300cacb991a459d527cb35e6e9cbd73879366d71"
        ),
        "source_audit": (
            "reports/training/"
            "m2-smolvla-athena-plan028-exact-audit-2026-08-15.json"
        ),
        "source_audit_sha256": (
            "c06412aba0d10fc0c7a21c04ece5352e2f1ff85d692a701d119a73d0cdcf6829"
        ),
        "source_exact_report_sha256": (
            "0848857b401a10635cd7d66da4dd7cd339f8417abd9982141669c01a05ccbb64"
        ),
        "reserve_metric": "maximum_tracking_overshoot_toward_limit_rad",
        "reserve_scope": "all_arm_joints_all_executed_train_exact_steps",
        "physical_joint_limit_margin_rad": 0.01,
        "tracking_reserve_rad": pytest.approx(0.03540462255477905),
        "command_joint_limit_margin_rad": pytest.approx(0.04540462255477905),
        "applies_to": [
            "mink.ConfigurationLimit",
            "moveit_msgs/JointConstraint",
        ],
        "affects_action_selection": True,
        "start_state_path_constraint_adapter": (
            "default_planner_request_adapters/FixStartStatePathConstraints"
        ),
        "recovery_entry_condition": (
            "physical_safe_but_tightened_path_constraint_invalid"
        ),
        "recovery_prefix_joint_limit_margin_rad": pytest.approx(0.01),
        "recovery_constrained_suffix_joint_limit_margin_rad": pytest.approx(
            0.04540462255477905
        ),
        "recovery_requires_positive_margin_progress": True,
        "recovery_below_command_margin_requires_monotonic_progress": True,
        "recovery_after_first_command_margin_entry_must_remain_inside": True,
        "recovery_forbidden_when_start_satisfies_command_margin": True,
    }
    assert plan["acceptance"]["maximum_commanded_margin_breach_events"] == 0
    assert plan["acceptance"]["maximum_observed_margin_breach_events"] == 0
    assert plan["acceptance"][
        "minimum_path_planner_terminal_control_completions"
    ] == 1
    assert plan["acceptance"]["minimum_task_contact_exemption_events"] == 1
    assert plan["trajectory_execution"]["source_exact_report_sha256"] == (
        "aaa033c6a1740bac40d9589b01372c059e9d242099b2aa6a5ab4c4f8d1029fa3"
    )
    assert plan["trajectory_execution"]["observed_global_plan_attempts"] == 131
    assert plan["trajectory_execution"]["affects_pose_gates"] is False
    assert plan["terminal_control"]["source_exact_report_sha256"] == (
        "f8ea0bb5514afe7c1bf64930b05ac40f52346b9c98ec35e9c2a539dbfc2acf5c"
    )
    assert plan["terminal_control"]["affects_pose_gates"] is False
    assert plan["sparse_actuator_moment_repair"] == {
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
    assert plan["joint_name_adapter_repair"]["gym_mujoco_names"] == [
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
    ]
    assert (
        plan["joint_name_adapter_repair"][
            "mapping_order_matches_arm_action_indices"
        ]
        is True
    )
    assert plan["terminal_completion_refresh"]["source_exact_report_sha256"] == (
        "2d1fa4e3afd3748a218c82220ea678972cc25a97514f9bacec6d71eca6a6174f"
    )
    assert plan["terminal_completion_refresh"][
        "source_final_arm_joint_l1_to_original_moveit_goal_rad"
    ] == pytest.approx(0.0000296434154734015)
    assert plan["task_contact_policy"]["phases"] == ["descend", "grasp"]
    assert plan["task_contact_policy"]["allowed_unordered_geom_pairs"] == [
        ["table", "vx300s_right/10_right_gripper_finger"]
    ]
    assert plan["evaluation"]["maximum_steps"] == 750
    assert plan["execution_horizon"]["source_maximum_steps"] == 500
    assert plan["execution_horizon"]["maximum_steps"] == 750
    assert plan["moveit_ik_budget"]["source_ik_timeout_s"] == pytest.approx(0.5)
    assert plan["moveit_ik_budget"]["ik_timeout_s"] == pytest.approx(2.0)
    assert plan["inverse_kinematics"]["path_planner_ik_timeout_s"] == pytest.approx(
        2.0
    )
    assert plan["deterministic_moveit_ik"]["search_mode"] == (
        "deterministic_seeded_moveit_subgroup_multistart_v1"
    )
    assert plan["inverse_kinematics"]["path_planner_ik_seed"] == 2210
    assert plan["inverse_kinematics"]["path_planner_ik_maximum_attempts"] == 256
    assert plan["inverse_kinematics"]["path_planner_ik_solver_base_frames"] == [
        "vx300s_left/base_link",
        "vx300s_right/base_link",
    ]
    assert plan["position_priority_waypoint"]["solver_mode"] == "position_only_ik"
    assert plan["position_priority_waypoint"]["orientation_weight"] == 0.0
    assert plan["position_priority_waypoint"]["activation_phases"] == [
        "approach"
    ]
    assert plan["position_priority_waypoint"][
        "cartesian_backoff_fractions"
    ] == [1.0, 0.75, 0.5, 0.25, 0.125]
    assert plan["position_priority_waypoint"][
        "ompl_seed_reset_per_request"
    ] is True
    assert plan["position_priority_waypoint"][
        "terminal_goal_normalization_limit_rad"
    ] == pytest.approx(1e-5)
    assert plan["inverse_kinematics"][
        "path_planner_position_priority_groups"
    ] == ["left_arm_position_priority", "right_arm_position_priority"]
    assert plan["acceptance"][
        "maximum_position_priority_orientation_relaxation_rad"
    ] == pytest.approx(0.04)
    assert plan["contact_phase_feedforward"]["phases"] == ["descend", "grasp"]
    assert plan["contact_phase_feedforward"]["maximum_correction_rad"] == (
        pytest.approx(0.05)
    )
    assert plan["acceptance"]["minimum_contact_phase_feedforward_commands"] == 1
    assert plan["acceptance"]["maximum_contact_phase_feedforward_failures"] == 0
    assert plan["inverse_kinematics"]["path_planner_runtime"]["image_id"] == (
        "sha256:82c7e6a8100ad485e22f28e0ab1ed36c551eb5ad877c0243fbbb31d29fe6540b"
    )
    exact_seed = plan["evaluation"]["exact_control"]["simulator_seed"]
    later_seeds = {
        *plan["evaluation"]["tuning_simulator_seeds"],
        *plan["evaluation"]["development_simulator_seeds"],
        *plan["evaluation"]["reserved_collection_simulator_seeds"],
        *plan["evaluation"]["reserved_policy_gate4_seeds"],
    }
    assert exact_seed not in later_seeds
    assert set(plan["evaluation"]["development_simulator_seeds"]) == {
        2000,
        2001,
        2002,
        2003,
        2004,
    }


def test_geometry_teacher_decision_api_has_no_time_or_step_input() -> None:
    parameters = inspect.signature(ObjectGeometryInsertionTeacher.decide).parameters

    assert tuple(parameters) == ("self", "geometry")


def test_full_pose_cartesian_waypoint_interpolates_position_and_orientation() -> None:
    current = GeometryPose(
        position=torch.tensor([0.0, 0.0, 0.0]),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    requested = GeometryPose(
        position=torch.tensor([2.0, 4.0, 6.0]),
        quaternion=torch.tensor(
            [math.cos(0.02), 0.0, 0.0, math.sin(0.02)]
        ),
    )

    waypoint = _full_pose_cartesian_waypoint(current, requested, 0.125)

    assert waypoint.position.tolist() == pytest.approx([0.25, 0.5, 0.75])
    assert _quaternion_distance(current.quaternion, waypoint.quaternion) == (
        pytest.approx(0.005, abs=5e-5)
    )
    assert _quaternion_distance(waypoint.quaternion, requested.quaternion) == (
        pytest.approx(0.035, abs=5e-5)
    )


def test_geometry_teacher_plan_rejects_moveit_identity_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["path_planner_ompl_seed"] = 2211

    with pytest.raises(ValueError, match="OMPL seed"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_moveit_adapter_identity_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["path_planner_request_adapters"] = []

    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_full_pose_group_selection_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["path_planner_full_pose_groups"].reverse()

    with pytest.raises(ValueError, match="path-planner identity mismatch"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_full_pose_group_evidence_drift() -> None:
    plan = _load_plan(PLAN)
    plan["full_pose_group_selection"]["runtime_audit_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="full-pose group evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_full_pose_backoff_fraction_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_full_pose_cartesian_backoff_fractions"
    ].append(0.025)

    with pytest.raises(ValueError, match="full-pose Cartesian backoff identity"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_joint_margin_selection_mode_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["path_planner_ik_candidate_selection_mode"] = (
        "first_valid"
    )

    with pytest.raises(ValueError, match="Deterministic MoveIt identity mismatch"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_joint_margin_evidence_drift() -> None:
    plan = _load_plan(PLAN)
    plan["joint_margin_candidate_selection"]["repeat_report_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="candidate-selection evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_trust_region_direction_drift() -> None:
    plan = _load_plan(PLAN)
    plan["active_set_cartesian_trust_region"]["coordinate_directions"][0][0] = 0.5

    with pytest.raises(ValueError, match="trust-region evidence differs"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_trust_region_identity_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_maximum_requested_position_relaxation_m"
    ] = 0.02

    with pytest.raises(ValueError, match="active-set trust-region identity mismatch"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_trust_region_source_hash_drift() -> None:
    plan = _load_plan(PLAN)
    plan["active_set_cartesian_trust_region"]["discarded_pick_ik_audit_sha256"] = (
        "0" * 64
    )

    with pytest.raises(ValueError, match="trust-region evidence differs"):
        _validate_plan_boundaries(plan)


def test_feedback_aligned_trust_region_basis_is_deterministic_and_orthonormal() -> None:
    basis = _feedback_aligned_orthonormal_basis(
        torch.tensor([0.2, 0.3, 0.4]),
        torch.tensor([0.1, 0.5, 0.45]),
    )

    vectors = [torch.tensor(direction, dtype=torch.float64) for direction in basis]
    assert len(vectors) == 6
    assert torch.allclose(vectors[0], -vectors[1], atol=1e-12, rtol=0.0)
    assert torch.allclose(vectors[2], -vectors[3], atol=1e-12, rtol=0.0)
    assert torch.allclose(vectors[4], -vectors[5], atol=1e-12, rtol=0.0)
    positive = torch.stack((vectors[0], vectors[2], vectors[4]))
    assert torch.allclose(
        positive @ positive.T,
        torch.eye(3, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert basis == _feedback_aligned_orthonormal_basis(
        torch.tensor([0.2, 0.3, 0.4]),
        torch.tensor([0.1, 0.5, 0.45]),
    )


def test_feedback_aligned_trust_region_basis_rejects_zero_position_error() -> None:
    position = torch.tensor([0.2, 0.3, 0.4])

    with pytest.raises(ValueError, match="needs position error"):
        _feedback_aligned_orthonormal_basis(position, position)


def test_geometry_teacher_plan_rejects_feedback_basis_drift() -> None:
    plan = _load_plan(PLAN)
    plan["feedback_aligned_trust_region_basis"]["candidate_basis"] = (
        "fixed_world_coordinate_basis"
    )

    with pytest.raises(ValueError, match="basis evidence differs"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_feedback_basis_identity_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_candidate_basis"
    ] = "fixed_world_coordinate_basis"

    with pytest.raises(ValueError, match="active-set trust-region identity mismatch"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_orientation_first_selection_drift() -> None:
    plan = _load_plan(PLAN)
    plan["orientation_first_trust_region_selection"]["selection_policy"] = (
        "margin_restoration_first"
    )

    with pytest.raises(ValueError, match="selection evidence differs"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_orientation_first_identity_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_selection_policy"
    ] = "margin_restoration_first"

    with pytest.raises(ValueError, match="active-set trust-region identity mismatch"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_constraint_restoration_drift() -> None:
    plan = _load_plan(PLAN)
    plan["constraint_anchored_restoration"]["restoration_buffer_rad"] = 0.0009

    with pytest.raises(ValueError, match="restoration evidence differs"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_constraint_restoration_identity_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_restoration_reference"
    ] = "current_margin"

    with pytest.raises(ValueError, match="active-set trust-region identity mismatch"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_orientation_target_budget_drift() -> None:
    plan = _load_plan(PLAN)
    plan["expanded_orientation_target_budget"]["orientation_target_budget_rad"] = 0.3

    with pytest.raises(ValueError, match="orientation target budget evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_orientation_target_budget_identity_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_active_set_trust_region_orientation_target_budget_rad"
    ] = 0.3

    with pytest.raises(ValueError, match="active-set trust-region identity mismatch"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_teacher_orientation_target_drift() -> None:
    plan = _load_plan(PLAN)
    plan["teacher"]["maximum_orientation_step_rad"] = 0.3

    with pytest.raises(ValueError, match="orientation target budget differs from teacher"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_lift_feedback_step_drift() -> None:
    plan = _load_plan(PLAN)
    plan["lift_grasp_feedback"]["step_m"] = 0.007

    with pytest.raises(ValueError, match="Lift grasp-feedback evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_teacher_lift_feedback_drift() -> None:
    plan = _load_plan(PLAN)
    plan["teacher"]["lift_feedback_step_m"] = 0.007

    with pytest.raises(ValueError, match="Lift feedback step differs from teacher"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_lift_feedback_anchor_gate_drift() -> None:
    plan = _load_plan(PLAN)
    plan["acceptance"]["minimum_lift_feedback_anchor_commands"] = 0

    with pytest.raises(ValueError, match="Lift feedback-anchor command"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_lift_contact_phase_drift() -> None:
    plan = _load_plan(PLAN)
    plan["lift_contact_exemption"]["phases"] = ["grasp"]

    with pytest.raises(ValueError, match="Lift contact-exemption evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_lift_contact_pair_drift() -> None:
    plan = _load_plan(PLAN)
    plan["lift_contact_exemption"]["allowed_unordered_geom_pairs"] = [
        ["table", "vx300s_left/10_left_gripper_finger"]
    ]

    with pytest.raises(ValueError, match="Lift contact-exemption evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_lift_contact_anchor_gate_drift() -> None:
    plan = _load_plan(PLAN)
    plan["acceptance"]["minimum_lift_contact_exemption_events"] = 0

    with pytest.raises(ValueError, match="Lift contact-exemption event"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_lift_gripper_bar_gate_drift() -> None:
    plan = _load_plan(PLAN)
    plan["acceptance"]["minimum_lift_gripper_bar_contact_exemption_events"] = 0

    with pytest.raises(ValueError, match="Lift gripper-bar contact exemption"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_lift_moveit_fallback_phase_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["path_planner_phases"] = ["approach", "orient"]

    with pytest.raises(ValueError, match="MoveIt fallback phase scope"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_lift_moveit_fallback_activation_drift() -> None:
    plan = _load_plan(PLAN)
    plan["lift_moveit_fallback"]["activation"] = "before_mink_qp"

    with pytest.raises(ValueError, match="Lift MoveIt fallback evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_lift_moveit_fallback_gate_drift() -> None:
    plan = _load_plan(PLAN)
    plan["acceptance"]["minimum_lift_moveit_fallback_events"] = 0

    with pytest.raises(ValueError, match="Lift MoveIt fallback event"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_position_priority_scope_drift() -> None:
    plan = _load_plan(PLAN)
    plan["position_priority_waypoint"]["activation_phases"].append("orient")

    with pytest.raises(ValueError, match="position-priority evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_position_priority_orientation_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_position_priority_maximum_orientation_relaxation_rad"
    ] = 0.05

    with pytest.raises(ValueError, match="position-priority identity"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_position_priority_repeat_drift() -> None:
    plan = _load_plan(PLAN)
    plan["position_priority_waypoint"]["repeat_report_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="position-priority evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_per_step_global_replanning() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["path_planner_replan_every_step"] = True

    with pytest.raises(ValueError, match="trajectory-execution identity"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_terminal_control_replanning() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_terminal_control_replan_on_failure"
    ] = True

    with pytest.raises(ValueError, match="trajectory-execution identity"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_terminal_completion_disable() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_replan_on_terminal_completion"
    ] = False

    with pytest.raises(ValueError, match="trajectory-execution identity"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_terminal_completion_tolerance_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_terminal_completion_goal_l1_tolerance_rad"
    ] = 0.002

    with pytest.raises(ValueError, match="terminal-completion tolerance"):
        _validate_plan_boundaries(plan)


def test_collision_classification_exempts_only_registered_unordered_pair() -> None:
    class FakeEnvironment:
        @staticmethod
        def contact_pairs() -> tuple[tuple[str, str], ...]:
            return (
                ("table", "vx300s_right/10_right_gripper_finger"),
                ("table", "vx300s_left/10_left_gripper_finger"),
                ("red_peg", "table"),
            )

        @staticmethod
        def is_unexpected_collision_pair(first: str, second: str) -> bool:
            return "vx300s_" in first or "vx300s_" in second

    allowed = frozenset(
        {frozenset({"table", "vx300s_right/10_right_gripper_finger"})}
    )
    count, exemptions = _collision_classification(  # type: ignore[arg-type]
        FakeEnvironment(),
        allowed,
    )

    assert count == 1
    assert exemptions == (("table", "vx300s_right/10_right_gripper_finger"),)


def test_geometry_teacher_plan_rejects_task_contact_scope_drift() -> None:
    plan = _load_plan(PLAN)
    plan["task_contact_policy"]["phases"].append("approach")

    with pytest.raises(ValueError, match="task-contact policy evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_execution_horizon_drift() -> None:
    plan = _load_plan(PLAN)
    plan["evaluation"]["maximum_steps"] = 751

    with pytest.raises(ValueError, match="maximum steps"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_moveit_ik_timeout_budget_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["path_planner_ik_timeout_s"] = 0.5

    with pytest.raises(ValueError, match="IK timeout"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_contact_phase_feedforward_scope_drift() -> None:
    plan = _load_plan(PLAN)
    plan["contact_phase_feedforward"]["phases"].append("lift")

    with pytest.raises(ValueError, match="Contact-phase feedforward evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_terminal_control_margin_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_terminal_control_joint_limit_margin_rad"
    ] = 0.01

    with pytest.raises(ValueError, match="changes the command joint margin"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_terminal_source_evidence_drift() -> None:
    plan = _load_plan(PLAN)
    plan["terminal_control"]["source_exact_report_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source evidence differs"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_sparse_moment_evidence_drift() -> None:
    plan = _load_plan(PLAN)
    plan["sparse_actuator_moment_repair"]["observed_actuator_moment_shape"] = [
        16,
        28,
    ]

    with pytest.raises(ValueError, match="sparse actuator-moment repair evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_joint_name_adapter_drift() -> None:
    plan = _load_plan(PLAN)
    plan["joint_name_adapter_repair"]["gym_mujoco_names"][0] = "left_waist"

    with pytest.raises(ValueError, match="joint-name adapter repair evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_simple_sampler_tolerance_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_waypoint_l1_tolerance_rad"
    ] = 0.21

    with pytest.raises(ValueError, match="SimpleSampler waypoint tolerance"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_physical_moveit_margin_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_physical_joint_limit_margin_rad"
    ] = 0.009

    with pytest.raises(ValueError, match="physical joint margin"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_start_bound_tolerance_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"][
        "path_planner_start_bound_reconciliation_tolerance_rad"
    ] = 0.00003

    with pytest.raises(ValueError, match="start-bound reconciliation tolerance"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_zero_collision_geometry() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["path_planner_collision_geometry_shape_count"] = 0

    with pytest.raises(ValueError, match="collision-geometry identity"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_oversized_joint_limit_margin() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["mink_joint_limit_margin_rad"] = 0.051

    with pytest.raises(ValueError, match="joint-limit margin"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_moveit_joint_margin_drift() -> None:
    plan = _load_plan(PLAN)
    plan["inverse_kinematics"]["path_planner_joint_limit_margin_rad"] = 0.009

    with pytest.raises(ValueError, match="joint path-constraint margin"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_execution_guard_source_hash_drift() -> None:
    plan = _load_plan(PLAN)
    plan["execution_guard"]["source_audit_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source audit identity"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_execution_guard_reserve_drift() -> None:
    plan = _load_plan(PLAN)
    plan["execution_guard"]["tracking_reserve_rad"] = 0.035

    with pytest.raises(ValueError, match="reserve differs from train-exact evidence"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_execution_guard_sum_drift() -> None:
    plan = _load_plan(PLAN)
    plan["execution_guard"]["command_joint_limit_margin_rad"] = 0.044

    with pytest.raises(ValueError, match="not physical plus reserve"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_nonofficial_execution_guard_limit() -> None:
    plan = _load_plan(PLAN)
    plan["execution_guard"]["applies_to"][1] = "custom_joint_clamp"

    with pytest.raises(ValueError, match="registered official limits"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_nonofficial_start_recovery_adapter() -> None:
    plan = _load_plan(PLAN)
    plan["execution_guard"]["start_state_path_constraint_adapter"] = (
        "custom/FixStartStatePathConstraints"
    )

    with pytest.raises(ValueError, match="official MoveIt adapter"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_recovery_prefix_margin_relaxation() -> None:
    plan = _load_plan(PLAN)
    plan["execution_guard"]["recovery_prefix_joint_limit_margin_rad"] = 0.009

    with pytest.raises(ValueError, match="retain the physical joint margin"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_requires_positive_recovery_progress() -> None:
    plan = _load_plan(PLAN)
    plan["execution_guard"]["recovery_requires_positive_margin_progress"] = False

    with pytest.raises(ValueError, match="must enable"):
        _validate_plan_boundaries(plan)


@pytest.mark.parametrize(
    "key",
    [
        "recovery_below_command_margin_requires_monotonic_progress",
        "recovery_after_first_command_margin_entry_must_remain_inside",
    ],
)
def test_geometry_teacher_plan_requires_safe_set_recovery_contract(key: str) -> None:
    plan = _load_plan(PLAN)
    plan["execution_guard"][key] = False

    with pytest.raises(ValueError, match="safe-set recovery must enable"):
        _validate_plan_boundaries(plan)


def test_geometry_teacher_plan_rejects_exact_seed_reuse() -> None:
    plan = _load_plan(PLAN)
    plan["evaluation"]["tuning_simulator_seeds"] = [10]

    with pytest.raises(ValueError, match="Exact-control seed"):
        _validate_plan_boundaries(plan)


def test_execution_diagnostic_separates_command_margin_from_observed_overshoot() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    pre_step = torch.zeros(contract.dimension)
    commanded = pre_step.clone()
    observed = pre_step.clone()
    joint_index = contract.dimension_names.index("right_forearm_roll")
    upper = contract.upper_bounds[joint_index]
    pre_step[joint_index] = upper - 0.10
    commanded[joint_index] = upper - 0.02
    observed[joint_index] = upper - 0.005

    diagnostic = _joint_execution_diagnostic(
        pre_step,
        commanded,
        observed,
        contract,
        registered_margin_rad=0.01,
        source="moveit2_ompl_lma_rrtconnect",
    )

    assert diagnostic["commanded"]["minimum_margin_rad"] == pytest.approx(
        0.02,
        abs=5e-7,
    )
    assert diagnostic["commanded"]["inside_registered_margin"] is False
    assert diagnostic["observed_post_step"]["minimum_margin_rad"] == pytest.approx(
        0.005,
        abs=5e-7,
    )
    assert diagnostic["observed_post_step"]["inside_registered_margin"] is True
    assert diagnostic["maximum_absolute_tracking_error_joint"] == (
        "right_forearm_roll"
    )
    assert diagnostic["maximum_absolute_tracking_error_rad"] == pytest.approx(
        0.015,
        abs=5e-7,
    )
    assert diagnostic["maximum_overshoot_toward_commanded_bound_rad"] == (
        pytest.approx(0.015, abs=5e-7)
    )
    assert diagnostic["maximum_margin_loss_command_to_observation_rad"] == (
        pytest.approx(0.015, abs=5e-7)
    )


def test_geometry_teacher_plan_rejects_diagnostic_action_feedback() -> None:
    plan = _load_plan(PLAN)
    plan["execution_diagnostics"]["affects_action_selection"] = True

    with pytest.raises(ValueError, match="must not affect action selection"):
        _validate_plan_boundaries(plan)
