#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <ompl/util/RandomNumbers.h>
#include <random_numbers/random_numbers.h>
#include <rclcpp/rclcpp.hpp>

#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_model/link_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>

namespace
{
using Json = nlohmann::json;

constexpr std::array<const char*, 12> kArmJointNames = {
  "left_waist",         "left_shoulder",    "left_elbow",       "left_forearm_roll",
  "left_wrist_angle",   "left_wrist_rotate", "right_waist",      "right_shoulder",
  "right_elbow",        "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate",
};
constexpr std::array<const char*, 2> kArmGroups = { "left_arm", "right_arm" };
constexpr std::array<const char*, 2> kPositionPriorityArmGroups = {
  "left_arm_position_priority", "right_arm_position_priority"
};
constexpr std::array<const char*, 2> kEeLinks = { "vx300s_left/ee_gripper_link",
                                                  "vx300s_right/ee_gripper_link" };
constexpr std::array<const char*, 2> kIkSolverBaseFrames = { "vx300s_left/base_link",
                                                             "vx300s_right/base_link" };
constexpr double kEeToGymCalibrationSiteXM = 0.0428;
constexpr const char* kBimanualGroup = "bimanual";
constexpr const char* kPlannerId = "RRTConnect";
constexpr const char* kPlannerType = "geometric::RRTConnect";
constexpr const char* kStartStatePathConstraintAdapter =
    "default_planner_request_adapters/FixStartStatePathConstraints";
constexpr const char* kDeterministicIkSearchMode =
    "deterministic_seeded_moveit_subgroup_multistart_v1";
constexpr const char* kIkCandidateSelectionMode =
    "deterministic_maximum_minimum_joint_limit_margin_v1";
constexpr double kPositionPriorityTerminalNormalizationLimitRad = 1e-5;

std::string read_text(const std::string& path)
{
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("cannot open file: " + path);
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (!stream.good() && !stream.eof())
    throw std::runtime_error("cannot read file: " + path);
  return buffer.str();
}

std::vector<double> finite_vector(const Json& value, std::size_t size, const std::string& name)
{
  if (!value.is_array() || value.size() != size)
    throw std::invalid_argument(name + " must be an array of length " + std::to_string(size));
  std::vector<double> result;
  result.reserve(size);
  for (const auto& item : value)
  {
    const double number = item.get<double>();
    if (!std::isfinite(number))
      throw std::invalid_argument(name + " contains a non-finite value");
    result.push_back(number);
  }
  return result;
}

double finite_positive(const Json& request, const char* name, double fallback)
{
  const double value = request.value(name, fallback);
  if (!std::isfinite(value) || value <= 0.0)
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
  return value;
}

Eigen::Isometry3d gym_calibration_target_to_ee(const Json& target, const std::string& name)
{
  const std::vector<double> position = finite_vector(target.at("position"), 3, name + ".position");
  const std::vector<double> quaternion =
      finite_vector(target.at("quaternion_wxyz"), 4, name + ".quaternion_wxyz");
  Eigen::Quaterniond rotation(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
  const double norm = rotation.norm();
  if (!std::isfinite(norm) || norm <= 1e-12)
    throw std::invalid_argument(name + ".quaternion_wxyz has zero norm");
  rotation.normalize();
  Eigen::Isometry3d world_from_calibration = Eigen::Isometry3d::Identity();
  world_from_calibration.linear() = rotation.toRotationMatrix();
  world_from_calibration.translation() = Eigen::Vector3d(position[0], position[1], position[2]);
  return world_from_calibration * Eigen::Translation3d(-kEeToGymCalibrationSiteXM, 0.0, 0.0);
}

Eigen::Isometry3d ee_to_gym_calibration_pose(const Eigen::Isometry3d& world_from_ee)
{
  return world_from_ee * Eigen::Translation3d(kEeToGymCalibrationSiteXM, 0.0, 0.0);
}

Json pose_json(const Eigen::Isometry3d& pose)
{
  Eigen::Quaterniond quaternion(pose.rotation());
  quaternion.normalize();
  if (quaternion.w() < 0.0)
    quaternion.coeffs() *= -1.0;
  return {
    { "position", { pose.translation().x(), pose.translation().y(), pose.translation().z() } },
    { "quaternion_wxyz", { quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z() } },
  };
}

double quaternion_distance(const Eigen::Matrix3d& first, const Eigen::Matrix3d& second)
{
  Eigen::Quaterniond left(first);
  Eigen::Quaterniond right(second);
  left.normalize();
  right.normalize();
  const double dot = std::abs(left.dot(right));
  return 2.0 * std::acos(std::clamp(dot, 0.0, 1.0));
}

class AlohaMoveItPlanner
{
public:
  AlohaMoveItPlanner(std::string urdf, std::string srdf, std::uint_fast32_t ompl_seed)
    : ompl_seed_(ompl_seed)
  {
    const rclcpp::NodeOptions options =
        rclcpp::NodeOptions().allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true);
    node_ = std::make_shared<rclcpp::Node>("rosetta_aloha_moveit_planner", options);
    const auto declare_lma_group = [this](const char* group, bool position_only) {
      // Humble's loader first checks group-local keys and then the usual
      // robot_description_kinematics namespace.  Raw-string RDF loading can
      // lose the latter prefix, so register both official lookup forms.
      const std::array<std::string, 2> roots = {
        std::string(group), std::string("robot_description_kinematics.") + group
      };
      for (const std::string& root : roots)
      {
        node_->declare_parameter(root + ".kinematics_solver",
                                 "lma_kinematics_plugin/LMAKinematicsPlugin");
        node_->declare_parameter(root + ".kinematics_solver_search_resolution", 0.005);
        node_->declare_parameter(root + ".kinematics_solver_timeout", 0.005);
        node_->declare_parameter(root + ".position_only_ik", position_only);
      }
    };
    for (const char* group : kArmGroups)
      declare_lma_group(group, false);
    for (const char* group : kPositionPriorityArmGroups)
      declare_lma_group(group, true);

    robot_model_loader::RobotModelLoader::Options loader_options(std::move(urdf), std::move(srdf));
    // The raw-string Options constructor in MoveIt 2.5.9 leaves this prefix
    // empty; set the canonical name so it reads robot_description_kinematics.
    loader_options.robot_description_ = "robot_description";
    loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(node_, loader_options);
    model_ = loader_->getModel();
    if (!model_)
      throw std::runtime_error("MoveIt failed to load the composed ALOHA robot model");
    bimanual_group_ = model_->getJointModelGroup(kBimanualGroup);
    if (!bimanual_group_)
      throw std::runtime_error("MoveIt model lacks the bimanual planning group");
    const auto validate_lma_groups = [this](const auto& group_names, const char* mode) {
      for (std::size_t index = 0; index < group_names.size(); ++index)
      {
        const char* group_name = group_names[index];
        const moveit::core::JointModelGroup* group = model_->getJointModelGroup(group_name);
        if (!group || !group->getSolverInstance())
          throw std::runtime_error(std::string("MoveIt failed to load official LMA IK for ") +
                                   group_name);
        const kinematics::KinematicsBaseConstPtr& solver = group->getSolverInstance();
        const auto normalized_frame = [](std::string frame) {
          if (!frame.empty() && frame.front() == '/')
            frame.erase(frame.begin());
          return frame;
        };
        if (normalized_frame(solver->getBaseFrame()) != kIkSolverBaseFrames[index])
          throw std::runtime_error(std::string("MoveIt LMA base frame differs for ") + group_name);
        if (normalized_frame(solver->getTipFrame()) != kEeLinks[index])
          throw std::runtime_error(std::string("MoveIt LMA tip frame differs for ") + group_name);
        if (group->getVariableNames() !=
            model_->getJointModelGroup(kArmGroups[index])->getVariableNames())
          throw std::runtime_error(std::string("MoveIt LMA joint ordering differs for ") + mode);
      }
    };
    validate_lma_groups(kArmGroups, "full_pose");
    validate_lma_groups(kPositionPriorityArmGroups, "position_priority");
    const std::vector<std::string>& model_variable_names = model_->getVariableNames();
    for (const char* joint_name : kArmJointNames)
    {
      if (std::find(model_variable_names.begin(), model_variable_names.end(), joint_name) ==
          model_variable_names.end())
        throw std::runtime_error(std::string("MoveIt model lacks expected joint variable ") + joint_name);
    }

    collision_geometry_link_count_ = model_->getLinkModelsWithCollisionGeometry().size();
    for (const moveit::core::LinkModel* link : model_->getLinkModelsWithCollisionGeometry())
      collision_geometry_shape_count_ += link->getShapes().size();
    if (collision_geometry_link_count_ == 0 || collision_geometry_shape_count_ == 0)
      throw std::runtime_error("MoveIt model loaded no collision geometry");

    scene_ = std::make_shared<planning_scene::PlanningScene>(model_);
    pipeline_ = std::make_shared<planning_pipeline::PlanningPipeline>(
        model_, node_, "ompl", "ompl_interface/OMPLPlanner",
        std::vector<std::string>{ kStartStatePathConstraintAdapter });
    pipeline_->displayComputedMotionPlans(false);
    pipeline_->publishReceivedRequests(false);
    pipeline_->checkSolutionPaths(true);

    planning_interface::PlannerConfigurationSettings default_config;
    default_config.group = kBimanualGroup;
    default_config.name = kBimanualGroup;
    default_config.config = { { "type", kPlannerType }, { "range", "0.0" } };
    planning_interface::PlannerConfigurationSettings named_config = default_config;
    named_config.name = std::string(kBimanualGroup) + "[" + kPlannerId + "]";
    planning_interface::PlannerConfigurationMap configs;
    configs.emplace(default_config.name, default_config);
    configs.emplace(named_config.name, named_config);
    pipeline_->getPlannerManager()->setPlannerConfigurations(configs);
  }

  Json identity() const
  {
    Json bounds = Json::object();
    for (const char* name : kArmJointNames)
    {
      const moveit::core::VariableBounds& limit = model_->getVariableBounds(name);
      bounds[name] = { limit.min_position_, limit.max_position_ };
    }
    return {
      { "status", "ok" },
      { "backend", "moveit2_ompl" },
      { "ros_distro", "humble" },
      { "moveit_version", ROSETTA_MOVEIT_VERSION },
      { "ompl_version", ROSETTA_OMPL_VERSION },
      { "planner_plugin", pipeline_->getPlannerPluginName() },
      { "planner_id", kPlannerId },
      { "planner_type", kPlannerType },
      { "kinematics_plugin", "lma_kinematics_plugin/LMAKinematicsPlugin" },
      { "ik_group_selection_mode", "explicit_registered_groups_v1" },
      { "full_pose_groups", kArmGroups },
      { "ik_task_modes", { "full_pose", "position_priority" } },
      { "position_priority_kinematics_plugin",
        "lma_kinematics_plugin/LMAKinematicsPlugin" },
      { "position_priority_groups", kPositionPriorityArmGroups },
      { "position_priority_orientation_weight", 0.0 },
      { "position_priority_ompl_seed_reset_per_request", true },
      { "position_priority_terminal_goal_normalization_limit_rad",
        kPositionPriorityTerminalNormalizationLimitRad },
      { "ik_search_mode", kDeterministicIkSearchMode },
      { "ik_candidate_selection_mode", kIkCandidateSelectionMode },
      { "ik_solver_base_frames", kIkSolverBaseFrames },
      { "ik_solver_tip_frames", kEeLinks },
      { "planning_group", kBimanualGroup },
      { "planning_frame", model_->getModelFrame() },
      { "joint_path_constraint_type", "moveit_msgs/JointConstraint" },
      { "planning_request_adapters", pipeline_->getAdapterPluginNames() },
      { "joint_names", kArmJointNames },
      { "joint_bounds", bounds },
      { "collision_geometry_link_count", collision_geometry_link_count_ },
      { "collision_geometry_shape_count", collision_geometry_shape_count_ },
      { "ompl_seed", ompl_seed_ },
      { "gym_calibration_site_offset_x_m", kEeToGymCalibrationSiteXM },
    };
  }

  Json forward_kinematics(const Json& request)
  {
    moveit::core::RobotState state = state_from_request(request);
    return {
      { "status", "ok" },
      { "within_bounds", state.satisfiesBounds() },
      { "self_collision", scene_->isStateColliding(state, "", false) },
      { "left", pose_json(calibration_pose(state, 0)) },
      { "right", pose_json(calibration_pose(state, 1)) },
    };
  }

  Json plan(const Json& request)
  {
    const std::string ik_task_mode = request.value("ik_task_mode", "full_pose");
    if (ik_task_mode != "full_pose" && ik_task_mode != "position_priority")
      throw std::invalid_argument("ik_task_mode must be full_pose or position_priority");
    const bool position_priority = ik_task_mode == "position_priority";
    const double allowed_planning_time_s =
        finite_positive(request, "allowed_planning_time_s", 0.25);
    const double ik_timeout_s = finite_positive(request, "ik_timeout_s", 0.025);
    const std::string ik_search_mode = request.value("ik_search_mode", "");
    if (ik_search_mode != kDeterministicIkSearchMode)
      throw std::invalid_argument("ik_search_mode differs from the deterministic registered mode");
    const std::uint64_t ik_seed_raw = request.value("ik_seed", std::uint64_t{ 0 });
    if (ik_seed_raw > std::numeric_limits<std::uint_fast32_t>::max())
      throw std::invalid_argument("ik_seed exceeds the registered integer range");
    const std::size_t ik_maximum_attempts = request.value("ik_maximum_attempts", std::size_t{ 0 });
    if (ik_maximum_attempts == 0 || ik_maximum_attempts > 4096)
      throw std::invalid_argument("ik_maximum_attempts must be within [1, 4096]");
    const std::uint_fast32_t ik_seed = static_cast<std::uint_fast32_t>(ik_seed_raw);
    const double maximum_joint_step_rad =
        finite_positive(request, "maximum_joint_step_rad", 0.23561944901923448);
    const double position_tolerance_m =
        finite_positive(request, "position_tolerance_m", 0.001);
    const double orientation_tolerance_rad =
        finite_positive(request, "orientation_tolerance_rad", 0.003);
    const double maximum_orientation_relaxation_rad =
        finite_positive(request, "maximum_orientation_relaxation_rad", 0.04);
    if (maximum_orientation_relaxation_rad > 0.04 + 1e-12)
      throw std::invalid_argument("maximum_orientation_relaxation_rad exceeds 0.04");
    const double rotation_weight = finite_positive(request, "rotation_weight", 0.2);
    const double maximum_accepted_error =
        finite_positive(request, "maximum_accepted_error", 0.001);
    const double maximum_accepted_projected_error =
        finite_positive(request, "maximum_accepted_projected_error", 0.003);
    const double start_bound_reconciliation_tolerance_rad =
        finite_positive(request, "start_bound_reconciliation_tolerance_rad", 0.00002);
    const double joint_limit_margin_rad =
        finite_positive(request, "joint_limit_margin_rad", 0.01);
    const double physical_joint_limit_margin_rad =
        finite_positive(request, "physical_joint_limit_margin_rad", 0.01);
    if (physical_joint_limit_margin_rad >= joint_limit_margin_rad)
      throw std::invalid_argument(
          "physical_joint_limit_margin_rad must be smaller than joint_limit_margin_rad");
    const bool include_trajectory = request.value("include_trajectory", false);

    moveit::core::RobotState start = state_from_request(request);
    const std::vector<double> requested_start = arm_positions(start);
    Json start_bound_violations = Json::array();
    double maximum_start_bound_violation_rad = 0.0;
    for (std::size_t index = 0; index < kArmJointNames.size(); ++index)
    {
      const char* name = kArmJointNames[index];
      const moveit::core::VariableBounds& limit = model_->getVariableBounds(name);
      if (!limit.position_bounded_)
        continue;
      const double reconciled =
          std::clamp(requested_start[index], limit.min_position_, limit.max_position_);
      const double delta = std::abs(reconciled - requested_start[index]);
      maximum_start_bound_violation_rad = std::max(maximum_start_bound_violation_rad, delta);
      if (delta > 0.0)
      {
        start_bound_violations.push_back({
          { "joint_name", name },
          { "requested_position_rad", requested_start[index] },
          { "nearest_bound_position_rad", reconciled },
          { "delta_rad", delta },
        });
      }
    }
    if (maximum_start_bound_violation_rad > start_bound_reconciliation_tolerance_rad)
    {
      return error("start_state_out_of_bounds",
                   { { "start_bound_violations", start_bound_violations },
                     { "maximum_start_bound_violation_rad", maximum_start_bound_violation_rad },
                     { "start_bound_reconciliation_tolerance_rad",
                       start_bound_reconciliation_tolerance_rad } });
    }
    start.enforceBounds(bimanual_group_);
    start.update();
    Json start_bound_reconciliations = Json::array();
    double maximum_start_bound_reconciliation_rad = 0.0;
    for (std::size_t index = 0; index < kArmJointNames.size(); ++index)
    {
      const double reconciled = start.getVariablePosition(kArmJointNames[index]);
      const double delta = std::abs(reconciled - requested_start[index]);
      maximum_start_bound_reconciliation_rad =
          std::max(maximum_start_bound_reconciliation_rad, delta);
      if (delta > 0.0)
      {
        start_bound_reconciliations.push_back({
          { "joint_name", kArmJointNames[index] },
          { "requested_position_rad", requested_start[index] },
          { "reconciled_position_rad", reconciled },
          { "delta_rad", delta },
        });
      }
    }
    const Json start_reconciliation = {
      { "start_bound_reconciliations", start_bound_reconciliations },
      { "maximum_start_bound_reconciliation_rad", maximum_start_bound_reconciliation_rad },
      { "start_bound_reconciliation_tolerance_rad", start_bound_reconciliation_tolerance_rad },
    };
    if (!start.satisfiesBounds())
      return error_with_start("start_state_out_of_bounds", start_reconciliation);
    if (scene_->isStateColliding(start, "", false))
      return error_with_start("start_state_in_collision", start_reconciliation);
    const moveit_msgs::msg::Constraints joint_path_constraints =
        make_joint_path_constraints(joint_limit_margin_rad);
    const double minimum_start_joint_limit_margin_rad = minimum_joint_limit_margin(start);
    if (minimum_start_joint_limit_margin_rad + 1e-12 < physical_joint_limit_margin_rad)
      return error_with_start(
          "start_state_outside_physical_joint_limit_margin", start_reconciliation,
          { { "physical_joint_limit_margin_rad", physical_joint_limit_margin_rad },
            { "joint_limit_margin_rad", joint_limit_margin_rad },
            { "minimum_start_joint_limit_margin_rad", minimum_start_joint_limit_margin_rad } });
    const bool start_state_satisfies_joint_path_constraint =
        scene_->isStateConstrained(start, joint_path_constraints, false);
    scene_->setCurrentState(start);

    const Json& targets = request.at("targets");
    EigenSTL::vector_Isometry3d target_poses;
    target_poses.push_back(gym_calibration_target_to_ee(targets.at("left"), "targets.left"));
    target_poses.push_back(gym_calibration_target_to_ee(targets.at("right"), "targets.right"));
    const std::vector<std::string> tips = { kEeLinks[0], kEeLinks[1] };
    moveit::core::RobotState goal(start);
    const moveit::core::GroupStateValidityCallbackFn validity =
        [this, &joint_path_constraints](moveit::core::RobotState* candidate,
                                       const moveit::core::JointModelGroup* group,
                                       const double* values) {
          candidate->setJointGroupPositions(group, values);
          candidate->update();
          return candidate->satisfiesBounds() &&
                 scene_->isStateConstrained(*candidate, joint_path_constraints, false) &&
                 !scene_->isStateColliding(*candidate, "", false);
        };
    std::vector<const moveit::core::JointModelGroup*> subgroups;
    for (std::size_t subgroup = 0; subgroup < kArmGroups.size(); ++subgroup)
    {
      const char* group_name = position_priority ?
          kPositionPriorityArmGroups[subgroup] : kArmGroups[subgroup];
      const moveit::core::JointModelGroup* group =
          model_->getJointModelGroup(group_name);
      if (!group)
        return error_with_start(
            "bimanual_registered_subgroup_missing", start_reconciliation,
            { { "missing_group", group_name } });
      subgroups.push_back(group);
    }
    if (subgroups.size() != kArmGroups.size())
      return error_with_start("bimanual_subgroup_count_differs", start_reconciliation);
    std::vector<kinematics::KinematicsBaseConstPtr> subgroup_solvers;
    subgroup_solvers.reserve(subgroups.size());
    for (std::size_t subgroup = 0; subgroup < subgroups.size(); ++subgroup)
    {
      const char* expected_group = position_priority ?
          kPositionPriorityArmGroups[subgroup] : kArmGroups[subgroup];
      if (subgroups[subgroup]->getName() != expected_group)
        return error_with_start("bimanual_subgroup_order_differs", start_reconciliation);
      const kinematics::KinematicsBaseConstPtr solver = subgroups[subgroup]->getSolverInstance();
      if (!solver)
        return error_with_start("bimanual_subgroup_solver_missing", start_reconciliation);
      subgroup_solvers.push_back(solver);
    }
    std::vector<geometry_msgs::msg::Pose> ik_queries;
    ik_queries.reserve(target_poses.size());
    for (std::size_t subgroup = 0; subgroup < target_poses.size(); ++subgroup)
    {
      Eigen::Isometry3d transformed_pose = target_poses[subgroup];
      if (!start.setToIKSolverFrame(transformed_pose, subgroup_solvers[subgroup]))
        return error_with_start("ik_solver_frame_transform_failed", start_reconciliation);
      const Eigen::Quaterniond quaternion(transformed_pose.rotation());
      geometry_msgs::msg::Pose query;
      query.position.x = transformed_pose.translation().x();
      query.position.y = transformed_pose.translation().y();
      query.position.z = transformed_pose.translation().z();
      query.orientation.x = quaternion.x();
      query.orientation.y = quaternion.y();
      query.orientation.z = quaternion.z();
      query.orientation.w = quaternion.w();
      ik_queries.push_back(query);
    }
    random_numbers::RandomNumberGenerator deterministic_rng(ik_seed);
    const auto ik_started = std::chrono::steady_clock::now();
    std::size_t ik_attempts_used = 0;
    std::size_t valid_ik_candidate_count = 0;
    std::size_t selected_ik_attempt = 0;
    double selected_ik_minimum_joint_limit_margin_rad =
        -std::numeric_limits<double>::infinity();
    double selected_ik_maximum_start_delta_rad =
        std::numeric_limits<double>::infinity();
    bool ik_found = false;
    for (std::size_t attempt = 0; attempt < ik_maximum_attempts; ++attempt)
    {
      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - ik_started).count();
      if (attempt > 0 && elapsed >= ik_timeout_s)
        break;
      ++ik_attempts_used;
      moveit::core::RobotState candidate(start);
      bool subgroup_solution_found = true;
      for (std::size_t subgroup = 0; subgroup < subgroups.size(); ++subgroup)
      {
        const moveit::core::JointModelGroup* group = subgroups[subgroup];
        if (attempt > 0)
        {
          std::vector<double> random_values;
          group->getVariableRandomPositions(deterministic_rng, random_values);
          candidate.setJointGroupPositions(group, random_values);
        }
        std::vector<double> initial_values;
        candidate.copyJointGroupPositions(group, initial_values);
        const std::vector<std::size_t>& bijection =
            group->getKinematicsSolverJointBijection();
        std::vector<double> solver_seed(bijection.size());
        for (std::size_t index = 0; index < bijection.size(); ++index)
          solver_seed[index] = initial_values[bijection[index]];
        std::vector<double> ik_solution;
        moveit_msgs::msg::MoveItErrorCodes error_code;
        if (!subgroup_solvers[subgroup]->getPositionIK(
                ik_queries[subgroup], solver_seed, ik_solution, error_code))
        {
          subgroup_solution_found = false;
          break;
        }
        std::vector<double> group_solution(bijection.size());
        for (std::size_t index = 0; index < bijection.size(); ++index)
          group_solution[bijection[index]] = ik_solution[index];
        candidate.setJointGroupPositions(group, group_solution);
      }
      if (!subgroup_solution_found)
        continue;
      candidate.update();
      std::vector<double> full_solution;
      candidate.copyJointGroupPositions(bimanual_group_, full_solution);
      if (!validity(&candidate, bimanual_group_, full_solution.data()))
        continue;
      bool candidate_task_valid = true;
      for (std::size_t arm = 0; arm < 2; ++arm)
      {
        const Eigen::Isometry3d achieved = calibration_pose(candidate, arm);
        const Eigen::Isometry3d requested =
            target_poses[arm] * Eigen::Translation3d(kEeToGymCalibrationSiteXM, 0.0, 0.0);
        const double position_error =
            (achieved.translation() - requested.translation()).norm();
        const double orientation_error =
            quaternion_distance(achieved.rotation(), target_poses[arm].rotation());
        const bool within_task_tolerance = position_priority ?
            (position_error <= position_tolerance_m &&
             orientation_error <= maximum_orientation_relaxation_rad) :
            (position_error <= position_tolerance_m &&
             orientation_error <= orientation_tolerance_rad &&
             position_error + rotation_weight * orientation_error <= maximum_accepted_error &&
             position_error + rotation_weight * orientation_error <=
                 maximum_accepted_projected_error);
        if (!within_task_tolerance)
        {
          candidate_task_valid = false;
          break;
        }
      }
      if (!candidate_task_valid)
        continue;
      ++valid_ik_candidate_count;
      const double candidate_margin = minimum_joint_limit_margin(candidate);
      const std::vector<double> candidate_positions = arm_positions(candidate);
      double candidate_maximum_start_delta_rad = 0.0;
      for (std::size_t index = 0; index < candidate_positions.size(); ++index)
        candidate_maximum_start_delta_rad = std::max(
            candidate_maximum_start_delta_rad,
            std::abs(candidate_positions[index] - requested_start[index]));
      const bool margin_better =
          candidate_margin > selected_ik_minimum_joint_limit_margin_rad + 1e-12;
      const bool margin_tied =
          std::abs(candidate_margin - selected_ik_minimum_joint_limit_margin_rad) <= 1e-12;
      const bool start_delta_better =
          candidate_maximum_start_delta_rad < selected_ik_maximum_start_delta_rad - 1e-12;
      if (!ik_found || margin_better || (margin_tied && start_delta_better))
      {
        goal = candidate;
        selected_ik_attempt = ik_attempts_used;
        selected_ik_minimum_joint_limit_margin_rad = candidate_margin;
        selected_ik_maximum_start_delta_rad = candidate_maximum_start_delta_rad;
        ik_found = true;
      }
    }
    goal.update();
    if (!ik_found)
      return error_with_start(
          position_priority ? "bimanual_position_priority_lma_ik_failed" :
                              "bimanual_lma_ik_failed",
          start_reconciliation,
          { { "ik_task_mode", ik_task_mode }, { "ik_search_mode", ik_search_mode },
            { "ik_candidate_selection_mode", kIkCandidateSelectionMode },
            { "ik_seed", ik_seed },
            { "ik_maximum_attempts", ik_maximum_attempts },
            { "ik_attempts_used", ik_attempts_used },
            { "valid_ik_candidate_count", valid_ik_candidate_count },
            { "ik_outer_timeout_s", ik_timeout_s },
            { "maximum_orientation_relaxation_rad",
              maximum_orientation_relaxation_rad } });
    if (!goal.satisfiesBounds())
      return error_with_start("ik_goal_out_of_bounds", start_reconciliation);
    if (!scene_->isStateConstrained(goal, joint_path_constraints, false))
      return error_with_start(
          "ik_goal_outside_joint_path_margin", start_reconciliation,
          { { "joint_limit_margin_rad", joint_limit_margin_rad },
            { "minimum_goal_joint_limit_margin_rad", minimum_joint_limit_margin(goal) } });
    if (scene_->isStateColliding(goal, "", false))
      return error_with_start("ik_goal_in_collision", start_reconciliation);

    Json goal_errors = Json::object();
    double maximum_position_error_m = 0.0;
    double maximum_orientation_error_rad = 0.0;
    double maximum_weighted_error = 0.0;
    for (std::size_t arm = 0; arm < 2; ++arm)
    {
      const Eigen::Isometry3d achieved = calibration_pose(goal, arm);
      const double position_error = (achieved.translation() -
                                     (target_poses[arm] * Eigen::Translation3d(
                                                              kEeToGymCalibrationSiteXM, 0.0, 0.0))
                                         .translation())
                                        .norm();
      const double orientation_error =
          quaternion_distance(achieved.rotation(), target_poses[arm].rotation());
      maximum_position_error_m = std::max(maximum_position_error_m, position_error);
      maximum_orientation_error_rad = std::max(maximum_orientation_error_rad, orientation_error);
      const double weighted_error = position_error + rotation_weight * orientation_error;
      maximum_weighted_error = std::max(maximum_weighted_error, weighted_error);
      goal_errors[arm == 0 ? "left" : "right"] = {
        { "position_m", position_error },
        { "orientation_rad", orientation_error },
        { "weighted", weighted_error },
      };
    }
    const bool goal_exceeds_tolerance = position_priority ?
        (maximum_position_error_m > position_tolerance_m ||
         maximum_orientation_error_rad > maximum_orientation_relaxation_rad) :
        (maximum_position_error_m > position_tolerance_m ||
         maximum_orientation_error_rad > orientation_tolerance_rad ||
         maximum_weighted_error > maximum_accepted_error ||
         maximum_weighted_error > maximum_accepted_projected_error);
    if (goal_exceeds_tolerance)
      return error_with_start("ik_goal_exceeds_registered_tolerance", start_reconciliation,
                              { { "ik_task_mode", ik_task_mode },
                                { "goal_errors", goal_errors },
                                { "maximum_orientation_relaxation_rad",
                                  maximum_orientation_relaxation_rad } });

    planning_interface::MotionPlanRequest planning_request;
    planning_request.group_name = kBimanualGroup;
    planning_request.planner_id = kPlannerId;
    planning_request.num_planning_attempts = 1;
    planning_request.allowed_planning_time = allowed_planning_time_s;
    planning_request.max_velocity_scaling_factor = 1.0;
    planning_request.max_acceleration_scaling_factor = 1.0;
    moveit::core::robotStateToRobotStateMsg(start, planning_request.start_state);
    planning_request.goal_constraints.push_back(
        kinematic_constraints::constructGoalConstraints(goal, bimanual_group_, 1e-6));
    planning_request.path_constraints = joint_path_constraints;

    planning_interface::MotionPlanResponse response;
    std::vector<std::size_t> adapter_added_state_indices;
    if (position_priority)
      ompl::RNG::setSeed(ompl_seed_);
    const bool planned = pipeline_->generatePlan(
        scene_, planning_request, response, adapter_added_state_indices);
    if (!planned || response.error_code_.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS ||
        !response.trajectory_)
    {
      return error_with_start("ompl_planning_failed", start_reconciliation,
                              { { "moveit_error_code", response.error_code_.val },
                                { "planning_time_s", response.planning_time_ } });
    }
    std::unique_ptr<robot_trajectory::RobotTrajectory> normalized_trajectory;
    double maximum_terminal_goal_normalization_rad = 0.0;
    if (position_priority)
    {
      normalized_trajectory = std::make_unique<robot_trajectory::RobotTrajectory>(
          *response.trajectory_, true);
      const std::vector<double> planned_terminal =
          arm_positions(normalized_trajectory->getLastWayPoint());
      const std::vector<double> exact_goal = arm_positions(goal);
      for (std::size_t index = 0; index < exact_goal.size(); ++index)
        maximum_terminal_goal_normalization_rad = std::max(
            maximum_terminal_goal_normalization_rad,
            std::abs(exact_goal[index] - planned_terminal[index]));
      if (maximum_terminal_goal_normalization_rad >
          kPositionPriorityTerminalNormalizationLimitRad)
        return error_with_start(
            "position_priority_terminal_goal_normalization_exceeds_limit",
            start_reconciliation,
            { { "maximum_terminal_goal_normalization_rad",
                maximum_terminal_goal_normalization_rad },
              { "terminal_goal_normalization_limit_rad",
                kPositionPriorityTerminalNormalizationLimitRad } });
      normalized_trajectory->getLastWayPointPtr() =
          std::make_shared<moveit::core::RobotState>(goal);
      if (!scene_->isPathValid(
              *normalized_trajectory, planning_request.path_constraints,
              planning_request.goal_constraints, kBimanualGroup, false))
        return error_with_start(
            "position_priority_normalized_path_invalid", start_reconciliation);
    }
    const robot_trajectory::RobotTrajectory& trajectory =
        normalized_trajectory ? *normalized_trajectory : *response.trajectory_;
    if (trajectory.getWayPointCount() < 2)
      return error_with_start("ompl_returned_no_motion", start_reconciliation);
    for (std::size_t index = 0; index < adapter_added_state_indices.size(); ++index)
    {
      if (adapter_added_state_indices[index] != index ||
          adapter_added_state_indices[index] >= trajectory.getWayPointCount())
        return error_with_start(
            "start_state_path_constraint_adapter_indices_invalid", start_reconciliation,
            { { "adapter_added_state_indices", adapter_added_state_indices } });
    }
    const std::size_t adapter_prefix_waypoint_count = adapter_added_state_indices.size();
    const bool start_state_path_constraint_recovery = adapter_prefix_waypoint_count > 0;
    if (start_state_satisfies_joint_path_constraint == start_state_path_constraint_recovery)
      return error_with_start(
          "start_state_path_constraint_adapter_activation_mismatch", start_reconciliation,
          { { "start_state_satisfies_joint_path_constraint",
              start_state_satisfies_joint_path_constraint },
            { "adapter_added_state_indices", adapter_added_state_indices } });

    Json trajectory_positions = Json::array();
    double path_length_rad = 0.0;
    double path_maximum_joint_delta_rad = 0.0;
    double minimum_path_joint_limit_margin_rad = std::numeric_limits<double>::infinity();
    double minimum_constrained_path_joint_limit_margin_rad =
        std::numeric_limits<double>::infinity();
    double minimum_adapter_prefix_physical_joint_limit_margin_rad =
        std::numeric_limits<double>::infinity();
    std::vector<std::size_t> recovery_joint_indices;
    std::vector<double> previous_recovery_joint_margins;
    if (start_state_path_constraint_recovery)
    {
      for (std::size_t index = 0; index < kArmJointNames.size(); ++index)
      {
        const moveit::core::VariableBounds& limit =
            model_->getVariableBounds(kArmJointNames[index]);
        const double position = start.getVariablePosition(kArmJointNames[index]);
        const double margin =
            std::min(position - limit.min_position_, limit.max_position_ - position);
        if (margin + 1e-12 < joint_limit_margin_rad)
        {
          recovery_joint_indices.push_back(index);
          previous_recovery_joint_margins.push_back(margin);
        }
      }
      if (recovery_joint_indices.empty())
        return error_with_start(
            "start_state_path_constraint_recovery_has_no_violating_joint",
            start_reconciliation);
    }
    std::vector<double> previous = arm_positions(trajectory.getWayPoint(0));
    if (include_trajectory)
      trajectory_positions.push_back(previous);
    for (std::size_t index = 0; index < trajectory.getWayPointCount(); ++index)
    {
      const moveit::core::RobotState& waypoint = trajectory.getWayPoint(index);
      if (!waypoint.satisfiesBounds() || scene_->isStateColliding(waypoint, "", false))
        return error_with_start("ompl_returned_invalid_waypoint", start_reconciliation,
                                { { "waypoint_index", index } });
      const double waypoint_joint_limit_margin_rad = minimum_joint_limit_margin(waypoint);
      minimum_path_joint_limit_margin_rad =
          std::min(minimum_path_joint_limit_margin_rad, waypoint_joint_limit_margin_rad);
      if (waypoint_joint_limit_margin_rad + 1e-12 < physical_joint_limit_margin_rad)
        return error_with_start(
            "ompl_returned_physical_joint_limit_margin_violation", start_reconciliation,
            { { "waypoint_index", index },
              { "physical_joint_limit_margin_rad", physical_joint_limit_margin_rad },
              { "waypoint_joint_limit_margin_rad", waypoint_joint_limit_margin_rad } });
      const bool adapter_prefix_waypoint = index < adapter_prefix_waypoint_count;
      if (adapter_prefix_waypoint)
      {
        minimum_adapter_prefix_physical_joint_limit_margin_rad =
            std::min(minimum_adapter_prefix_physical_joint_limit_margin_rad,
                     waypoint_joint_limit_margin_rad);
        for (std::size_t recovery_index = 0;
             recovery_index < recovery_joint_indices.size(); ++recovery_index)
        {
          const std::size_t joint_index = recovery_joint_indices[recovery_index];
          const char* joint_name = kArmJointNames[joint_index];
          const moveit::core::VariableBounds& limit = model_->getVariableBounds(joint_name);
          const double position = waypoint.getVariablePosition(joint_name);
          const double margin =
              std::min(position - limit.min_position_, limit.max_position_ - position);
          const double previous_margin = previous_recovery_joint_margins[recovery_index];
          const bool previous_below_command_margin =
              previous_margin + 1e-9 < joint_limit_margin_rad;
          const bool waypoint_below_command_margin =
              margin + 1e-9 < joint_limit_margin_rad;
          if ((previous_below_command_margin && margin + 1e-9 < previous_margin) ||
              (!previous_below_command_margin && waypoint_below_command_margin))
            return error_with_start(
                "start_state_path_constraint_recovery_not_monotonic",
                start_reconciliation,
                { { "waypoint_index", index }, { "joint_name", joint_name },
                  { "previous_margin_rad", previous_margin },
                  { "waypoint_margin_rad", margin } });
          previous_recovery_joint_margins[recovery_index] = margin;
        }
      }
      else
      {
        minimum_constrained_path_joint_limit_margin_rad =
            std::min(minimum_constrained_path_joint_limit_margin_rad,
                     waypoint_joint_limit_margin_rad);
      }
      if (!adapter_prefix_waypoint &&
          !scene_->isStateConstrained(waypoint, planning_request.path_constraints, false))
        return error_with_start(
            "ompl_returned_joint_path_constraint_violation", start_reconciliation,
            { { "waypoint_index", index },
              { "joint_limit_margin_rad", joint_limit_margin_rad },
              { "waypoint_joint_limit_margin_rad", waypoint_joint_limit_margin_rad } });
      if (index == 0)
        continue;
      const std::vector<double> current = arm_positions(waypoint);
      double squared = 0.0;
      for (std::size_t joint = 0; joint < current.size(); ++joint)
      {
        const double delta = current[joint] - previous[joint];
        squared += delta * delta;
        path_maximum_joint_delta_rad = std::max(path_maximum_joint_delta_rad, std::abs(delta));
      }
      path_length_rad += std::sqrt(squared);
      previous = current;
      if (include_trajectory)
        trajectory_positions.push_back(current);
    }
    if (start_state_path_constraint_recovery)
    {
      const moveit::core::RobotState& recovery_goal =
          trajectory.getWayPoint(adapter_prefix_waypoint_count - 1);
      if (!scene_->isStateConstrained(
              recovery_goal, planning_request.path_constraints, false))
        return error_with_start(
            "start_state_path_constraint_recovery_goal_outside_margin",
            start_reconciliation);
    }
    else
    {
      minimum_adapter_prefix_physical_joint_limit_margin_rad =
          minimum_start_joint_limit_margin_rad;
    }
    if (!std::isfinite(minimum_constrained_path_joint_limit_margin_rad))
      return error_with_start(
          "ompl_returned_no_constrained_path_waypoint", start_reconciliation);

    const moveit::core::RobotState& first = trajectory.getWayPoint(0);
    const moveit::core::RobotState& second = trajectory.getWayPoint(1);
    const std::vector<double> first_positions = arm_positions(first);
    const std::vector<double> second_positions = arm_positions(second);
    double first_segment_maximum_delta = 0.0;
    for (std::size_t joint = 0; joint < second_positions.size(); ++joint)
      first_segment_maximum_delta =
          std::max(first_segment_maximum_delta,
                   std::abs(second_positions[joint] - first_positions[joint]));
    double interpolation =
        first_segment_maximum_delta <= maximum_joint_step_rad ?
            1.0 :
            maximum_joint_step_rad / first_segment_maximum_delta;
    const auto requested_start_to_interpolated_maximum_delta =
        [&first_positions, &second_positions, &requested_start](double fraction) {
          double maximum = 0.0;
          for (std::size_t joint = 0; joint < second_positions.size(); ++joint)
          {
            const double position =
                first_positions[joint] +
                fraction * (second_positions[joint] - first_positions[joint]);
            maximum = std::max(maximum, std::abs(position - requested_start[joint]));
          }
          return maximum;
        };
    if (requested_start_to_interpolated_maximum_delta(interpolation) > maximum_joint_step_rad)
    {
      double feasible = 0.0;
      double infeasible = interpolation;
      for (std::size_t iteration = 0; iteration < 60; ++iteration)
      {
        const double candidate = 0.5 * (feasible + infeasible);
        if (requested_start_to_interpolated_maximum_delta(candidate) <= maximum_joint_step_rad)
          feasible = candidate;
        else
          infeasible = candidate;
      }
      interpolation = feasible;
    }
    const double maximum_requested_start_to_next_joint_delta_rad =
        requested_start_to_interpolated_maximum_delta(interpolation);
    moveit::core::RobotState next(first);
    first.interpolate(second, interpolation, next, bimanual_group_);
    next.update();
    if (!next.satisfiesBounds() || scene_->isStateColliding(next, "", false))
      return error_with_start("bounded_execution_waypoint_invalid", start_reconciliation);
    const double minimum_next_joint_limit_margin_rad = minimum_joint_limit_margin(next);
    if (minimum_next_joint_limit_margin_rad + 1e-12 < physical_joint_limit_margin_rad)
      return error_with_start(
          "bounded_execution_waypoint_violates_physical_joint_limit_margin",
          start_reconciliation,
          { { "physical_joint_limit_margin_rad", physical_joint_limit_margin_rad },
            { "minimum_next_joint_limit_margin_rad", minimum_next_joint_limit_margin_rad } });
    if (!start_state_path_constraint_recovery &&
        !scene_->isStateConstrained(next, planning_request.path_constraints, false))
      return error_with_start(
          "bounded_execution_waypoint_violates_joint_path_constraint", start_reconciliation,
          { { "joint_limit_margin_rad", joint_limit_margin_rad },
              { "minimum_next_joint_limit_margin_rad", minimum_next_joint_limit_margin_rad } });
    double minimum_recovery_progress_rad = 0.0;
    if (start_state_path_constraint_recovery)
    {
      minimum_recovery_progress_rad = std::numeric_limits<double>::infinity();
      for (std::size_t recovery_index = 0;
           recovery_index < recovery_joint_indices.size(); ++recovery_index)
      {
        const char* joint_name = kArmJointNames[recovery_joint_indices[recovery_index]];
        const moveit::core::VariableBounds& limit = model_->getVariableBounds(joint_name);
        const double start_position = start.getVariablePosition(joint_name);
        const double next_position = next.getVariablePosition(joint_name);
        const double start_margin = std::min(start_position - limit.min_position_,
                                             limit.max_position_ - start_position);
        const double next_margin = std::min(next_position - limit.min_position_,
                                            limit.max_position_ - next_position);
        minimum_recovery_progress_rad =
            std::min(minimum_recovery_progress_rad, next_margin - start_margin);
      }
      if (!std::isfinite(minimum_recovery_progress_rad) ||
          minimum_recovery_progress_rad <= 0.0)
        return error_with_start(
            "bounded_recovery_waypoint_has_no_positive_margin_progress",
            start_reconciliation,
            { { "minimum_recovery_progress_rad", minimum_recovery_progress_rad } });
    }

    Json result = {
      { "status", "ok" },
      { "backend", "moveit2_ompl" },
      { "planner_plugin", pipeline_->getPlannerPluginName() },
      { "planner_id", kPlannerId },
      { "joint_names", kArmJointNames },
      { "ik_task_mode", ik_task_mode },
      { "maximum_orientation_relaxation_rad", maximum_orientation_relaxation_rad },
      { "position_priority_ompl_seed_reset_per_request", position_priority },
      { "position_priority_terminal_goal_normalized", position_priority },
      { "maximum_terminal_goal_normalization_rad",
        maximum_terminal_goal_normalization_rad },
      { "terminal_goal_normalization_limit_rad",
        kPositionPriorityTerminalNormalizationLimitRad },
      { "ik_search_mode", ik_search_mode },
      { "ik_candidate_selection_mode", kIkCandidateSelectionMode },
      { "ik_seed", ik_seed },
      { "ik_maximum_attempts", ik_maximum_attempts },
      { "ik_attempts_used", ik_attempts_used },
      { "valid_ik_candidate_count", valid_ik_candidate_count },
      { "selected_ik_attempt", selected_ik_attempt },
      { "selected_ik_minimum_joint_limit_margin_rad",
        selected_ik_minimum_joint_limit_margin_rad },
      { "selected_ik_maximum_start_delta_rad",
        selected_ik_maximum_start_delta_rad },
      { "ik_outer_timeout_s", ik_timeout_s },
      { "goal", arm_positions(goal) },
      { "next", arm_positions(next) },
      { "goal_errors", goal_errors },
      { "maximum_goal_position_error_m", maximum_position_error_m },
      { "maximum_goal_orientation_error_rad", maximum_orientation_error_rad },
      { "maximum_goal_weighted_error", maximum_weighted_error },
      { "waypoint_count", trajectory.getWayPointCount() },
      { "planning_time_s", response.planning_time_ },
      { "path_length_rad", path_length_rad },
      { "path_maximum_waypoint_joint_delta_rad", path_maximum_joint_delta_rad },
      { "first_segment_interpolation", interpolation },
      { "maximum_joint_step_rad", maximum_joint_step_rad },
      { "maximum_requested_start_to_next_joint_delta_rad",
        maximum_requested_start_to_next_joint_delta_rad },
      { "joint_path_constraint_type", "moveit_msgs/JointConstraint" },
      { "joint_path_constraint_count", kArmJointNames.size() },
      { "joint_limit_margin_rad", joint_limit_margin_rad },
      { "physical_joint_limit_margin_rad", physical_joint_limit_margin_rad },
      { "start_state_satisfies_joint_path_constraint",
        start_state_satisfies_joint_path_constraint },
      { "start_state_path_constraint_recovery", start_state_path_constraint_recovery },
      { "planning_request_adapters", pipeline_->getAdapterPluginNames() },
      { "adapter_added_state_indices", adapter_added_state_indices },
      { "adapter_prefix_waypoint_count", adapter_prefix_waypoint_count },
      { "minimum_recovery_progress_rad", minimum_recovery_progress_rad },
      { "minimum_start_joint_limit_margin_rad", minimum_start_joint_limit_margin_rad },
      { "minimum_goal_joint_limit_margin_rad", minimum_joint_limit_margin(goal) },
      { "minimum_path_joint_limit_margin_rad", minimum_path_joint_limit_margin_rad },
      { "minimum_constrained_path_joint_limit_margin_rad",
        minimum_constrained_path_joint_limit_margin_rad },
      { "minimum_adapter_prefix_physical_joint_limit_margin_rad",
        minimum_adapter_prefix_physical_joint_limit_margin_rad },
      { "minimum_next_joint_limit_margin_rad", minimum_next_joint_limit_margin_rad },
      { "start_bound_reconciliations", start_bound_reconciliations },
      { "maximum_start_bound_reconciliation_rad", maximum_start_bound_reconciliation_rad },
      { "start_bound_reconciliation_tolerance_rad", start_bound_reconciliation_tolerance_rad },
    };
    if (include_trajectory)
      result["trajectory"] = std::move(trajectory_positions);
    return result;
  }

private:
  moveit::core::RobotState state_from_request(const Json& request) const
  {
    const std::vector<double> positions = finite_vector(request.at("start"), 12, "start");
    const std::vector<double> fingers =
        finite_vector(request.value("finger_positions", Json::array({ 0.024, 0.024 })), 2,
                      "finger_positions");
    moveit::core::RobotState state(model_);
    state.setToDefaultValues();
    for (std::size_t index = 0; index < kArmJointNames.size(); ++index)
      state.setVariablePosition(kArmJointNames[index], positions[index]);
    state.setVariablePosition("left_left_finger", fingers[0]);
    state.setVariablePosition("right_left_finger", fingers[1]);
    state.update();
    return state;
  }

  std::vector<double> arm_positions(const moveit::core::RobotState& state) const
  {
    std::vector<double> result;
    result.reserve(kArmJointNames.size());
    for (const char* name : kArmJointNames)
      result.push_back(state.getVariablePosition(name));
    return result;
  }

  moveit_msgs::msg::Constraints make_joint_path_constraints(double margin_rad) const
  {
    moveit_msgs::msg::Constraints constraints;
    constraints.name = "rosetta_arm_joint_limit_margin";
    for (const char* name : kArmJointNames)
    {
      const moveit::core::VariableBounds& limit = model_->getVariableBounds(name);
      if (!limit.position_bounded_)
        throw std::invalid_argument(std::string("arm joint is not position bounded: ") + name);
      const double constrained_minimum = limit.min_position_ + margin_rad;
      const double constrained_maximum = limit.max_position_ - margin_rad;
      if (!std::isfinite(constrained_minimum) || !std::isfinite(constrained_maximum) ||
          constrained_minimum >= constrained_maximum)
        throw std::invalid_argument(std::string("joint-limit margin empties range for ") + name);
      moveit_msgs::msg::JointConstraint constraint;
      constraint.joint_name = name;
      constraint.position = 0.5 * (constrained_minimum + constrained_maximum);
      constraint.tolerance_below = constraint.position - constrained_minimum;
      constraint.tolerance_above = constrained_maximum - constraint.position;
      constraint.weight = 1.0;
      constraints.joint_constraints.push_back(std::move(constraint));
    }
    return constraints;
  }

  double minimum_joint_limit_margin(const moveit::core::RobotState& state) const
  {
    double minimum = std::numeric_limits<double>::infinity();
    for (const char* name : kArmJointNames)
    {
      const moveit::core::VariableBounds& limit = model_->getVariableBounds(name);
      if (!limit.position_bounded_)
        throw std::runtime_error(std::string("arm joint is not position bounded: ") + name);
      const double position = state.getVariablePosition(name);
      minimum = std::min(
          minimum, std::min(position - limit.min_position_, limit.max_position_ - position));
    }
    return minimum;
  }

  Eigen::Isometry3d calibration_pose(const moveit::core::RobotState& state, std::size_t arm) const
  {
    return ee_to_gym_calibration_pose(state.getGlobalLinkTransform(kEeLinks.at(arm)));
  }

  Json error(const std::string& reason, Json details = Json::object()) const
  {
    Json result = { { "status", "error" }, { "reason", reason } };
    for (auto& item : details.items())
      result[item.key()] = std::move(item.value());
    return result;
  }

  Json error_with_start(const std::string& reason, const Json& start_reconciliation,
                        Json details = Json::object()) const
  {
    for (const auto& item : start_reconciliation.items())
      details[item.key()] = item.value();
    return error(reason, std::move(details));
  }

  std::uint_fast32_t ompl_seed_;
  rclcpp::Node::SharedPtr node_;
  robot_model_loader::RobotModelLoaderPtr loader_;
  moveit::core::RobotModelPtr model_;
  const moveit::core::JointModelGroup* bimanual_group_ = nullptr;
  std::size_t collision_geometry_link_count_ = 0;
  std::size_t collision_geometry_shape_count_ = 0;
  planning_scene::PlanningScenePtr scene_;
  planning_pipeline::PlanningPipelinePtr pipeline_;
};

void emit(const Json& payload)
{
  std::cout << payload.dump() << std::endl;
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3 || argc > 4)
  {
    std::cerr << "usage: aloha_moveit_planner COMPOSED_URDF COMPOSED_SRDF [OMPL_SEED]\n";
    return 2;
  }
  try
  {
    const auto seed = argc == 4 ? static_cast<std::uint_fast32_t>(std::stoul(argv[3])) : 2210U;
    ompl::RNG::setSeed(seed);
    rclcpp::init(0, nullptr);
    AlohaMoveItPlanner planner(read_text(argv[1]), read_text(argv[2]), seed);
    std::string line;
    while (std::getline(std::cin, line))
    {
      if (line.empty())
        continue;
      try
      {
        const Json request = Json::parse(line);
        const std::string command = request.at("command").get<std::string>();
        Json response;
        bool should_shutdown = false;
        if (command == "identity")
          response = planner.identity();
        else if (command == "fk")
          response = planner.forward_kinematics(request);
        else if (command == "plan")
          response = planner.plan(request);
        else if (command == "shutdown")
        {
          response = { { "status", "ok" }, { "shutdown", true } };
          should_shutdown = true;
        }
        else
          response = { { "status", "error" }, { "reason", "unknown_command" } };
        if (request.contains("request_id"))
          response["request_id"] = request.at("request_id");
        emit(response);
        if (should_shutdown)
          break;
      }
      catch (const std::exception& error)
      {
        emit({ { "status", "error" }, { "reason", "invalid_request" }, { "detail", error.what() } });
      }
    }
    rclcpp::shutdown();
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "aloha_moveit_planner startup failed: " << error.what() << '\n';
    if (rclcpp::ok())
      rclcpp::shutdown();
    return 1;
  }
}
