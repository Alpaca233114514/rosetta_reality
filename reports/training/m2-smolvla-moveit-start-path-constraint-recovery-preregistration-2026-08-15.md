# M2 SmolVLA MoveIt start-path-constraint recovery preregistration (2026-08-15)

## Decision

Plan `029` is locally preregistered and ready for a future AutoDL train-only
exact run. No exact, tuning, development, collection, policy Gate, validation,
hidden-test, label-writing, model-download, dataset-download or CUDA-training
stage was run while preparing it.

The only research-axis change from plan `028` is the official MoveIt 2
`default_planner_request_adapters/FixStartStatePathConstraints` adapter. When a
start state is still outside the physical `0.01`-rad joint margin but has left
the tightened `0.04540462255477905`-rad command set, the adapter may prepend a
collision-checked path that returns to the existing `moveit_msgs/JointConstraint`.
Every prefix waypoint must stay physically safe and monotonically improve each
violating joint margin; every non-prefix waypoint must satisfy the tightened
margin. The first command must make positive margin progress. Starts inside the
physical margin fail closed.

## Why this is upstream rather than a custom planner

MoveIt documents `FixStartStatePathConstraints` as a planning-request adapter
for the case where a valid start state violates path constraints: it first plans
to a state satisfying those constraints and then continues the original plan.
The sidecar loads that plugin through the official
`planning_pipeline::PlanningPipeline`, keeps upstream
`ompl_interface/OMPLPlanner` / `RRTConnect`, and records the adapter-added state
indices returned by the official pipeline.

Primary references:

- <https://moveit.picknik.ai/humble/doc/concepts/motion_planning.html>
- <https://moveit.picknik.ai/humble/api/html/classdefault__planner__request__adapters_1_1FixStartStatePathConstraints.html>
- <https://moveit.picknik.ai/humble/api/html/fix__start__state__path__constraints_8cpp_source.html>
- <https://moveit.picknik.ai/humble/api/html/classplanning__pipeline_1_1PlanningPipeline.html>

## Frozen identity

- Plan: `configs/sim/aloha_insertion_geometry_teacher_029.yaml`
- Plan SHA-256: `4fc4bff41966022591057bf19c9246b5f511ef3009fde66b1e87a59aca70cde0`
- Runtime image:
  `rosetta-reality-aloha-moveit2:humble-2.5.9-start-path-constraints-001`
- Runtime image ID:
  `sha256:9b8903623be1b75263a24042aa801723339bbdc08b670fbe4caf7d453dffedeb`
- Sidecar executable SHA-256:
  `3e89d68b92365ed3391101f20ed0e79aa13a08e0675ca53b4151cce672cec5da`
- Plan `028` source audit SHA-256:
  `c06412aba0d10fc0c7a21c04ece5352e2f1ff85d692a701d119a73d0cdcf6829`
- Plan `028` exact-report SHA-256:
  `0848857b401a10635cd7d66da4dd7cd339f8417abd9982141669c01a05ccbb64`

## Local evidence

The first FK attempt stopped before planner launch because Bash nounset was
active while sourcing ROS's official setup script. That attempt remains
preserved. Attempt `002` used the setup script's supported shell boundary and
passed:

- `47` relevant pytest cases;
- Ruff on the evaluator, client and protocol tests;
- official-image C++ compilation and plugin loading;
- five-sample MoveIt/Gym FK parity, maximum position error
  `3.188872858294072e-16` m and maximum orientation error `0` rad;
- normal planning: no adapter prefix, `8` waypoints, constrained-path minimum
  margin `0.25577631244600063` rad;
- recovery planning: `29` adapter-prefix waypoints and `59` total waypoints,
  prefix minimum physical margin `0.01998734641020672` rad, constrained suffix
  minimum `0.17818474371155113` rad, next-command margin
  `0.04194725447779968` rad and positive recovery progress
  `0.02195990806759296` rad;
- physical-negative planning: a start with only `0.00658000000000003` rad
  margin was rejected as `start_state_outside_physical_joint_limit_margin`.

The bound parity report is
`reports/training/m2-smolvla-aloha-moveit-model-parity-plan029-2026-08-15.json`
with SHA-256
`ed8110a68729813ea0c8e7784b5e5439a643af214a28208dc13436cc6b6788d9`.

## Next gate

The next permitted action is one future AutoDL run of calibration plus
train-only exact episode `2`, simulator seed `10`, from a new content-addressed
workspace after both independent shutdown watchdogs are installed and verified
across reconnect. Exact must pass before tuning seed `1900` can open. All other
later seeds and recovery labels remain sealed.
