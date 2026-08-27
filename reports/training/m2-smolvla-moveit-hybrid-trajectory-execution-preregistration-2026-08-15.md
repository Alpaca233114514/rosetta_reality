# M2 SmolVLA MoveIt hybrid trajectory execution preregistration

Status: **locally implemented and preregistered; no Plan031 remote exact has run**.

## Finding

Plan030 did not fail because OMPL could not find a safe path. It accepted 131
official RRTConnect paths, kept the registered physical and tightened joint
margins, and still exhausted 500 steps in `orient`. The evaluator used only the
bounded first command of each accepted path. On the next fallback step it asked
LMA plus RRTConnect for a new goal configuration, discarding the remaining
trajectory. The trace consequently alternated among redundant IK branches
instead of executing one accepted route to completion.

## Official execution model

MoveIt 2.5.9 Hybrid Planning already defines the needed architecture:

- `SimpleSampler` retains the global reference trajectory, advances at most one
  waypoint when the current joint state is within its official `0.2` rad L1
  tolerance, and otherwise continues to forward the same waypoint;
- `ForwardTrajectory` forwards the selected local waypoint to the controller
  and can stop on an invalid local trajectory;
- ordinary MoveIt execution similarly sends a complete planned trajectory to a
  trajectory controller instead of solving a new global plan for every control
  tick.

Plan031 ports those semantics into the existing process boundary. It does not
replace MoveIt or introduce a new search algorithm. The official OMPL
RRTConnect path, LMA goal IK, `FixStartStatePathConstraints` recovery prefix and
all existing C++ path validation remain authoritative.

Upstream sources:

- [Hybrid Planning concepts](https://moveit.picknik.ai/humble/doc/concepts/hybrid_planning/hybrid_planning.html)
- [MoveIt 2.5.9 Hybrid Planning plugins](https://github.com/moveit/moveit2/tree/2.5.9/moveit_ros/hybrid_planning)
- [SimpleSampler source](https://github.com/moveit/moveit2/blob/2.5.9/moveit_ros/hybrid_planning/local_planner/trajectory_operator_plugins/src/simple_sampler.cpp)
- [ForwardTrajectory source](https://github.com/moveit/moveit2/blob/2.5.9/moveit_ros/hybrid_planning/local_planner/local_constraint_solver_plugins/src/forward_trajectory.cpp)
- [TrajectoryExecutionManager API](https://moveit.picknik.ai/humble/api/html/classtrajectory__execution__manager_1_1TrajectoryExecutionManager.html)

The upstream plugin code is BSD-3-Clause. The Rosetta implementation mirrors
its small state machine and names the source/tag in the registered plan; it does
not vendor or silently fork the MoveIt planner.

## Registered single axis

`configs/sim/aloha_insertion_geometry_teacher_031.yaml` freezes one change:
retain and follow the complete accepted global trajectory until the teacher
changes phase. The reference is not replaced on every fallback step.

Frozen unchanged:

- exact episode 2 / simulator seed 10 and every later sealed seed group;
- 1 mm / 3 mm solver gates and every teacher pose gate;
- physical `0.01` rad and command `0.04540462255477905` rad margins;
- `0.23561944901923448` rad maximum joint target delta;
- OMPL seed 2210, RRTConnect, LMA, official start-state recovery and collision
  resources;
- no labels, hidden/validation episodes, optimizer steps, CUDA training, model
  download or data download.

## Local evidence

- Ruff passed.
- 57 focused tests passed in the network-disabled
  `rosetta-reality-smolvla-sim-xpu:mink-1.2.0` Linux container.
- The pinned official MoveIt image returned complete trajectories with
  `include_trajectory=true`: normal path 8 waypoints / maximum waypoint joint
  delta `0.0993591379230041` rad; recovery path 41 waypoints / maximum delta
  `0.11261942148813398` rad. Both remain below the frozen command delta.
- The actual Python client parsed and validated the pinned sidecar's complete
  trajectory and sampled its next retained waypoint successfully.
- `git diff --check` passed.

This is local preregistration evidence, not an exact-control pass. The next
remote action, only after the user explicitly opens SSH, is one new
content-addressed workspace followed by static reproduction and Plan031
train-only exact. Tuning, development, collection, policy-Gate, validation,
hidden and recovery-label gates remain closed.
