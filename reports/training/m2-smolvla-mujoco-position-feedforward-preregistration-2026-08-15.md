# M2 SmolVLA MuJoCo position-feedforward preregistration

Status: **locally implemented, preregistered and Linux-container statically
verified; Plan033 train-only exact has not run**.

## Finding

Plan032 removed the repeated-planning failure. It made one accepted official
RRTConnect plan, retained it for 402 commands and reached all 16 waypoints. The
MoveIt goal itself was accurate to `8.258796e-06` m position and
`0.000211779` rad orientation, and all IK, clip, collision and joint-margin
failure counts were zero. Nevertheless, the final observed joint state stayed
`0.03446431288757733` rad L1 from the final position reference, leaving the
teacher `0.02851971797645092` m from its unchanged approach target. Repeating
the same absolute setpoint for the rest of the 500-step rollout did not remove
that residual.

This is a controller/actuator boundary, not evidence that another path-search
algorithm is needed.

## Official control boundary

The upstream layers say the same thing from complementary directions:

- ROS 2 `joint_trajectory_controller` forwards a `position` command directly;
  its PID error adapter is for velocity-only or effort command interfaces.
- MoveIt Servo is the standard local high-rate Cartesian/joint jogging layer,
  with joint-limit, singularity and collision handling. Its Humble source adds
  a bounded joint increment to the current measured joint state before sending
  the controller command.
- MuJoCo defines each SISO actuator force as an affine function of control,
  actuator length and actuator velocity. At a static target, the arm actuator
  force must balance generalized bias minus passive force. Because the current
  registered Action Contract must remain absolute joint-position control,
  this is the matching official low-level equation to solve; switching the
  simulator to velocity/effort control would be a different Action Contract.

Official sources:

- [ROS 2 Joint Trajectory Controller behavior](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)
- [ROS 2 controller parameters and PID law](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/parameters.html)
- [MoveIt Servo tutorial](https://moveit.picknik.ai/humble/doc/examples/realtime_servo/realtime_servo_tutorial.html)
- [MoveIt Servo 2.5.9 source](https://github.com/moveit/moveit2/blob/2.5.9/moveit_ros/moveit_servo/src/servo_calcs.cpp)
- [MuJoCo actuation model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model)
- [MuJoCo passive-force computation](https://mujoco.readthedocs.io/en/stable/computation/index.html#passive-forces)

## Registered single axis

`configs/sim/aloha_insertion_geometry_teacher_033.yaml` changes one execution
axis. After the retained reference is at its final waypoint and the observed
bimanual joint state is within the unchanged upstream `0.2`-rad
`SimpleSampler` L1 tolerance, the evaluator:

1. evaluates the final accepted joint target in the pinned Gym MuJoCo model;
2. requires one direct joint transmission per arm joint, fixed gain, affine
   bias, no activation dynamics and no cross-joint actuator moment;
3. solves the official affine force equation so the static position reference
   supplies `qfrc_bias - qfrc_passive`;
4. rejects actuator force/control saturation, any correction above `0.05` rad,
   or any command below the existing `0.04540462255477905`-rad joint margin;
5. latches that reference for the teacher phase and never re-enters global
   planning after terminal-control failure.

This does not change OMPL, RRTConnect, LMA, Mink, the official start-state
adapter, the protected Gym adapter, the Action Contract, 1 mm / 3 mm solver
gates, 12 mm approach gate, physical/command margins, seeds or labels.

## Current evidence boundary

The modified Python files pass a read-only AST parse, `git diff --check` passes,
and the protected `gym_aloha.py` and `geometry_teacher.py` hashes remain
unchanged. New pure-equation, actuator-identity, correction-bound, executor
handoff and no-replan tests are present.

Ruff and all 67 focused tests passed in the network-disabled
`rosetta-reality-smolvla-sim-xpu:mink-1.2.0` Linux container with a read-only
repository mount. That suite also passed runtime YAML validation and the
implementation hash binding. The configured WSL distribution has no injected
Docker CLI, so explicit WSL Bash invoked the existing Windows Docker 29.6.1
engine. FK parity and the actual sidecar smoke have **not** run after this
change, and the current local implementation is not an exact pass.

The next authorized execution is read-only resolution of the existing local
dataset cache and official MoveIt 2.5.9 sidecar runtime, followed by FK parity,
the sidecar smoke and only then Plan033 train-only exact episode 2 / seed 10.
Tuning, development, collection, policy-Gate, validation, hidden and
recovery-label gates remain closed.
