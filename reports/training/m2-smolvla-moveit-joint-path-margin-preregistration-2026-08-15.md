# M2 SmolVLA MoveIt joint-path-margin preregistration (2026-08-15)

## Decision

Athena plan `025` remains immutable negative evidence. It executed 21 of 22
collision-checked official RRTConnect fallback waypoints, then stopped at step
168 when the live `right_forearm_roll` state exceeded its physical bound by
`0.0005873297882081907` rad. Its exact report SHA-256 is
`bfd55bc72f9f85cc19a8e3de58d2a15a72e65f41ded6818910c353a961acac2e`.

Plan `026` changes one axis: the official MoveIt request now carries twelve
native `moveit_msgs/JointConstraint` path constraints, each inset by `0.01`
rad from the physical arm-joint bounds. The same constraints are checked for
the start, every LMA IK candidate, the accepted goal, every OMPL trajectory
waypoint and the bounded execution waypoint. The returned JSON evidence
records the constraint identity, count and minimum physical margin for start,
goal, path and next waypoint.

This does not widen the `1` mm / `3` mm pose gates, change the physical Action
Contract, add orientation relaxation or replace official MoveIt/LMA/RRTConnect.
Teacher geometry, upstream Mink/DAQP and its `0.01`-rad margin, collision
resources, calibration episode, exact seed and every later seed identity stay
frozen.

## Frozen identities

| Item | SHA-256 / identity |
|---|---|
| plan `026` | `7e87a23663cd66af72c8957221e5e87b7b98744efbc68009bb1cb2ceea461a53` |
| C++ sidecar source | `15b83bd725036b58b1d3b8c05e7f216bc759adf9e2d33bf310c54e4a971684a3` |
| C++ sidecar executable | `e5da00ce9fe665d9d28edbd0bfa075df608418f2e5a471685b94136ff310cc6d` |
| Python client | `5ce1bdbb4c06d6bf59dd80ae073a6de3dc88ae0651421042dfe98d95047631db` |
| evaluator | `d89a5a636234c3d968d3e71a1366faa3fbbb2491acd902eec0e44a77bc7c305f` |
| Gym ALOHA adapter | `e9c1005d0ae085e82e0c96e9d18527dce7d4749268a71756116cdffbb98d6e7d` |
| MoveIt image ID | `sha256:063a59e9a9f4501fb768e85dbef3f91f4abee42aee83cbf39eddc46f14a71368` |
| model-parity report | `ccdaafadf0fe900b2c77c7e926ec444598ecbbee33afd6e8ae08e280530a5d68` |

The runtime remains ROS Humble, MoveIt `2.5.9`, OMPL `1.7.0`, official
`ompl_interface/OMPLPlanner`, `geometric::RRTConnect`, official
`lma_kinematics_plugin/LMAKinematicsPlugin`, and 22 collision links / 22
collision shapes.

## Local static evidence

The new image compiled successfully. Container Ruff passed and all 37 focused
tests passed. Five-sample Gym/MoveIt FK parity passed with maximum position
error `3.188872858294072e-16` m and zero orientation error.

A network-disabled direct sidecar smoke returned an eight-waypoint official
RRTConnect path with 12 active joint constraints. Its minimum physical margin
across all returned path waypoints was `0.25577550429170226` rad, its next
waypoint margin was `0.4185679291845288` rad, and its maximum start-to-next
joint delta was `0.09935916874209885` rad. The same executable rejected a
`right_forearm_roll = 3.135` start because its `0.00658000000000003`-rad
physical margin was below the registered `0.01` rad. The create-only smoke
record SHA-256 is
`f6dbe5cd6fba2a7797831b3572f825fd77e753afbb6e9c9008a29a71eba76f09`.

## Gate state and next execution

Status is `train_exact_preregistered_not_executed`. The next SSH workspace may
run only train episode 2 / seed 10 calibration plus exact after verifying the
new content-addressed workspace, official runtime identities and watchdogs.
Tuning seed `1900`, development seeds `2000`--`2004`, collection seeds
`3000`--`3004`, policy-Gate seeds `1000`--`1004`, validation, hidden,
recovery-label writes, optimizer work and CUDA training remain sealed.
