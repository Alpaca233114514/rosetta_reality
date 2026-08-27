# M2 SmolVLA official MoveIt path-planner preregistration — 2026-08-15

## Decision and boundary

Geometry-teacher plan `022` replaces no teacher, simulator or Action Contract
logic. It adds one fallback after upstream Mink/DAQP fails during `approach` or
`orient`: the official MoveIt 2 planning pipeline calls its official OMPL
plugin with `geometric::RRTConnect`, using the official MoveIt LMA kinematics
plugin to solve the two-arm goal. Rosetta only composes the two pinned vendor
VX300S descriptions, maps Gym-ALOHA frames/joints, transports one JSON-lines
request and samples one bounded waypoint from the returned path.

This is the mainstream ROS 2 manipulation architecture rather than a new
Rosetta motion planner. The relevant upstream authorities are the
[MoveIt motion-planning concepts](https://moveit.picknik.ai/main/doc/concepts/motion_planning.html),
[MoveIt planning-pipeline tutorial](https://moveit.picknik.ai/main/doc/examples/motion_planning_pipeline/motion_planning_pipeline_tutorial.html),
[Trossen MoveIt configuration](https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros2_packages/moveit_motion_planning_configuration.html),
and pinned
[Interbotix manipulator descriptions](https://github.com/Interbotix/interbotix_ros_manipulators/tree/b66d5b905725351dd71d3251a06cd3f4c777940f).

The unchanged acceptance boundary is a `0.001` maximum weighted IK error, a
`0.003` maximum projected error, the registered Action Contract and a maximum
per-command arm-joint step of `0.23561944901923448` rad. Orientation relaxation
is zero. Tuning, development, collection, policy-Gate, validation and hidden
boundaries remain sealed. No recovery label is authorized.

## Frozen upstream and runtime identity

| Item | Frozen identity |
|---|---|
| ROS distribution | Humble |
| MoveIt | `2.5.9` |
| OMPL | `1.7.0` |
| planning plugin | `ompl_interface/OMPLPlanner` |
| planner | `RRTConnect` / `geometric::RRTConnect` |
| kinematics plugin | `lma_kinematics_plugin/LMAKinematicsPlugin` |
| Interbotix source commit | `b66d5b905725351dd71d3251a06cd3f4c777940f` |
| source archive SHA-256 | `d22c67bf76a83de275e547f07ed9959bbf5a4335fe0da4ff092efa7094ab7637` |
| MoveIt image ID | `sha256:a48a28875f358f087fa83f6819c720d66f74e9ff9f3bdb28c8997fcd54f566b2` |
| sidecar executable SHA-256 | `e2d51c1aad413bf58111a0aa2c750c1083175eb80a8a23fdda5c78ed0173181b` |
| composed URDF SHA-256 | `544b5f299a5955b4c705cd42f952f4ab4fb898327f6206ea37117533190f63f4` |
| composed SRDF SHA-256 | `65bd2a440fb3565fae4373d0218a95afd9564fcb2e4bfdcece06ac4e82c0321d` |
| description manifest SHA-256 | `65d733588c7d6a7f78ac2a2eacb62b4b14e7337151e4190d046d957edbd5572b` |

## Local non-training evidence

Five joint configurations were evaluated independently through Gym-ALOHA
MuJoCo FK and the composed MoveIt model. Maximum position disagreement was
`3.188872858294072e-16` m and maximum orientation disagreement was `0` rad.
The create-only parity report is
`reports/training/m2-smolvla-aloha-moveit-model-parity-2026-08-15.json` with
SHA-256
`09d5872bae5cb8ff34d724c7d02c7c9c1c852a2c3dad4c54a97575ff4f7f54a0`.
It loaded no dataset rows and no hidden-test material.

A direct official sidecar smoke moved both neutral calibration sites down
`0.005` m. LMA produced a goal with maximum position error
`2.3459061185342847e-05` m, maximum orientation error
`0.0005538440389254536` rad and maximum weighted error
`0.0001342278689703837`. OMPL reported the
`ompl_interface/OMPLPlanner` RRTConnect path with two waypoints, path length
`0.03678845529387281` rad and a collision/bounds-valid next waypoint. The
observed planning time was `0.00315584` s; timing is observational, not an
acceptance identity.

Ruff passed and 22 focused Linux-container tests passed. They cover description
composition, model identity, hash-bound plan loading, exact-seed isolation,
Mink constraints, sidecar identity/protocol, official-failure preservation and
Action Contract waypoint mapping.

An optional local combined evaluation image was not accepted because Docker's
route to the official PyPI file CDN repeatedly ended with TLS EOF while fetching
software dependencies. TLS verification was not disabled. This network failure
does not count as exact evidence and does not alter the preregistered Athena
runtime.

## Athena authorization

The only authorized remote run name is **Athena**. Its allowed work is:

1. read-only instance/task/runner/process audit;
2. create-only low-privilege three-hour watchdog, followed by an independent
   SSH reconnect that verifies both watchdog and sleep child;
3. install authorized Python/ROS/MoveIt software without touching CUDA or the
   driver and without downloading model/data content;
4. verify every workspace, plan, source, binary, model and cache identity;
5. reproduce Ruff, focused tests, model parity and the plan-`022` train-only
   exact stage under a durable session.

No optimizer step, CUDA training, tuning seed `1900`, development seed,
collection seed, policy-Gate seed, validation/hidden episode or recovery-label
write is authorized. A failed exact result ends Athena at a clean evidence
boundary. A passed exact result authorizes only a new, separately reviewed
decision about whether to open tuning; it does not open tuning automatically.

