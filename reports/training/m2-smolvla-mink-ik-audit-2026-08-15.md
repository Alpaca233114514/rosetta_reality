# M2 SmolVLA upstream Mink/DAQP IK audit — 2026-08-15

## Outcome

The current geometry-teacher execution path no longer uses the historical
handwritten active-set/waypoint solver. Plans `015`--`021` delegate dual-arm
pose IK to upstream Mink `1.2.0`, QPsolvers `4.13.0` and DAQP `0.8.7` on MuJoCo
`3.8.1`. The Rosetta-owned code is now a thin ALOHA/Action Contract adapter plus
the event-driven task teacher and sealed evaluation protocol.

This is an architecture replacement, not a passed gate. Train-only exact still
fails at approach step 98 with `0.01189256245974505` maximum projected pose
error under plan `021`. Tuning, development, collection, policy-Gate,
validation and hidden-test boundaries remain sealed. No recovery label was
written and no CUDA training was started.

## Upstream structure used

The active solver follows Mink's published ALOHA example:

- two full-pose `FrameTask`s for the left and right end effectors;
- one low-cost `PostureTask` fixed to the initial neutral configuration;
- MuJoCo-native `ConfigurationLimit` and named-arm `VelocityLimit` inequalities;
- DAQP through QPsolvers;
- Mink `DofFreezingTask` equality constraints for fingers and free objects that
  are not variables of the arm pose solve.

The adapter projects only solver-internal ingress states onto the slightly
narrower MJCF joint bounds. The returned arm and gripper commands still pass
through the unchanged Action Contract. The legacy custom branch remains only
for reproduction of immutable plans `003`--`014`; `solver_backend: mink_qp`
dispatches before that branch.

Upstream references:

- <https://github.com/kevinzakka/mink>
- <https://github.com/kevinzakka/mink/blob/main/examples/arm_aloha.py>
- <https://kevinzakka.github.io/mink/api/inverse_kinematics.html>
- <https://kevinzakka.github.io/mink/api/limits.html>

## Train-only exact evidence

| Plan | Single registered axis | Last step / phase | Maximum projected error | Result |
|---|---|---:|---:|---|
| `015` | five-iteration upstream ALOHA baseline | 15 / approach | `0.003425959918725615` | failed |
| `016` | ten measured Mink iterations | 15 / approach | `0.0015426735626533628` | failed: 1 mm solver gate |
| `017` | fifteen measured iterations | aborted | no create-only report | strict JSON rejected infinity |
| `018` | finite failure observability only | 33 / approach | `0.01200002059340477` | failed: non-arm finger ingress |
| `019` | upstream non-arm DOF-freezing constraint | 98 / approach | `0.012138118074065505` | failed: Action Contract/MJCF wrist-bound delta |
| `020` | project all limited ingress joints to MJCF bounds | 98 / approach | `0.011913858596623727` | failed: genuine constrained residual |
| `021` | upstream neutral-posture target lifecycle | 98 / approach | `0.01189256245974505` | failed: genuine constrained residual |

Plan `017` remains an aborted identity with no report; it was not rewritten.
The first plan-`019` invocation omitted the required `ROSETTA_RUN_ROOT` and
exited after evaluation but before destination creation. No run directory or
report existed, and the same immutable plan was rerun with the required root;
the resulting create-only report is the authority recorded in the JSON
companion.

## Frozen identities and verification

- local container image:
  `rosetta-reality-smolvla-sim-xpu:mink-1.2.0`, image SHA-256
  `8313a310d42fdefc5e8f069be5732e2a97b6f0772aa1c8b2033934a1b489596f`;
- current Mink adapter SHA-256:
  `373373f6dda5b5f1fa86a95357cd1c20e45a66b47ebf788d95f519afa44e14b5`;
- evaluator SHA-256:
  `9ec759ad6de5ff9af2d888a6f987603a4a55f278991ed151e2408139e27c872f`;
- teacher SHA-256:
  `74b788fd316ef0f723d871db5c5cf550999998c37826843a268a31c4edddf5fd`;
- Gym-ALOHA adapter SHA-256:
  `e9c1005d0ae085e82e0c96e9d18527dce7d4749268a71756116cdffbb98d6e7d`.

Ruff and 15 focused geometry/Mink/adapter/protocol container tests pass. The
Mink tests load the native Gym-ALOHA MJCF, hold a dual-arm pose, make a 2 mm
translation, check native joint limits, freeze non-arm DOFs and cover the
Action Contract/MJCF wrist-bound mismatch.

## Architecture boundary after this audit

Mink is the upstream MuJoCo-native constrained differential-IK controller; it
is not a global sampling-based motion planner. Its official ALOHA structure did
not find a feasible local continuation for the unchanged exact target at the
right wrist limit. Further handwritten waypoint, active-set or blind weight
heuristics are not supported by this evidence.

If the next requirement is a true global geometric planner, preregister a
separate upstream planning stack such as MoveIt 2 with its default OMPL
pipeline, using the official Trossen VX300S MoveIt configuration as the
single-arm model source. That is a new ROS 2, dual-arm URDF/SRDF, collision and
MJCF-parity axis; it must not be hidden inside the current Mink plan or started
on a paid server before static model-parity tests pass.

