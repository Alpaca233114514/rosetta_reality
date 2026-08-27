# M2 SmolVLA Athena Plan032 exact audit

Status: **train-only exact failed `0/1`; all later gates remain sealed**.

## Identity and prerequisites

- Plan: `configs/sim/aloha_insertion_geometry_teacher_032.yaml`, SHA-256
  `435630a5f5e3037a3580e4296e38d4bd9b2b5f9ea2c49f55bb4e261dca7e5645`.
- Workspace archive SHA-256:
  `7ee5ddb43dc037b704e35a9b03423587a46717e8ca182363dcaae720246093a9`.
- Static attempt passed Ruff, `61/61` focused tests, five-sample MoveIt/Gym FK
  parity and the actual pinned-sidecar complete-trajectory client.
- Exact used only train episode `2` and simulator seed `10`; calibration reached
  reward `4` in `294` steps.
- No optimizer step, CUDA training, download, validation/hidden episode, later
  seed or recovery-label write was authorized or executed.

## Exact result

The exact report has SHA-256
`f8ea0bb5514afe7c1bf64930b05ac40f52346b9c98ec35e9c2a539dbfc2acf5c`.
The rollout exhausted all `500` steps with reward `0`, final phase `approach`
and phase visits `{open: 15, approach: 485}`.

Plan032 validated the retained-reference diagnosis:

- global planner attempts: `1`;
- accepted trajectory waypoints: `16`;
- retained-reference commands: `402`;
- reference waypoint advancements: `15`;
- final reference waypoint index: `15`;
- total planning time: `0.016344363` s.

The reference reached its last waypoint by trace step `125`, then remained there
through step `499`. At the final trace the joint-space L1 distance to the last
reference was `0.03446431288757733` rad, but the task-space target position error
remained `0.02851971797645092` m, above the frozen approach gate. This moves the
failure boundary from repeated global replanning to final-waypoint/controller
tracking feedback. It does not yet prove which controller-side correction is
appropriate.

## Safety evidence

- geometry-teacher, IK and adapter-clip failures: all `0`;
- joint-limit projections: `0`;
- commanded and observed margin breaches: both `0`;
- minimum commanded / observed margins:
  `0.05830754287719708` / `0.052193536834716614` rad;
- minimum constrained-path margin: `0.052193536834716614` rad;
- maximum MoveIt goal position / orientation errors:
  `8.258796088854016e-06` m / `0.0002117788971842651` rad;
- unexpected collisions: `0`.

The official sidecar stderr has SHA-256
`a96f0fccef291bb34e1e971e41f43e552d21c8e89354bb87a62c07df1e2f3f05`.
The exact execution log and runner have SHA-256
`160d7eb8747465d3f12e19f0296a4bf0fbbd8b74d532a0825b28cfdd6734e8fa`
and `0b9a02520385e805012fc57dbe047aa6ad73ae3a778fe519ed91d8cc25a90e4c`.

## Gate decision

Plan032 is negative evidence, not an exact pass. Tuning seed `1900`, development
seeds `2000--2004`, collection seeds `3000--3004`, policy-Gate seeds
`1000--1004`, validation/hidden episodes and recovery labels remain closed. A
next plan must be designed locally as one newly preregistered controller-feedback
axis without relaxing pose gates, margins or Action Contract limits.
