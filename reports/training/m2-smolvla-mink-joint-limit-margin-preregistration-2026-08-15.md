# M2 SmolVLA Mink joint-limit margin preregistration (2026-08-15)

## Authority and predecessor

Athena plan `023` is immutable valid negative evidence. Its official MoveIt
runtime loaded 22 collision links and 22 shapes, but train-only exact failed at
approach step 98 because the live `right_wrist_rotate` observation exceeded the
registered upper bound by `0.005985603256225769` rad. The exact report SHA-256
is `ec0ed7a2e910eabbc9fefc3b9369a990b82d4ab37c588055f085a53e73af1234`.

## Single-axis hypothesis

Plan `024` adds one standard safety-control axis to the upstream Mink
`ConfigurationLimit`: all twelve arm-joint ranges are inset by exactly `0.01`
rad in the solver model before commands are generated. The margin is larger
than the measured train-only dynamic overshoot and remains well inside the
physical Action Contract. It does not change Gym physics, the Action Contract,
teacher geometry, official MoveIt/LMA/RRTConnect, collision resources,
`0.001`/`0.003` pose gates, maximum command delta, or the `0.00002`-rad
representation-only reconciliation allowance.

| Item | Frozen value |
|---|---|
| plan | `configs/sim/aloha_insertion_geometry_teacher_024.yaml` |
| plan SHA-256 | `dcccc4bcbdaab7384b9c3334e0418470a48911f907084f43be3474be47c699d8` |
| Mink adapter SHA-256 | `cdff063ce164dc5eb8137c6c8cbd66f5c102eb06f0842a0f92b64727f9e70583` |
| evaluator SHA-256 | `5b22d41e6c87c340da264579c598cb5212bb2e8373cacdda94af621168e1fa25` |
| joint-limit margin | `0.01` rad |
| exact episode / seed | train episode `2` / seed `10` |

Static validation and the train-only exact run must use a new create-only,
content-addressed workspace. Tuning `1900`, development `2000`--`2004`,
collection `3000`--`3004`, policy-Gate `1000`--`1004`, validation, hidden and
recovery-label gates remain sealed regardless of runtime smoke. No optimizer
step or CUDA training is authorized.

Create-only attempt `athena-plan024-static-validation-020` passed Ruff and all
32 focused tests. It also reverified every official mesh/URDF resource. Its
workspace archive SHA-256 is
`a6cc9810857346a379428181f382ce209e8b5f9d3eb03d34d3153de5a5f0a3fd`;
the immutable execution-log SHA-256 is
`2f4411fe52509ae5c8ae19edd01106633be0d3913ddaedc8c7fe4dff40723ae7`.
