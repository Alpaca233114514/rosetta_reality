# M2 SmolVLA Athena Plan033 local exact audit

Status: **failed safely at the MuJoCo 3.8 actuator-moment storage boundary;
later gates remain sealed**.

Plan033 ran locally in one network-disabled Linux container with the repository
mounted read-only and no CUDA device. The official MoveIt 2.5.9 sidecar identity,
22/22 collision geometry, resource manifests and five-sample MoveIt-to-Gym FK
parity all passed. Train-only calibration reproduced reward `4` in 294 steps.

The exact rollout made one global plan, advanced through all 15 subsequent
waypoints and reached the registered final-handoff condition at step 127:
waypoint index `15` and observed L1 distance `0.16979807420839654` rad, inside
the unchanged `0.2`-rad SimpleSampler tolerance. The command was not executed
because the new feedforward boundary failed closed before latching.

The failure is an implementation/API representation defect. The Plan033 code
required a dense `nu x nv` `actuator_moment` matrix. MuJoCo 3.8.1 exposed the
same official moment map in CSR form: 16 scalar values with one entry per row,
plus `moment_rownnz`, `moment_rowadr` and `moment_colind`. All observed values
were `1.0` and all column indices were the matching direct DOF. MuJoCo's own
forward-dynamics source consumes these four arrays together when multiplying
actuator forces into generalized force.

Plan033 is preserved as negative evidence. Plan034 may change only this storage
adapter: read and validate the official CSR row metadata, while retaining the
dense test boundary. It may not change the planner, teacher, thresholds, joint
margins, Action Contract, seeds or label policy.

Exact report SHA-256:
`a7516fb47d386ba3f87d63eda7ef787edf434aa914d1f2975e77eed49af65f52`.
Tuning, development, collection, policy-Gate, validation, hidden and recovery
labels remain closed.
