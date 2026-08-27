# M2 SmolVLA MuJoCo sparse actuator-moment repair preregistration

Status: **Plan034 is locally implemented, preregistered and statically
verified; train-only exact has not run**.

## Single repair axis

Plan033 reached its final SimpleSampler waypoint handoff at step 127 but failed
closed before terminal-control activation. The pinned MuJoCo 3.8.1 runtime
stores `actuator_moment` as CSR values with `moment_rownnz`, `moment_rowadr` and
`moment_colind`; Plan033 incorrectly required a dense `nu x nv` array.

Plan034 changes only that storage adapter. Each sparse row must have valid
bounds, unique in-range columns and exactly one nonzero at the registered joint
DOF. The existing dense representation remains covered by unit tests. The
affine force equation, correction bound, command margin, official MoveIt path,
teacher gates, Action Contract, seeds and labels are unchanged.

## Evidence before exact

Ruff and 69 focused tests passed in a network-disabled Linux container with a
read-only repository mount. The official MoveIt 2.5.9 sidecar initialized with
22 collision links and 22 collision shapes; both complete resource manifests
matched. Five-sample MoveIt-to-Gym FK parity passed with maximum position error
`3.188872858294072e-16` m and orientation error `0.0` rad.

Plan034 SHA-256 is
`6dc5019b80522eca489b2d1473f0f801ef6c596cd5cbf38acfe1401baaa883ad`.
The next authorized action is only episode 2 / seed 10 train-only exact.
Tuning, development, collection, policy-Gate, validation, hidden and recovery
labels remain closed.
