# M2 SmolVLA Plan032 output-schema repair preregistration

Status: **schema-only repair preregistered; no Plan032 remote exact has run**.

## Failure boundary

Athena Plan031 static attempt 002 passed Ruff, 60 focused tests, five-sample
MoveIt/Gym FK parity and the actual official-sidecar full-trajectory client.
Its train-only exact attempt then completed the fixed episode-2/seed-10
calibration but stopped before the planner rollout with `KeyError: 'output'`.
The registered Plan031 YAML had accidentally omitted the `output`, remaining
`implementation_files` and `stop_conditions` tail inherited from Plan030. No
exact report was created, no later seed was opened and no label was written.

The durable Plan031 exact execution log has SHA-256
`f1b88ba641a09e43600b28f25cf55e2e809f90c75a0a9390015584bf94ec7b35`.
The failed runner and exit record have SHA-256
`15c49e1d240cbeed0f188ce108e5663c2f142775424fef969c2f37dbd0cd24dd`
and `b9a7d4ad53d77c3dc9966a91cf4b1735f7e482e4acc5101fb1167aa26a75e2f1`.

## Registered repair

`configs/sim/aloha_insertion_geometry_teacher_032.yaml` has SHA-256
`435630a5f5e3037a3580e4296e38d4bd9b2b5f9ea2c49f55bb4e261dca7e5645`.
It restores the omitted Plan030 tail, changes the output directory to the
Plan032 identity and otherwise preserves Plan031's retained official MoveIt
trajectory execution axis. The evaluator now validates the output mapping,
safe relative run directory and report scoping before calibration. A regression
test removes `output` and requires that preflight to fail immediately.

Frozen unchanged: episode 2 / seed 10, all later sealed seed groups, the 1 mm /
3 mm pose gates, physical and command joint margins, maximum Action Contract
delta, OMPL RRTConnect, LMA IK, `FixStartStatePathConstraints`, SimpleSampler's
0.2-rad L1 waypoint tolerance, collision resources and label authority.

The next authorized action is a new content-addressed workspace, remote static
reproduction and one Plan032 train-only exact. Tuning, development, collection,
policy-Gate, validation, hidden and recovery-label stages remain closed.
