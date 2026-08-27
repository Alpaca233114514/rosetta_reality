# Deterministic MoveIt subgroup IK frame preregistration (Plan043)

The Plan042 dry-run stopped before planning because it incorrectly asserted
that each official subgroup LMA solver uses `world` as its base. MoveIt 2.5.9's
subgroup implementation accepts model-frame targets and calls
`RobotState::setToIKSolverFrame` before invoking each subgroup solver.

Plan043 preserves the deterministic Plan042 scheduler—request state first,
then fixed-seed restarts, at most 256 bimanual attempts inside the existing
2.00-second outer bound—and restores that official coordinate conversion. The
solver remains the official LMA plugin and every complete candidate remains
subject to the frozen bounds, joint-path constraint, and self-collision checks.
All targets, tolerances, OMPL settings, controller behavior, contacts, margins,
horizon, seeds, and label gates remain frozen from Plan041.
