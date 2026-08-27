# M2 SmolVLA execution-margin diagnostic preregistration (2026-08-15)

## Outcome

Plan `027` is preregistered as an instrumentation-only train-exact diagnostic.
It does not change the teacher, action, constrained solvers, MoveIt request,
controller adapter, pose gates or physical `0.01`-rad joint margin. It records
the commanded-versus-observed arm-joint evolution that plan `026` could not
distinguish. Tuning, development, collection, policy-Gate, validation, hidden
and recovery-label boundaries remain sealed.

The frozen plan is
`configs/sim/aloha_insertion_geometry_teacher_027.yaml`, SHA-256
`b655c2956a7b7ce84ba882796748f8877b6229ff25a385262852a3dfd139123c`.
The evaluator SHA-256 is
`46e375a40b2cc8445a524069105f63146f0d40ddf3b090f31dbbb66258a28b5e`.

## Why this is the next axis

Plan `026` kept every accepted official path outside the registered margin,
then stopped at step 113 because the later observed start state had only
`0.00961627014160138` rad of margin. Its sparse trace did not retain the
preceding absolute command or the immediate post-step robot state. Therefore
the evidence does not yet distinguish an unsafe Mink command from safe-command
tracking overshoot or another command/observation timing effect.

Plan `027` records every executed exact-control step, including all twelve arm
joints in the pre-step observation, absolute command and next observation. It
also records per-joint lower/upper margins, tracking error, same-direction
overshoot beyond the command, margin loss from command to observation, and the
first command and observation margin-breach steps. An unexecuted IK failure
retains its full pre-step margin snapshot. The code path is explicitly
registered as non-causal instrumentation and cannot feed values back into the
teacher, IK, path planner or environment action.

## Decision boundary after exact

- If the command itself first enters the protected band, the next single axis
  belongs at solver-output command enforcement.
- If the command remains safe but the next observation enters the band, the
  next single axis is a measured execution reserve or feedback guard. The
  physical `0.01`-rad margin still does not change.
- If the failure does not reproduce and exact passes, tuning may only be opened
  through the existing staged gate; no later seed is opened by this
  preregistration alone.

## Local verification

The protected Gym adapter still has SHA-256
`e9c1005d0ae085e82e0c96e9d18527dce7d4749268a71756116cdffbb98d6e7d`.
After the WSL/Docker relocation, the existing
`rosetta-reality-smolvla-sim-xpu:mink-1.2.0` Linux image ran with networking
disabled and the repository mounted read-only. Ruff passed and all 39 focused
geometry/Mink/MoveIt/Gym tests passed. The first test attempt exposed only a
new float32 assertion whose default pytest tolerance was narrower than one
float32 ULP at the registered joint value; the test tolerance was bounded to
`5e-7` without changing runtime code, after which the complete set passed.

No SSH, CUDA, optimizer, model/data download or label write was used. The
train-only exact reproduction is the remaining gate.
