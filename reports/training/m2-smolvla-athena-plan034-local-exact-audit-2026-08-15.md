# M2 SmolVLA Athena Plan034 local exact audit

Status: **failed safely before exact-report creation at the Action
Contract-to-Gym joint-name adapter; later gates remain sealed**.

Plan034 passed the original CSR-storage failure and entered the terminal
feedforward function after train-only calibration again reached reward `4` in
294 steps. It then passed `left_waist` to MuJoCo `model.name2id`. That is the
MoveIt and Action Contract dimension name; the pinned Gym model joint is
`vx300s_left/waist`. The evaluator already owns the complete ordered mapping in
its frozen `LEFT_JOINTS` and `RIGHT_JOINTS` constants and uses it throughout
the Mink and FK paths.

This is a deterministic namespace-adapter defect, not a new planner, teacher,
CSR or force-balance result. No terminal command was executed and the evaluator
raised before it could create `exact.json`. The execution-log SHA-256 is
`f1552754b36e1d15588cfb23dcb59a9dbf06f7428be91d7d92dff9b54c26697d`.

Plan035 may replace only the 12 terminal feedforward lookup names with the
existing ordered Gym names. It may not change path planning, force balance,
teacher gates, thresholds, margins, Action Contract, seeds or labels. This is
the third and final local exact attempt in the current user-authorized bound.
