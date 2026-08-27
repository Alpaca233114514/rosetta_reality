# M2 SmolVLA MoveIt finger-boundary preregistration (2026-08-15)

Plan `024` is immutable negative evidence: its `0.01`-rad upstream Mink margin
removed the arm start-bound violation, but official MoveIt still returned
`start_state_out_of_bounds` at step 97 with zero arm-joint violation. Direct
identity inspection proved the remaining boundary: Gym open gripper `1.0`
maps to `0.058` m while the pinned official Interbotix finger upper bound is
`0.057` m.

Plan `025` changes only that representation adapter. It reconciles each MoveIt
request finger position into the official `[0.021, 0.057]` m range and fails
closed above `0.001` m. Plan SHA-256 is
`21ca069b55e22bf72e5e922e679c45f6f21a196217c0a2a190665fc331df29d6`;
evaluator SHA-256 is
`f46dded67df9ca36814652ebb06052fbb042430f7e55e95c150ec068eb7dbf8f`.
The Mink margin, Action Contract, official MoveIt/LMA/RRTConnect, collision
resources, pose gates and all seed identities remain unchanged. Only train
episode 2 / seed 10 exact is authorized. Tuning, development, collection,
policy-Gate, validation, hidden, labels, CUDA and optimizer work remain sealed.

Create-only static attempt `athena-plan025-static-validation-022` passed Ruff
and all 32 focused tests. Its workspace archive SHA-256 is
`00353be012e786946bf66e3fcd43e112e46db961b13e27b0103e0f80e53396fb`;
execution-log SHA-256 is
`b4451a0aaba9199813ba8b4f3553fcd0fa056a112338bcc3640ae673714c5b1e`.
