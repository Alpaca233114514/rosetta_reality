# Athena Plan036 local exact audit

Plan036 failed safely at step 427 in `descend`, after proving the new lifecycle behavior: 3 MoveIt plans, 3 terminal-control activations and 2 completed-reference resets, with no IK, adapter-clip, commanded-margin or observed-margin failures.

The sole failure was `table` against `vx300s_right/10_right_gripper_finger`. Static reconstruction of the final state reproduces exactly that pair. A read-only replay of the registered train calibration episode 2, seed 10 shows the same pair at steps 175–183 and 185 immediately before the first successful grasp at step 186; that replay ultimately reaches reward 4. The generic robot-versus-table classifier is therefore over-broad for this task-specific pre-grasp geometry.

No later seed, validation/hidden episode, label write, download, CUDA execution or optimizer step occurred. The JSON companion binds all run and diagnostic hashes.
