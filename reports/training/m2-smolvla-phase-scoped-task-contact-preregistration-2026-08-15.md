# Phase-scoped task contact preregistration (Plan 037)

Plan036's lifecycle repair worked, then train-only exact stopped at step 427 on one pair: `table` and `vx300s_right/10_right_gripper_finger`. The immutable successful calibration replay contains that same pair at steps 175–183 and 185, immediately before its first reward-1 grasp at step 186 and eventual reward 4.

Plan037 changes only contact classification. That exact unordered pair is treated as a registered task contact while the teacher is already in `descend` or `grasp`. Raw contact reporting remains enabled. Every other robot/table, robot/object and self-collision remains fail-closed, and the pair remains forbidden in every other phase.

The 500-step budget, teacher targets and tolerances, Action Contract, joint margins, planner/controller identities, seed 10 and all data/label seals stay frozen.
