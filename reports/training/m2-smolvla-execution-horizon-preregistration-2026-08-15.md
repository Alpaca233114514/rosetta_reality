# Execution horizon preregistration (Plan 038)

Plan037 reached `descend` at step 478 with zero teacher, IK, clipping, unexpected-collision or joint-margin failures, then exhausted the 500-step horizon at 9.54 mm position error. Only 22 descend steps were available. The successful train calibration uses 108 steps from first grasp through terminal reward 4.

Plan038 changes only `maximum_steps`, from 500 to 750. The extra 250 steps cover the measured safe-path overhead and demonstrated post-grasp duration. Every pose tolerance, collision rule, Action Contract, joint margin, planner/controller setting, seed and label seal remains unchanged.
