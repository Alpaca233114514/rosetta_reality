# Athena Plan039 local exact audit

Plan039 fixed the prior LMA timeout boundary: three official MoveIt requests succeeded, two retained trajectories completed and refreshed, and there were no IK, clipping, or joint-margin failures. It reached `descend` and recorded 15 occurrences of the exact registered right-finger/table task contact.

At step 441 it stopped fail-closed on a different pair: `table` with the left gripper finger. That pair does not occur in the successful train calibration and cannot be added to the allowlist. Static reconstruction showed that the commanded pose remained free of this left contact while the observed pose did not, identifying near-table tracking/gravity droop rather than a geometric target or IK failure.

A one-step counterfactual using the already implemented official MuJoCo static inverse-dynamics feedforward needed only 4.12 mrad correction, retained 0.633 rad command margin, and removed the unregistered left contact. This is diagnostic evidence, not a pass. Plan040 may apply that controller only in `descend` and `grasp`; all collision, pose, joint-limit, planner, horizon, seed, and label gates remain frozen. The JSON companion is the exact authority.
