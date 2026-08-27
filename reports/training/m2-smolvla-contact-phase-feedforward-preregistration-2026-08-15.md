# Contact-phase feedforward preregistration (Plan040)

Plan039 showed an unregistered left-finger/table contact only in the observed state; the corresponding commanded pose remained collision-free. A reconstructed one-step counterfactual using the already registered official MuJoCo static inverse-dynamics position feedforward removed the left contact with a 4.12 mrad correction and 0.633 rad remaining command margin.

Plan040 changes only controller scope: the same fail-closed feedforward used at MoveIt's terminal handoff will also wrap successful `descend` and `grasp` arm commands. It retains the existing 0.05 rad correction bound and 0.04540462255477905 rad command margin. The task-contact allowlist is unchanged; the left-finger/table pair remains forbidden. IK, planner, targets, 1/3 mm gates, 750-step horizon, seed partitions, and labels remain frozen.
