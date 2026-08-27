# MoveIt 2-second IK timeout preregistration (Plan041)

Plan040 completed two official retained trajectories, then its third `orient` request exhausted the 0.50-second bimanual LMA restart budget before `descend`; the new contact-phase controller therefore had zero commands and remains untested. No collision, clipping, teacher, controller, or joint-margin failure occurred.

Plan041 changes only `path_planner_ik_timeout_s`, from 0.50 to 2.00 seconds. The LMA plugin, OMPL RRTConnect planner, 0.25-second planning budget, 5.0-second response timeout, Plan040 contact-phase feedforward, targets, pose gates, collision policy, margins, 750-step horizon, seeds, and labels remain unchanged.
