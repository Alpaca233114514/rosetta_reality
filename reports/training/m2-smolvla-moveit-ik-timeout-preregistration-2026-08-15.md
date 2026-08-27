# MoveIt IK timeout preregistration (Plan039)

Plan038 completed its first official retained trajectory, refreshed correctly, and stopped at the second global-path request because the unchanged MoveIt LMA solver did not return a bimanual goal within its separately configured 0.10-second IK timeout. No collision, clipping, teacher, command-margin, observation-margin, or episode-horizon boundary caused the stop.

Plan039 changes only `path_planner_ik_timeout_s`, from 0.10 to 0.50 seconds. This is the bounded timeout passed to MoveIt's official `RobotState::setFromIKSubgroups` call. The LMA plugin, OMPL RRTConnect planner, 0.25-second planning budget, goals, frozen 1/3 mm pose gates, joint margins, controller, collision policy, 750-step horizon, seed partitions, and label seals remain unchanged.
