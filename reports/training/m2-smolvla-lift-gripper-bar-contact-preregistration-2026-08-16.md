# M2 SmolVLA lift gripper-bar contact preregistration — 2026-08-16

Plan058 changes one lift contact pair. Plan057's official MoveIt lift fallback
removed the IK failure and retained both grasps, then failed at step 444 because
`table` / `vx300s_right/9_gripper_bar` touched during lift. That pair is part
of the same right gripper assembly whose finger/table contact is already
registered for lift.

Plan058 adds only that observed gripper-bar/table pair to the lift contact
exemption. The finger/table pair remains, no other pair is allowed, and every
other contact still fails closed. Lift feedback, MoveIt fallback, pose gates,
margins, horizon, seeds and label boundaries remain unchanged.
