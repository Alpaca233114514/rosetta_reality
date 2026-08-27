# M2 SmolVLA Athena Plan057 local exact audit — 2026-08-16

Status: **failed train-only exact**. Immutable negative evidence.

Plan057 extended the official MoveIt/OMPL fallback to lift after the unchanged
Mink QP failure. The fallback fired exactly once, advanced one additional
waypoint, and removed the Plan056 IK failure. The run failed at step 444 in
`lift` because the right gripper bar (`vx300s_right/9_gripper_bar`) touched the
table; only the right finger/table pair is currently registered for lift.

No IK, clip, projection, margin breach, trust-region event or other unexpected
collision occurred. Both objects remained grasped.
