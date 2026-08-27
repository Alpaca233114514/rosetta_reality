# M2 SmolVLA Athena Plan058 local exact audit — 2026-08-16

Status: **failed train-only exact**. Immutable negative evidence.

Plan058 added the observed table / right-gripper-bar pair to the lift-only
contact exemption. It ran all 750 registered steps in `lift` with no teacher
failure, IK failure, clip, projection, margin breach, trust-region event, or
unexpected collision. Both grasps remained in contact, but the right peg stayed
on/near the table and the registered lift height was not reached.

The feedback-anchored lift preserves grasp but does not generate enough upward
motion for the right peg when it remains on the table. The current local repair
plan stops here.
