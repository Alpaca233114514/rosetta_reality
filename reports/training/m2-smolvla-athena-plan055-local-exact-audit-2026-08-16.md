# M2 SmolVLA Athena Plan055 local exact audit — 2026-08-16

Status: **failed train-only exact**. Immutable negative evidence.

Plan055 changed only the lift target rule to a feedback-anchored 0.006 m
object-lift increment. It preserved both grasp contacts for 50 lift decisions,
but failed at step 438 in `lift` because the right gripper finger touched the
table while the peg was still near the table; that contact is registered only
for `descend` and `grasp`, so it was correctly classified as unexpected.

No IK failure, joint-limit projection, adapter clip, commanded or observed
margin breach, or trust-region event occurred. The right peg remained grasped
and began rising near the end of the trace. The axis was exercised
(`lift_feedback_anchor_commands = 50`) and is rejected only at the next
boundary: lift-phase task-contact scoping.
