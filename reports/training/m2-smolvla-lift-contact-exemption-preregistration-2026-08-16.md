# M2 SmolVLA lift contact-exemption preregistration — 2026-08-16

Plan056 changes one lift-phase contact boundary. Plan055's feedback-anchored
lift preserved both grasp contacts for 50 lift decisions but failed at step 438
because the already-evidenced `table` / `vx300s_right/10_right_gripper_finger`
contact appeared in `lift`; that contact is currently registered only for
`descend` and `grasp`.

Plan056 reuses the exact same unordered contact pair and extends its phase scope
to include only `lift`. No new contact pair is allowed, and every other contact
still fails closed. Lift feedback step, pose gates, margins, orientation budget,
horizon, seeds and label boundaries remain unchanged.
