# M2 SmolVLA lift grasp-feedback preregistration — 2026-08-16

Plan055 changes one lift-phase target rule. Plan054 reached `lift` with both
objects grasped, then lost the right peg grasp between trace samples 416 and
422; the observed peg-to-right-EEF transform exceeded the unchanged 45 mm
grasp-drift limit. The registered expanded-orientation trust-region event was
still zero.

The new rule replaces the fixed lift endpoint
(`socket/peg` world pose at `lift_object_height_m` composed with the grasp-time
captured transform) with a feedback-anchored increment:

- for each lift decision, the virtual object target starts from the currently
  observed object pose;
- its z coordinate advances by at most
  `teacher.lift_feedback_step_m = 0.006` m, clamped to the unchanged
  `lift_object_height_m`;
- x/y and quaternion remain anchored to the current observed object pose;
- the EEF target is that virtual object pose composed with the unchanged
  grasp-time captured transform.

The final lift height, 45 mm grasp-drift limit, 0.012 m global Cartesian step,
0.20 rad orientation budget, joint margins, 1 mm / 3 mm pose gates, exact seed,
750-step horizon, and later seed/label boundaries remain unchanged.
