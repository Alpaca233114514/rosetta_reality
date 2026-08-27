# Official MoveIt full-pose Cartesian backoff preregistration

Plan045's remaining train-only failure is registered as a target-step failure
at orient step 225. Direct full-pose LMA exhausted 256 attempts while the
nearest joint retained 0.06096 rad margin. A diagnostic using the same binary,
official full-pose LMA and OMPL RRTConnect passed deterministically when both
positions and quaternions were interpolated to fractions 0.125, 0.1 or 0.05.

Plan046 may add one axis only: after Mink and direct full-pose MoveIt fail in
approach or orient, try the frozen fractions largest-first, using linear
position interpolation and shortest-arc quaternion slerp. A successful
candidate remains a strict full-pose LMA goal, is planned by the unchanged
collision-checked OMPL path, and is followed by feedback replanning toward the
original teacher target. The existing approach-only position-priority fallback
remains last in the order.

Final 1 mm / 3 mrad pose gates, collision policy, all joint margins, Action
Contract, seeds, label seals and CUDA prohibition remain unchanged.
