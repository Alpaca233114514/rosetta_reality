# M2 SmolVLA Athena Plan045 local exact audit

Status: **failed train-only exact; all later gates remain sealed**.

Plan045 fixed the Plan044 subgroup-selection bug. The new runtime executed
ordinary full-pose LMA, the official position-only LMA fallback, 76 OMPL path
waypoints, and one retained-reference terminal completion without a collision,
clip, commanded-margin breach, or observed-margin breach.

The remaining failure moved to orient step 225. The direct full-pose request
asked both arms to move about 11 mm while rotating exactly 0.04 rad from the
current pose; official LMA exhausted all 256 registered attempts with the
nearest joint still retaining 0.06096 rad physical margin. Position-only LMA
could not preserve the already saturated orientation-relaxation boundary.

An attempt-scoped train-only probe then interpolated both translation and
quaternion while retaining the official full-pose LMA and OMPL contracts. The
bimanual request passed on its first IK attempt at fractions 0.125, 0.1 and
0.05. This is negative evidence for direct target stepping and positive
evidence for a bounded full-pose Cartesian waypoint; it is not an exact pass.

Tuning, development, collection, policy Gate seeds, validation/hidden episodes,
recovery labels and CUDA training were not opened.
