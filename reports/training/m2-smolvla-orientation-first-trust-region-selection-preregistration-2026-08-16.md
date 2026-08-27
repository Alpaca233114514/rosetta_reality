# M2 SmolVLA orientation-first trust-region selection preregistration — 2026-08-16

Plan 052 changes one axis: the trust-region candidate-selection order. It tests the registered orientation fractions from largest to smallest and selects the maximum-minimum-joint-margin candidate within the first feasible fraction. Margin restoration runs only when no orientation-progress fraction is feasible.

Plan 051 established the failure mode cleanly: 8/8 feedback-basis activations chose margin restoration, 0 chose orientation progress, no IK failure occurred, and the run exhausted 750 steps in `orient`. Earlier sealed probes found valid 0.125 orientation-progress candidates above the unchanged command margin at both the initial and late failure boundaries.

The feedback basis, radii, 12 mm position envelope, 1 mrad restoration requirement, official LMA/OMPL stack, hard path constraints, Action Contract margins, 1/3 mm pose gates, horizon, seed, and all later seed/label boundaries remain unchanged.
