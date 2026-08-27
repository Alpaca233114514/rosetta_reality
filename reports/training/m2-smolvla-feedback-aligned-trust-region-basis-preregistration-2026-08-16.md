# M2 SmolVLA feedback-aligned trust-region basis preregistration — 2026-08-16

Plan 051 changes one axis: the Plan 050 active-set trust region replaces its fixed world-coordinate directions with a deterministic local orthonormal basis aligned to the current bounded feedback target. The radial axis points from the active arm's current position to its requested position; two tangents are derived by a deterministic cross-product construction. Both signs of all three axes are evaluated.

The Plan 050 failure reconstruction evaluated 65 official LMA requests and found 11 valid candidates. A 3 mm negative second-tangent waypoint restored the minimum joint margin from `0.04560698516845685` to `0.051032575992705276` rad. The same basis also admitted a `0.125` orientation-progress fraction with sub-millimetre/sub-milliradian solver error.

The radii, 12 mm requested-position envelope, 1 mrad minimum margin improvement, orientation fractions, official LMA and OMPL identities, Action Contract margins, pose gates, horizon, exact seed, and all later seed/label boundaries remain unchanged. Plan 051 exact must exercise the new basis and pass before tuning seed 1900 can open.
