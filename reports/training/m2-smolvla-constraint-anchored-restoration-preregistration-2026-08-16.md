# M2 SmolVLA constraint-anchored restoration preregistration — 2026-08-16

Plan 053 keeps the 1 mrad restoration buffer but measures it from the unchanged command-margin boundary. This is the active-set hysteresis invariant: a restoration goal must reach at least `command margin + 0.001 rad`. It does not require another full milliradian relative to an already-safe current state.

At the sealed Plan 052 failure, the only official LMA candidate reached `0.0467601859750193` rad, which is `0.00135556342024025` rad inside the command boundary while preserving the original pose tolerances. The candidate was rejected only by the prior current-relative comparison.

Orientation-first scheduling, feedback basis, radii, position/orientation envelopes, official planner, hard margins, pose gates, horizon, seed, and later seed/label seals remain unchanged.
