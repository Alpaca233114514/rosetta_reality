# M2 SmolVLA expanded orientation target budget preregistration — 2026-08-16

Plan 054 changes one numeric planner input: `teacher.maximum_orientation_step_rad` from `0.04` to `0.20`. This is the bounded intermediate target supplied to the existing active-set planner, not an acceptance tolerance. The existing largest registered trust-region fraction remains `0.5`, so a selected progress candidate can request at most `0.10 rad` toward the current orientation target before official LMA, collision, path-constraint, and OMPL validation.

The sealed Plan 053 exact run used 601 planned waypoints and exhausted all 750 steps in `orient` despite 18 safe progress events. At its sealed step-711 state, the diagnostic generated 54 requests for the larger target and the official MoveIt server returned 30 valid constrained plans; nine used the unchanged `0.5` fraction.

The 1 mm / 3 mm pose gates, joint margins, waypoint joint-step cap, candidate basis, radii, fraction list, restoration rule, horizon, exact seed, and later seed/label seals remain unchanged.
