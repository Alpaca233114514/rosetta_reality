# M2 SmolVLA lift MoveIt fallback preregistration — 2026-08-16

Plan057 changes one IK fallback phase boundary. Plan056 preserved both grasp
contacts and the lift contact scope for 56 lift decisions, then failed once at
step 443 because local Mink QP returned weighted IK error
`0.0015606494501644484`, above the unchanged `0.001` gate.

Plan057 extends the already-registered official MoveIt/OMPL
`FixStartStatePathConstraints` fallback, currently `[approach, orient]`, to
include only `lift`. It activates only after the unchanged Mink QP solve fails.
All MoveIt identities, collision checks, joint margins, path constraints,
trajectory execution and acceptance gates remain unchanged.
