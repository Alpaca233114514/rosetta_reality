# Deterministic MoveIt subgroup IK preregistration (Plan042)

MoveIt 2.5.9's official subgroup implementation tries the request start state first, then draws unseeded random restart states until a wall-clock timeout. Plan040 and Plan041 failed at the same third-request boundary with 0.50 and 2.00 second budgets, so increasing time does not create reproducible exact evidence.

Plan042 copies the official subgroup structure into the sidecar while changing only seed scheduling: start state first, then `random_numbers::RandomNumberGenerator(2210)`, capped at 256 bimanual candidates and the existing 2.00-second outer limit. Each subgroup still uses the loaded official LMA `getPositionIK` implementation. Every complete candidate must pass the unchanged full-state bounds, joint-path-constraint, and self-collision checks before OMPL RRTConnect sees it.

The sidecar must fail closed unless subgroup order, solver base/tip frames, seed, attempt cap, and solver identity match. The same request must return the same IK goal on repetition. Plan040's contact-phase feedforward and all pose, collision, margin, horizon, seed, and label gates remain frozen.
