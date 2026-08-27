# MoveIt terminal completion refresh preregistration (Plan 036)

Plan 035 did not fail because the position actuator missed the MoveIt goal. Its final 12-axis state was only `2.96434154734015e-05 rad` L1 from the original, uncompensated MoveIt goal. The larger `0.039295138703090406 rad` diagnostic was measured against the gravity-compensated controller reference. Static feedforward therefore did its intended job.

The actual failure is one layer higher. At step 97 the geometry teacher still had `0.0381825380027294 m` to travel but exposed one bounded `0.012 m` task-space increment. MoveIt planned and executed that increment, terminal control held it for 347 commands, and the evaluator never admitted the next current-state target. The final `0.026163499802350998 m` error is consistent with the unserved remainder.

Plan 036 changes one axis: terminal completion semantics. MoveIt 2.5.9 `SimpleSampler` reports full progress at the final reference waypoint, and the official `LocalPlannerComponent` then succeeds and resets the trajectory operator. The local adapter will mirror that lifecycle only after terminal control is active and the observed arm state is within `0.001 rad` L1 of the original, uncompensated MoveIt goal. It will then run the newly observed bounded target through the existing Mink-first, MoveIt-fallback pipeline. This is neither per-step global replanning nor failure-triggered retry.

Frozen teacher pose gates, the Action Contract, joint-limit margins, seed 10, MoveIt/OMPL identity, static feedforward, labels, validation, hidden episodes and later seeds remain unchanged. The machine-readable companion is the acceptance authority.
