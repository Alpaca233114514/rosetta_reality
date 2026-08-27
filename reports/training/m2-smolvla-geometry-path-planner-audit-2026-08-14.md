# M2 SmolVLA geometry path-planner audit — 2026-08-14

## Scope and sealed boundary

This report records the create-only train-exact evidence for geometry-teacher
plans `010`--`014`. It supplements, and does not modify or supersede, the
historical plans `003`--`009` audit. All five plans kept the Action Contract,
the 3 mm maximum projected IK error, the 12 mm approach/per-step position
bounds, the train episode identity and the hidden-test boundary unchanged.

No tuning, development, collection or policy-Gate seed was executed. No
validation or hidden-test episode was loaded and no recovery label was written.
No CUDA training was started.

## Exact results

| Plan | Single planner axis | Last step / phase | Attempts / accepted waypoints | Maximum projected IK error | Result |
|---|---|---:|---:|---:|---|
| `010` | approach position priority with bounded orientation relaxation | 125 / orient | 28 / 27 | `0.006960180828197437` | failed |
| `011` | add orient priority with per-step position relaxation | 155 / orient | 58 / 57 | `0.006882439298032348` | failed |
| `012` | freeze the first orient target as a 12 mm position anchor | 129 / orient | 32 / 31 | `0.00803323605687403` | failed |
| `013` | include the fixed anchor in active-set pose IK at rotation weight `0.2` | 125 / orient | 28 / 27 | `0.006960180828197437` | failed |
| `014` | raise only the constrained orient solve weight to `1.0` | 125 / orient | 28 / 27 | `0.006960180828197437` | failed |

Plan `010` proved that active-set position waypoints can cross the historical
step-98 approach failure. Plan `011` proved that orientation-only active-set
waypoints can make additional progress, but its moving reference admitted
cumulative position drift. Plan `012` closed that defect and demonstrated the
remaining constrained-manifold boundary. Plans `013` and `014` showed that the
dm_control single-call weighted pose IK did not produce even the first accepted
orient waypoint at either registered weight. Further blind weight scanning is
not supported by this evidence.

## Immutable identities

The teacher implementation remained
`74b788fd316ef0f723d871db5c5cf550999998c37826843a268a31c4edddf5fd`
and `gym_aloha.py` remained
`e9c1005d0ae085e82e0c96e9d18527dce7d4749268a71756116cdffbb98d6e7d`.
Per-plan evaluator, plan and report hashes are recorded in the JSON companion.
Plan `010` is remote instance evidence; plans `011`--`014` are local
`vla-sim-xpu` container evidence.

## Current gate and next admissible hypothesis

Train-only exact remains failed, so tuning seed `1900` and every later group
remain sealed. A next plan must introduce a genuine constrained solver—for
example, a Jacobian null-space or bounded least-squares step that optimizes
orientation subject to an explicit position inequality and active joint
limits—and must preregister its convergence and feasibility checks. It must not
weaken the 1 mm IK, 3 mm projected IK, 12 mm position or Action Contract gates.

