# M2 SmolVLA Athena joint-limit follow-up audit (2026-08-15)

## Outcome

Athena did not pass train-only exact, so no later seed or label gate opened.
The furnace nevertheless converted the plan `022` infrastructure ambiguity
into three controlled results:

| Plan | Single axis | Exact result |
|---|---|---|
| `023` | official Interbotix resources, collision identity and bounded arm-state reconciliation | failed at step 98; live `right_wrist_rotate` was `0.005985603256225769` rad outside the physical bound |
| `024` | `0.01`-rad upstream Mink ConfigurationLimit arm margin | removed the arm start-bound violation; failed at step 97 on the separate Gym-open `0.058` m / official finger-upper `0.057` m representation mismatch |
| `025` | bounded `0.001`-m Gym-to-official finger adapter | executed 21 of 22 official RRTConnect fallback waypoints and advanced to step 168; then failed when live `right_forearm_roll` exceeded the physical bound by `0.0005873297882081907` rad |

Plan `025` is the strongest result: official MoveIt 2.5.9, OMPL RRTConnect,
LMA, 22 collision links / 22 shapes, the original Action Contract and the
original 1 mm / 3 mm pose gates all remained active. Its exact report SHA-256
is `bfd55bc72f9f85cc19a8e3de58d2a15a72e65f41ded6818910c353a961acac2e`.
Maximum official goal errors among successful fallbacks were
`3.773987103663825e-05` m and `0.0009541714104034111` rad.

## Next safe axis

Do not widen the start-state reconciliation or pose gates. The measured
remaining issue is that MoveIt-generated executable waypoints can approach a
hard joint bound even though Mink commands use a safety margin. The next plan
should use official MoveIt joint path constraints (or an equivalently
collision-checked official constraint mechanism) to apply the same registered
margin to the complete planned path and every executed waypoint. It must be a
new plan/workspace with the exact seed only; tuning, development, collection,
policy-Gate, validation, hidden and recovery-label gates stay sealed.
