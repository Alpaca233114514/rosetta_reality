# M2 T2 seating-feasibility probe verdict (2026-09-03)

Privileged, read-only verdict on the open question recorded in the -003
closure (`m2-smolvla-t2-geometric-teacher-003-closure-2026-09-03.md` §4):
are the failing G2 poses seatable at all under this simulator's position
control, or is the remaining wall teacher-internal? Instrument:
`scripts/diagnose_g2_seating_feasibility.py` (teacher-free verdict; no
gate evidence written; no seeds outside the opened groups touched).

## Method

1. Calibration: the registered candidate-003 teacher solved G2 sibling
   1903 (step 305, reward 4); the seated peg->socket relative transform was
   captured at terminal success.
2. Probe (seed 1901, light run under the user's no-heavy-tasks boundary):
   the teacher ran to its INSERT stall (step 298, reward 2 — the stall
   detector fires at the align->insert transition pause, so the privileged
   controller took over EARLY, which only strengthens the test), then a
   privileged seater controlled every subsequent step: ideal socket target
   recomposed from the LIVE peg frame each step (no frozen captures), the
   left-site target anchored as ideal position plus the live socket->site
   world offset with the live orientation (position-priority — no
   orientation task pressure), the right arm held at its live pose,
   multistart constrained IK (unshifted first, shifts only above 5 mm
   residual), and the registered 0.06 rad/step action bound with closed
   grippers — the registered execution semantics, minus every teacher-side
   limitation.

## Result

- The privileged seater advanced the task one reward increment (2 -> 3,
  peg-socket contact achieved under live-geometry composition).
- It then floored at a **minimum socket-to-seated gap of 0.0269 m** with
  **no pin contact and no success** over the 60-step privileged budget —
  the same ~3 cm wall every teacher candidate measured, reproduced by a
  controller with no stage machine, no frozen transforms, no exit
  tolerances and full multistart.

## Verdict

The seating wall at seed 1901 is **not teacher-internal**. A contract-free
privileged controller with live recomposition cannot close the final
increment under the registered position-control action bound either; the
binding constraint sits in the physics/protocol envelope (position control
through the compliant cage at that loaded configuration), not in any
candidate's command generation. Caveats recorded honestly: the light run
used a 60-step privileged budget and a single seed (1904 unprobed); the
budget equals up to tens of centimetres of potential commanded travel, so
the floored gap indicates a genuine equilibrium, not insufficient time —
but a longer-budget confirmation remains open.

## Consequence

- Rebuilding another teacher of the reactive Cartesian-target family
  against the frozen G2 set has a low prior of reaching 5/5: the measured
  wall reproduces without any teacher limitation present.
- The evidence now supports treating this as a protocol-level finding
  (the G2 pose set demands precision beyond the position-control envelope
  at 1901, and plausibly 1904): any change to the frozen G2 seeds,
  physics, or actuation semantics is the user's decision and requires a
  new registration — it is not something a candidate may work around.
- A final teacher design (`-004`, live-geometry composition) is drafted at
  `m2-smolvla-t2-live-geometry-teacher-004-design-2026-09-03.md` with this
  risk stated in the document itself; it is ready to implement only if the
  user chooses to spend the compute despite the low prior.

## Boundaries

No gate thresholds, seeds or criteria were modified; no gate evidence was
written; only the already-opened seeds (10, 1900-1904) were touched; the
hidden test and all sealed groups stay sealed; no AutoDL, SSH, download or
external service was used.
