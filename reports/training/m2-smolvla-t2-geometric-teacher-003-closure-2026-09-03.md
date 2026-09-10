# M2 T2 geometric-teacher candidate 003 closure report (2026-09-03)

Final report for candidate `mink-geometric-insertion-teacher-003`
(`reports/training/m2-smolvla-t2-geometric-teacher-003-design-2026-09-03.md`)
against the frozen staged gate `m2-t2-teacher-gate-001`. JSON companion:
`m2-smolvla-t2-geometric-teacher-003-closure-2026-09-03.json`.

Verdict: **the gate is not passed; candidate 003 closes at G2 3/5 (G1
passed) — the best teacher result recorded, but the stall-triggered escape
axis is spent and the remaining wall is structural for this command
layer.** All gate thresholds, seeds and criteria stayed frozen; every stage
run is create-only; no collection, furnace or AutoDL action was authorized
or performed.

## 1. Registered axis

The candidate-002 coupling table showed every pose-constant escape pulled
in opposite directions by different poses. Candidate 003 replaced the
constants with stall-triggered, state-conditioned escapes over
teacher-internal EEF-position history (contract-legal — the scripted
descend-stall precedent):

1. **transport stall unlock** — static arms with the peg parked hold the
   meet target at the current pose inside `_meet_left_target` (the method
   the registered exit reads), so the exit fires through its own condition
   and the normal freeze captures run;
2. **align stall escape** — shifted branch seeds engage after ALIGN stalls
   (best-residual with early acceptance break), plus the final
   floor-acceptance form: when the achieved socket gap is within the
   registered 0.03 m physical margin, the achieved pose becomes the
   alignment reference and the exit fires (the measured floor: the
   carrying arm tracks the site composition to 8 mm while the socket hangs
   14 mm off the 12 mm park tolerance);
3. **insert stall press** — with `peg_socket_contact` and a stall, the
   target presses 0.02 m deeper along the peg axis.

## 2. Gate evidence (create-only, immutable)

- G0-001 PASSED (`e9920249…`); G1-001 PASSED (`c9f1797c…`: seed 10, 379
  steps, reward 4.0, zero violations/refusals/collisions — the escapes
  never fire on the non-stalling calibration pose).
- G2-001 FAILED (`ae6103f6…`): **3/5** —

| seed | result | mechanism |
|---|---|---|
| 1900 | **pass** (287 steps) | first 1900 solution across every candidate; the full escape chain end-to-end |
| 1901 | fail, r3 | INSERT stall: socket ~3 cm short of the seating target at the 13 mm loaded QP floor; the press-through does not close it; no exit condition exists to unlock — the success criterion is physical pin contact |
| 1902 | fail, step 40 | APPROACH self-collision (left upper-arm/lower-forearm); the round-A-measured fix (APPROACH glide) is not part of this candidate's registered base |
| 1903 | **pass** (306 steps) | unchanged from G2-001 — the escapes never fire |
| 1904 | fail, r2 | INSERT stall after the align floor-acceptance advanced with the achieved alignment |

## 3. What the escapes proved

- State-conditioned acceptance works where a virtual-referenced exit is
  the binding constraint: transport (1900/1901 deadlock unlocked; 1900
  solved) and align (1901/1904 advanced past the 14 mm-vs-12 mm floor
  conflict) — with zero interference on non-stalling poses (seed 10 and
  1903 byte-stable across the whole line).
- The remaining wall is different in kind: INSERT seating demands physical
  pin contact, and at these poses the dual-arm loaded QP floor (8-14 mm)
  plus the compliant-cage hang leaves the socket centimetres short with
  the arm demonstrably at its floor. No exit to unlock; the command layer
  cannot manufacture force beyond the registered press-through.

## 4. Succession

Teacher tally against the frozen gate: scripted adapter 1/5 (2026-08-30),
geometric command layer G1-fail (2026-09-02), geometric composition 2/5
(2026-09-03), geometric stall-escapes 3/5 (this report). Each closure's
residual wall moved deeper into the task (transport deadlock → align
floor → insert seating). Before any further teacher design (the standing
user authorization for a teardown), the sharp unanswered question is
whether the G2 poses 1901/1904 are seatable AT ALL under this simulator's
position control: a privileged read-only feasibility diagnostic
(global-plan the insertion offline for those poses, teacher-free) would
distinguish "the teacher line is insufficient" from "the G2 pose set
demands precision beyond the simulator's controllable envelope" — the
latter is a protocol-level finding that belongs to the user's decision,
not another candidate.

## 5. Boundaries

Gate thresholds, seeds and acceptance criteria were never modified; all
gate evidence is create-only; the hidden test, development/collection/
policy-gate seeds and the six-identity Gate 4 record stay sealed/untouched;
no AutoDL, SSH, download or external service was used; the frozen
observation contract and Action Contract were not modified.
