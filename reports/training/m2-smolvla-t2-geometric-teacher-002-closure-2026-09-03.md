# M2 T2 geometric-teacher candidate 002 closure report (2026-09-03)

Final report for candidate `mink-geometric-insertion-teacher-002`
(`reports/training/m2-smolvla-t2-geometric-teacher-002-design-2026-09-03.md`)
against the frozen staged gate `m2-t2-teacher-gate-001`. JSON companion:
`m2-smolvla-t2-geometric-teacher-002-closure-2026-09-03.json`.

Verdict: **the gate is not passed; per the preregistered stop condition the
candidate closes as method-level negative evidence with its best
gate-measured result at G2 2/5 (G1 passed).** This is the strongest teacher
result recorded so far — the scripted-adapter class reached 1/5 and never
beyond reward 3 on two poses, while this candidate passes G1 and solves two
poses outright — but the gate requires 5/5 and the repair-budget
experiments prove no single registered composition reaches it. All gate
thresholds, seeds and criteria stayed frozen; every stage run is
create-only; no collection, furnace or AutoDL action was authorized or
performed.

## 1. Registered composition (the branch's final code state)

The scripted task plan verbatim plus the geometric command layer: exact
stage-target constrained IK (Mink/DAQP), full-gain direct command, one
direction-preserving common scale (0.06 rad bound), contact-phase glide in
DESCEND/GRASP/TRANSPORT (0.025 m / 0.12 rad per step), the carrying height
guard (no downward target while the carried object rests on the table), the
contact-phase scale cap (0.5), and ALIGN/INSERT shifted multistart seeds
under lowest-weighted-residual selection (`rotation_weight = 0.2`).

## 2. Gate evidence (create-only, immutable)

- G0-001 PASSED (`be0097ad…`).
- G1-001 PASSED (`5677ac8e…`): seed 10, success at step 379, reward 4.0,
  zero refusals, zero joint-limit violations, zero unexpected collisions.
- G2-001 FAILED (`f1ae3442…`): 2/5 —

| seed | result | detail |
|---|---|---|
| 1900 | fail | 500 steps, reward 2, TRANSPORT deadlock (left arm ~1.2 cm outside the 0.012 meet tolerance; peg genuinely parked, peg2park 0.013 m) |
| 1901 | fail | 500 steps, reward 2, same TRANSPORT deadlock |
| 1902 | fail | refused at step 40 (`unsafe_state`): left upper-arm/lower-forearm self-collision during the APPROACH swing |
| 1903 | **pass** | 306 steps |
| 1904 | **pass** | 418 steps — the first pose 1904 solution across every teacher candidate |

## 3. Post-registration repair exploration (read-only previews; no gate
evidence consumed)

- **Round A** — transport exit tolerance 0.02 + APPROACH glide: unblocked
  1900/1901 transport and removed the 1902 self-collision (the dual-arm
  loaded QP tracking floor is ~9-14 mm, just above the scripted 0.012 meet
  tolerance).
- **Round B** — descend-entry object recapture (1902's socket was displaced
  ~4 cm from its OPEN-time capture, grasp closing on empty table), align
  tolerance 0.02, insert press-through 0.02 on `peg_socket_contact`, and
  ALIGN/INSERT branch-selection experiments: 1901's align stall is a
  multistart artifact (unshifted align reaches INSERT), 1900 needs the
  branch escape (unshifted align stalls), and an accepted-branch flip
  mid-carry without the glide swung the socket out of the cage (left
  orientation residual 1.789 rad, socket dropped). With the ALIGN/INSERT
  glide and acceptance-only switching, 1900/1901 reached INSERT at reward 3
  but 1903/1904 regressed from solved to reward 3 and seed 10 struck the
  table in TRANSPORT.
- **Round C** — the G2-001 branch behaviour plus the round-A/B mechanical
  fixes: preview 1/5 with the seed-10 transport strike. Worse than G2-001.

## 4. Coupling verdict (why the class closes)

Each registered constant is pulled in opposite directions by different
poses:

| constant | value A | value B | conflict |
|---|---|---|---|
| transport exit tolerance | 0.012 | 0.02 | seed 10 safe vs 1900/1901 unblocked |
| align branch selection | best-residual | unshifted / acceptance-only | 1900 rescue vs 1901 pin |
| ALIGN/INSERT glide | off | on | 1903/1904 solved vs slowed to reward 3 |

No single registered composition satisfies all five poses. Further
composition search would be fitting the gate, not testing a hypothesis.

## 5. Residual hypotheses (recorded, not exercised)

1. A stall-triggered branch-escape state machine: keep the unshifted solve
   and engage the multistart only after N steps without target progress,
   making the branch selection state-conditioned instead of
   pose-constant.
2. Per-phase tolerance scheduling tied to the measured QP tracking floor
   with a subsequent precision re-approach, instead of one static
   tolerance.
3. The transport-tolerance/entry-state coupling: understanding why seed
   10's transport entry strikes the table under the loosened tolerance
   (the freeze captures the carry transforms earlier) may decouple the
   conflict.

The T2 teacher-gate stage remains the blocker for the recovery-data axis:
three candidate classes are now closed (scripted adapter 2026-08-30,
geometric command layer 2026-09-02, geometric composition family
2026-09-03 — this report). A future teacher needs a new registered plan and
must clear G1-G5 before any collection or furnace is preregistered.

## 6. Boundaries

Gate thresholds, seeds and acceptance criteria were never modified; all
gate evidence is create-only; the hidden test, development/collection/
policy-gate seeds and the six-identity Gate 4 record stay sealed/untouched;
no AutoDL, SSH, download or external service was used; the frozen
observation contract and Action Contract were not modified.
