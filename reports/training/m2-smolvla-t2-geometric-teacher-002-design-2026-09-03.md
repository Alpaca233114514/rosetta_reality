# M2 T2 geometric-teacher candidate 002 design (2026-09-03)

Registered follow-up plan for the T2 teacher gate
`m2-t2-teacher-gate-001`, opening the successor candidate
`mink-geometric-insertion-teacher-002` after candidate 001 closed on its
preregistered stop condition
(`m2-smolvla-t2-geometric-teacher-candidate-diagnostic-2026-09-02.md`).
Authorization: explicit user decision 2026-09-03 to repair or rework the
candidate; this document is the new registered plan that authorization
requires. Gate thresholds remain frozen in the protocol; this candidate's
identity becomes immutable only at the create-only `register` step.

## Carried-over axis

Candidate 002 keeps the candidate-001 command layer exactly as measured at
its closure: exact stage-target constrained IK (Mink/DAQP, evaluator-proven
settings), full-gain direct command, one direction-preserving common scale
`s = min(1, 0.06 / max|joint delta|)`, and shifted multistart seeds gated to
the ALIGN/INSERT stages. The task plan stays the registered scripted set.

## Registered repairs (each targets one measured mechanism)

1. **Carrying height guard** (mechanism: transport-entry table press,
   G1-002 — both right fingers contacted the table because the park target
   sits ~7 mm below the carried pose). In stage TRANSPORT, while
   `peg_on_table` is observed the right-arm target z is clamped to the
   current right EEF z; while `socket_on_table` is observed the left-arm
   target z is clamped to the current left EEF z. No downward command is
   generated while the carried object rests on the table; the clamp is a
   no-op for upward targets and releases as soon as the object lifts
   (state-conditioned, no new constant, deterministic).
2. **Contact-phase command scale cap** (mechanism: descend side-loading —
   the left finger joint was levered to 0.0578 rad past its 0.057 limit
   during the fast bottom squeeze, steps 213-216 of the seed-10 diagnostic).
   In stages DESCEND, GRASP and TRANSPORT the common command scale is capped
   at `contact_phase_scale_cap = 0.5`, bounding contact-adjacent execution
   speed to roughly the scripted chain's effective rate while free-space
   phases (OPEN/APPROACH/ORIENT, and the ALIGN/INSERT precision chase) keep
   the full direct-command speed.

Both repairs are command-layer constants owned by the candidate; neither
touches the frozen gate thresholds, stage machine, tolerances or the Action
Contract.

## Budget and stop condition

At most two further hypothesis-driven repair rounds after this design,
create-only attempt increments, same semantics as before. If no registered
single-axis hypothesis produces a G1 pass within that budget, candidate 002
closes as method-level negative evidence like its predecessor, and the
scripted/geometric/002 chain of closures becomes the recorded evidence that
the state-conditioned insertion teacher gate blocks the recovery-data axis
pending a genuinely different teacher design.

## Repair round 1 measured outcome (2026-09-03, read-only diagnostic)

The two registered repairs of this design were implemented and measured on
seed 10 before any gate attempt; both failed to change their mechanisms:

- the contact-phase scale cap did not stop the descend wedge (the left
  finger joint was still levered to 0.0575-0.0579 rad at steps 213-217) —
  the press is contact-driven and speed-insensitive, because the direct
  command's target never yields;
- the carrying height guard did not stop the transport-entry table press —
  the left-arm lift tilts the socket inside its cage, the right wrist
  tracks the tilting carry frame, and the swinging fingers contact the table
  even with the target z clamped.

Root cause shared by both: the direct full-target command keeps pressing
into contact, where the scripted chain's glide made the target follow the
achieved pose so a blocked arm stops pressing.

## Repair round 2 (registered): contact-phase glide inheritance

In stages DESCEND, GRASP and TRANSPORT the command target is first bounded
by the registered scripted glide (`_bounded_target` with the unchanged
`maximum_cartesian_step_m = 0.025` and `maximum_orientation_step_rad =
0.12`), so a blocked arm sees its demand collapse onto itself and stops
pressing; the scale cap and the carrying height guard stay layered on top.
Free-space stages (OPEN/APPROACH/ORIENT) and the precision stages under the
registered axis (ALIGN/INSERT) keep the exact full-target command. The
seed-10 read-only diagnostic then completed the full task: reward 4.0,
success at step 379, zero refusals, zero joint-limit violations, zero
unexpected collisions.

## Gate outcome (2026-09-03)

- `register` + G0-001 PASSED; **G1-001 PASSED** (reward 4.0, success at
  step 379, zero violations/refusals/collisions) — the first geometric
  command-layer teacher to pass G1.
- **G2-001 FAILED at 2/5**: 1903 solved (306 steps) and 1904 solved for the
  first time across all candidates (418 steps); 1900/1901 deadlocked in
  TRANSPORT (reward 2, both arms static at their targets, the left arm
  ~1.2 cm outside the 0.012 meet tolerance while the peg is genuinely
  parked); 1902 refused at step 40 on a left upper-arm/lower-forearm
  self-collision during the APPROACH swing.

## Post-registration repair rounds (create-only, read-only previews)

**Round A** (transport exit tolerance 0.012 -> 0.02; APPROACH added to the
glide stages): unblocked the 1900/1901 transport deadlock (the dual-arm
loaded QP tracking floor is ~9-14 mm) and removed the 1902 self-collision;
seed 10 was not re-validated (later measured a regression, see round C).

**Round B** (descend-entry object recapture; align tolerance 0.02;
insert press-through 0.02 on `peg_socket_contact`; ALIGN/INSERT branch
selection experiments): the diagnostics measured, per seed —

- 1902's grasp closed on empty table because the socket was displaced
  ~4 cm from its OPEN-time capture (fixed by the descend-entry recapture);
- 1901's align stall was a multistart artifact (the shifted seeds pinned
  the carrying arm; unshifted align reached INSERT at reward 3);
- 1900's align phase, with the branch flip engaged mid-carry, swung the
  socket out of the cage (left orientation residual 1.789 rad) and dropped
  it — the flip needs the glide's gradual rotation;
- with the ALIGN/INSERT glide extended and an acceptance-only branch
  discipline, 1900/1901 both reached INSERT at reward 3 but 1903/1904
  regressed from solved to reward 3 and seed 10 struck the table in
  TRANSPORT (the loosened transport tolerance interacts with the entry
  state).

**Round C** (the G2-001 branch behaviour plus the round-A/B mechanical
fixes, align tolerance restored to 0.012): preview 1/5 with the seed-10
transport strike — worse than G2-001.

## Stop-condition verdict (2026-09-03)

Across the G2-001 composition and the round-A/B/C alternatives every
registered constant is pulled in opposite directions by different poses:
the transport tolerance that unblocks 1900/1901 breaks seed 10; the align
branch escape that rescues 1900 breaks 1901; the acceptance-only discipline
that saves 1901 breaks 1903/1904. **No single registered composition
reaches 5/5; per the stop condition candidate 002 closes as method-level
negative evidence with its best gate-measured result at G2 2/5 (G1
passed).** The branch code is left at the G2-001 composition and the test
suite documents it. The closure report
(`m2-smolvla-t2-geometric-teacher-002-closure-2026-09-03.md`) records the
residual hypotheses (a stall-triggered branch-escape state machine,
per-phase tolerance scheduling, the transport-tolerance/entry-state
coupling) — reopening requires a new registered plan.

## Boundaries

No gate threshold may be widened; no collection, label manifest or furnace
is authorized by any stage result; the hidden test, development/collection/
policy-gate seeds and the six-identity Gate 4 record stay sealed/untouched;
AutoDL remains shut down; no SSH or external service is used.
