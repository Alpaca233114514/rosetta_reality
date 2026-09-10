# M2 T2 geometric-teacher candidate design (2026-09-02)

Design note for the second real teacher candidate of the frozen staged gate
`m2-t2-teacher-gate-001`
(`reports/training/m2-smolvla-t2-teacher-gate-protocol-preregistration-2026-08-29.md`).
This is a design, not a preregistration: the gate thresholds are already frozen
in the protocol; the candidate identity becomes immutable only at the
create-only `register` step. Work branch: `codex/t2-scripted-teacher-candidate`.

## Motivation and single registered axis

The scripted-adapter candidate class closed 2026-08-30 as method-level negative
evidence with a complete failure decomposition
(`m2-smolvla-t2-scripted-teacher-align-branch-diagnostic-2026-08-30.md`):

- the align wrist-branch IK trap was solved by the registered fixed-order
  alignment multistart (parallel -> 180-degree world-Z flip, frozen-state
  probe);
- the descend live glide is load-bearing (the entry-anchored ladder was a
  measured regression and was withdrawn);
- 1900/1902 stall 0.02-0.03 m short of the pin-press position in INSERT and
  1904 stalls at the seed-random drifted dock, with a ~0.03 m residual floor
  in the flipped branch;
- 1901 strikes the table at step 177 identically in every attempt: ~2-3 cm of
  QP/physics execution scatter against a 2 cm-wide object, which the
  command-layer candidates could not repair.

The registered next alternative per that report is the **Mink geometric
teacher stack**: full constrained IK with proper multistart instead of greedy
per-step tracking, targeting exactly the measured execution-scatter failure
mode. This candidate implements that alternative as one axis:

> **Single axis = the action-conversion (command) layer.** The task plan
> (stage machine, stage targets, tolerances, advancement conditions, the
> transport-freeze alignment multistart, gripper targets, descend stall/floor
> guards, workspace guards and refusal classes) is reused verbatim from the
> registered scripted candidate `gym-aloha-scripted-insertion-adapter-001`.
> The command layer changes from {glide the stage target toward the live pose
> -> one greedy IK solve -> damped 0.7-gain joint pursuit} to {deterministic
> multistart full-pose constrained IK on the exact stage target -> direct
> joint command bounded only by the registered smoothness clamp}.

Everything else — the frozen observation contract, the Action Contract, the
observer module, the gate thresholds and seeds — is unchanged.

## Candidate

- Module `src/rosetta_reality/sim/geometric_teacher.py`; class
  `MinkGeometricInsertionTeacher` (subclass of `ScriptedInsertionTeacher`
  overriding only `_task_target_action`); factory `build_teacher()`.
- Suggested identity: `mink-geometric-insertion-teacher-001` (frozen at
  registration). Observer module: the existing
  `rosetta_reality.sim.scripted_teacher_observer` (unchanged frozen field
  extraction).
- Solver: the registered `MinkAlohaIkSolver` with the evaluator-proven
  `default_ik_settings()` (unchanged, including the 0.0454 rad joint-limit
  margin and DAQP backend).

### Registered command-layer constants (`GeometricCommandSettings`)

- **IK seed multistart** (deterministic fixed order, applied to the expanded
  16-D qpos; indices are expanded arm joints — 3 left forearm_roll, 5 left
  wrist_rotate, 11 right forearm_roll, 13 right wrist_rotate):
  1. the current state (no shift) — tried first so unchanged behavior is
     preserved when the current branch already converges;
  2. left wrist_rotate +90 deg;
  3. left wrist_rotate -90 deg;
  4. right wrist_rotate +90 deg;
  5. right wrist_rotate -90 deg;
  6. left wrist_rotate +180 deg;
  7. right wrist_rotate +180 deg;
  8. left forearm_roll +180 deg;
  9. right forearm_roll +180 deg.
  The solver clips each seed into the inset joint limits itself. The seed set
  is the discrete wrist-branch family measured by the branch diagnostic
  (the same position converges under rotZ180 flip, +/-90 wrist variants and
  from the START pose; forearm_roll shifts realize the same world-Z flip
  through the other wrist DOF).
- **Residual metric** for seed selection: max over both arms of
  `position_error_m + rotation_weight * orientation_error_rad` with the
  evaluator-registered `rotation_weight = 0.2`
  (`configs/sim/aloha_insertion_geometry_teacher_003.yaml`).
- **Seed acceptance rule**: the first seed in the fixed order whose solve
  reaches position <= 0.004 m AND orientation <= 0.05 rad on BOTH arms is
  accepted immediately; otherwise the lowest-residual solve is kept. If every
  seed raises, the teacher refuses `out_of_support` (never degrades).
- **Direct command**: the accepted/best solve's arm joints are commanded
  directly — `gain = 1.0` (the scripted 0.7 damped pursuit is removed) —
  clamped per joint by the unchanged registered smoothness bound
  `maximum_joint_target_delta_rad = 0.06`. No target gliding: the IK target
  is the exact stage target.

### Targeted failure modes

| Measured failure | Mechanism |
|---|---|
| insert final approach stall (1900/1902/1904) | per-decision multistart escapes the residual-floor wrist branch the greedy chase sat in; direct pursuit removes the damped lag that left a 0.02-0.03 m gap |
| descend execution scatter (1901) | commands are full-IK solutions of the exact object-center target instead of one-step damped chases, so lateral scatter shrinks toward the solver residual |
| align branch trap | inherited stage-layer alignment multistart (unchanged) plus per-decision seed multistart as defense in depth |

## Refusal semantics (inherited + command layer)

Unexpected collisions refuse `unsafe_state`; object poses outside the
registered workspace refuse `workspace_violation` (both inherited verbatim).
A solver failure on every multistart seed refuses `out_of_support`. Corrupted
geometry cannot reach the class (the frozen observation constructor rejects
non-finite inputs). Nothing degrades silently.

## Stop condition and repair budget

The gate thresholds (G1 seed 10; G2 seeds 1900-1904 all solved) are frozen in
`m2-t2-teacher-gate-001` and never modified. This candidate gets at most two
hypothesis-driven repair rounds (create-only attempt increments, same
semantics as the scripted candidate). If no registered single-axis hypothesis
reaches 5/5 within that budget, the geometric candidate class closes as
method-level negative evidence and the residual hypotheses are recorded,
not exercised. Read-only diagnosis may reuse the existing probe modes
(`scripts/diagnose_scripted_teacher_g1.py`) and new read-only modes bound to
this candidate.

## Boundaries

Single axis = one new teacher candidate measured by the frozen gate. No
threshold may be widened; no collection, label manifest or furnace is
authorized by any stage result; the six-identity Gate 4 failure record stays
untouched; the hidden test, development seeds 2000-2004, collection seeds
3000-3004 and policy-gate seeds 1000-1004 stay sealed; AutoDL remains shut
down; no SSH or external service is used.

## Implementation session outcome and closure (2026-09-02)

The candidate is implemented, registered and measured; the gate is NOT
passed and the class is closed. Evidence under
`runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/teacher_gate/mink-geometric-insertion-teacher-001/`:

- `register` + G0-001/002/003 PASSED (one per code identity);
- G1-001 FAILED: step 71 approach-phase collision (left gripper bar x
  socket-2); 8-10 of 12 arm joints saturated the per-joint clamp every step
  and both arms wandered non-monotonically;
- repair round 1 (direction-preserving common command scale, replacing the
  per-joint clamp): read-only diagnostic — approach/orient/descend/grasp
  clean, transport entered at reward 2, then a branch switch while carrying
  lost the peg (workspace_violation at step 259);
- repair round 2 (shifted seeds gated to ALIGN/INSERT): G1-002 FAILED —
  transport first step pressed both right fingers into the table
  (unsafe_state refusal at step 227, reward 1.0, 4 joint-limit violations);
  the read-only diagnostic and the formal evidence agree step-for-step.

**Stop condition reached: the two-round budget is spent without a G1 pass;
the geometric command-layer class closes as method-level negative evidence**
(`m2-smolvla-t2-geometric-teacher-candidate-diagnostic-2026-09-02.md`).
Residual hypotheses recorded there (transport-entry height guard, registered
branch-switch selection margin, minimum command-scale floor) are not
exercised; reopening requires a new registered plan. Both T2 candidate
classes (scripted adapter 2026-08-30, geometric command layer 2026-09-02)
are now closed negative against the frozen gate.
