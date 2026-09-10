# M2 T2 state-conditioned teacher gate design — 2026-08-28

## 1. Scope and authority

This is a **design document only**.  It defines the teacher gate and the
teacher contract that the next T2 (recovery-distribution) axis must satisfy
before any data collection, label manifest, DAgger rollout or furnace may even
be discussed.  It authorizes nothing: no implementation plan, no seed group, no
exact/tuning/development/collection run, no recovery label, no policy training,
and no reopening of the geometry-planner scope lock (no Plan `059` follows from
this document).  All numeric thresholds here are **design proposals**; they
become binding only when frozen in the teacher's own preregistration.

Architecture-map reference: section 10 item 10 — recovery-distribution data
remains the only untried primary axis, but it requires a state-conditioned
teacher that passes its own gate first.

## 2. Why the previous two teachers failed, and what the gate must block

| Prior teacher | Core mechanism | Failure | Gate criterion that blocks a repeat |
|---|---|---|---|
| recovery oracle (`sim/recovery_oracle.py`, plans `001`/`002`) | robot-state nearest-neighbor retrieval against one translated successful reference | passed exact seed 10 (reward 4 in 294 actions) but stalled at reward 0 on the cross-pose tuning seed 1900 — robot-state proximity is not pose-general | G2 cross-pose generalization must pass **before** anything downstream opens |
| object-geometry teacher (plans `003`–`058`) | event/phase conditions, object-pose targets, Mink IK + official MoveIt paths, joint-margin and feedforward execution | increasingly safe but non-progressing: final chain ran 750 safe steps in `lift` with the right peg never leaving the table; every earlier failure was a feasibility/feedback boundary, never solved by another planner tweak | G3 recovery-from-deviated-states is measured directly, not inferred from safe-but-stalled trajectories; the gate forbids judging a teacher by safety alone |
| time-indexed expert actions | dataset replay labels | divergence begins at step zero under every trained policy; post-deviation expert actions describe the expert trajectory, not the correct action for the deviated state | the teacher contract (section 3) forbids time/trajectory/episode inputs by construction |

Two structural lessons are binding for the design: **safety without progress
is failure** (the Plan `058` boundary), and **one exact-seed success predicts
nothing about pose generalization** (the seed-1900 boundary).

## 3. Teacher contract (proposed)

```text
inputs : robot joint state (both arms + grippers),
        object geometry (peg pose, socket pose, grasped-contact flags),
        task milestone state (derived from observed reward transitions only)
output : one absolute standard-space action under the unchanged Action Contract
forbidden inputs: wall-clock or step time, trajectory/frame index,
        episode id, seed, dataset identity, any policy state
behavior: deterministic given inputs; fail-closed refusal outside the
        registered support neighborhood (refusal is an explicit output,
        never a silent default action)
```

The teacher may internally use the Rosetta-owned event/phase structure and the
already-validated safe components (bounded targets, joint-margin awareness,
official solvers as fallbacks), but its acceptance is defined by the gate
below, not by the internals.  A teacher that only executes a prerecorded plan
cannot pass G3 by construction; a teacher that only reacts cannot pass G2
without pose-general perception — the gate is what makes this claim testable
instead of rhetorical.

## 4. The teacher gate (staged, fail-closed, policy-free)

Every stage runs through a dedicated create-only runner, writes immutable JSON
evidence, and involves no policy checkpoint.  A failed stage stops the gate;
nothing later opens.  Seed groups stay strictly disjoint and in order.

- **G0 — contract conformance (static).** The teacher binary/entry refuses
  forbidden inputs (property-tested), is deterministic under fixed inputs, and
  its refusal region is non-degenerate (some off-support state must refuse).
- **G1 — exact calibration reproduction.** On the registered calibration
  episode/seed (episode 2 / seed 10 semantics), the teacher reaches task
  reward and terminates within the registered step budget, with zero
  joint-limit, unexpected-collision and out-of-contract actions.  This is the
  same bar the previous teachers met; it stays because it is cheap and
  necessary, and it is **insufficient by design**.
- **G2 — cross-pose generalization.** On the reserved tuning-seed group
  (seed 1900 and at least four further registered poses from the same fixed
  `sample_insertion_pose` ranges), the teacher must solve the task on **every**
  pose within the per-pose budget (proposal: 5/5; a single failure fails the
  gate).  This is the criterion the recovery oracle failed 0/1.
- **G3 — recovery from deviated states.** From each G2 pose, at a registered
  set of perturbed states (proposal: at least 3 per pose, sampled from
  registered bounded perturbations of successful trajectories and from the
  completed first-deviation trace states), the teacher must restore task
  progress — reach the next task milestone or increase reward within a
  registered recovery budget (proposal: ≤ 150 actions) — while keeping every
  safety criterion.  Refusals are recorded and must stay below a registered
  bound (proposal: ≤ 10% of perturbed states).  This stage is the reason the
  teacher exists: it directly measures the supervision the policy lacks.
- **G4 — boundary honesty.** Off-support inputs (out-of-reach poses, corrupted
  geometry) must refuse, not degrade.  A registered negative suite must show
  zero silent defaults.
- **G5 — independent audit.** All stages re-runnable from evidence by a
  reviewer without the authoring session; reports carry checksums, code
  identity, seed inventory and the full refusal/failure ledger.

## 5. Explicitly not authorized by a gate pass

A passed teacher gate permits exactly one thing: **proposing** the next
preregistration, which must separately define the recovery-data contract
(label semantics, state coverage, DAgger or offline mixing ratios, manifest
schema, train/validation isolation) and the policy-side single-axis comparison.
It does not open collection seeds, does not write any label manifest, does not
authorize a furnace, and does not modify the five-identity Gate 4 failure
record.

## 6. Relation to the current work sequence

The visual-conditioning state-dropout axis (preregistration
`m2-smolvla-visual-conditioning-state-dropout-preregistration-2026-08-28.md`)
proceeds independently and first: it needs no teacher.  T2 teacher
implementation work may proceed in parallel locally, but its gate execution,
like every ML run, requires the container path and its own registered
authorization.  If the visual-conditioning axis changes what the policy
actually conditions on, the teacher gate is unaffected — the gate evaluates
the teacher against the simulator contract, not against any policy.
