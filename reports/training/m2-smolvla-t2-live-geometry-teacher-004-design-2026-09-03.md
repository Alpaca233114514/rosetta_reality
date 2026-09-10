# M2 T2 live-geometry teacher candidate 004 design (2026-09-03, drafted pending the compute decision)

Teardown-and-redo design per the standing user authorization ("完全不行的话
你就根据相关文档来重做"), drafted after the seating-feasibility probe
verdict (`m2-smolvla-t2-seating-feasibility-probe-2026-09-03.md`). This is
a design, not a preregistration: the gate thresholds stay frozen in
`m2-t2-teacher-gate-001`; the candidate identity becomes immutable only at
the create-only `register` step. It is ready to implement only if the user
chooses to spend the compute — see the honest risk assessment.

## Honest risk assessment (read first)

The privileged probe measured that a contract-free controller with
per-step live-frame recomposition, position-priority anchoring and full
multistart STILL floors ~2.7 cm short of seating on seed 1901 under the
registered action bound. Candidate 004 removes every teacher-side
limitation the probe removed, so by construction it cannot beat the probe
at the seating increment. Its realistic ceiling is the probe's own: reward
3 on 1901-class poses. **The design below is therefore worth implementing
only as the complete, evidence-clean embodiment of the reactive-family
endgame (and to clear 1902), not as a likely 5/5.** If the goal is a gate
pass, the evidence points at the protocol question (G2 pose set vs the
position-control envelope), which is the user's decision under a new
registration, not a teacher redesign.

## Registered axis

Replace the frozen-capture target composition of the -003 stack with
**per-decision live-geometry composition**: every stage target composes
from the current observation's object frames (the probe's proven
mechanics), with the stage machine, stall escapes, contact-phase glide,
height guard and scale cap retained from the registered -003
composition. Concretely, relative transforms are re-derived from the live
observation each decision instead of captured once at stage transitions:

- grasp/insert targets: `compose(live_socket/peg, live_eef-relative)`
  compositions recomputed per decision;
- the transport park reference stays the registered socket-frame park
  body (a task definition, not a capture);
- the alignment multistart probe and the stall escapes keep their
  registered semantics.

Plus the one measured uncoupled fix from the candidate-002 round-A
evidence: APPROACH joins the glide stages (the 1902 self-collision
mechanism).

## First-step fidelity axis (added 2026-09-03 after measurement)

Two independent measurements established that every teacher rollout has
been starting on the wrong configuration manifold:

1. **Teacher step-0 self-deviation 0.117 rad** (teacher first action vs
   its own current state, seed-10 calibration replay,
   `scripts/diagnose_teacher_first_step.py`): the OPEN-phase "hold home"
   command was re-solved through the constrained QP, whose redundancy
   resolution lands on a different arm configuration than the reset pose;
   the state then drifts to that configuration during the opening phase
   (state-vs-expert MAE 0.019 -> 0.094 over 11 steps). Every episode of
   every candidate inherited this.
2. **Dataset frame-0 degeneracy** (`scripts/diagnose_first_action_spread.py`):
   all 50 episodes share bit-identical frame-0 states (pairwise max
   difference 0.000000) while their expert first actions spread
   MAD 0.0134 / pairwise 0.0185 — the same magnitude as the policies'
   recorded step-0 deviation (Aster 0.020, Zen 0.033), quantitatively
   closing the "vision ignored -> first action is the episode mean"
   mechanism for the policy line.

The candidate-004 command layer therefore implements the **zero-delta
hold**: a stage target that coincides with the live end-effector pose
(within `hold_position_epsilon_m = 1e-3`, `hold_orientation_epsilon_rad =
1e-2`) is commanded as the current joint state — no IK re-solve, no
first-step yank. The hold branch returns the phase's gripper targets
unchanged.

## Frame-0 vision probe (pending at drafting time)

`scripts/diagnose_frame0_vision_probe.py` feeds the deployed artifacts
each non-hidden episode's frame-0 image with the identical frame-0 state
(zero noise, deterministic flow) and measures whether the emitted first
action is episode-specific (vision used) or collapses to the episode mean
(vision content unused at the only frames where it is the sole
disambiguating signal). Its verdict on the vcdropout artifact vs the
Zen-uniform baseline decides the fix-vs-rebuild split for the policy
line and is recorded in the closure/probe reports when it completes.

## Budget and stop condition

At most two hypothesis-driven repair rounds; if G2 does not reach 5/5,
the reactive Cartesian-target family closes with this candidate as its
final, evidence-clean member and the protocol question stands as the
recorded blocker.

## Boundaries

No gate threshold may be widened; no collection, label manifest or furnace
is authorized by any stage result; the hidden test, development/
collection/policy-gate seeds and the six-identity Gate 4 record stay
sealed/untouched; AutoDL remains shut down; no SSH or external service is
used; the frozen observation contract and Action Contract are not
modified.
