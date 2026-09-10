# M2 T2 geometric-teacher candidate 003 design (2026-09-03)

Registered successor plan for the T2 teacher gate `m2-t2-teacher-gate-001`,
opening candidate `mink-geometric-insertion-teacher-003` after candidate 002
closed on its preregistered stop condition
(`m2-smolvla-t2-geometric-teacher-002-closure-2026-09-03.md`). Authorization:
explicit user decision 2026-09-03 — repair further, and if it still cannot
pass, tear the design down and redo without sentiment. This document is the
new registered plan that authorization requires. Gate thresholds remain
frozen in the protocol; the candidate identity becomes immutable only at
the create-only `register` step.

## Registered axis

The candidate-002 closure proved that every pose-constant escape is pulled
in opposite directions by different poses (its coupling table). The
single axis of candidate 003 replaces those pose-constants with
**stall-triggered, state-conditioned escapes**: each escape engages only
when the observed geometry shows the system has stopped making progress,
so poses that never stall keep the exact candidate-002 G2-001 behaviour
(the composition that passes G1 and solves 1903/1904).

The stall detector is teacher-internal history over observed
end-effector positions (the same contract-legal pattern as the scripted
descend-stall guard): no time, index, episode or seed input; deterministic
given the observation sequence; reset with the teacher.

## Escapes (each targets one measured coupling)

1. **Transport stall unlock** (coupling: transport exit tolerance 0.012
   keeps seed 10 safe vs 0.02 unblocks 1900/1901). In TRANSPORT, when both
   end-effectors have moved ≤ `stall_progress_m` per step over
   `stall_window_steps` decisions AND the peg is parked within the
   unchanged 0.04 transport tolerance, the meet targets are replaced by
   the current end-effector poses for that decision. The frozen stage
   machine then exits through its own registered condition on the next
   observation (the arms are static, so the 0.012 meet check passes) and
   executes the normal freeze captures and alignment probe. The exit
   tolerance itself is untouched, so seed 10 (which never stalls) is
   unaffected.
2. **Align stall branch escape** (coupling: best-residual branch selection
   rescues 1900 vs pins 1901). ALIGN solves unshifted by default. When the
   stall detector fires in ALIGN, the shifted multistart seeds engage under
   acceptance-only selection (a shifted seed is used only if it converges
   within the registered tolerances; otherwise the unshifted solve is
   kept), and the targets inherit the scripted glide for the remainder of
   the stage so the wrist branch flip rotates gradually instead of
   sweeping the socket out of the cage (the measured 1.789-rad residual
   drop). Poses that progress through align unshifted (1901, 1903, 1904)
   never trigger the escape.
3. **Insert stall press** (the r3 wall: 1900/1901 reach one reward
   increment below the terminal pin contact). In INSERT, once
   `peg_socket_contact` is observed and the arms stall, the left target
   presses `insert_press_through_m = 0.02` deeper along the peg axis
   (latched), converting the position command into insertion force through
   the compliant cage. The measured always-on variant did not convert r3
   on any pose; the stall-triggered variant applies it only where the
   free track has already failed, and the diagnostic will measure the
   remaining peg-to-pin gap before any depth change is proposed.

Base composition (unchanged from the candidate-002 G2-001 state): scripted
task plan verbatim, exact stage-target constrained IK, full-gain direct
command with the direction-preserving common scale, contact-phase glide in
DESCEND/GRASP/TRANSPORT, carrying height guard, contact-phase scale cap,
INSERT branch seeds under best-residual selection.

## Repair rounds (2026-09-03, pre-registration previews; recorded before
the formal chain like the candidate-002 precedent)

**Round 1 — escape-1 layer correction.** The initial transport unlock held
the command-layer target, but the registered exit evaluates the unmodified
meet target inside the stage machine, so the exit never fired and the
0.002/step stall bound mis-triggered on normal deceleration (1903
regressed). Fix: the hold is applied in `_meet_left_target` — the method
the exit itself reads — with the stall bound tightened to the registered
precedent 0.0015/step. Preview: seed 10 unchanged (379 steps, solved — the
escapes never fire on non-stalling poses), **1900 solved for the first
time** (286 steps), 1903 preserved (305 steps).

**Round 2 — align escape final form.** The acceptance-only selection
(round-2 first attempt) and the cage-transform recapture (round-2 second
attempt) did not move 1901/1904: the measured stall shows the carrying arm
tracking the site composition to its 8 mm QP floor while the socket body
sits 14 mm off the park pose against the 12 mm exit tolerance — the
demanded precision is below the achievable floor. Final form: when ALIGN
stalls with the escape engaged and the achieved gap is within the
registered 0.03 m physical margin (anchored to the scripted dock-sweep
residual floor), the achieved socket pose becomes the alignment reference
and the exit fires through its own condition; the insert channel composes
from the frozen peg frame and absorbs the residual. Preview: 1901 reached
INSERT at reward 3, 1904 reached INSERT at reward 2.

## Measured wall at closure time

10/1900/1903 solve; 1901 stalls in INSERT with the socket ~3 cm short of
the seating target at the 13 mm loaded QP floor (the 0.02 press-through
does not close it); 1904 stalls earlier in INSERT; 1902 strikes its own
upper arm during the APPROACH swing (step 39, the round-A-measured fix —
APPROACH glide — is not part of this candidate's registered base). The
insert success criterion is physical pin contact: unlike transport/align
there is no exit condition to unlock, so the loaded-floor wall is
structural for this command layer at these poses.

## Gate outcome (2026-09-03)

- `register` + G0-001 PASSED; G1-001 PASSED (seed 10, 379 steps, reward
  4.0, all-zero safety counts — the escapes never fire on the
  non-stalling calibration pose).
- G2-001 FAILED at **3/5** — the best teacher result recorded: 1900 solved
  for the first time (287 steps, the full escape chain end-to-end) and
  1903 preserved (306 steps); 1901 stalled in INSERT at reward 3 (socket
  ~3 cm short of seating at the 13 mm loaded QP floor), 1902 struck its
  own upper arm in APPROACH (step 40), 1904 stalled in INSERT at reward 2.
- Closure: `m2-smolvla-t2-geometric-teacher-003-closure-2026-09-03.md`.
  The stall-escape axis is spent; the remaining wall (physical insert
  seating at the loaded QP floor) is structural for this command layer.

## Repair budget and stop condition

At most two hypothesis-driven repair rounds after this design (create-only
attempt increments). If no registered single-axis hypothesis reaches a G2
5/5 pass within that budget, candidate 003 closes like its predecessors
and — per the standing user authorization — the next step is a teardown:
a genuinely different teacher design (not another composition of this
command layer), registered as a new plan.

## Boundaries

No gate threshold may be widened; no collection, label manifest or furnace
is authorized by any stage result; the hidden test, development/collection/
policy-gate seeds and the six-identity Gate 4 record stay sealed/untouched;
AutoDL remains shut down; no SSH or external service is used; the frozen
observation contract and Action Contract are not modified.
