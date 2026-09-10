# M2 T2 geometric-teacher candidate diagnostic and stop-condition verdict (2026-09-02)

Execution of the registered candidate `mink-geometric-insertion-teacher-001`
(`reports/training/m2-smolvla-t2-geometric-teacher-candidate-design-2026-09-02.md`)
against the frozen staged gate `m2-t2-teacher-gate-001`: initial
implementation, two preregistered repair rounds, and the stop-condition
assessment. JSON companion:
`m2-smolvla-t2-geometric-teacher-candidate-diagnostic-2026-09-02.json`.

Verdict: **G1 was never passed; the two-round repair budget is spent, and per
the stop condition recorded in the candidate design the geometric command
layer closes as method-level negative evidence.** All gate thresholds, seeds
and criteria stayed frozen; every stage run is create-only (G0-001/002/003,
G1-001/002); no collection, furnace or AutoDL action was authorized or
performed; the scripted candidate class closure is unaffected.

## 1. Registered axis

The candidate reuses the scripted task plan verbatim
(`gym-aloha-scripted-insertion-adapter-001`) and changes one layer: the
action conversion solves the exact stage target with the constrained
Mink/DAQP QP and commands the solved arm joints directly at full gain,
replacing the scripted {target glide -> greedy solve -> 0.7-gain pursuit}
chain that produced the measured ~2-3 cm execution scatter. Deterministic
wrist-branch multistart seeds target the measured align-branch trap.

## 2. Initial implementation (G1-001, failed)

Full-target multistart plus a per-joint 0.06 rad clamp. Seed 10 refused at
step 71 (`unsafe_state`, one unexpected collision): the left gripper bar
struck `socket-2` during APPROACH. Trace: 8-10 of 12 arm joints saturated the
per-joint clamp every step, no seed converged from the far approach state
(weighted residual 0.067-0.148), and both end-effectors wandered
non-monotonically. The per-joint clamp distorts the joint-space direction
(joints with different levers are truncated to equal magnitude), turning the
approach into a flail. This re-measures why the scripted glide existed.

## 3. Repair round 1: direction-preserving scale (read-only diagnostic)

Registered change: the smoothness bound is applied as one common scale
`s = min(1, 0.06 / max|joint delta|)` instead of per-joint clamping, keeping
the commanded path on the solved IK direction at the same worst-case speed.

Result: the approach became clean (seed-0 convergence, scale 0.59), ORIENT
converged (residual 0.291 -> 0.009), DESCEND/GRASP completed, and TRANSPORT
entered with reward 2. New failure: with both arms carrying, the shifted
wrist-branch seeds won the residual selection (deltas ~1.2-1.5 rad -> common
scale 0.04-0.06, crawling), the arm wandered while dragging the held peg, the
grasp was lost, and at step 259 the peg rested at z = -0.0001 m — a
`workspace_violation` refusal. A branch switch while carrying is unsafe.

## 4. Repair round 2: stage-gated branch seeds (G1-002, failed)

Registered change: the shifted multistart seeds engage only in the
alignment-critical stages (ALIGN/INSERT) where the branch trap was measured;
carrying and early stages solve the current branch only.

Result: ORIENT crawl reduced (scale >= 0.25), DESCEND/GRASP clean
(residual 0.003), and TRANSPORT failed on its first step — both right
fingers contacted the table (2 unexpected collisions), refusing at step 227
with reward 1.0 and 4 joint-limit violations. The park target sits ~7 mm
below the carried pose at transport entry, and the direct full-gain command
presses the fingertips into the table where the scripted 0.7-gain + glide
chain survived the same geometry marginally. The read-only diagnostic and
the formal G1-002 evidence agree step-for-step (deterministic rollout).

## 5. Stop-condition assessment

The gate requires G1 (seed 10) before G2. Across the initial implementation
and both repair rounds the command layer produced three distinct failure
mechanisms (approach flail, carry-phase branch loss, transport-entry table
press) and never completed the calibration episode. The two-round budget is
exhausted; no registered single-axis hypothesis reaches a G1 pass, so the
geometric candidate class closes as method-level negative evidence.

Residual hypotheses are recorded here and are NOT exercised; reopening the
class requires a new registered plan:

1. transport-entry height guard: lift or z-clamp the park target while
   carrying (mirrors the scripted descend floor guard);
2. a registered selection margin requiring a shifted seed to beat the
   unshifted solve before a branch switch is accepted while carrying;
3. a minimum per-decision command scale floor to bound the crawl observed
   in ORIENT/TRANSPORT.

The T2 teacher-gate stage remains the blocker for the recovery-data axis:
two candidate classes (scripted adapter, geometric command layer) are now
closed negative against the frozen gate, and the historical recovery-oracle
and geometry-teacher lines failed earlier. A future teacher candidate needs
a new registered plan and must clear G1-G5 before any collection or furnace.

## 6. Boundaries

Gate thresholds, seeds and acceptance criteria were never modified; all gate
evidence is create-only (attempts preserved on failure); the hidden test,
development/collection/policy-gate seeds and the six-identity Gate 4 record
stay sealed/untouched; no AutoDL, SSH, download or external service was
used; the frozen observation contract and Action Contract were not modified.
