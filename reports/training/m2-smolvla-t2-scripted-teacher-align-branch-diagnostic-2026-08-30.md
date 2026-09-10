# M2 T2 scripted-teacher align-branch diagnostic and stop-condition verdict (2026-08-30)

Execution of the preregistered diagnosis queue from
`m2-smolvla-t2-scripted-teacher-candidate-design-2026-08-29.md` for candidate
`gym-aloha-scripted-insertion-adapter-001` against the frozen staged gate
`m2-t2-teacher-gate-001`, plus two hypothesis-driven repair rounds and the
preregistered stop-condition assessment. JSON companion:
`m2-smolvla-t2-scripted-teacher-align-branch-diagnostic-2026-08-30.json`.

Verdict: **G2 remains failed after the repair budget; the failure
decomposition is complete and converged, and no registered single-axis
hypothesis reaches the 5/5 gate criterion. Per the stop condition recorded in
the design document, the scripted-adapter candidate class closes as
method-level negative evidence.** All gate thresholds, seeds and criteria
stayed frozen; every stage re-run is create-only; no collection, furnace or
AutoDL action was authorized or performed.

## 1. Probe instrument

`scripts/diagnose_scripted_teacher_g1.py` gained two read-only modes (no gate
evidence is written):

- `--mode align`: per-seed spawn pose/yaw extraction, the frozen
  socket-approach/align left-site targets captured at the transport freeze,
  the achieved-residual window, and iterated constrained-IK chases at the
  exact targets and at registered orientation variants, from the live frozen
  state and from `START_ARM_POSE`.
- `--mode dock`: a sweep of the peg-park dock distance `P` against both
  arms — the left insert target and the right dock target chased at each `P`
  in both wrist branches.

All runs used the pinned local image
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`
(`vla-sim-xpu`, network disabled, WSL Docker), matching the evidence-chain
identity of attempts 001–004.

## 2. Probe findings (align mode, seeds 1900–1904)

- **Spawn yaw is zero for every seed** (identity quaternions; the pose
  sampler randomizes position only). The literal "per-seed spawn yaw"
  hypothesis is dead.
- The per-seed variable is the **in-cage settling at the transport freeze**:
  for seed 1900 both objects flop heavily (peg tilt ~77°, socket tilt ~94°,
  target yaw +34.4°); for 1902/1903/1904 the frozen frames stay near-identity
  (target yaw −2.5°/+3.3°/−0.9°).
- The parallel align target **stalls from the live state** at
  0.174 m (1900) and 0.034 m (1902/1904) while the same position converges
  with the wrist yawed ±90°/180° and from `START_ARM_POSE` for every seed.
  The position is reachable; the greedy chase sits in the wrong wrist branch.
- Recomposing the site target rigidly through a 180°-flipped socket frame
  **stalls** (position shift 2.6–4 cm) — the measured convergent flipped
  branch keeps the parallel-composed site position and flips only the wrist
  orientation. The socket hangs near-upright in the compliant cage
  regardless of wrist tilt, so the socket body target is identical for both
  branches.
- The flipped wrist cannot reach the +0.10 m hover climb (0.03–0.12 m stall
  for every seed); the flipped candidate therefore bypasses
  SOCKET_APPROACH and glides directly to its align target.

## 3. Repair round 1 (G2 attempt 003)

Registered adaptations: (a) deterministic fixed-order alignment multistart —
parallel first, then a 180°-about-world-Z wrist flip, selected by a
frozen-state constrained-IK probe (thresholds 0.008 m / 0.05 rad, ≤40
iterations); (b) descend glide converted to an entry-anchored ladder with a
tightened 0.012 m step (the candidate fix for seed 1901's lateral drift).

Chain: G0-002 passed (`dae1407f…`), G1-007 passed (seed 10, 306 steps,
reward 4.0, zero violations — unchanged path, parallel branch). G2 attempt
003 failed (`ce5761a7…`): 1903 solved (269 steps); 1900/1902 died with
**new descend-phase table strikes** (right-finger/table at steps 172/171);
1901 unchanged (step 177); 1904 flipped through align (entered at step 237,
aligned within 0.007 m at 244) and stalled in INSERT (0.034 m, reward 2).

Attribution: the descend ladder **removed the live glide's lateral homing** —
seeds whose approach exits 2–3 cm off-center now drive the fingers straight
into the table beside the object; it is a measured regression and was
withdrawn. The flip works online up to align. The 1901 outcome is untouched
by command-layer changes (identical step 177 across attempts 001–003).

## 4. Dock-distance sweep (dock mode, seed 1904; 1900/1902 declined to freeze)

The dock distance `P` moves the right arm's park target and — with it — the
left arm's insert target (`dock − 0.09`); the align target is P-independent
(`dock − P` returns to the socket's own position). Sweep at the frozen state
(left insert residual, flipped branch): 0.042 (P=0.16) → 0.035 (0.14) →
0.030 (0.12) → 0.028 (P=0.10/0.08). **A ~0.03 m residual floor remains**;
reducing `P` cannot rescue the insert. Parallel-branch insert residuals stay
~0.010–0.013 at every `P`. The constructed right-dock chase converges at
every `P`, but the **real** transport stage parks the peg with a
seed-random ±3.5 cm error inside the 0.04 position tolerance (measured dock
error: 1903 −31 mm in y, 1904 +36 mm in y), and the left arm's insert reach
correlates with the resulting dock y (1903 y=0.460 converges; 1904 y=0.527
stalls).

## 5. Repair round 2 (G2 attempt 004)

Registered change: withdraw the descend ladder (restore the registered
live-pose glide); keep the alignment multistart. Chain: G0-003 passed
(`d397aacb…`), G1-008 passed (seed 10, 305 steps, reward 4.0). G2 attempt
004 failed (`7656b80d…`):

| seed | success | length | max reward | refusals / collisions |
|---|---|---|---|---|
| 1900 | false | 500 | **3.0** | none |
| 1901 | false | 177 | 0.0 | table strike at 177 (unchanged) |
| 1902 | false | 500 | **3.0** (2 joint-limit violations) | none |
| 1903 | true | 269 | 4.0 | none |
| 1904 | false | 500 | 2.0 | none |

The failure picture is now converged: 1900/1902 survive descend, flip at the
freeze, align, slide the socket onto the peg and stall **one reward increment
below the terminal pin contact**; 1904 stalls earlier at the drifted dock;
1901 is the independent descend-execution failure; 1903 is preserved
byte-compatibly by the fixed order.

## 6. Stop-condition assessment

Across attempts 002–004 the failure decomposition is complete:

1. **Align branch trap** (1900/1902/1904) — solved by the registered
   multistart flip (round 1).
2. **Descend ladder** — registered, measured as a regression, withdrawn
   (round 2); the live-glide homing is load-bearing.
3. **Insert final approach** (1900/1902/1904) — the left arm stalls 0.02–0.03
   m short of the pin-press position at the drifted dock; the dock carries
   the right arm's ±3.5 cm seed-random park error; the dock sweep bounds the
   reachable improvement short of zero.
4. **Descend execution drift** (1901) — identical table strike at step 177 in
   every attempt; the registered command-layer candidate (ladder + tightened
   step) was measured ineffective; the drift is QP/physics-layer execution
   scatter (~2–3 cm) against a 2 cm-wide object.

The gate requires all five poses solved. Reaching 5/5 would require fixing
(3) and (4) simultaneously; (4) has no surviving registered hypothesis after
two rounds, and the documented press-through idea for (3) (deepen the insert
command once `peg_socket_contact` is observed, converting the position
command into a holding force through the compliant cage) has a best case of
3/5 — it cannot touch 1901 or 1904. **The repair budget cannot produce a gate
pass; per the preregistered stop condition the scripted-adapter candidate
class closes as method-level negative evidence.** The residual hypotheses are
recorded here and are not exercised; reopening the class would require a new
registered plan.

Next registered alternative per the design document: the Mink geometric
teacher stack (dependencies already pinned in the sim image), which performs
full constrained IK with proper multistart instead of greedy per-step
tracking, targeting exactly the execution-scatter failure mode measured here.

## 7. Boundaries

Gate thresholds, seeds and acceptance criteria were never modified; all gate
evidence is create-only (attempts numbered, failures preserved); the hidden
test, collection seeds 3000–3004 and the six-identity Gate 4 record stay
sealed/untouched; no AutoDL, SSH, download or external service was used; the
frozen observation contract was not modified.
