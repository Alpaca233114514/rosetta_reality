# M2 SmolVLA Zen first-deviation trace — 2026-08-28

## 1. Authority and scope

Completion report for the preregistered diagnostic
`reports/training/m2-smolvla-zen-first-deviation-preregistration-2026-08-28.md`
(option 3 of the Zen audit's next-step list). Non-gating diagnostic; no
training, no checkpoint reuse, no hidden-test access, no closed-loop
acceptance claim. Executed after the two selected Zen deploy artifacts were
transferred to the local artifact root with full SHA256 verification
(`runs/…/artifact_backup/m2-smolvla450m-zen-deploy-artifacts-local-transfer-001.json`)
and the AutoDL instance was shut down again under the registered procedure.

## 2. Execution identity

| Field | Value |
|---|---|
| artifact | `m2-smolvla450m-zen-cuda-b64-firstaction-001-step0316-deploy-001`, manifest sha256 `d6b2a7ff…` (matches the preregistration and `gate4-smolvla-sim-422.json`) |
| protocol | train episode 2 / simulator seed 10 / policy noise seed 10 / ≤320 steps |
| runtime | local WSL Docker `vla-sim-xpu`, image `sha256:f4a71c40…`, memory 6g, networking disabled |
| frozen checks | all `_load_artifact` validations passed (selection derivation, manifest checksums, contract equality, precedents, resource boundary) |
| resets | cross-environment reset state MAE `0.0` (deterministic) |
| evidence | `runs/…/diagnostics/zen-trajectory-a4acabf885059bd2.json`; log/status `runs/…/orchestration/zen-trajectory-trace-001.{log,status}` (exit 0) |
| rendered plan / selection | `runs/…/plans/m2-smolvla450m-zen-firstaction-trace-sim-001.yaml`, `runs/…/selection/…-selection-trace-gate.json` (create-only, drift-checked) |

## 3. Result

The loop executed 294 steps and ended when the expert replay terminated with
task success at step 293 (expert maximum reward `4.0`, first nonzero reward at
step 186) — the registered expert replay baseline reproduced exactly. The
policy never received any reward (`policy_maximum_reward = 0.0`), never
terminated, and committed **zero** joint-limit violations and zero unexpected
collisions, consistent with its clean Gate 4 safety profile.

| Measure | Zen firstaction | Aster baseline (immutable trace) |
|---|---|---|
| step-zero action MAE | `0.032943226397037506` | `0.0204168` (+61% for Zen) |
| step-zero post-state MAE | `0.015197002328932285` | `0.0055942` (~2.7x for Zen) |
| state-MAE first crossings | `0.005`/`0.01` at step 0, `0.025` at step 1, `0.05` at step 18, `0.1` at step 29 | `0.01` at step 1, `0.025` at step 4, `0.05` at step 24, `0.1` at step 28 |
| maximum state MAE | `0.31062379479408264` | `0.222958` |
| final state MAE | `0.09667164832353592` | `0.097614` |
| mean time-indexed action MAE | `0.1736299404430957` | `0.119910` |
| policy maximum reward | `0.0` | `1.0` |

## 4. Interpretation (non-gating)

- The Zen first-action treatment artifact departs from the expert trajectory
  **earlier and faster** than Aster at the same reset: its very first executed
  action is already 61% further from the expert action, and the state error
  crosses both `0.005` and `0.01` immediately at step 0 (Aster crossed `0.01`
  at step 1 and `0.025` only at step 4). By the `0.05`/`0.1` thresholds the
  two profiles converge (step 18 vs 24, 29 vs 28) — divergence onset differs
  more than divergence magnitude.
- This is fully consistent with, and strengthens, the established
  training/closed-loop generalization classification: the failure again
  begins at step 0, before any simulator anomaly, limit violation or contact,
  and the expert replay succeeds in the same environment at the same seed.
  The batch-64 / 316-update regime did not improve the executed first action
  relative to the batch-8 / 2,500-update Aster regime — in line with the Zen
  campaign's rejection of the temporal-weight axis and the decoupling of
  offline MAE from closed-loop behavior.
- The zero-violation, zero-reward profile of this artifact in the trace
  matches its Gate 4 safety profile exactly; nothing here implicates the
  action boundary or the adapter chain.

## 5. What not to conclude

- This trace does not gate M2, does not rank the two Zen arms (only the
  first-action arm was traced, per the audit's safety-profile recommendation),
  and does not measure closed-loop task success.
- Time-indexed expert actions after divergence remain references, not
  recovery targets; no relabeling is authorized.
- The earlier divergence does not by itself authorize T2 recovery-data
  collection; that axis still requires a state-conditioned teacher that
  passes its own gate under a separate preregistration.

## 6. Evidence ledger

- `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/zen-trajectory-a4acabf885059bd2.json`
- `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/orchestration/zen-trajectory-trace-001.log`
- `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/orchestration/zen-trajectory-trace-001.status`
- `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/plans/m2-smolvla450m-zen-firstaction-trace-sim-001.yaml`
- `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/selection/m2-smolvla450m-zen-cuda-b64-firstaction-001-selection-trace-gate.json`
- `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/artifact_backup/m2-smolvla450m-zen-deploy-artifacts-local-transfer-001.json`
- local deploy artifacts under
  `artifacts/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/m2-smolvla450m-zen-cuda-b64-{uniform,firstaction}-001-step0316-deploy-001/`
