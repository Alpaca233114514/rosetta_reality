# M2 SmolVLA Faust trainer and optimizer audit — 2026-08-12

## 1. Authority, scope and reading order

This is the current source of truth for the completed Faust run, its repaired
action boundary, and the remaining trainer/optimizer defects. It supersedes the
status sections of
`reports/training/m2-smolvla-action-repair-handoff-2026-08-12.md`, but preserves
that document as the initial diagnosis boundary.

An AI resuming this work should read, in order:

1. `AGENTS.md`;
2. `docs/m2-smolvla-architecture.md` for the stable component and control-flow map;
3. this document;
4. `reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.json`;
5. `configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml`;
6. `configs/vla/smolvla_450m_aloha_insertion_faust_batch8_002.yaml`;
7. `configs/vla/smolvla_450m_aloha_insertion_faust_sim_001.yaml`;
8. the immutable evidence files in section 11.

Do not edit the hash-bound Faust plans or reinterpret Gate 4 failure as M2
acceptance. New fixes need new IDs, new configs and controlled comparisons.
Hidden-test episodes `[31, 6, 1, 24, 5]` remained sealed.

## 2. Executive verdict

Faust completed one fresh, revision-pinned SmolVLA-base pass on XPU with
`batch_size = 8`. AdamW and the scheduler remained finite, all four checkpoints
were saved, and the selected step-1875 checkpoint improved validation action
MAE from `0.17051533` to `0.04465636` (`73.81%`). Export/reload reproduced the
action exactly, Gate 3 passed, and raw, projected and executed action-contract
violation rates were all zero.

Faust nevertheless failed Gate 4: `0/5` task successes across five 500-step
closed-loop rollouts. Four seeds had maximum reward `0`; seed `1003` reached
maximum reward `2` but did not succeed. The repaired action representation is
therefore effective as a safety and serialization fix, but it did not solve the
policy-quality problem.

The main remaining problem is not an AdamW crash. It is a mismatch between what
the trainer optimizes and what the simulator executes, amplified by expert-only
behavior cloning distribution shift and a state-dominant shortcut. Optimizer
instrumentation and resume coverage also need repair before changing
hyperparameters.

## 3. Faust identity and result

| Field | Registered value or result |
|---|---|
| codename | `Faust` |
| formal run | `m2-smolvla450m-faust-b8-002` |
| base initialization | pinned original `lerobot/smolvla_base`, not an old furnace or overfit checkpoint |
| formal plan SHA-256 | `7f8215b23c21b1b5685cbcc86ee74c9bd691ebf26214e562204bfa135d592c46` |
| batch / updates / exposures | `8 / 2,500 / 20,000` |
| accelerator | Intel XPU, BF16, Docker Linux launched from WSL Bash |
| peak XPU allocated | `3,428,736,512` bytes, below the `7,516,192,768`-byte guardrail |
| checkpoints | `625`, `1250`, `1875`, `2500` |
| selected checkpoint | step `1875`, model SHA-256 `b1ec752c8ec78689a737624454bd01578a3d92c19ddbc899420af02a602bbb73` |
| export artifact | `m2-smolvla450m-faust-b8-step1875-001` |
| hidden test | not loaded |
| public Trackio sync | not performed; local durable metrics retained |

The prior batch-1 Faust attempt is immutable interruption evidence, not a
failed/OOM run. It was stopped by explicit request at observed optimizer step
`3071` with `SIGTERM`/exit `143`, `oom_killed = false`, then replaced by the
batch-8 run.

Validation checkpoints:

| Checkpoint | Action MAE | First-action MAE | Fixed flow loss |
|---:|---:|---:|---:|
| base | 0.17051533 | 0.18022069 | 3.05913060 |
| 625 | 0.06827670 | 0.05012882 | 0.17708165 |
| 1250 | 0.04869220 | 0.03073250 | 0.12605575 |
| **1875** | **0.04465636** | 0.02756259 | 0.11213726 |
| 2500 | 0.04472017 | **0.02734508** | **0.11101718** |

Step 1875 won the preregistered primary metric. Step 2500 was slightly better
on the two secondary numbers but slightly worse on action MAE; this is not a
large regression and does not establish overtraining.

## 4. Confirmed engineering bugs repaired

These are completed fixes, not proposed trainer experiments:

1. **Action representation and gripper legality.** Standard-space grippers are
   projected to `[0, 1]`, trained through the registered bounded-sine mapping,
   and decoded through `(sin(x) + 1) / 2`. Gate 3/4 raw decoder outputs remained
   legal without relying on final clipping.
2. **Checkpoint/resume transient memory.** Explicit garbage collection, XPU
   cache release and allocator trim were added around save/load boundaries. The
   fixed overfit resume completed after the earlier transient OOM.
3. **No-execute compiler cache.** Triton/Inductor cache was moved from a
   no-execute temporary mount to the run-scoped executable cache.
4. **Export/reload boundary loss.** Export now discovers and copies the
   serialized processor states and reinstalls the registered action boundary.
   Reload action equality was exact (`max_abs_diff = 0`).
5. **Simulation reload boundary loss.** The simulation wrapper now loads the
   source Action Contract and repaired processor boundary rather than treating
   an exported dataclass JSON file as the source YAML.
6. **Quarter-check inspection.** A window inspector handles checkpoint steps
   that have no exact log row. This repairs monitoring, but section 5 finding
   `T9` describes the underlying trainer/logging contract still to fix.

## 5. Trainer findings

### T1 — P1: loss horizon does not match executed horizon

**Status:** confirmed contract mismatch; contribution to Gate 4 failure is a
strong, not isolated, hypothesis.

The pinned SmolVLA forward computes elementwise flow-matching MSE and averages
uniformly across all valid `50 x 14` chunk elements. Faust simulation executes
only the first action from each predicted chunk (`n_action_steps = 1`,
receding-horizon first action). There is no first-action, early-horizon,
gripper, contact-phase or task-progress weighting.

**Why it matters:** most training gradient is spent on future chunk elements
that this controller never executes. A low aggregate chunk loss can coexist
with an error in the single action that closes the loop.

**Required repair:** expose unreduced `[batch, horizon, action_dim]` loss and
register a controlled weighting contract. First compare uniform loss against
an early-horizon weighting while keeping data, base checkpoint, exposures and
optimizer fixed. Log total, first-action, arm and gripper losses separately.
Do not silently change the existing Faust artifact.

**Acceptance test:** the selected metric must be deployment-aligned and the new
candidate must improve fixed-seed Gate 4, not only offline MAE.

### T2 — P1: expert-only BC has no recovery distribution

**Status:** confirmed training regime; causal mechanism is strongly consistent
with the evidence but has not been isolated experimentally.

Faust is one-pass expert-only behavior cloning. Teacher-forced validation stays
on expert observations, while closed-loop evaluation observes states induced by
the policy. Gate 4 produced safe actions and good offline MAE but no successes.

**Required repair:** add recovery demonstrations or a DAgger-style loop using a
state-conditioned oracle. A time-indexed expert action is not a valid recovery
label after the rollout has deviated. First add trajectory/deviation evidence,
then collect a small, revisioned recovery set with provenance.

### T3 — P1: checkpoint selection noise differs from deployment noise

**Status:** confirmed protocol mismatch.

Post-hoc validation selects on `noise = zeros` and fixed `flow_time = 0.5`.
Gate 3/4 uses seeded standard-normal inference noise, matching normal SmolVLA
sampling. The selected offline checkpoint is therefore not necessarily the
best checkpoint under deployment stochasticity.

**Required repair:** evaluate each checkpoint on a preregistered ensemble of
fixed Gaussian noise seeds, report mean/worst-case first-action and chunk
metrics, and keep a deterministic zero-noise diagnostic as a separate metric.

### T4 — P1: policy is state-dominant despite using images

**Status:** confirmed diagnostic result.

At step 1875, normal teacher-forced chunk MAE was `0.04469046`. Cross-episode
same-phase image shuffling raised it to `0.05812579` (`+30.06%`), zeroing images
raised it to `0.06889522` (`+54.16%`), and state shuffling raised it to
`0.10253406` (`+129.43%`). Images are genuinely present and used; the policy is
nevertheless much more dependent on state.

The run also used `n_obs_steps = 1`, disabled image transforms, a frozen vision
encoder, `train_expert_only = true`, and a trainable state projection. These
facts do not prove that unfreezing vision is the right fix.

**Required repair:** add per-module gradient/update norms, controlled visual
augmentation, temporal/history ablations and off-trajectory/recovery data.
Treat backbone adaptation as a separate experimental axis.

### T5 — P2: batch-1 to batch-8 changed the optimization path

**Status:** confirmed confound, not a runtime failure.

Keeping 20,000 sample exposures while moving from batch 1 to batch 8 reduced
the planned parameter updates from 20,000 to 2,500 and changed the LR schedule.
The batch-1 run did not complete, so Faust does not prove batch equivalence.

**Required repair:** run small controlled comparisons with (a) equal exposures
and (b) equal optimizer updates. Do not infer optimizer quality from throughput.

### T6 — P1: bounded-sine gripper decoder is safe but periodic

**Status:** confirmed representation risk.

The decoder guarantees standard-space bounds for arbitrary internal outputs,
but many internal angles map to the same gripper command. In fixed-overfit
acceptance, `33.25%` of right-gripper internal predictions were outside the
training support even though standard outputs were legal.

**Required repair:** test a dedicated bounded gripper head or add an internal
support penalty. A sigmoid replacement must explicitly handle the near-closed
and near-open endpoints; do not replace the mapping without an endpoint test.

### T7 — P1: formal training has no tested resume mode

**Status:** confirmed protocol gap.

The formal action-repair runner exposes only `preflight`, `smoke` and `train`.
The fixed-overfit resume test passed, but the formal batch-8 optimizer,
scheduler, RNG and dataloader state were not resumed and compared.

**Required repair:** add a create-only formal resume plan that asserts exact
model, optimizer, scheduler, RNG, dataset-view and next-batch identities. Compare
an uninterrupted control against a stop/resume run at the same final step.

### T8 — P2: no validation or early stopping during training

**Status:** intentional Faust protocol limitation, not an accidental bypass.

`eval_steps = 0`, `env_eval_freq = 0` and validation occurred post hoc at the
four quarter checkpoints. This followed the requested quarter-only inspection
protocol, but the trainer could not detect quality changes between checkpoints.

**Required repair:** keep expensive simulation outside the hot loop, but add a
small deterministic validation slice at registered checkpoint boundaries.

### T9 — P2: checkpoint and metric logging schedules are misaligned

**Status:** confirmed instrumentation defect; workaround exists.

Checkpoints occur every `625` steps while metrics log every `10`, so steps 625
and 1875 have no exact metric row. The window inspector chooses nearby rows,
which is sufficient for monitoring but not ideal provenance.

**Required repair:** require `save_freq % log_freq == 0` or write an exact
metric/resource snapshot transactionally with every checkpoint.

### T10 — P2: Gate 4 lacks first-deviation diagnostics

**Status:** confirmed evidence gap.

Gate 4 records reward maxima, action legality, collisions, smoothness and
latency, but not first reward step, object/EEF poses, first deviation from an
expert/state-conditioned oracle, or a compact raw-internal gripper trace.

**Required repair:** add a non-gating trajectory trace and extract the first
deviation window. This must precede another long full-data furnace.

## 6. Optimizer and scheduler findings

### O1 — Healthy: AdamW executed stably

The registered contract was AdamW `lr=1e-4`, betas `(0.9, 0.95)`,
`eps=1e-8`, `weight_decay=1e-10`, global norm clip `10`, with 125-step warmup
and cosine decay to `2.5e-6` over 2,500 updates. All logged losses and gradients
were finite, all checkpoints were saved, final LR matched the expected endpoint,
and peak XPU allocation stayed inside the guardrail. There is no evidence of an
AdamW/XPU numerical failure.

### O2 — P1 instrumentation: the logged gradient norm is pre-clip only

The maximum logged gradient norm was `21.93872`, greater than the clip threshold
`10`. This does **not** mean clipping failed: Accelerate returns the total norm
from `clip_grad_norm_`, and the pinned trainer logs that returned value.

The actual defect is missing evidence: there is no post-clip norm, clipping
fraction, nonfinite count per module, or expert/projection gradient breakdown.

**Required repair:** log pre-clip global norm, post-clip global norm, whether
clipping fired, clipping coefficient, and per-module norms/update ratios.

### O3 — P2 scheduler contract: nominal peak LR is never reached

The pinned cosine-with-warmup implementation uses `current_step < warmup_steps`
for warmup, then computes cosine decay from the absolute step rather than from
`current_step - warmup_steps`. At the transition, it has already entered the
cosine. Faust's maximum **logged** LR was `9.929468e-5`, not the declared
`1e-4` peak.

This sub-percent discrepancy is not a credible sole explanation for Gate 4
failure, but the configuration contract is misleading.

**Required repair:** either define the field as a nominal scale or shift cosine
progress to start at zero after warmup and cover only the remaining decay
interval. Add an exact LR-sequence unit test at steps `0`, `warmup-1`,
`warmup`, `warmup+1` and final.

### O4 — P2 observability: one optimizer group hides module imbalance

SmolVLA exposes all trainable parameters to one AdamW group. The expert and
state/action projections therefore share LR and decay, and current logs cannot
show which group dominates updates. This is a risk given the state-shuffle
result, not proof that differential LRs are needed.

**Required repair:** instrument named-module gradients and update/parameter
ratios first; only then preregister a differential-LR experiment.

### O5 — Not proven faulty: weight decay is effectively disabled

`weight_decay = 1e-10` is operationally negligible. Faust provides no controlled
evidence that stronger decay would improve closed-loop behavior. Record it as
an explicit no-regularization choice and test it only after T1–T4 diagnostics.

### O6 — Not proven faulty: EMA is disabled

EMA decay is `None`. EMA may smooth candidate checkpoints, but no Faust evidence
shows it is required. Treat it as a separate candidate experiment, not a bug fix.

### O7 — Future compatibility risk: scheduler stepping with accumulation

The pinned trainer calls scheduler `step()` per micro-batch. Faust used gradient
accumulation `1`, so this did not affect the completed run. Before enabling
accumulation greater than one, verify that the prepared scheduler advances only
on real optimizer updates and that skipped mixed-precision updates do not
advance LR.

## 7. What not to conclude

- `0/5` Gate 4 means Faust is not an accepted M2 checkpoint.
- Good offline MAE does not prove a usable closed-loop policy.
- Legal gripper output does not prove the internal bounded-sine latent stayed
  in distribution.
- The `21.94` gradient norm does not prove clipping failed.
- Batch 8 is faster here, but is not proven statistically equivalent to batch 1.
- Image-shuffle degradation proves images are used; it does not prove visual
  reliance is sufficient.
- Do not unfreeze the backbone, change AdamW, add EMA and change the loss in one
  experiment. Those are separate axes.

## 8. Recommended repair order

1. Add exact checkpoint metrics plus pre/post-clip and per-module optimizer
   diagnostics; add formal resume parity.
2. Add deployment-noise validation and first-deviation trajectory traces.
3. Run a small controlled horizon/loss-weighting experiment aligned to the
   first executed action.
4. Add a bounded-gripper latent-support experiment.
5. Add small recovery/state-conditioned-oracle data; then compare against the
   expert-only control.
6. Only after those results, test history/augmentation, differential LR,
   regularization, EMA or backbone adaptation one axis at a time.

Stop any candidate on nonfinite values, action-contract regression, resume
identity mismatch, hidden-test access, or loss of reproducible Gate 3 behavior.

## 9. External primary references

- SmolVLA paper: <https://arxiv.org/abs/2506.01844>
- Pinned SmolVLA model/loss source:
  <https://github.com/huggingface/lerobot/blob/c903b114a90e703b3f7d0c46cb38727c328c55ff/src/lerobot/policies/smolvla/modeling_smolvla.py>
- Pinned LeRobot trainer source:
  <https://github.com/huggingface/lerobot/blob/c903b114a90e703b3f7d0c46cb38727c328c55ff/src/lerobot/scripts/lerobot_train.py>
- Pinned scheduler source:
  <https://github.com/huggingface/lerobot/blob/c903b114a90e703b3f7d0c46cb38727c328c55ff/src/lerobot/optim/schedulers.py>
- Pinned optimizer source:
  <https://github.com/huggingface/lerobot/blob/c903b114a90e703b3f7d0c46cb38727c328c55ff/src/lerobot/optim/optimizers.py>
- Accelerate gradient clipping API:
  <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_>
- ACT / action chunking and compounding error:
  <https://arxiv.org/abs/2304.13705>
- DAgger / induced observation distribution:
  <https://proceedings.mlr.press/v15/ross11a.html>

## 10. Implemented repair entry points

- Action boundary: `src/rosetta_reality/vla/processor.py`
- Checkpoint memory handling: `src/rosetta_reality/vla/checkpoint_memory.py`
- Formal Faust protocol: `scripts/run_smolvla_action_repair_formal.py`
- Quarter inspection: `scripts/inspect_smolvla_faust_window.py`
- Repaired export: `scripts/export_smolvla_action_repair.py`
- Repaired simulation gates: `scripts/smolvla_action_repair_sim_gate.py`
- Modality diagnostic: `scripts/diagnose_smolvla_action_repair_modalities.py`

## 11. Immutable evidence ledger

Paths below are relative to `runs/` or `artifacts/` as shown. Do not edit them.

- Selection:
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/selection/m2-smolvla450m-faust-b8-002-selection.json`
- Final quarter state:
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/monitoring/m2-smolvla450m-faust-b8-002-step-002500-window.json`
- Gate 3:
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/gates/gate3-smolvla-sim-002.json`
- Gate 4:
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/gates/gate4-smolvla-sim-002.json`
- Modality audit:
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/teacher-forced-modalities-step001875-32f1c4b80472280b.json`
- Fixed-overfit acceptance:
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/acceptance/repair-bounded-overfit-xpu-003.json`
- Batch-1 interruption:
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/orchestration/m2-smolvla450m-faust-001-interrupted.json`
- Artifact:
  `artifacts/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/m2-smolvla450m-faust-b8-step1875-001`

Key evidence hashes:

| Object | SHA-256 |
|---|---|
| experiment config | `0e9dd0499d0708939ac73cc5d517849f133cf6deab072d9cde09f2880ae22210` |
| formal Faust plan | `7f8215b23c21b1b5685cbcc86ee74c9bd691ebf26214e562204bfa135d592c46` |
| simulation plan | `20386f7ad5dda19d9d5ffa1668bfa45df5ae3f6441d94f45661eb9f10fa442f3` |
| selected model | `b1ec752c8ec78689a737624454bd01578a3d92c19ddbc899420af02a602bbb73` |
| artifact manifest | `39409420eab02f7fbb7d4c27644392496e1a7e10a7e6eeb0311e5a090279f137` |
| Gate 3 report | `ded31186c6b198c6c002b299ae42a8900370f0021a86770a69049979ee082a92` |
| Gate 4 report | `af7c75dfb5c8bd975d65c6db559812b25b8725e35ce33c9e7448c12f766b7e00` |

## 12. Resume boundary

No further furnace is authorized by this document. The next agent should first
implement instrumentation and a new preregistered, small controlled experiment.
It must preserve original caches, old furnace checkpoints, Faust artifacts and
hidden-test isolation. Do not commit, push, upload or publish without explicit
authorization.
