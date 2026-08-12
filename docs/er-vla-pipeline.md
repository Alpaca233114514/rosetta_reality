# ER / VLA Pipeline

> Current M2 implementation and status navigation lives in
> [`m2-smolvla-architecture.md`](m2-smolvla-architecture.md). This document
> retains the role, reuse and gate design; plan-era statements below are not
> evidence that the current formal run is still pending or that M2 passed.

This document is the canonical role and execution map for the next Rosetta
Reality development cycle. It supersedes the earlier assumption that a frozen
Qwen backbone plus a small action head is the project VLA reference.

## System roles

```text
Observation + Instruction
          |
          v
Qwen ER / System 2 (low frequency)
          |
          v
ActionPlan v1
          |
          v
SmolVLA 450M / System 1 (high frequency)
          |
          v
Rosetta Action Contract
          |
          v
Simulation Adapter -> MuJoCo -> Next Observation
```

- **ER** owns scene understanding, task decomposition, progress checks and
  recovery decisions. Qwen checkpoints produced for this role must be trained
  and evaluated on ER supervision, not on the legacy action-head objective.
- **VLA** owns continuous action generation. The development reference is
  `lerobot/smolvla_base` 450M.
- **Integration** owns the versioned `ActionPlan` schema and translation from a
  grounded plan into VLA conditioning. Natural language alone is not the full
  interface.
- **Shared infrastructure** owns dataset identity, fixed splits, action
  semantics, simulator adapters, metrics and provenance.

## Incremental repository layout

Existing code and evidence stay in place. New work uses these logical entry
points without moving or deleting legacy files:

```text
configs/
  er/                    # Qwen ER-only experiment identities
  vla/                   # SmolVLA experiment identities
  experiments/           # legacy Qwen-as-VLA experiments, read-only evidence
integration/
  schemas/               # ER -> VLA wire contracts
scripts/                 # Docker/WSL orchestration and explicit phase runners
src/rosetta_reality/
  data/                  # shared dataset contracts and adapters
  sim/                   # shared action contract and simulator adapters
  eval/                  # role-specific and end-to-end metrics
reports/training/        # append-only experiment evidence
```

## Reuse boundary

| Asset | SmolVLA use | Qwen ER use | Rule |
| --- | --- | --- | --- |
| ALOHA dataset cache and immutable revision | Reuse after current inspection | Only if an ER task explicitly needs it | Never infer role compatibility from file presence |
| Fixed train/validation/test episode split | Reuse | May reference, but ER must define its own supervision | Test remains sealed until selection |
| Action Contract and simulation adapter | Reuse and revalidate | Read-only context | Physical semantics remain authoritative |
| Gate 1 scripted action and Gate 2 replay | Reuse as prior evidence, then rerun under the SmolVLA contract | Not ER acceptance | Same action dimension is insufficient |
| Qwen frozen Feature Cache | Do not reuse | Reuse only with an exact ER cache identity | Qwen features are not SmolVLA inputs |
| Qwen action-head checkpoints | Do not reuse | Historical negative evidence only | Action loss is not ER training |
| Metrics, reports and provenance schemas | Reuse | Reuse | Keep role labels explicit |

## Pinned first VLA identity

- LeRobot source: `huggingface/lerobot` commit
  `c903b114a90e703b3f7d0c46cb38727c328c55ff`.
- VLA base: `lerobot/smolvla_base` commit
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- Dataset: `lerobot/aloha_sim_insertion_human` commit
  `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`.
- Data scope: all 50 episodes with the existing 40/5/5 episode split.
- Camera/state/action: `observation.images.top`, 14-D state and 14-D absolute
  joint-position action at 50 Hz.
- Policy horizon: retain SmolVLA's 50-action training chunk and execute only the
  first action before observing again. A derived Action Contract must be
  revalidated before policy rollout.
- Initial adaptation: upstream defaults `freeze_vision_encoder=true`,
  `train_expert_only=true`, `train_state_proj=true`. This is not LoRA or full
  fine-tuning.
- Tracking: Trackio `0.28.0`, project `rosetta-reality-vla`, public static Space
  `LAlpaca/rosetta-reality-trackio`, durable local storage under the ignored
  `runs/trackio/` boundary. The account cannot allocate a dynamic Gradio Space,
  so sanitized snapshots sync at checkpoint boundaries instead of claiming
  live server-side ingestion.

## Execution gates

1. Static configuration and `ActionPlan` schema checks.
2. Docker image build from WSL and environment/import checks.
3. Read-only dataset, model metadata and license inspection.
4. Derived 50-step Action Contract validation, Gate 1 and Gate 2 replay.
5. Trackio local write/query plus static Space sync/read smoke.
6. SmolVLA batch-1, two-step forward/backward/checkpoint smoke with a hard
   memory limit.
7. Fixed tiny-sample overfit and explicit checkpoint resume.
8. Resource review and a separately preregistered formal run.
9. Validation selection, independent reload/export, Gate 3 and Gate 4.
10. Independent Qwen ER training/evaluation, followed by ActionPlan integration.

No formal training may start merely because the model downloads successfully.
Any OOM, non-finite value, dataset/model identity mismatch, unsupported XPU
operation, missing Trackio durability or action-semantic mismatch stops the
current phase.

## Measured XPU training path

The preregistered optimized plan is
`configs/vla/smolvla_450m_aloha_insertion_formal_optimized_001.yaml`. Its bounded
30-step benchmark selected batch 12 with `torch.compile(mode="reduce-overhead")`,
a revision-scoped compiler cache, and exact-parity skipping of the vision encoder
for two fully masked camera placeholders. This amortizes the frozen 350M vision-
language prefix across a batch instead of invoking it once per sample.

The selected candidate measured 4.6415 samples/s, projected 83.48 minutes for
1,680 optimizer steps (20,160 sample exposures, or 1.008 train-set passes), and
peaked at 4.01 GiB of XPU allocation. The projection includes cold compilation,
startup, and checkpoint allowance and remains under the registered 8 GiB
container limit and 7 GiB XPU guard. It is benchmark evidence, not a claim that
the formal run or closed-loop acceptance has completed; the ignored prerequisite
reports must be regenerated or checksum-verified in the execution environment.

A custom Triton RMSNorm forward probe was also measured, but it is not integrated
because backward and end-to-end training parity have not been established. The
training path therefore keeps the documented upstream/Inductor implementation.

## Hugging Face boundaries

The Trackio Space is a public read-only snapshot dashboard. Local SQLite remains
the durable source during a run; a sanitized sync refreshes the Space at safe
checkpoint boundaries. Only metrics, non-sensitive hyperparameters, immutable
revisions, resource statistics and run status may be public. Tokens, environment
variables, absolute host paths, samples, private conversation content, complete
console logs, checkpoints, weights and unreviewed media/artifacts are forbidden.

Model publication uses a separate model repository and happens only after
export/reload and license checks. Hub tokens remain outside the repository and
are never printed in commands, logs, configs or manifests.
