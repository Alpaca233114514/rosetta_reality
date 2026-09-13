# Rosetta Reality

Rosetta Reality is an early-stage monorepo for Embodied Reasoning (ER),
Vision-Language-Action (VLA), and their structured integration. It includes a
revision-pinned, bounded robot dataset pipeline, action/simulation contracts, and
reproducible training/evaluation foundations; it does not provide autonomous
physical-robot control.

## Status

Experimental / early-stage. **M0 — Repository Skeleton** and **M1 — Dataset
Pipeline** are complete for their bounded acceptance scopes. The accepted M1
slice is episode 0 of `lerobot/aloha_sim_insertion_human`, verified on
2026-08-09. The earlier frozen Qwen action-policy experiments did not pass
closed-loop M2 selection. The current M2 reference is revision-pinned **SmolVLA
450M**. As of 2026-09-11, M2 remains incomplete: seven policy identities failed
Gate 4 `0/5`. The completed Hestia frame-0 experiment passes training-fit checks
but fails development visual generalization; it has no new Gate 3/4 result.

## Goal

Build a replaceable two-system stack: Qwen ER performs low-frequency embodied
reasoning and emits a versioned `ActionPlan`; SmolVLA performs high-frequency
continuous control. Data, action semantics, simulation, evaluation and
provenance remain model-agnostic.

## Architecture

```text
Observation + Instruction
          |
          v
Qwen ER / System 2
          |
          v
ActionPlan v1
          |
          v
SmolVLA 450M / System 1
          |
          v
Action Contract -> Simulation Adapter -> Robot Motion
```

The runnable M0 path remains available for offline contract tests. Existing
Qwen action-head code and artifacts are retained as historical VLA evidence;
new Qwen work belongs to the independent ER track.

## Repository Layout

- `configs/er/`: Qwen ER-only identities and gates.
- `configs/vla/`: SmolVLA identities and phase gates.
- `configs/experiments/`: legacy Qwen-as-VLA experiments retained as evidence.
- `configs/data/`: dataset registry and bounded preparation configurations;
  entries without acceptance evidence remain preparatory.
- `integration/schemas/`: versioned ER-to-VLA wire contracts.
- `src/rosetta_reality/models/`: replaceable backbones and generic VLA policy components.
- `src/rosetta_reality/data/`: robot-agnostic frames, action chunks, batches,
  dataset adapters, and online normalization.
- `src/rosetta_reality/train/`: action loss and minimal training-step helper.
- `scripts/`: environment inspection, M0 CPU dry-runs, M1 data preparation and
  inspection, and conservative cache auditing.
- `tests/`: offline, CPU-compatible import and shape tests.
- `docs/`: architecture decisions and staged roadmap.

## Quick Start

Run machine-learning commands in Linux Docker containers launched from WSL
Bash. Windows is limited to editing, Git, and non-ML static checks. The existing
offline baseline can be checked without downloading model weights or data:

```bash
scripts/run_m2_container.sh build-ml
scripts/run_m2_container.sh ml python scripts/check_env.py
scripts/run_m2_container.sh ml python -m pytest -q
scripts/run_m2_container.sh ml python scripts/train.py --dry-run
scripts/run_m2_container.sh ml ruff check .
```

The stable current M2 component, training, export, closed-loop and evidence map
is [`docs/m2-smolvla-architecture.md`](docs/m2-smolvla-architecture.md).
[`docs/er-vla-pipeline.md`](docs/er-vla-pipeline.md) retains the original role,
reuse and gate design. Faust, Aster, Way, Zen-uniform, Zen-firstaction,
vcdropout and vfunfreeze all failed development Gate 4 `0/5`; the latest
completed policy-gate closure is
[`vfunfreeze closure`](reports/training/m2-smolvla-vfunfreeze-gate34-closure-2026-09-06.md).
Offline visual diagnostics are separate from these closed-loop results.

## M1 Dataset Preparation

M1 uses
[`lerobot/aloha_sim_insertion_human`](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human),
an MIT-licensed ALOHA simulation dataset with 50 episodes, 25,000 frames, and
14-dimensional state/action vectors. The first bounded target is episode 0
(500 frames), camera `observation.images.top`, and action chunks of length 8.

The preparation command resolves `main` to an immutable Hub commit SHA before
loading. Each SHA receives its own ignored `data/` cache and manifest. LeRobot
v3 consolidates multiple episodes into shared Parquet/video files, so selecting
episode 0 may still cache close to the full dataset size of approximately
91.3 MB.

`python scripts/prepare_data.py` is the explicit preparation command and may
download the configured dataset into the ignored cache. `inspect` is read-only,
does not use the network, and reports the manifest, statistics, and checksums:

Run preparation and the explicit real-data smoke test through Docker from WSL:

```bash
scripts/run_m2_container.sh data python scripts/prepare_data.py
scripts/run_m2_container.sh ml python scripts/prepare_data.py inspect
scripts/run_m2_container.sh ml python -m pytest -m data
```

For an existing LeRobot v3 cache, the conservative audit can be run with
`python scripts/clean_data.py --config configs/data/aloha_sim_insertion.yaml`.
It writes a JSON quality report but never rewrites source Parquet or video
files; row-level problems require manual review.

The smoke test uses RGB channel means as explicit three-dimensional dummy
features. Real normalized state and action targets pass through
`DummyBackbone + StateEncoder + ContinuousActionHead` for one CPU optimizer
step. It does not download model weights or start a full training run.

The bounded M1 acceptance slice is complete for episode 0 of
`lerobot/aloha_sim_insertion_human`; see [docs/m1-acceptance.md](docs/m1-acceptance.md)
for the recorded evidence. The additional dataset configurations remain
preparatory and are not represented as completed M1 caches.

The recorded M1 acceptance verification for this slice was:

| Check | Result |
| --- | --- |
| Environment | Python 3.13.5, PyTorch 2.11.0+cpu, CUDA unavailable |
| Offline tests | 33 passed, 1 skipped, 4 deselected |
| CPU dry-run | Prediction shape `(2, 8, 7)`, finite Smooth L1 loss `0.404727` |
| Cache inspection | Revision `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`, metadata/statistics, 9 checksums |
| Real-data smoke | 1 passed, 3 skipped for unprepared additional configurations |

These checks validate the M1 data and CPU-smoke boundary only. They do not
claim model-weight integration, formal training, action semantics, simulator
control, or physical-robot control.

## Milestones

M0 established stable interfaces and an offline optimizer path. M1 closed a
revision-pinned, robot-agnostic dataset path. Historical Qwen VLA experiments
extended the training and simulation tooling but did not pass M2 closed-loop
acceptance. Their reusable infrastructure now feeds the SmolVLA M2 path.

For the temporary AutoDL RTX 4090 worker, the platform container itself replaces
the local nested-Docker wrapper. The offline, benchmark-first staging and CUDA
preflight procedure is documented in
[`docs/autodl-rtx4090.md`](docs/autodl-rtx4090.md); formal CUDA training remains
locked until live doctor, benchmark and two-step smoke evidence are registered.

The completed Faust run, Gate 4 failure, trainer/optimizer findings, evidence
identities and AI repair order are recorded in
[`reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.md`](reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.md),
with a machine-readable
[`JSON companion`](reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.json).
The historical two-arm Zen temporal-weighting comparison, which rejected its
registered hypothesis, is audited in
[`reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md`](reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md)
and its
[`JSON companion`](reports/training/m2-smolvla-zen-formal-audit-2026-08-27.json).
The latest completed frame-0 fit experiment is
[`Hestia main-run recovery`](reports/training/m2-smolvla-hestia-main-recovery-result-2026-09-11.md):
1280 updates completed, with exact historical B/C full-chunk independent reload
evidence. C passes train40 checks in all action groups under four fixed inference
noise conditions (`4/4`), while dev5 fails in every group (`0/4`). These four
conditions are not independent training runs or task-success counts.
The [completed CUDA checkpoint curve](reports/training/m2-smolvla-hestia-cuda-checkpoint-curve-closure-2026-09-12.md)
reproduces historical 1280 arrays exactly; development full-chunk error regresses
after 640 despite better training fit. The
[640/1280 localization](reports/training/m2-smolvla-hestia-checkpoint-localization-result-2026-09-12.md)
places 92.25% of net gripper MAE regression in the last ten slots, identifies
episode 13's large contribution to left-arm regression, and finds broader
right-arm regression. Late gripper opening events mismatch development targets.
The [scene/phase check](reports/training/m2-smolvla-hestia-scene-phase-result-2026-09-12.md)
confirms scene-dependent early/late opening using the next 50 local target frames.
Fixed geometric neighbors do not beat constant onset-time references. An oracle
using true target phases reveals partial left-arm geometric correspondence, but
left-wrist-angle mismatch persists. The subsequent
[commanded-pose check](reports/training/m2-smolvla-hestia-command-pose-result-2026-09-12.md)
finds better left-end-effector position correspondence but a remaining development
orientation deficit, concentrated in episode 45 for this neighbor predictor.
Changing joint labels to poses alone does not remove it. These oracle/FK controls
are not deployable policy or achieved-pose evidence; no policy repair, checkpoint
selection or new training acceptance is claimed.
The [readout-family control](reports/training/m2-smolvla-hestia-pose-readout-result-2026-09-12.md)
then finds that quadratic readout of the same positions beats orientation
constants under the true-phase oracle in train holdouts and development.
The neighbor deficit is method-dependent; observable onset prediction and
original-clock full-chunk actions still do not jointly pass. Native FK also
separates orientation regression from improved right-end-effector position.
The [phase-mechanism bridge](reports/training/m2-smolvla-hestia-phase-bridge-result-2026-09-12.md)
then rejects single-left-event timing as a sufficient explanation: native
event-aligned orientation and same-clock oracle reconstruction still fail
their comparisons. The [native prefix collection](reports/training/m2-smolvla-hestia-prefix-path-closure-2026-09-12.md)
has since completed: all 180 outputs exactly reproduce C and all parameters
remain unchanged. The [eight-arm readout](reports/training/m2-smolvla-hestia-prefix-readout-result-2026-09-12.md)
retains object-position information after projection, while timing and full
actions still fail their joint comparison. The corrected reciprocal K/V experiment
has completed 720 forwards: both native endpoints are exact, and the 16 changed
K/V weights have bidirectional causal effects on later development regression.
Returning older K/V recovers 74.7% of left-joint regression; reciprocal substitution
recreates 89.7%. The hybrid still fails the constant-baseline comparison, so the
unique root of all generalization failure remains open. See the
[causal result](reports/training/m2-smolvla-hestia-parameter-crossover-result-2026-09-12.md).
The six-condition K-only/V-only split has now completed 1080 forwards and all
30 files are recovered. V-only accounts for 59.1% recovery / 69.7% reciprocal damage
of left-joint regression, versus K-only 19.5% / 13.8%; hybrids still fail the
relevant constant baselines. See the
[K/V split result](reports/training/m2-smolvla-hestia-kv-split-result-2026-09-12.md).
The eighteen-condition V-layer test completed 3240 forwards, 78 verified files,
and 193,664 independent numeric checks. Effects are distributed across layers. The subsequent image-V component test completed 1800 forwards, 52 verified files and 95,744 independent checks: a shared image-V offset contributes to late regression, but the generalization gap already present at step 640 remains unresolved. See the [component result](reports/training/m2-smolvla-hestia-image-value-component-result-2026-09-12.md) and [next-session handoff](reports/training/m2-smolvla-hestia-next-session-handoff-2026-09-12.md). No new training or checkpoint selection is accepted.
The initial no-optimizer diagnosis is
preserved in
[`reports/training/m2-smolvla-action-repair-handoff-2026-08-12.md`](reports/training/m2-smolvla-action-repair-handoff-2026-08-12.md).

## Planned Roadmap

- M1 — Dataset Pipeline (complete for the bounded acceptance slice)
- M2 — SmolVLA 450M Development VLA
- M3 — Qwen ER and structured ER/VLA integration
- M4 — Robust ER/VLA evaluation
- M5 — Multi-dataset / Cross-embodiment
- M6 — Controlled action-expert research
- M7 — Controlled ER/VLA scale-up
- M8 — Sim-to-real experiments

See [docs/roadmap.md](docs/roadmap.md) for milestone boundaries.

## Safety / Scope

Rosetta Reality is simulation-first. Model/data downloads, Hub writes and real
training require explicit authorization and execute through the Docker/WSL
boundary. No workflow issues commands to a physical robot unless separately
authorized and validated.
