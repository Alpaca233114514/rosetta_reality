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
closed-loop M2 selection. The next M2 reference is revision-pinned **SmolVLA
450M**.

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
reuse and gate design. Five formal SmolVLA campaigns (Faust, Aster, Way and the
two-arm Zen comparison) have completed training, selection, export/reload and
Gate 3, but all five failed development Gate 4 `0/5`, so M2 remains blocked;
the architecture map routes to the current audit instead of treating any
earlier projected plan as current status.

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
The newest completed campaign — the two-arm Zen temporal-weighting comparison,
which rejected the registered hypothesis and extended the Gate 4 failure tally
to five identities — is audited in
[`reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md`](reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md)
and its
[`JSON companion`](reports/training/m2-smolvla-zen-formal-audit-2026-08-27.json).
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
