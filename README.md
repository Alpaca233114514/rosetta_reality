# Rosetta Reality

Rosetta Reality is an early-stage monorepo for Embodied Reasoning (ER),
Vision-Language-Action (VLA), and their structured integration. It includes a
revision-pinned robot dataset pipeline, action/simulation contracts, and
reproducible training/evaluation foundations; it does not provide autonomous
physical-robot control.

## Status

Experimental / early-stage. **M0 — Repository Skeleton** and **M1 — Dataset
Pipeline** are complete for their bounded acceptance scopes. The earlier frozen
Qwen action-policy experiments did not pass closed-loop M2 selection. The next
M2 reference is revision-pinned **SmolVLA 450M**.

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
- `integration/schemas/`: versioned ER-to-VLA wire contracts.
- `src/rosetta_reality/models/`: replaceable backbones and generic VLA policy components.
- `src/rosetta_reality/data/`: robot-agnostic frames, action chunks, batches,
  dataset adapters, and online normalization.
- `src/rosetta_reality/train/`: action loss and minimal training-step helper.
- `scripts/`: read-only environment inspection and safe M0 entry points.
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

The SmolVLA and Trackio image/runner is governed by
[`docs/er-vla-pipeline.md`](docs/er-vla-pipeline.md). Formal training remains
blocked until its Space, dataset, action contract, tiny smoke, and overfit gates
pass. The same document records the bounded XPU benchmark and the preregistered
sub-two-hour training plan without treating a projection as a completed run.

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

Run preparation and the explicit real-data smoke test through Docker from WSL:

```bash
scripts/run_m2_container.sh data python scripts/prepare_data.py
scripts/run_m2_container.sh ml python scripts/prepare_data.py inspect
scripts/run_m2_container.sh ml python -m pytest -m data
```

The smoke test uses RGB channel means as explicit three-dimensional dummy
features. Real normalized state and action targets pass through
`DummyBackbone + StateEncoder + ContinuousActionHead` for one CPU optimizer
step. It does not download model weights or start a full training run.

The bounded M1 acceptance slice is complete for episode 0 of
`lerobot/aloha_sim_insertion_human`; see [docs/m1-acceptance.md](docs/m1-acceptance.md)
for the recorded evidence. The additional dataset configurations remain
preparatory and are not represented as completed M1 caches.

## Milestones

M0 established stable interfaces and an offline optimizer path. M1 closed a
revision-pinned, robot-agnostic dataset path. Historical Qwen VLA experiments
extended the training and simulation tooling but did not pass M2 closed-loop
acceptance. Their reusable infrastructure now feeds the SmolVLA M2 path.

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
