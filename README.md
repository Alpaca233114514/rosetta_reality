# Rosetta Reality

Rosetta Reality is an early-stage research codebase for Vision-Language-Action
(VLA) and embodied foundation-model experiments. The current implementation
includes a CPU-testable policy skeleton and the first bounded dataset pipeline;
it does not provide autonomous robot control.

## Status

Experimental / early-stage. **M0 — Repository Skeleton** and **M1 — Dataset
Pipeline** are complete for their bounded acceptance scopes. **M2 — Development
VLA** is next.

## Goal

Build a backbone-agnostic research pipeline that can combine visual and
language context with robot state to predict configurable chunks of continuous
actions. Qwen3.5 is the current default backbone family, but the policy, data,
training, evaluation, and simulation layers are intentionally not Qwen-specific.

## Architecture

```text
Image / Video ---------+
                       |
Language Instruction --+--> Replaceable VLM Backbone --+
                                                        |
Robot State ----------------> State Encoder -------------+--> Fusion
                                                               |
                                                               v
                                                         Action Head
                                                               |
                                                               v
                                                         Action Chunk
```

The runnable M0 path uses `DummyBackbone`, which has no network, model-weight,
or GPU requirements. `Qwen35Backbone` is only a lazy-loading adapter skeleton.

## Repository Layout

- `configs/`: model, training, data, and simulation configuration examples.
- `src/rosetta_reality/models/`: replaceable backbones and generic VLA policy components.
- `src/rosetta_reality/data/`: robot-agnostic frames, action chunks, batches,
  dataset adapters, and online normalization.
- `src/rosetta_reality/train/`: action loss and minimal training-step helper.
- `scripts/`: read-only environment inspection and safe M0 entry points.
- `tests/`: offline, CPU-compatible import and shape tests.
- `docs/`: architecture decisions and staged roadmap.

## Quick Start

All local machine-learning operations run in WSL. Use a WSL-only virtual
environment; do not reuse a native Windows environment. Installing the editable
package does not download model weights or datasets.

```bash
python3.13 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,data]"

python scripts/check_env.py
pytest
python scripts/train.py --dry-run
ruff check .
```

The Qwen adapter is optional and deliberately uses local files by default:

```bash
python -m pip install -e ".[qwen]"
```

Installing the optional dependency still does not fetch any model checkpoint.

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

Run preparation and the explicit real-data smoke test inside WSL:

```bash
source .venv-wsl/bin/activate
python scripts/prepare_data.py
python scripts/prepare_data.py inspect
pytest -m data
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

M0 established stable interfaces and proved that a dummy policy can complete a
forward pass, Smooth L1 loss, backward pass, and one optimizer step on CPU.
M1 closed a revision-pinned, robot-agnostic dataset path for the bounded
insertion episode. The remaining model, action-contract, and simulation gates
belong to M2 and later.

## Planned Roadmap

- M1 — Dataset Pipeline (complete for the bounded acceptance slice)
- M2 — Development VLA
- M3 — LoRA
- M4 — Action Chunk Transformer
- M5 — Closed-loop Simulation
- M6 — Multi-dataset / Cross-embodiment
- M7 — Diffusion or Flow-Matching Action Expert
- M8 — Sim-to-real experiments

See [docs/roadmap.md](docs/roadmap.md) for milestone boundaries.

## Safety / Scope

Rosetta Reality is simulation-first. M1 downloads only the explicitly configured
dataset cache. It does not download Qwen weights, install simulators, configure
CUDA/ROCm, start real training, or issue commands to physical robots.
