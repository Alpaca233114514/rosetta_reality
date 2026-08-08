# Rosetta Reality

Rosetta Reality is an early-stage research codebase for Vision-Language-Action
(VLA) and embodied foundation-model experiments. The current implementation is
a small, CPU-testable repository skeleton; it does not provide autonomous robot
control.

## Status

Experimental / early-stage. The current milestone is **M0 — Repository
Skeleton**.

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
- `src/rosetta_reality/data/`: robot-agnostic sample schema and small utilities.
- `src/rosetta_reality/train/`: action loss and minimal training-step helper.
- `scripts/`: read-only environment inspection and safe M0 entry points.
- `tests/`: offline, CPU-compatible import and shape tests.
- `docs/`: architecture decisions and staged roadmap.

## Quick Start

Use a fresh virtual environment. Installing the editable package does not
download model weights or datasets.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# POSIX shells: source .venv/bin/activate
python -m pip install -e ".[dev]"

python scripts/check_env.py
pytest
python scripts/train.py --dry-run
```

The Qwen adapter is optional and deliberately uses local files by default:

```bash
python -m pip install -e ".[qwen]"
```

Installing the optional dependency still does not fetch any model checkpoint.

## Current Milestone

M0 establishes stable interfaces and proves that a dummy policy can complete a
forward pass, Smooth L1 loss, backward pass, and one optimizer step on CPU.

## Planned Roadmap

- M1 — Dataset Pipeline
- M2 — Frozen Backbone + State Encoder + Action Head
- M3 — LoRA
- M4 — Action Chunk Transformer
- M5 — Closed-loop Simulation
- M6 — Multi-dataset / Cross-embodiment
- M7 — Diffusion or Flow-Matching Action Expert
- M8 — Sim-to-real experiments

See [docs/roadmap.md](docs/roadmap.md) for milestone boundaries.

## Safety / Scope

Rosetta Reality is simulation-first. M0 does not download Qwen weights or robot
datasets, install simulators, configure CUDA/ROCm, start real training, or issue
commands to physical robots.
