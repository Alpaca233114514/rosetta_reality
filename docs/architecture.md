# Architecture

Rosetta Reality separates low-frequency embodied reasoning, high-frequency
action generation, and physical execution contracts.

```text
Observation + instruction
           |
           v
Replaceable ER (Qwen reference)
           |
           v
ActionPlan v1 -- grounded subtask / target / constraints / recovery
           |
           v
Replaceable VLA (SmolVLA 450M reference) + robot state
           |
           v
Rosetta Action Contract -> Simulation Adapter -> next observation
```

## Current technical direction

The current research stack combines:

- a Qwen3.5 ER reference for reasoning and planning;
- a revision-pinned SmolVLA 450M VLA reference;
- a versioned structured ER/VLA interface;
- existing robot-state, action and embodiment contracts;
- a robot-agnostic dataset pipeline; and
- simulation and evaluation infrastructure.

Neither Qwen nor SmolVLA is the Rosetta architecture itself. ER and VLA may be
replaced independently if they continue to satisfy `ActionPlan`, observation,
action, artifact and evaluation contracts.

Historical Qwen frozen-feature action policies remain in their original paths
as negative VLA evidence. They do not initialize SmolVLA and are not accepted
as ER checkpoints.

## M0 implementation

M0 uses a pooled `[batch, hidden_size]` representation from the backbone. The
state encoder maps `[batch, state_dim]` into its own hidden representation. A
small learned projection fuses the two tensors, and the action head emits
`[batch, chunk_size, action_dim]`.

This deliberately simple fusion establishes tensor contracts without
prematurely introducing state tokens or cross-attention. Later milestones can
replace the fusion and action expert while preserving the public policy path.

## M1 data path

```text
LeRobot v3 record
       |
       v
DatasetAdapter --> RosettaFrame --> episode-safe chunking --> RosettaSample
                                                               |
                                                               v
                                                        RosettaBatch
                                                               |
                         +-------------------------------------+------+
                         |                                            |
                         v                                            v
              online state statistics                     online action statistics
```

The LeRobot adapter owns source field names, immutable Hub revision selection,
and image conversion. Generic chunking reads only `DatasetAdapter` methods and
therefore cannot depend on ALOHA, LeRobot, camera names, action dimensions, or
storage layout. An anchor is valid only when all requested future actions have
consecutive frame indices inside the same episode; no cross-episode padding is
introduced.

## Boundaries

- `configs/er/` owns ER-only experiment identity and supervision gates.
- `configs/vla/` owns SmolVLA experiment identity and phase gates.
- `integration/schemas/` owns the model-independent ER/VLA wire format.
- `models/backbones/` owns model-family-specific loading and input processing.
- `models/vla.py` owns generic policy composition.
- `data/adapters/` owns third-party field mapping.
- `data/schema.py`, `data/dataset.py`, and `data/normalization.py` own generic
  frame/sample/batch, chunking, and population-statistics contracts.
- `train/` owns loss and optimization mechanics, not model downloading.
- `sim/` owns action semantics and simulator adapters without coupling ER or
  the core policy to a particular simulator.

See [`er-vla-pipeline.md`](er-vla-pipeline.md) for the pinned first experiment,
reuse matrix and execution gates.
