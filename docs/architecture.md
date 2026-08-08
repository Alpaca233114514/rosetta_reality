# Architecture

Rosetta Reality separates perception and language representation from robot
state processing and action prediction.

```text
Visual input + instruction
           |
           v
Replaceable VLM backbone -----> pooled hidden representation
                                      |
Robot state ---> State Encoder -------+--> simple fusion --> Action Expert
                                                                |
                                                                v
                                                continuous action chunk
```

## Current technical direction

The planned research stack combines:

- a Qwen3.5 vision-language backbone;
- a robot-state encoder;
- a replaceable action expert;
- future embodiment adapters;
- a robot-agnostic dataset pipeline; and
- simulation and evaluation infrastructure.

Qwen is the current default backbone, not the Rosetta Reality architecture
itself. Generic policy components depend only on the `VLABackbone` interface.
A future Gemma or other VLM adapter should be able to reuse the same state
encoder, action head, data schema, trainer, and evaluation code.

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

- `models/backbones/` owns model-family-specific loading and input processing.
- `models/vla.py` owns generic policy composition.
- `data/adapters/` owns third-party field mapping.
- `data/schema.py`, `data/dataset.py`, and `data/normalization.py` own generic
  frame/sample/batch, chunking, and population-statistics contracts.
- `train/` owns loss and optimization mechanics, not model downloading.
- `sim/` will provide simulator adapters without coupling the core policy to a
  particular simulator.
