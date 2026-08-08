# Roadmap

## M0 — Repository Skeleton (current)

Define the package layout, replaceable backbone interface, runnable dummy
policy, robot-agnostic sample schema, minimal action-regression training step,
offline tests, and read-only environment inspection.

## M1 — Dataset Pipeline

Add dataset adapters and normalization workflows while keeping the internal
schema independent of LeRobot, DROID, BridgeData, Open X-Embodiment, and
simulation sources.

## M2 — Frozen Qwen + State Encoder + Action Head

Integrate a locally available Qwen checkpoint, freeze the backbone, and train
only the state encoder and continuous action head on one bounded task.

## M3 — LoRA

Introduce parameter-efficient backbone adaptation after the frozen-backbone
pipeline is stable.

## M4 — Action Chunk Transformer

Replace the MLP action head with a temporal action expert while preserving the
`[batch, chunk_size, action_dim]` contract.

## M5 — Closed-loop MuJoCo Evaluation

Evaluate repeated observation-to-action execution in simulation using task
success, collision, validity, smoothness, and latency metrics.

## M6 — Multi-dataset / Cross-embodiment

Train across normalized datasets and embodiments with explicit embodiment
metadata and adapters.

## M7 — Diffusion / Flow-Matching Action Expert

Investigate richer continuous-action distributions only after deterministic
action chunking has a trustworthy baseline.

## M8 — Sim-to-real Experiments

Explore carefully bounded transfer experiments after simulation safety and
evaluation gates are established.
