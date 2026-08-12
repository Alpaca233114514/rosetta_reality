# Roadmap

## M0 — Repository Skeleton (complete; draft review)

Define the package layout, replaceable backbone interface, runnable dummy
policy, robot-agnostic sample schema, minimal action-regression training step,
offline tests, and read-only environment inspection.

## M1 — Dataset Pipeline (complete; bounded acceptance slice)

Add dataset adapters and normalization workflows while keeping the internal
schema independent of LeRobot, DROID, BridgeData, Open X-Embodiment, and
simulation sources.

The first slice uses only episode 0 from the MIT-licensed
`lerobot/aloha_sim_insertion_human` dataset. It resolves the Hub branch to an
immutable commit, stores revision-scoped ignored cache data, maps LeRobot v3
records into `RosettaFrame`, creates episode-safe action chunks as
`RosettaSample`, and collates them into `RosettaBatch`. State/action population
statistics are computed online and persisted separately. The M1 acceptance path
ends after one offline CPU optimizer smoke step; it does not load model weights
or begin formal training. The acceptance evidence is recorded in
`docs/m1-acceptance.md`. Additional dataset configurations may remain
unprepared and are tracked separately from this bounded M1 result.

## M2 — SmolVLA 450M Development VLA

Use revision-pinned `lerobot/smolvla_base` 450M as the development VLA. Reuse
the accepted ALOHA data/action/simulation infrastructure, then complete tiny
smoke, small-data overfit, formal training, validation, checkpoint/resume,
evaluation, export/reload and MuJoCo closed-loop gates. The historical frozen
Qwen action policies are negative evidence and do not satisfy this milestone.

## M3 — Qwen ER and structured ER/VLA integration

Train and evaluate a Qwen ER model independently, producing `ActionPlan v1`
rather than continuous actions. Connect a selected ER checkpoint to the M2
SmolVLA policy and separately measure plan quality, execution quality, recovery
behavior and end-to-end success.

## M4 — Robust ER/VLA evaluation

Evaluate ER, VLA and their integration under spatial, semantic, temporal and
recovery perturbations. Avoid treating a single offline loss or benchmark as a
complete system result.

## M5 — Multi-dataset / Cross-embodiment

Expand the revision-pinned data and adapter matrix only after the first M2/M3
loop is reproducible. Keep embodiment-specific mappings outside ER, VLA and the
shared action schema.

## M6 — Controlled action-expert research

Compare action horizons, execution horizons, adaptation choices and alternative
action experts one axis at a time while preserving fixed data and evaluation.

## M7 — Controlled ER/VLA scale-up

Scale Qwen ER or the VLA only after the matching development pipeline has passed
its own gates. A larger model is not a substitute for interface, data or
closed-loop correctness.

## M8 — Sim-to-real Experiments

Explore carefully bounded transfer experiments after simulation safety and
evaluation gates are established.
