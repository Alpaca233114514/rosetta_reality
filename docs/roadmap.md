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

Status on 2026-09-12: **incomplete**. Seven policy identities have measured
Gate 4 failures (`0/5`). Hestia completed its 1280-update frame-0 experiment:
training-fit checks pass under all four fixed inference noise conditions,
development visual acceptance remains `0/4`, and new Gate 3/4 is not measured.
The four-checkpoint CUDA curve has been recovered and compared locally. The
640/1280 decomposition localizes gripper regression to late opening events and
separates scene-concentrated left-arm regression from broader right-arm regression.
The subsequent scene/phase check confirms timing mismatch from bounded local
labels; target-phase oracle alignment exposes partial geometric correspondence,
but left-wrist-angle mismatch persists. The subsequent commanded-pose check finds
useful left-position correspondence and a remaining development orientation
deficit; merely changing output coordinates has not repaired it. These are
oracle command-pose diagnostics, not achieved poses or policy improvements.
The subsequent fixed-coordinate quadratic readout improves oracle orientation
beyond both constants in train holdouts and development, so the neighbor
deficit is method-dependent. Onset prediction and full clock-time actions
still do not jointly pass. The subsequent phase bridge also fails to explain
native error using a single opening event or to recover full clock-time actions
with true timing. Native C's prefix collection has since completed with all 180
outputs exactly reproduced. Eight fixed readouts retain object-position information
before and after projection, but onset and full-action comparisons still fail.
A local byte audit confirms that 345 frozen VLM tensors are identical at 640/1280;
16 cross-attention K/V and 106 other action-path tensors changed. Corrected reciprocal
K/V substitution 002 completed 720 forwards, both native endpoints exactly reproduced,
and all 22 result files were recovered. K/V updates have bidirectional causal effects
on seven full-trajectory development regressions under all four noises. The hybrid
still fails joint/gripper/orientation constant comparisons. This localizes part of
the late regression; the unique root of all generalization failure remains open.
The K-only/V-only split has since completed 1080 forwards with all 30 files verified.
V-only contributes more to full left-joint regression (59.1% recovery / 69.7%
reciprocal damage) than K-only (19.5% / 13.8%), while both contribute and the
remaining generalization failure persists. The per-layer V test completed 3240
forwards and 193,664 independent checks; layer 7 has the largest individual joint
recovery but does not repair generalization. The completed route diagnostic
used exact V-output routing across image, language and state prefix positions. The subsequent image-component diagnostic completed 1800 forwards and 52 verified files: the shared image-V offset explains part of late regression, while the earlier 320/640 generalization gap remains open. See the [component result](../reports/training/m2-smolvla-hestia-image-value-component-result-2026-09-12.md) and the
[split result](../reports/training/m2-smolvla-hestia-kv-split-result-2026-09-12.md).
No new training axis or checkpoint selection is accepted. See
[the current M2 evidence map](m2-smolvla-architecture.md#2-current-m2-status) and
[the prefix-readout report](../reports/training/m2-smolvla-hestia-prefix-readout-result-2026-09-12.md).

## M3 — Qwen ER and structured ER/VLA integration

Train and evaluate a Qwen ER model independently, producing `ActionPlan v1`
rather than continuous actions. Connect a selected ER checkpoint to the M2
SmolVLA policy and separately measure plan quality, execution quality, recovery
behavior and end-to-end success.

Integration acceptance requires both M2 closed-loop acceptance and independent
ER acceptance first; the current offline diagnostics do not unlock M3.

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
