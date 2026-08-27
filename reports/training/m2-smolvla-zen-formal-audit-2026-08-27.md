# M2 SmolVLA Zen formal furnace audit — 2026-08-27

## 1. Authority and scope

This is the completion audit for the Zen formal furnace: a preregistered
two-arm single-axis comparison executed end to end on the AutoDL RTX 4090D
container through the version-2 plan-driven harness. Reading order: `AGENTS.md`,
`docs/m2-smolvla-architecture.md`, `docs/m2-smolvla-training-harness-v2.md`, the
Zen trial handoff (`runs/.../handoff/m2-smolvla450m-zen-smoke-handoff-001.json`),
this document, then its JSON companion.

| Field | Value |
|---|---|
| codename | `Zen` (formal program, two arms, Prometheus-style) |
| arm A (control) | `m2-smolvla450m-zen-uniform-002`, run `m2-smolvla450m-zen-cuda-b64-uniform-001`, plan sha `07cc24bf…` |
| arm B (treatment) | `m2-smolvla450m-zen-firstaction-001`, run `m2-smolvla450m-zen-cuda-b64-firstaction-001`, plan sha `3b9c62f5…` |
| single axis | `first_action_only` temporal weight profile (pinned `horizon_loss.py`, upstream sha `37b1d56f…`) vs uniform |
| shared contract | batch 64, 316 steps (20,224 exposures), fresh pinned base `c83c3163…`, AdamW/cosine identical, checkpoints [79,158,237,316], seed 20260809 |
| runtime | AutoDL container, CUDA, bf16, nested_docker_used=false, final workspace `20260827T104500Z-6fe7535-zen-formal-r42b` |
| hidden test | `[31,6,1,24,5]` never loaded |

## 2. Executive verdict

**The registered hypothesis is rejected, and both arms are honest negative
results.** Both arms trained to convergence, exported with bit-exact reload,
passed Gate 3, and failed Gate 4 with `0/5` success and maximum reward `0` on
every fixed seed 1000–1004. Offline selection ironically favored the *control*:
uniform first-action MAE `0.021573` vs treatment `0.022151` (base `0.290538`).
At this development scale (one registered exposure pass, 316 updates), the
first-action temporal weighting neither improves the offline target nor moves
closed-loop success — contradicting the Aster-era offline gain, which was
obtained at batch 8 / 2,500 updates and must not be generalized across
regimes. M2 remains incomplete; Gate 4 has now failed for Faust, Aster, Way,
Zen-uniform and Zen-firstaction under one identical protocol.

## 3. Training results

| Arm | loss @79 → @316 | grad norm @79 → @316 | wall | peak CUDA |
|---|---|---|---|---|
| uniform | 0.905 → 0.156 | 3.162 → 0.814 | 48.4 min | 18.29 GB |
| firstaction | 1.032 → 0.150 | 4.203 → 1.526 | 47.9 min | 18.29 GB |

All losses/gradients finite; LR schedule verified at quarter checkpoints
(7.7e-05 → 7.3e-06). Checkpoints on the system disk
`/root/zen-runtime/checkpoints` (deviation from the durable root, forced by a
7.3 GB durable-disk budget; instance must not be released).

## 4. Validation-only selection (fixed zeros-noise protocol)

| Arm | selected step | first-action MAE | improvement over base |
|---|---|---|---|
| uniform | 316 | 0.021572770214905695 | 92.57% |
| firstaction | 316 | 0.022150604739519103 | 92.38% |

Read-only context (different identities, not same-tree): Aster control
`0.022510`, Prometheus early-horizon `0.029148`. Both Zen arms beat both
contexts offline, underscoring that offline MAE does not transfer to
closed-loop success.

## 5. Export and independent reload

Both deploy artifacts live under the durable artifact root with the
gate-required layout (`pretrained_model/`, `config.json`,
`normalization.json`, `action_contract.json`, `manifest.json`). Reload proof =
re-running the frozen fixed-validation engine against the exported copy and
comparing seven deterministic action metrics: both arms exact
(`maximum_absolute_metric_difference = 0.0`).

## 6. Gates

| Arm | Gate 3 | Gate 4 | task successes | max reward (5 seeds) | unexpected collisions | invalid actions | joint-limit violations |
|---|---|---|---|---|---|---|---|
| uniform (411) | passed | **failed** | 0/5 | all 0.0 | 0 | 0 | 4/0/0/0/1 |
| firstaction (422) | passed | **failed** | 0/5 | all 0.0 | 0 | 0 | 0/0/0/0/0 |

The only failed acceptance criterion in both Gate 4 reports is
`minimum_task_success_rate`. The firstaction arm is the cleanest closed-loop
run in the campaign so far (zero violations of every safety class) while still
scoring zero reward — strengthening the earlier classification that the
failure is training/closed-loop generalization, not the action boundary.

## 7. Engineering defects found and fixed during execution (r30–r42b)

All fixes are create-only additions to the Zen native layer; the frozen
historical stack is untouched:

1. Frozen validation engine hardcodes `policy_preprocessor_step_5_…`; v2 saves
   `step_7`. Zen wrapper resolves processor files by glob (exactly-one rule).
2. Engine report naming is `<prefix>-step-%06d.json` (hyphen); selector/driver
   assumed `step%06d`.
3. Selection source key is `step`, not the way-injected `checkpoint_step`.
4. An inverted `load_vlm_weights is not False` identity check (introduced
   during the reload refactor) failed every artifact reload; corrected.
5. Synthetic-probe reload checks (three attempts) were abandoned in favor of
   the Way-proven engine-rerun comparison, which passed immediately.
6. `tarfile` default gzip level 9 on a 1.2 GB artifact stalled the gate ~10
   minutes on this CPU; backup evidence is now a checksum inventory (the
   artifact already lives on the durable disk, Way precedent
   `same_durable_data_disk: true`).
7. The gate engine binds Gate 4 to the Gate 3 report's `code_identity` and
   `simulation_plan_sha256`; every workspace re-stage broke the pair. Final
   identity r42b re-ran both gates per arm with fresh 3-digit suffixes
   (uniform 411, firstaction 422); the earlier 401 reports are retained as
   cross-identity discontinuity evidence.
8. Durable data disk exhausted (50G/50G) mid-gate; freed by removing this
   session's own unreferenced duplicate backup archive and partial workspace.
9. Prior-failure report checksums were mistranscribed from memory; re-read
   from the immutable files.
10. Artifact `config.json` initially omitted `adapt_to_pi_aloha`, breaking
    `SmolVLAActionSpace(**config)` in the CUDA policy class; the exporter now
    merges the policy flag, and the defective artifacts were superseded by
    rename (preserved, not deleted).

## 8. FFmpeg / torchcodec maintenance decision

Per user instruction the FFmpeg 5/6/7 runtimes were installed (conda prefixes
`/root/ff5|6|7`, libavutil 57/58/59) plus the complete FFmpeg 4 set from apt.
Activation failed for environmental reasons, recorded as negative results:
conda-forge libraries require `GLIBCXX_3.4.30`, which the miniconda-python
RPATH-pinned `libstdc++` does not provide; torchcodec 0.11.1 + torch 2.8.0
lacks `torch_dtype_float4_e2m1fn_x2` on the FFmpeg-4 core; 0.9.0 core-dumps
on import. torchcodec was reverted to the registered 0.11.1 and the
environment re-verified (protocol tests pass). Training itself never used
torchcodec (torchvision decode backend, ~9.3 s/step), so no retraining is
warranted; proper activation belongs to a coordinated torch+torchcodec
environment upgrade in a future maintenance session.

## 9. What not to conclude

- Do not read uniform's offline edge over firstaction as a stable ranking: one
  seed, one scale; it only invalidates the registered single-axis claim here.
- Do not treat clean safety metrics as task progress; zero-violation 0-reward
  is still Gate 4 failure.
- Do not reuse Zen checkpoints/optimizer state for another hypothesis.
- The pinned registry still contains only `first_action_only`; Prometheus's
  half-weighting remains unreproduced.

## 10. Next-step options (each needs its own preregistration)

1. Recovery-distribution data (T2) remains the only untried primary axis.
2. Longer-schedule / smaller-batch replication of the temporal-weight axis,
   if the offline contradiction matters.
3. Deploy one Zen artifact (firstaction recommended by safety profile) for
   first-deviation tracing against the existing expert replay baseline.

## 11. Immutable evidence ledger (remote durable root)

- selections: `runs/<exp>/selection/m2-smolvla450m-zen-cuda-b64-{uniform,firstaction}-001-selection.json`
- gates: `runs/<exp>/gates/gate{3,4}-smolvla-sim-{411,422}.json` + episode dirs
- artifacts: `artifacts/<exp>/m2-smolvla450m-zen-cuda-b64-{uniform,firstaction}-001-step0316-deploy-001/`
- reload reports: `runs/<exp>/validation/*-reload*-step-000316.json`
- driver log/events: `runs/orchestration/zen-furnace-events.jsonl`, `zen-phase-*.log`
- superseded (defective-layout) artifacts retained under `*-superseded*-<ts>` names
