# Canonical full-frame 001 fixed-endpoint post-training and Gate plan

Identity: `canonical-fullframes-20260914-001-posttrain-001`.
This is a separate, create-only post-training acceptance registration for the
fixed endpoint of `canonical-fullframes-20260914-001`. It does not authorize a
resume, another optimizer update, checkpoint search, data download, dependency
change, hidden-test access, or a new remote worker.

## Bound endpoint and prerequisites

- Source run: `canonical-fullframes-20260914-001`.
- Source workspace: `20260914T114222Z-95cf9cf9483b-db29920cea05`.
- Dispatch composite SHA-256:
  `db29920cea05745645247e55e6d7a73e8f881388d8898fb8c08d413ab7a751ce`.
- Template SHA-256:
  `8c91ddc172f09dcaf21b3c63b5e12ce200e949b9abd11784d751eb04b389b19c`.
- Fixed endpoint: complete checkpoint at update `5000`; checkpoint `2500` is
  retained as recovery evidence and is not a selection candidate.
- The endpoint is bound to the action-repair training config
  `configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml`
  (SHA-256
  `0e9dd0499d0708939ac73cc5d517849f133cf6deab072d9cde09f2880ae22210`), the
  train-only normalization report (SHA-256
  `263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988`), and
  its dataset-view manifest (SHA-256
  `9853c191ae87016379fc1a16ebfbb87e05ab5147cd8a03d82f9c2a894c9b531e`).
  The recovered endpoint model file is SHA-256
  `d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef` and
  the complete pretrained directory occupies `1,201,362,478` bytes.
- The endpoint is admissible only after the recovered handoff manifest,
  transfer receipt, every recovered-file SHA-256, exact checkpoint metric
  records for 2500/5000, model-ingress seal, schedule identity, and both
  independent reload-process outputs have been verified.
- The recovery manifest must prove `20000` unique `(episode, frame)` inputs,
  the sealed `Random(20260809)` 40-episode × 500-frame schedule, `5000`
  successful optimizer updates, finite loss/gradients, and unchanged frozen
  tensors. The post-training runner must fail closed on any mismatch.

The current recovery window is a separate no-GPU file-transfer operation,
registered in `reports/training/canonical-recovery-window-2026-09-14-001.json`
(SHA-256
`e46cea4f7d73f846a8ce2d8cffe379fe4225a72dd6761cffa4d87fc2fb28b066`). It has
a 1,800-second deadline, 0.5 CPU / 2 GiB budget, zero model forwards and no
remote mutation beyond platform startup/shutdown. It must never run offline
evaluation or Gates.

Any later model evaluation requires a separately admitted AutoDL CUDA worker:
the registered CUDA container boundary, `nested_docker_used=false`, 2 CPU,
6 GiB container memory and swap, process RSS ≤8 GiB, CUDA allocated ≤8 GiB,
CUDA reserved ≤10 GiB, and at least `1,073,741,824` durable free bytes before
each stage, a 3,600-second work deadline and a 4,200-second protected shutdown
deadline. The worker must bind the recovered endpoint and this plan before
loading a model. A new GPU window is a separate parent-agent resource
decision.

The lower storage admission is evidence-based and uses a zero-copy endpoint
artifact: the `1,201,362,478`-byte pretrained directory remains at its
checkpoint path and the artifact contains symlinks plus at most 1 MiB of
metadata. Offline reports are capped at 16 MiB, Gate reports and per-seed
summaries at 32 MiB, simulation recordings and trajectory artifacts are
disabled, runtime scratch is capped at 256 MiB, and runner/shutdown metadata
has a 17 MiB cap. A 702 MiB safety reserve brings the required floor to 1 GiB.
The completed training snapshot reported `2,318,667,776` free bytes (`2.1596`
GiB), leaving `1,244,925,952` measured bytes above the floor. A full
checkpoint or export copy is forbidden; the runner checks free bytes and
output caps before and after every stage and stops before the budget can be
crossed.

The executable recovery/preflight adapter is
`scripts/canonical_fullframes_posttrain.py` (SHA-256
`f046afdb2a909d736edac32d9e4d536b695bc32b2ed380005ae7c8f72abff7ba`), with
static contract tests in `tests/test_canonical_fullframes_posttrain.py`
(SHA-256
`921cc13cebca61f539a87c64fd4ac97de70fd290d75e43aab5474d15f5e1e4d7`). The
runtime chain is sealed by `scripts/canonical_fullframes_endpoint.py`
(SHA-256
`35d21b2730b3618cfdcf8a51b66fb659208149c05ac20fb22975ac89d4ca1da3`),
`scripts/canonical_fullframes_offline.py` (SHA-256
`972c801e835480d42d3ac8dabee694d0049bd9afda71cc3b6b1a0480f802ad9d`),
`scripts/canonical_fullframes_sim_gate.py` (SHA-256
`db2316999dac965ab73829a1e65c2dadb746b47bac92e186445e50d2182eb131`), and
`scripts/run_canonical_fullframes_posttrain.sh` (SHA-256
`8c60524d730da932531a9fe71f9d68326060fb965a3c95a433016c71540135e3`). The
versioned recovery entry points are `runs/canonical-furnace-preparation-20260914-001/receive-002.sh`
and `backup-checkpoints-002.sh`; both check SSH readiness before creating a
target, use at most four concurrent read-only chunk transfers, and verify each
reconstructed file against a remote SHA-256 inventory.

## Offline endpoint evaluation

Run in the pinned Linux Docker boundary with networking disabled, using the
recovered native checkpoint and its saved processor state. Use the repaired
canonical image scaling and the real input chain: raw uint8 camera bytes,
14-dimensional state, projected 14-dimensional absolute action targets,
complete 50-action chunks and padding masks. Normalization is train-only.

- Evaluate the 40 registered training episodes and the five development
  episodes separately using this fixed allowlist: train episodes, in the
  registered order, are `[49,4,23,43,21,37,18,34,0,47,38,29,3,26,14,17,44,30,15,42,10,35,25,32,19,36,41,28,8,27,16,11,2,20,9,39,46,48,12,40]`;
  development episodes are `[22,13,7,33,45]`; frame offsets are exactly
  `[0,125,250,375]` for every listed episode. This is 160 train samples and
  20 development samples, each with a complete 50×14 target chunk.
- Run the primary offline pass with zero inference noise and flow time `0.5`.
  Run a separately labelled, non-selecting deployment-noise diagnostic with
  seeded standard-normal policy-noise seeds `[1000,1001,1002,1003,1004]` over
  the same allowlist. Do not mix the two cohorts or use the diagnostic to
  select the endpoint.
- Report full-chunk and first-action action MAE/RMSE, fixed-flow loss,
  finite/action-limit rates, smoothness, latency, and internal gripper-support
  diagnostics for each `(cohort, noise)` pair.
- Recompute train-only constant baselines from the training split and retain
  the development comparison. Report fit and development generalization as
  separate evidence; neither offline metric selects a checkpoint or establishes
  task success.
- Offline acceptance is an integrity gate: exact cohort/frame/noise identity,
  finite outputs/loss, zero invalid actions and zero Action Contract limit
  violations, complete 50-action chunks, train-only statistics, and hidden-test
  exclusion. MAE improvement is reported evidence and has no invented pass
  threshold.
- The sealed hidden episodes `[31, 6, 1, 24, 5]` must not be materialized or
  loaded. Any hidden-test or non-finite/input/padding identity violation stops
  the run while preserving the failure report.

## Export and independent reload

Create one endpoint-bound research artifact for step 5000 only. Preserve the
complete policy config, model weights, tokenizer, preprocessor/postprocessor
state, train-only normalization, Action Contract, source identities and file
manifest. Re-run the fixed native input probe in two separate model processes
and require exact full-array action equality for every registered probe; record
parameter hashes before/after and require zero optimizer creation or updates.
The artifact is research-only and has no physical-robot validation.

## Closed-loop Gates

Bind the Gates to the existing executable protocol instead of reconstructing
thresholds from this prose: `scripts/sim_gate.py` SHA-256
`33ee39c897232baa876dfd55a6e76a7b1a9373784293f75f1a5778886a6147cd`, the
Faust protocol config
`configs/vla/smolvla_450m_aloha_insertion_faust_sim_001.yaml` SHA-256
`20386f7ad5dda19d9d5ffa1668bfa45df5ae3f6441d94f45661eb9f10fa442f3`, and
the Action Contract
`configs/sim/aloha_insertion_smolvla.yaml` SHA-256
`fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62`.
The protocol config is a historical threshold/source authority only; its XPU
runtime fields are not the execution runtime for this endpoint. Actual Gate
execution must use the separately budgeted AutoDL CUDA container
(`nested_docker_used=false`, platform-managed memory/swap) with seeded
standard-normal policy noise and receding-horizon first-action execution.
Preserve the protocol config's explicit projection and adapter criteria; the
canonical endpoint receives a new plan/config identity.

- Gate 3: environment/noise seed `20260809`, maximum 20 steps. Require finite
  actions, projected policy actions within the Action Contract, no additional
  adapter clipping, zero unexpected collisions, non-empty rollout, and exact
  artifact reload identity.
- Gate 4: environment/noise seeds `1000,1001,1002,1003,1004`, maximum 500
  steps per seed, with matching policy-noise seeds. Require task-success rate
  `>= 0.2`, finite actions, projected policy actions within the Action Contract,
  no additional adapter clipping, zero joint-limit violations, and zero
  unexpected collisions. Preserve per-seed reward, rollout length, action
  smoothness and inference/simulation latency. A failed Gate 3 blocks Gate 4.
- Gate reports must bind the endpoint artifact manifest, this post-training
  plan, the Gate 3 report, the Action Contract and current code identity.
  Passing Gate 3 is a safety/reload result; only Gate 4 task success is the
  M2 policy acceptance criterion.

## Stop and lifecycle rules

Stop on non-finite values, identity drift, hidden access, input/processor/
padding mismatch, checkpoint or reload mismatch, action-contract violation,
OOM, or resource failure. Keep all failed reports and logs. Do not lower any
threshold, retry in place, release the instance, or infer success from loss,
offline fit, positive reward, or Gate 3. If a new CUDA worker is required after
the registered shutdown, stop at this boundary and have the parent agent make a
separate resource/authorization decision.

Status at registration: `preregistered`; no post-training endpoint, offline
metrics, export, reload, Gate 3 or Gate 4 result is claimed by this document.
