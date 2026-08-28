# M2 SmolVLA training harness v2

This document describes the version-2 training composition layer introduced on
branch `codex/smolvla-training-harness-v2`. It complements
`docs/m2-smolvla-architecture.md` (the stable M2 navigation map) and does not
replace it.

## 1. Why the harness was rebuilt

The historical SmolVLA training path grew one launcher plus one trainer entry
per experiment family (Faust action-repair, Aster horizon-loss, Way
state-robustness CUDA). Each layer wrapped the previous one through private
cross-script functions and environment variables, and each launcher duplicated
roughly the same plan-validation pile. Two concrete failure modes motivated the
rebuild:

- `aster-b8-002` was invalidated because a wrapper covered only the raw loss
  boundary and left the upstream all-valid-horizon denominator in place —
  exactly the kind of partial-boundary mistake a layered monkeypatch stack
  invites;
- every new single-axis experiment paid the duplication cost again and risked
  silently diverging from the registered protocol.

The v2 harness keeps the pinned LeRobot trainer (revision
`c903b114a90e703b3f7d0c46cb38727c328c55ff`) as the only training loop and
rebuilds only the Rosetta-owned composition layer around it.

**What this rewrite is not:** it does not change any learning semantics, it
does not repair the open Gate 4 research failure (training / closed-loop
generalization), and it does not authorize a new furnace by itself. It is
engineering restructuring that makes the audit's recommended single-axis
experiments (optimizer instrumentation, resume parity, noise-aligned
validation, gradient diagnostics) structurally harder to get wrong.

## 2. Ownership map

| Path | Owns | Must not own |
|---|---|---|
| `src/rosetta_reality/vla/training/plan.py` | version-2 plan schema: structure, split guards, optimizer/scheduler contract, quarter monitoring policy, feature declarations, T9 `save_freq % log_freq == 0` | file checksum resolution (launcher duty) |
| `src/rosetta_reality/vla/training/features.py` | feature registry: install/restore, declaration order, double-install fail-closed, rollback | learning semantics beyond the frozen feature modules |
| `src/rosetta_reality/vla/training/context.py` | immutable `TrainingContext` handed to every feature | mutable global state |
| `src/rosetta_reality/vla/training/masked_camera.py` | masked-camera encoder skip migrated verbatim from the frozen `train_smolvla_trackio` | new vision behavior |
| `src/rosetta_reality/vla/training/launch.py` | plan-to-CLI assembly, optimizer CLI fragment, runtime-experiment composition | training-loop implementation |
| `scripts/run_smolvla_v2.py` | single launcher: offline guard, plan/parent/split/resource/prerequisite/normalization validation, launch manifest, environment assembly, mode dispatch | bypassing prerequisite evidence |
| `scripts/train_smolvla_v2.py` | single trainer entry: plan-declared feature installation with rollback, pinned `lerobot_train.main()`, reverse-order restore | experiment selection or evaluation |

Feature implementations are reused, not rewritten: `vla/horizon_loss.py`,
`vla/state_robustness.py`, `vla/processor.py`, `vla/fixed_samples.py` and the
Trackio bridge are consumed read-only. `checkpoint_memory.py` and
`checkpoint_accelerator_memory.py` remain frozen for provenance; the v2
`checkpoint_memory_trim` feature is one new device-aware implementation.

## 3. The feature registry

A version-2 plan declares an **ordered** `features` list. The registry
resolves each declaration, installs in exactly that order, rolls back on the
first failure and restores in reverse order:

| Feature | Replaces / reuses |
|---|---|
| `trackio_logging` | `WandBLogger` swap with a plan-bound `TrackioLogger` subclass; no module-state patching, runtime experiment read from the launcher-written file |
| `train_only_statistics` | migrated `_install_train_only_statistics`; report path from the plan, not the environment |
| `masked_camera_skip` | migrated `_install_masked_camera_encoder_skip` with restore support |
| `action_boundary_projection` | migrated `_install_projection` around `ensure_smolvla_action_boundary` |
| `fixed_frame_sampler` | migrated fixed-frame `EpisodeAwareSampler` replacement, phase-parameterized |
| `horizon_weight_profile` | delegates to `vla/horizon_loss.py` (upstream SHA fail-closed) |
| `state_robustness_jitter` | delegates to `vla/state_robustness.py` (upstream SHA fail-closed) |
| `state_conditioning_dropout` | delegates to the new `vla/visual_conditioning.py`; drops complete normalized-state samples with a dedicated RNG and leaves validation/deployment clean |
| `checkpoint_memory_trim` | new device-aware merge of the two historical memory modules |

Guarantees: unknown or duplicated declarations fail closed at schema
validation and again at stack construction; double installation of an
installed feature raises instead of silently stacking wrappers; every feature
restores the untouched upstream surface (tests and diagnostics only — the
formal process ends with the trainer).

Cross-declaration invariants enforced by the schema:
`masked_camera_skip` requires
`training.policy.skip_fully_masked_camera_encoding: true`;
`checkpoint_memory_trim` requires `resources.checkpoint_memory_trim: true`;
`horizon_weight_profile` requires a `loss_contract`; `state_robustness_jitter`
requires a `state_robustness_contract`; `trackio_logging` requires a tracking
section; `state_conditioning_dropout` requires a
`visual_conditioning_contract`.  The dropout implementation does not consume
the global model/dataloader RNG and forbids formal resume until its dedicated
generator state participates in the registered T7 parity contract.

## 4. Version-2 plan schema

`schema_version: 2`, `role: vla`, `status: preregistered`, checksum-pinned
single-level `extends` inheritance, and the sections: `plan_id` / `run_name`,
`parent_experiment`, `training` (episodes, batch, steps, save/log frequencies,
checkpoint grid, optimizer + cosine-warmup scheduler contract, policy
overlay), `validation`, `resources`, optional quarter-only `monitoring`
(sleep 300 mandatory), ordered `features`, `prerequisites`, `normalization`,
`implementation_files`, `stop_conditions`. Structural differences from the
historical formal plans:

- the ordered `features` list is the single declaration of local extensions —
  environment variables carry only paths and runtime identity, never training
  semantics;
- `save_freq % log_freq == 0` is enforced so every checkpoint has an exact
  metric row (audit finding T9);
- training step counts must be divisible by four for the quarter-checkpoint
  policy.

The launcher additionally enforces, per mode: formal training must use the
registered train split exactly; smoke/preflight episodes must be train-split
subsets disjoint from validation and hidden test; active Docker memory limits
must match the plan; prerequisite evidence files must match their declared
checksums (with the frozen `run_smolvla_phase` deep validators reused
read-only for benchmark/gates/trackio/smoke-acceptance reports); the
normalization report, dataset-view manifest, per-file inventory and view stats
must all agree; and `train` mode additionally requires validated no-optimizer
preflight and base-validation reports.

## 5. Execution flow

```text
run_smolvla_v2 {preflight|smoke|train} --plan <v2 plan>
    -> offline guard -> schema validation -> parent binding -> split guards
    -> resource guard -> prerequisite evidence -> normalization identity
    -> coverage (train) -> mode reports (train)
    -> runtime experiment file + launch manifest under the durable run root
    -> environment assembly -> lerobot-train CLI (single construction point)
    -> train_smolvla_v2: feature stack install (with rollback)
    -> pinned lerobot_train.main()
    -> finally: reverse-order restore + finish_trackio
```

`preflight` delegates to the existing no-optimizer `smolvla_forward_check`
entry exactly as the historical launchers did. Outputs are create-only: an
existing smoke/train output directory fails closed.

## 6. Frozen provenance

All historical `run_smolvla_*` / `train_smolvla_*` scripts, their hash-bound
configs and the completed `runs/` / `artifacts/` evidence stay untouched. They
remain the authority for the completed Faust, Aster and Way identities. New
experiments that want the v2 harness must register new plan identities with
their own prerequisites; no historical plan is migrated or edited.

## 7. Verification entry points

Run from the repository root in WSL Bash through the pinned Docker path:

```bash
scripts/run_m2_container.sh vla-xpu \
  python -m pytest -q \
  tests/test_smolvla_training_plan_schema.py \
  tests/test_smolvla_training_features.py \
  tests/test_smolvla_training_launch.py \
  tests/test_smolvla_action_repair.py \
  tests/test_smolvla_horizon_loss.py \
  tests/test_smolvla_state_robustness.py \
  tests/test_smolvla_visual_conditioning.py \
  tests/test_smolvla_formal_protocol.py \
  tests/test_smolvla_faust_protocol.py

scripts/run_m2_container.sh vla-xpu \
  python -m ruff check \
  src/rosetta_reality/vla/training \
  scripts/run_smolvla_v2.py \
  scripts/train_smolvla_v2.py
```

These are code/protocol checks only. Before any formal plan adopts the v2
harness, a separately authorized tiny no-optimizer forward plus two-step
optimizer smoke through `run_smolvla_v2` must pass, mirroring the historical
per-capability ladder.
