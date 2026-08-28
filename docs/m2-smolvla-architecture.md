# M2 SmolVLA architecture and navigation

This is the stable architecture and navigation entry point for the current
SmolVLA M2 work. It is intentionally not named after a furnace or date. Update
this file when component ownership, execution boundaries, the current evidence
source, or the next repair stage changes.

Document updated: 2026-08-28. Current Faust evidence snapshot: 2026-08-12;
current Aster implementation audit: 2026-08-13; current Way CUDA evidence:
2026-08-14; current object-geometry teacher/official planner evidence:
2026-08-16; current Zen two-arm campaign completion audit: 2026-08-27.

## 1. Mandatory reading order and authority

Before changing SmolVLA data, processors, trainers, optimizers, checkpoints,
exports, evaluation or simulation, read in this order:

1. `AGENTS.md` — safety, runtime, stage-gate and repository rules;
2. this file — stable component, control-flow and evidence map;
3. `reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.md`
   — current empirical interpretation and repair order;
4. `reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.json`
   — machine-readable result and finding registry;
5. `reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md` and its JSON
   companion — the newest completed formal campaign (two-arm Zen) and its
   Gate 4 negative results, superseding nothing above but extending the
   failure tally and next-step options; the follow-up diagnostic is
   preregistered in
   `reports/training/m2-smolvla-zen-first-deviation-preregistration-2026-08-28.md`
   and its JSON companion;
6. for object-geometry work,
   `reports/training/m2-smolvla-geometry-teacher-audit-2026-08-14.md` and its
   JSON companion, then
   `reports/training/m2-smolvla-mink-ik-audit-2026-08-15.md` and its JSON
   companion, then
   `reports/training/m2-smolvla-moveit-path-planner-preregistration-2026-08-15.md`
   and its JSON companion, then
   `reports/training/m2-smolvla-moveit-runtime-boundary-preregistration-2026-08-15.md`
   and its JSON companion, then
   `reports/training/m2-smolvla-moveit-joint-path-margin-preregistration-2026-08-15.md`,
   `reports/training/m2-smolvla-athena-plan026-exact-audit-2026-08-15.md`,
   `reports/training/m2-smolvla-execution-margin-diagnostic-preregistration-2026-08-15.md`,
   `reports/training/m2-smolvla-athena-plan027-exact-audit-2026-08-15.md`, and
   `reports/training/m2-smolvla-execution-reserve-preregistration-2026-08-15.md`,
   then `reports/training/m2-smolvla-athena-plan028-exact-audit-2026-08-15.md`,
   then
   `reports/training/m2-smolvla-moveit-start-path-constraint-recovery-preregistration-2026-08-15.md`,
   then `reports/training/m2-smolvla-athena-plan030-exact-audit-2026-08-15.md`,
   then
   `reports/training/m2-smolvla-moveit-hybrid-trajectory-execution-preregistration-2026-08-15.md`
   then `reports/training/m2-smolvla-athena-plan032-exact-audit-2026-08-15.md`,
   then
   `reports/training/m2-smolvla-mujoco-position-feedforward-preregistration-2026-08-15.md`
   then
   `reports/training/m2-smolvla-athena-plan033-local-exact-audit-2026-08-15.md`,
   then
   `reports/training/m2-smolvla-mujoco-sparse-actuator-moment-repair-preregistration-2026-08-15.md`
   then
   `reports/training/m2-smolvla-athena-plan034-local-exact-audit-2026-08-15.md`,
   then
   `reports/training/m2-smolvla-gym-joint-name-adapter-repair-preregistration-2026-08-15.md`
   then
   `reports/training/m2-smolvla-athena-plan035-local-exact-audit-2026-08-15.md`
   with their JSON companions, then the immutable Plan `050`--`054` exact
   audits ending at
   `reports/training/m2-smolvla-athena-plan054-local-exact-audit-2026-08-16.md`,
   its JSON companion, the Athena remote exact reproduction at
   `reports/training/m2-smolvla-athena-plan054-remote-exact-audit-2026-08-16.md`
   and its JSON companion, then the repaired content-addressed Athena package
   report at
   `reports/training/m2-smolvla-athena-plan054-workspace-package-repair-2026-08-16.md`
   and its JSON companion, then the authorized local repair-chain audits
   `m2-smolvla-athena-plan055-local-exact-audit-2026-08-16`,
   `m2-smolvla-athena-plan056-local-exact-audit-2026-08-16`,
   `m2-smolvla-athena-plan057-local-exact-audit-2026-08-16`, and
   `m2-smolvla-athena-plan058-local-exact-audit-2026-08-16` with their JSON
   companions;
7. the registered experiment, formal-run and simulation configs named below;
8. immutable `runs/` and `artifacts/` evidence referenced by those reports;
9. implementation code and tests for the component being changed.

Authority is layered rather than interchangeable:

| Question | Source of truth |
|---|---|
| What is allowed and where may it run? | `AGENTS.md` |
| Which component owns a behavior? | this architecture document plus executable code |
| What was preregistered for a run? | the hash-bound config for that run |
| What physical action semantics apply? | `configs/sim/aloha_insertion_smolvla.yaml` |
| What actually completed? | immutable machine-readable evidence under `runs/` and `artifacts/` |
| Why did the result pass or fail? | the current audit Markdown and JSON |

If prose conflicts with a hash-bound config, Action Contract, code assertion or
evidence file, stop and reconcile the mismatch. Do not silently select the
convenient source. Historical reports remain provenance and must not be edited
to make a later result appear successful.

## 2. Current M2 status

The current work line is **VLA / System 1**, not Qwen ER.

| Field | Current state |
|---|---|
| development policy | revision-pinned `lerobot/smolvla_base` 450M |
| upstream LeRobot revision | `c903b114a90e703b3f7d0c46cb38727c328c55ff` |
| base model revision | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| dataset | `lerobot/aloha_sim_insertion_human` |
| dataset revision | `cc571a3c661df81b566dbfde3d5c1e85fcdf7884` |
| split | 40 train / 5 validation / 5 sealed hidden test episodes |
| repaired experiment | `m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003` |
| completed formal runs | Faust `-002`; corrected Aster `-003`; Way CUDA batch-64/default `formal-002`; Zen two-arm `uniform-002` / `firstaction-001` |
| latest selected checkpoint | Zen uniform step 316, validation first-action MAE `0.021572770214905695`; Zen firstaction step 316 `0.022150604739519103`; Way step 316 `0.030136355795964066`; Aster remains the `0.02250973408226855` read-only offline control |
| export/reload | Faust, Aster, Way and both Zen arms passed with exact action equality |
| Gate 3 | Faust, Aster, Way and both Zen arms passed |
| Gate 4 | Faust, Aster, Way, Zen-uniform (`411`) and Zen-firstaction (`422`) all failed `0/5` |
| Aster T1 attempt | `aster-b8-002` completed, but is not valid single-axis evidence |
| current T1 result | `aster-b8-003` selected/exported and Gate 3 passed; Gate 4 failed |
| current Way result | fresh-base train-only normalized-state jitter `std=0.05`; selected/exported and Gate 3 passed; Gate 4 failed |
| current Zen result | preregistered two-arm single-axis comparison (uniform control vs `first_action_only` treatment) at batch 64 / 316 updates through the v2 harness; both arms trained to convergence, exported bit-exact, Gate 3 passed, Gate 4 failed `0/5` with reward `0` on every seed 1000--1004; the registered hypothesis is rejected |
| current recovery-teacher result | Plan `054` failed grasp drift in `lift` locally and on Athena. Authorized local repair chain: Plan `055` feedback-anchored lift preserved both grasps but hit an unregistered table contact; Plan `056` lifted the contact scope but hit one Mink IK failure; Plan `057` extended the official MoveIt fallback to lift but hit the right gripper-bar/table contact; Plan `058` added that observed contact and ran all 750 steps safely in `lift` without losing grasp, but the right peg never left the table. Exact remains failed and all later gates remain sealed |
| M2 completion | **not complete** |
| hidden test | not loaded |

Faust proves that the repaired standard-action boundary is legal and survives
checkpoint export/reload. It does not prove a successful closed-loop policy.
No new full furnace is authorized merely by this status snapshot.

The current geometry-planner authority is the local repair chain ending at
Plan `058`:
`reports/training/m2-smolvla-athena-plan058-local-exact-audit-2026-08-16.md`
and its JSON companion, with predecessors
`m2-smolvla-athena-plan055-local-exact-audit-2026-08-16`,
`m2-smolvla-athena-plan056-local-exact-audit-2026-08-16`, and
`m2-smolvla-athena-plan057-local-exact-audit-2026-08-16` reports plus their
JSON companions. Plans `055`--`058` are immutable negative evidence. Plan `054`
reached `lift` in 423 steps but lost the peg grasp and failed the unchanged
45 mm grasp-drift invariant; its expanded-budget trust-region counter remained
zero locally and on Athena. The local repair chain moved the boundary from
grasp drift to lift contact scope, then to lift IK fallback, and finally to
right-peg lift progress. Plan `058` ran the full 750-step horizon in `lift`
with no safety breach and both grasps retained, but the right peg did not leave
the table. No Plan `059` is authorized in this wrap-up scope. Tuning,
development, collection, policy Gate, validation, hidden-test and
recovery-label boundaries remain closed.

`aster-b8-002` produced a real validation first-action MAE of
`0.022567120325818125` at step 2500, but its model-boundary mask left the
upstream all-valid-horizon denominator unchanged. The reported training loss
and gradients were therefore scaled by the remaining valid horizon (50x on an
unpadded chunk), so the run changed both temporal weighting and effective
gradient scale. Preserve its checkpoints and reports as exploratory evidence;
do not use them as acceptance evidence for the registered single-axis T1 claim.

The corrected `aster-b8-003` run completed 2500 steps, passed public Trackio
sync and validation-only selection, and selected step 2500. Its registered
first-action MAE improved from the Faust selected control's
`0.027562587232595043` to `0.02250973408226855` (18.33%). The exported artifact
passed exact deterministic reload. These are offline and artifact-integrity
results only. Aster's Gate 3 passed, but its fixed five-seed Gate 4 produced
`0/5` success with reward `0` in every rollout, so Aster is not accepted.

The fresh-base Way CUDA batch-64/default run completed 316 steps and selected
step 316 without hidden-test access. Its clean-state validation first-action MAE
was `0.030136355795964066`, 33.88% worse than the read-only Aster control, while
its exported deploy artifact passed exact independent reload. Way's Gate 3
passed, but the same fixed Gate 4 seeds 1000--1004 all completed 500 steps with
reward `0` and no success. All actions were finite; invalid-action, raw/action
limit, joint-limit and corrected unexpected-collision counts were zero. Way is
therefore not accepted: train-only normalized-state Gaussian jitter with
`std=0.05` did not repair closed-loop generalization.

## 3. System boundary

Rosetta Reality separates low-frequency reasoning, high-frequency control and
embodiment execution:

```mermaid
flowchart LR
    O["Observation + task"] --> ER["ER / System 2"]
    ER --> AP["Versioned ActionPlan"]
    AP --> VLA["VLA / System 1"]
    O --> VLA
    VLA --> AC["Rosetta Action Contract"]
    AC --> SA["Simulation / embodiment adapter"]
    SA --> R["Robot and world state"]
    R --> O
```

The current M2 evaluation does not claim ER/VLA integration. It conditions
SmolVLA with a fixed insertion instruction and directly evaluates the VLA,
Action Contract and simulator loop. M3 remains blocked until M2 Gate 4 passes
and Qwen ER independently passes its own evaluation.

## 4. Repository ownership map

| Path | Owns | Must not own |
|---|---|---|
| `docs/architecture.md` | model-independent ER/VLA system overview | current furnace status |
| `docs/er-vla-pipeline.md` | original role, reuse and gate design | current completed-result authority |
| `docs/m2-smolvla-architecture.md` | stable current M2 navigation and component map | immutable run evidence |
| `configs/vla/` | VLA experiment, formal-run and evaluation identities | simulator actuator implementation |
| `configs/sim/aloha_insertion_smolvla.yaml` | complete physical Action Contract | model training hyperparameters |
| `src/rosetta_reality/vla/action_space.py` | experiment/action-space schema loading and identity checks | MuJoCo calls |
| `src/rosetta_reality/vla/processor.py` | dataset-to-model and model-to-standard-action boundary | task success logic |
| `src/rosetta_reality/vla/checkpoint_memory.py` | save/resume memory-boundary handling | optimizer policy |
| `src/rosetta_reality/vla/fixed_samples.py` | deterministic diagnostic sample identity | train/validation split selection |
| `src/rosetta_reality/vla/horizon_loss.py` | checksum-bound temporal mask plus selected-valid reduction | optimizer/scheduler policy |
| `src/rosetta_reality/vla/state_robustness.py` | checksum-bound, train-only normalized-state jitter | validation/deployment mutation or recovery labels |
| `src/rosetta_reality/vla/runtime_compatibility.py` | versioned post-training normalization/tokenizer/root and CUDA compile guards | mutation of completed hash-bound runners or learning semantics |
| `src/rosetta_reality/sim/` | simulator-neutral action contract and Gym-ALOHA adapter | SmolVLA internals |
| `src/rosetta_reality/sim/geometry_teacher.py` | object/EEF/contact/reward-conditioned event teacher and bounded task-space targets | time-indexed source actions or simulator-specific IK |
| `src/rosetta_reality/sim/moveit_aloha_planner.py` | process-isolated, collision-identity-checked JSONL client for the complete accepted official MoveIt path; validates trajectory metrics and executes a retained reference with upstream `SimpleSampler`/`ForwardTrajectory` semantics while preserving observable `2e-5`-rad start-bound reconciliation and all-path joint-margin evidence | motion-planning algorithms, IK implementation or task-space tolerance relaxation |
| `src/rosetta_reality/sim/mujoco_position_feedforward.py` | fail-closed inversion of MuJoCo's official affine SISO actuator equation for direct fixed-gain joint-position actuators at a retained static target; preserves the tightened command margin and registered correction bound | path search, pose-gate relaxation, learned controller tuning or changes to `gym_aloha.py` |
| `integration/aloha_moveit2/` | compose the pinned dual-VX300S planning scene, load official LMA plus OMPL, apply native MoveIt joint path constraints and reject any invalid start/goal/path/next state | custom path search or simulator policy logic |
| `docker/Dockerfile.aloha-moveit2` | hash-bound ROS Humble/MoveIt/OMPL/Interbotix build boundary | Python simulator dependencies, model/data content or nested AutoDL Docker |
| `src/rosetta_reality/eval/` | metrics and trajectory diagnostics | optimizer updates |
| `src/rosetta_reality/tracking/` | durable Trackio bridge and sanitized payloads | checkpoint weights or secrets |
| `scripts/run_smolvla_action_repair_formal.py` | plan/prerequisite validation and formal launch assembly | upstream flow-loss implementation |
| `scripts/train_smolvla_action_repair_formal.py` | runtime injection into the pinned LeRobot trainer | experiment selection decisions |
| `scripts/evaluate_smolvla_action_repair_validation.py` | offline validation reports | hidden-test selection |
| `scripts/select_smolvla_action_repair_checkpoint.py` | validation-only checkpoint selection | Gate 4 acceptance |
| `scripts/export_smolvla_action_repair.py` | deploy artifact and exact independent reload | further training |
| `scripts/smolvla_action_repair_sim_gate.py` | Gate 3/4 closed-loop execution and reports | training loss |
| `scripts/run_smolvla_horizon_loss_formal.py` | corrected Aster plan, prerequisite and implementation-hash validation | dependency-cache mutation |
| `scripts/train_smolvla_horizon_loss_formal.py` | plan-authorized temporal-loss injection | checkpoint selection |
| `scripts/select_smolvla_aster_checkpoint.py` | Faust-control comparison plus public-sync provenance | hidden-test or Gate 4 acceptance |
| `scripts/run_smolvla_state_robustness_smoke.py` | Way plan/prerequisite validation and isolated two-step launch | formal training authorization |
| `scripts/train_smolvla_state_robustness_smoke.py` | train-only state-jitter and Aster-loss injection | validation/deployment input mutation |
| `scripts/accept_smolvla_way_state_jitter_smoke.py` | immutable Trackio/checkpoint/plan acceptance | closed-loop efficacy claim |
| `scripts/run_smolvla_state_robustness_cuda_smoke.py` plus CUDA verify/accept wrappers | AutoDL batch feasibility, registered runtime-repair identity and independent smoke reload | formal training authorization or failed-run state reuse |
| `scripts/run_smolvla_state_robustness_cuda_formal.py` | smoke-bound fresh-base Way CUDA formal launch and resource identity | fallback mutation or checkpoint reuse |
| `scripts/run_smolvla_state_robustness_cuda_formal_v2.py` | future-plan CUDA compile guard before delegation to a separately registered formal runner | plan registration or smoke acceptance fabrication |
| `scripts/evaluate_smolvla_way_validation.py` | clean-input Way validation on the registered validation split | state jitter or hidden-test selection |
| `scripts/evaluate_smolvla_way_validation_runtime_repair.py` | create-only processor/tokenizer compatibility repair around the immutable Way validator | formal-plan mutation or validation semantics changes |
| `scripts/select_smolvla_way_checkpoint.py` | validation-only Way selection, public-sync provenance and Aster comparison | closed-loop acceptance |
| `scripts/export_smolvla_way.py` | Way deploy artifact plus exact independent reload | further training or state jitter at deployment |
| `scripts/export_smolvla_way_runtime_repair.py` | create-only export compatibility repair around the immutable Way exporter | selected-model mutation or relaxed reload checks |
| `scripts/smolvla_autodl_way_sim_gate.py` | AutoDL CUDA Gate wrapper with durable-run evidence resolution | workspace-local ignored evidence |
| `scripts/smolvla_autodl_way_sim_gate_runtime_repair.py` | create-only Gate runtime repair around the immutable Way Gate wrapper | protocol, seed, threshold or Action Contract changes |
| `scripts/evaluate_smolvla_way_validation_v2.py`, `scripts/export_smolvla_way_v2.py`, `scripts/smolvla_autodl_way_sim_gate_v2.py` | reusable post-Way compatibility entry points for future plans | retroactive mutation of Way evidence or automatic experiment authorization |
| `scripts/run_autodl_posttrain_v2.sh` | isolated future validation/export/Gate dispatch without changing the completed Way runner identity | optimizer or formal-training authorization |
| `scripts/run_autodl.sh` | registered AutoDL doctor/smoke/formal/validation/selection/export/Gate command dispatch | bypassing verified plans or nested Docker |
| `scripts/smolvla_zen_protocol.py`, `scripts/smolvla_zen_validate.py` | preregistered Zen two-arm protocol identity, specs and plan validation | mutating completed Zen identities or authorizing new arms |
| `scripts/select_smolvla_zen_checkpoint.py`, `scripts/export_smolvla_zen.py` | Zen validation-only selection and deploy export with derived gate-facing records | hidden-test selection or Gate 4 acceptance |
| `scripts/smolvla_autodl_zen_sim_gate.py` | AutoDL CUDA Gate 3/4 wrapper rendering per-arm sim plans with derived selection and inventory backup evidence | protocol, seed, threshold or Action Contract changes |
| `scripts/run_smolvla_zen_furnace.sh` | the guarded tmux Zen furnace ladder (doctor through Gate 4 with per-phase guards) | unguarded execution or in-place parameter edits |
| `scripts/diagnose_smolvla_zen_trajectory.py`, `scripts/run_zen_trajectory_trace.sh` | the preregistered Zen first-deviation trace diagnostic and its local XPU runner | gating claims or recovery-label authorization |
| `src/rosetta_reality/vla/training/` | version-2 plan-driven composition layer for the pinned LeRobot trainer: plan schema, ordered feature registry with install/restore and rollback, launch assembly (see `docs/m2-smolvla-training-harness-v2.md`) | the upstream training loop, learning semantics, or mutation of the frozen historical trainer stack |
| `scripts/run_smolvla_v2.py` | the single version-2 launcher validation chain, launch manifest and mode dispatch | bypassing prerequisite evidence or authorizing a furnace by itself |
| `scripts/train_smolvla_v2.py` | the single version-2 trainer entry installing plan-declared features on the pinned LeRobot trainer | experiment selection or evaluation semantics |
| `scripts/evaluate_aloha_geometry_teacher.py` | train-only rigid calibration, joint-limit-aware IK/path-planner boundary and staged create-only teacher reports | label collection or opening a later seed stage after failure |
| `reports/training/` | human and machine-readable interpretation | mutable checkpoints |
| ignored `runs/` and `artifacts/` | immutable local runtime evidence and deploy artifacts | tracked source code |

Important boundary: `src/rosetta_reality/train/losses.py` contains generic and
historical Rosetta action losses. Faust does **not** use those functions for its
SmolVLA flow-matching objective. The active Faust loss, trainer, AdamW builder
and scheduler come from the pinned LeRobot source. Changing the generic loss
file alone cannot fix finding T1.

Trainer composition boundary: future SmolVLA training plans use the version-2
harness (`src/rosetta_reality/vla/training/` plus `scripts/run_smolvla_v2.py`
and `scripts/train_smolvla_v2.py`), where an ordered, hash-bound `features`
list is the only way local extensions enter the pinned trainer. The historical
`train_smolvla_*` / `run_smolvla_*` stack is frozen as provenance for the
completed Faust, Aster and Way identities and must not be extended or edited.
The v2 rewrite changes no learning semantics and does not authorize a new
furnace by itself; see `docs/m2-smolvla-training-harness-v2.md`.

Do not edit a dependency cache in place. A trainer/loss/scheduler experiment
must be implemented as an explicit local extension or controlled injection,
covered by tests, checksum-bound to a new plan and compared against the frozen
Faust control.

## 5. Registered configuration chain

The current repaired chain is:

```text
configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml
        |
        +-- action-space and repaired processor contract
        |
        v
configs/vla/smolvla_450m_aloha_insertion_faust_batch8_002.yaml
        |
        +-- immutable formal optimizer/scheduler/resource plan
        |
        v
Faust checkpoints -> validation selection -> exported deploy artifact
        |
        v
configs/vla/smolvla_450m_aloha_insertion_faust_sim_001.yaml
        |
        v
Gate 3 -> Gate 4

Faust read-only control
        |
        +-- Aster `-002`: completed exploratory run; invalid single-axis reduction
        |
        v
Aster `-003`: first-action mask + mean over selected non-padding entries
        |
        v
preflight -> two-step optimizer smoke -> formal train -> validation -> sync
        -> Faust-control selection -> export -> Gate 3 -> Gate 4

Aster `-003` Gate 4 failure -> first-deviation and inference diagnostics
        |
        v
Way XPU state-jitter smoke `-002`
        |
        v
AutoDL batch-128 smoke -> structured pre-step CUDA OOM
        |
        v
isolated batch-64 fallback -> failed reduce-overhead CUDA Graph recapture
        |
        v
fresh batch-64/default smoke `-002` -> accepted two-step reload
        |
        v
fresh-base formal `-002` -> clean validation -> public sync -> selection
        -> exact export/reload -> Gate 3 pass -> Gate 4 fail `0/5`
        |
        v
Zen two-arm v2 campaign (`zen_cuda_b64_{uniform_002,firstaction_001}`)
        |
        v
doctor -> benchmark -> preflight -> smoke -> formal (batch 64, 316 steps)
        -> validation -> selection -> exact export/reload -> Gate 3 pass
        -> Gate 4 fail `0/5` (both arms) -> first-deviation trace
        preregistration 2026-08-28 (pending artifact transfer)
```

The formal Faust config and completed evidence are historical identities. A new
loss, optimizer, batch, scheduler, adaptation, data or evaluation hypothesis
requires a new config and run identity. Never mutate Faust and resume under the
same name.

## 6. Training data and action architecture

### 6.1 Data identity

```mermaid
flowchart LR
    D["Pinned LeRobot dataset revision"] --> S["Episode-disjoint split"]
    S --> T["Train-only dataset view"]
    T --> P["Projection + representation adapter"]
    P --> N["Train-only normalization"]
    N --> M["SmolVLA forward"]
    M --> L["Flow-matching loss"]
    L --> O["AdamW + scheduler"]
    O --> C["Checkpoint + processor + optimizer state"]
```

- Train episodes alone create normalization statistics.
- Validation episodes select checkpoints.
- Hidden-test episodes `[31, 6, 1, 24, 5]` stay sealed until the registered
  protocol allows them. Faust did not load them.
- Episode identity and frame alignment are preserved; action chunks never cross
  episode boundaries.

### 6.2 Standard action to model space

The registered action space is 14-D absolute ALOHA joint position at 50 Hz.
Dimensions 6 and 13 are left and right grippers.

Training targets pass through this order:

```text
raw standard-ALOHA target
    -> Action Contract projection
    -> reject source values beyond registered tolerance
    -> pi-Aloha arm representation
    -> bounded-sine gripper representation
    -> train-only mean/std normalization
    -> SmolVLA model space
```

For a projected standard gripper target `g` in `[0, 1]`, the repaired internal
target is `asin(2g - 1)`. At inference, any internal gripper value `x` decodes
through `(sin(x) + 1) / 2`, guaranteeing a standard-space result in `[0, 1]`.

The model config keeps upstream `adapt_to_pi_aloha = false` because the Rosetta
processor boundary owns the conversion. Turning both paths on would double-
transform state/actions and violate the registered identity.

The sine decoder fixes physical output bounds but is periodic. Legal output is
therefore not proof that the internal gripper latent stayed inside training
support. The audit's T6 diagnostic remains required.

### 6.3 Active training objective

The pinned SmolVLA model predicts a 50-action chunk and computes elementwise
flow-matching MSE over valid horizon/action entries, followed by a uniform mean.
Faust used:

- `n_obs_steps = 1`;
- `chunk_size = 50`;
- `n_action_steps = 1` at execution;
- `freeze_vision_encoder = true`;
- `train_expert_only = true`;
- `train_state_proj = true`;
- disabled dataset image transforms.

This creates the confirmed T1 contract mismatch: training weights the full
chunk uniformly while the current controller executes only the first action
before observing again. A repair must change the active SmolVLA flow-loss path,
not merely an offline MAE calculation.

The first Aster implementation (`aster-b8-002`) zeroed steps 2-50 at the raw
loss boundary but let the upstream policy divide by every valid horizon entry.
That is not equivalent to a first-action mean, especially when episode-tail
padding changes the number of valid future steps. The corrected
`aster-b8-003` contract wraps both boundaries:

1. mask the unreduced `[batch, chunk, action]` loss to the first action;
2. divide by selected, non-padding entries rather than all valid horizon
   entries;
3. require the exact pinned upstream source SHA before installing either
   wrapper;
4. keep the Faust optimizer and scheduler contract unchanged.

Unit evidence must cover unpadded loss scale, uneven padding, per-sample
reduction, gradients, double-install rejection and upstream SHA drift.

### 6.4 Optimizer and checkpoint boundary

Faust's registered optimizer contract was AdamW with LR `1e-4`, betas
`(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-10`, global clip norm `10`,
125 warmup steps and cosine decay to `2.5e-6` over 2,500 updates.

The local formal runner validates and assembles this contract. The pinned
LeRobot trainer performs forward/backward, clipping, optimizer stepping and
scheduler stepping. A complete training checkpoint must preserve:

- model weights and policy config;
- preprocessor/postprocessor state and statistics;
- optimizer state;
- scheduler state;
- run/config identity;
- RNG and dataloader continuation identity when formal resume is implemented.

The fixed-overfit path tested resume memory handling. The formal Faust runner
still has only `preflight`, `smoke` and `train` modes, so exact formal batch-8
resume parity remains unproven.

## 7. Inference, export and closed-loop architecture

```mermaid
flowchart LR
    E["Gym-ALOHA observation"] --> CM["Camera/state mapping"]
    I["Fixed instruction"] --> PP["Saved preprocessor"]
    CM --> PP
    PP --> P["Reloaded SmolVLA policy"]
    N["Seeded Gaussian noise"] --> P
    P --> CH["50-action model chunk"]
    CH --> PO["Saved postprocessor + bounded decoder"]
    PO --> CP["Action Contract safety projection"]
    CP --> A["Gym-ALOHA adapter"]
    A --> W["MuJoCo step"]
    W --> E
```

Current simulation semantics:

- simulator camera `top` maps to policy camera
  `observation.images.camera1`;
- instruction is `Insert the peg into the socket.`;
- the runtime robot-state dimension (14-D ALOHA) is read from the artifact
  dataset features; the policy-config `observation.state` shape is an upstream
  base-model placeholder (6-D) that `make_policy` never rebuilds and must not
  be used as the simulator state contract;
- policy sampling uses seeded upstream standard-normal noise;
- only the first action of each predicted chunk is executed;
- the next observation is collected immediately after that action;
- unprojected decoder output is retained as a diagnostic;
- Action Contract projection remains the final safety boundary before the
  simulator adapter;
- adapter-side additional clipping, invalid actions, joint-limit violations and
  unexpected collisions are reported;
- collision classification exempts only same-arm internal finger contact and
  the explicit Gym-ALOHA insertion grasps (either right-arm finger with
  `red_peg`; either left-arm finger with `socket-1..4`). Every other
  robot-scene contact, including a
  gripper touching the table, wrong object or unrelated geometry, is
  unexpected.

Export is a semantic operation, not just weight copying. A deploy artifact must
contain the saved preprocessor/postprocessor and bounded action boundary, then
pass independent reload. Faust's selected artifact produced exact action
equality after reload.

## 8. Gate and evidence state machine

```mermaid
flowchart LR
    G1["Gate 1 scripted action"] --> G2["Gate 2 dataset replay"]
    G2 --> SM["Tiny optimizer smoke"]
    SM --> OF["Fixed-sample overfit + resume"]
    OF --> FT["Formal training"]
    FT --> V["Validation-only selection"]
    V --> X["Export + independent reload"]
    X --> G3["Gate 3 short closed loop"]
    G3 --> G4["Gate 4 development task"]
    G4 -->|"pass"| A["M2 candidate"]
    G4 -->|"fail"| D["Diagnose; new single-axis plan"]
```

Current state:

| Stage | Result | Meaning |
|---|---|---|
| Gate 1 scripted action | passed | actuator ordering, direction and limits are usable |
| Gate 2 dataset replay | passed | expert action semantics reach the task in the simulator |
| bounded processor diagnostics | passed | repaired representation round-trips and bounds hold |
| smoke / fixed overfit / resume | passed | optimizer path can learn the bounded fixed sample set |
| Faust formal training | completed | finite batch-8 run with four checkpoints |
| validation selection | passed | step 1875 selected without hidden-test access |
| export/reload | passed | deploy artifact reproduces the action exactly |
| Gate 3 | passed | short observation-action-observation loop is safe |
| Gate 4 | **failed** | zero successful tasks in five 500-step rollouts |
| Aster `-002` validation | exploratory only | first-action MAE improved, but loss normalization violated the single-axis claim |
| Aster `-003` training | passed | 2500 finite steps and all four checkpoints completed |
| Aster `-003` selection | passed | step 2500 improved first-action MAE over the registered Faust control |
| Aster `-003` export/reload | passed | 13 manifest files verified and deterministic action equality is exact |
| Aster `-003` Gate 3 | passed | 20-step reloaded closed loop was finite, within the Action Contract and collision-free |
| Aster `-003` Gate 4 | **failed** | `0/5` success and reward `0` in every 500-step rollout; 8 joint-limit violations; the historical 45-contact field is rejected below |
| Aster first-deviation trace | diagnostic completed | at seed 10, policy/expert state MAE starts at step 0 and exceeds `0.025` by step 4 |
| inference-only remedies | diagnostic only | zero noise and four-sample ensemble still reached reward 0; temporal aggregation reached reward 3 but not success |
| Way state-jitter smoke `-002` | passed | exact plan-bound two-step optimizer/checkpoint smoke; no validation or closed-loop claim |
| Way batch-128 CUDA smoke | structured fallback trigger | CUDA OOM occurred before step 1; no optimizer/checkpoint was reused |
| Way batch-64 reduce-overhead CUDA smoke | failed evidence | step 1 was finite, then CUDA Graph recapture failed; its optimizer/checkpoint remains isolated |
| Way batch-64/default CUDA smoke `-002` | passed | two finite steps, complete checkpoints, independent reload and acceptance passed |
| Way batch-64/default formal `-002` | passed | 316 finite steps, 20,224 registered exposures and four checkpoints completed from the fresh base |
| Way validation/selection | passed offline gate | step 316 first-action MAE `0.030136355795964066`; worse than the Aster control and no hidden-test access |
| Way export/reload | passed | selected model and 14-file deploy artifact passed exact independent reload |
| Way Gate 3 | passed | 20-step reloaded closed loop was finite and inside all registered safety limits |
| Way Gate 4 | **failed** | seeds 1000--1004 all returned reward `0` and success `false`; safety criteria passed, task-success criterion failed |
| Zen training/validation/selection/export | passed | both arms converged (batch 64, 316 steps), selected step 316, exact independent reload (audit 2026-08-27) |
| Zen Gate 3 | passed | both arms through the rendered per-arm plans (suffixes `411` uniform, `422` firstaction) under final workspace `r42b` |
| Zen Gate 4 | **failed** | both arms `0/5` with reward `0` on every seed 1000--1004; firstaction recorded zero violations of every safety class; uniform additionally failed `joint_limits_respected` (5 violations) — the audit markdown's "only failed criterion" sentence is a prose slip, the gate JSONs are authoritative |
| Zen first-deviation trace | completed 2026-08-28 | local XPU diagnostic on the firstaction deploy artifact: divergence begins at step zero (action MAE `0.032943`, post-state MAE `0.015197`; Aster `0.0204168`/`0.0055942`), crossings `0.005`/`0.01`/`0.025` at steps 0/0/1, `0.05` at 18, `0.1` at 29; expert replay reproduced reward 4 at step 293; policy reward 0 with zero violations; report `m2-smolvla-zen-first-deviation-trace-2026-08-28` |
| recovery-oracle exact control | diagnostic passed | train episode 2/seed 10 reproduced reward 4 in 294 actions with no OOD or IK failure |
| recovery-oracle cross-pose tuning | **failed** | two registered robot-state progression thresholds both returned reward 0 on dedicated seed 1900; development/collection/Gate seeds remained unopened |
| object-geometry teacher exact | **plan 030 failed `0/1`; joint-limit safety held; later gates sealed** | calibration reached reward 4, but exact exhausted 500 steps in `orient` with reward 0; 131 planner attempts and 23 recovery events produced zero IK, clip or joint-margin failures |

Gate 4 failure returns the workflow to diagnosis. It does not authorize a larger
model, more seeds, more epochs or a new optimizer by itself.

### Aster `-003` Gate 4 diagnosis

The primary failure classification is **training / closed-loop generalization**,
not a broken simulator, reward function or Action Contract adapter:

- the registered Gate 2 expert replay completed episode 2 at seed 10 in 294
  steps, reached reward 4 and terminated with task success in the same
  Gym-ALOHA insertion task;
- Gym-ALOHA's `sample_insertion_pose(seed)` draws seed 10 and Gate 4 seeds
  1000--1004 from the same fixed peg/socket pose ranges. Faust also reached
  reward 2 on the exact Gate seed 1003 under the same pinned simulator image,
  proving that the Gate distribution can trigger intermediate task rewards;
- all Aster outputs and executed actions were finite and inside the Action
  Contract, with no adapter-side clipping. The failures are therefore not an
  action-shape, NaN or output-boundary crash;
- the original 45-contact aggregate used a one-sided task-contact whitelist and
  is not valid collision evidence: expert replay itself contacts both finger
  geoms while grasping the socket and peg. The corrected classifier permits
  both fingers of the task-assigned gripper but still rejects table, wrong-arm,
  wrong-object and unrelated-scene contacts. A create-only reclassification of
  the five immutable episode histograms maps all 45 contacts to permitted task
  grasps and records 0 corrected unexpected contacts in
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/aster-gate4-003-collision-reclassification-001.json`.
  The 8 recorded joint-limit violations remain valid adapter diagnostics;
- Aster improved the selected teacher-forced first-action MAE by about 18% over
  Faust, but regressed from Faust's maximum closed-loop reward 2 to reward 0 on
  every seed. The first-action-only objective therefore improved its offline
  target without supplying state-conditioned recovery supervision.
- Aster's own read-only 20-context modality diagnostic confirms the shortcut:
  normal first-action MAE was `0.02288`, image shuffle raised it only to
  `0.02565`, while state shuffle raised it to `0.10819`. The first-action
  prediction shift from state shuffle (`0.10398`) was about 8.4 times the image
  shuffle shift (`0.01241`). This is teacher-forced, non-gating evidence, but it
  directly establishes that the selected Aster checkpoint is state-dominant.

This classification has one explicit evidence limit: there is no
state-conditioned expert oracle for each exact Gate seed, so simulator
reachability has not been independently replayed at all five poses. That gap is
not enough to explain the policy-specific wrong-side contacts or the successful
same-task expert replay, but it prevents claiming that every individual reset
has been oracle-proven. The current Gate reports also lack object/EEF and
first-deviation traces; that is an observability gap, not evidence that the
simulator caused the failure.

#### First-deviation trace and attempted remedies

The bounded seed-10 trace closes the earlier observability gap without treating
time-indexed expert actions as a recovery oracle. It executes the registered
expert replay and the independently reloaded Aster policy from the same reset
and records actions, post-step state, reward, joints, object poses and contacts:

- the expert reaches reward 4 and terminates at step 293; Aster reaches only
  reward 1 and never terminates; the trace's step-134 collision flag used the
  rejected one-sided task-contact whitelist and is not retained as evidence;
- step-zero action MAE is `0.0204168`, and the very first post-step state already
  differs by `0.0055942`; state MAE crosses `0.01` at step 1, `0.025` at step 4,
  `0.05` at step 24 and `0.1` at step 28;
- maximum state MAE is `0.222958`, final state MAE is `0.097614`, and mean
  time-indexed action MAE after deviation is `0.119910`;
- the trace is stored as
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/aster-trajectory-520c8ec87c1618fc.json`.

The divergence therefore begins before any simulator anomaly or limit
violation and then compounds under closed-loop execution. Time-indexed expert
actions after that point describe the expert trajectory, not the correct action
for Aster's deviated state, so they must not be relabeled as recovery targets.

Three inference-only families were tested as non-gating diagnostics. Removing
diffusion noise and averaging four Gaussian samples both remained at reward 0.
Exponential temporal aggregation, including the original ACT ordering with
decay `0.01`, improved seed 10 to reward 3 but still produced no insertion in
500 steps. Its historical contact aggregate also used the rejected one-sided
whitelist and is not safety evidence. Smoothing and sampling variance are
therefore not sufficient repairs, although reward 3 shows that aggregation
changes the closed-loop trajectory materially.

Way tested one bounded state-robustness axis: Gaussian jitter with standard
deviation `0.05` was applied only to train-normalized `observation.state`;
validation/deployment state and the absolute expert action label remained clean.
The local XPU smoke `-002` first established the implementation contract.

On AutoDL, batch 128 triggered the registered fallback with a structured CUDA
OOM before step 1. The first isolated batch-64 attempt completed one finite step
but then failed `compile_mode=reduce-overhead` CUDA Graph recapture; its
checkpoint and optimizer state were preserved as failed evidence and never
reused. A separate fresh-base batch-64/default smoke `-002` passed two steps,
complete checkpoint checks and independent reload. Its accepted plan SHA-256 is
`b8a23372ae0f4006d773cf9b35db03722e7b871f7dee2179520ff050e3f96508`.

The preregistered fresh-base formal run then completed 316 steps over 20,224
registered exposures with four checkpoints. Step 316 was selected on clean
validation, publicly synchronized through the sanitized Trackio bridge, and
exported with exact independent reload. Gate 3 passed. Gate 4 failed `0/5` with
reward `0` for all five fixed seeds while every finite/action-limit/joint-limit/
corrected-collision criterion passed. This result rejects the tested Way
augmentation as a sufficient repair. It does not establish recovery learning:
unchanged time-indexed labels under perturbed state are still not a
state-conditioned recovery oracle.

The Zen campaign (completion audit
`reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md`) then executed the
first preregistered two-arm single-axis comparison through the version-2
harness on the AutoDL RTX 4090D container: arm A uniform control
`m2-smolvla450m-zen-uniform-002` vs arm B `first_action_only` treatment
`m2-smolvla450m-zen-firstaction-001`, sharing batch 64, 316 steps (20,224
exposures), the fresh pinned base, identical optimizer/scheduler and seed
20260809. Both arms converged (uniform loss `0.905 -> 0.156`, firstaction
`1.032 -> 0.150`, peak CUDA 18.29 GB), selected step 316, exported with
bit-exact independent reload, and passed Gate 3. Gate 4 failed `0/5` with
reward `0` for both arms (reports `411`/`422`). Uniform additionally recorded
five joint-limit violations; firstaction recorded zero violations of every
safety class — the cleanest closed-loop run of the campaign while still
scoring zero reward. Offline selection favored the control arm (uniform
first-action MAE `0.021572770214905695` vs treatment `0.022150604739519103`,
both ~92% over base `0.290538`), so the registered temporal-weighting
hypothesis is rejected at this development scale: the Aster-era offline gain
(batch 8 / 2,500 updates) does not generalize across training regimes, and
offline MAE remains decoupled from closed-loop success. Note one prose slip
in the audit markdown (its section 6 calls `minimum_task_success_rate` the
only failed criterion for both arms): the immutable `411` report also fails
`joint_limits_respected`; the gate JSON files remain the authority. Zen
checkpoints (8 quarter checkpoints) remain on the AutoDL system disk
`/root/zen-runtime/checkpoints` — the instance was shut down after the audit
and must not be released; the durable data disk reached its 50G budget during
execution and was recovered by removing this session's own unreferenced
duplicates. The FFmpeg 5/6/7 runtime install could not activate torchcodec
(GLIBCXX/ABI negatives, recorded in audit section 8) and is deferred to a
coordinated torch+torchcodec upgrade; training itself never used torchcodec.

#### Recovery-oracle boundary

The first state-conditioned recovery implementation now exists at
`src/rosetta_reality/sim/recovery_oracle.py`, with its diagnostic runner at
`scripts/evaluate_aloha_recovery_oracle.py` and a create-only future-data
contract at `src/rosetta_reality/data/recovery_manifest.py`. The core oracle has
no time-step or timestamp input: it selects a monotonic successful reference
from the current robot state and unlocks post-contact motion only from observed
task reward. It fails closed outside the reference neighborhood.

This boundary is **not yet a recovery-data source**. Eleven train-only source
episode/seed pairs replayed to reward 4, and the exact episode-2/seed-10 control
closed successfully in 294 actions without OOD or IK failures. But sparse local
IK retargeting plus robot-state retrieval failed on the separately reserved
tuning seed 1900. Plan `001` with progress distance `0.01` stalled at reference
index 20; plan `002`, which changed only that distance to `0.02`, stalled at
index 25. Both completed 500 steps at reward 0 with no OOD or IK failure.

The result rejects robot-state proximity plus translated reference paths as a
cross-pose oracle. Development seeds 2000--2004, collection seeds 3000--3004,
policy Gate seeds 1000--1004, validation episodes and hidden-test episodes were
not opened. No recovery label or manifest was written. Full identities and
report checksums live in
`reports/training/m2-smolvla-recovery-oracle-audit-2026-08-14.md` and its JSON
companion. A future oracle must condition on object/task geometry and pass an
independent teacher gate before it can authorize DAgger labels or a new furnace.

That object/task-geometry boundary now exists in
`src/rosetta_reality/sim/geometry_teacher.py`, with staged evaluation in
`scripts/evaluate_aloha_geometry_teacher.py`. Plans `003`--`009` repaired
orientation bounding, post-IK verification, redundant joint-delta limiting,
joint-limit active-set solving, phase-driven translation/orientation, and a
duplicate numerical-convergence gate. The latest exact run nevertheless fails
at step 98 before the action is executed: fixing `right_wrist_rotate` at its
registered limit leaves `0.0119651` projected error against the unchanged
`0.003` threshold. Tuning seed 1900, development/collection/policy-Gate seeds,
validation episodes and hidden-test episodes were not opened, and no label was
written. The current authority is
`reports/training/m2-smolvla-geometry-teacher-audit-2026-08-14.md` and its JSON
companion for plans `003`--`009`.

Plan `010` added a joint-limit-aware position-priority/orientation-relaxation
path planner without changing the Action Contract, 3 mm projected-pose gate or
0.04-rad orientation step. Its remote exact run executed 27 precise planner
waypoints, moved the failure from approach step 98 to orient step 125, and then
failed when both wrist-rotate joints were limited (`0.00696018` projected
error). Its create-only report SHA-256 is
`64044579a3cb66613748172b77f577b59ed0bf080616a5b0ce4611cd794c57ce`.
Tuning and all later seed groups remained sealed and no label was written.

Plans `011`--`014` are additional immutable local exact negative evidence.
Plan `011` extended the planner through orientation and accepted 57 waypoints,
but its per-step position reference allowed cumulative drift before failure at
orient step 155 (`0.00688244` maximum projected error). Plan `012` froze the
first orient target as a position anchor and constrained every waypoint to the
existing 12 mm approach tolerance; it accepted 31 waypoints, then failed at
orient step 129 (`0.00803324`). Plans `013` and `014` supplied that anchor
inside active-set pose IK with rotation weights `0.2` and `1.0`; both rejected
the first orient waypoint at step 125 (`0.00696018`). These runs did not change
the 3 mm projected-pose gate, the 12 mm anchor/per-step bounds or the Action
Contract. Current Ruff and 15 focused geometry/Mink/adapter/protocol container
tests pass. Tuning and all later seed groups remain sealed and no label has been
written. A next plan needs a new constrained-solver hypothesis rather than
another unprincipled weight scan.

Plans `015`--`021` implement and audit that architectural replacement. The
event-driven teacher, Action Contract and staged evaluator remain Rosetta-owned,
while dual-arm pose IK is delegated to upstream Mink `1.2.0` with QPsolvers
`4.13.0` and DAQP `0.8.7`. The active path follows Mink's published ALOHA
structure: two full-pose `FrameTask`s, a low-cost neutral-pose `PostureTask`,
native MuJoCo `ConfigurationLimit` and `VelocityLimit` inequalities, and a DAQP
solve. Mink's `DofFreezingTask` equality constraint freezes non-arm fingers and
free-object DOFs. MuJoCo remains pinned to `3.8.1`. The adapter only reconciles
Action Contract ingress with the slightly narrower native MJCF limits; returned
actions still pass through the unchanged Action Contract.

The legacy active-set/waypoint code remains solely to reproduce immutable plans
`003`--`014`; it is not on the `mink_qp` execution path and must not be selected
for a new plan. Plan `015` failed its five-iteration upstream baseline, plan
`016` measured ten-iteration convergence, plan `017` aborted before report
creation on strict JSON serialization, and plan `018` exposed a non-arm finger
configuration mismatch. Plan `019` used upstream DOF-freezing constraints and
reached step 98; plan `020` reconciled the exact `-pi` Action Contract bound
with the MJCF `-3.14158` bound; plan `021` aligned the posture target lifecycle
with Mink's official ALOHA example. Plans `020` and `021` still failed at
approach step 98 with approximately `0.0119` projected pose error and no adapter
clip. Therefore the solver architecture has been replaced, but train-only exact
has not passed. The current authority is
`reports/training/m2-smolvla-mink-ik-audit-2026-08-15.md` and its JSON companion.

Plan `022` preregistered the next single axis without extending the historical
custom waypoint branch. Mink/DAQP remains the first constrained local solver.
Only after it fails during `approach` or `orient`, a process-isolated adapter
loads ROS Humble, MoveIt `2.5.9`, OMPL `1.7.0`, the official
`ompl_interface/OMPLPlanner`, `geometric::RRTConnect` and the official LMA
kinematics plugin. Rosetta composes two copies of the pinned Interbotix VX300S
description, maps Gym calibration-site frames, validates joint order, hashes,
bounds and collision state, and executes at most one Action-Contract-bounded
waypoint from the official path. It does not implement graph search, sampling,
IK or orientation relaxation.

Local non-training evidence includes five-sample Gym/MoveIt FK parity with
maximum position disagreement `3.188872858294072e-16` m and zero orientation
disagreement, a real two-waypoint RRTConnect smoke whose maximum weighted goal
error was `0.0001342278689703837`, Ruff, and 29 focused remote Linux tests.
The exact seed remains isolated from every later seed group. The
original preregistration authority is
`reports/training/m2-smolvla-moveit-path-planner-preregistration-2026-08-15.md`
and its JSON companion.

Athena plan `022` then reproduced the train-only calibration at reward `4` in
294 steps, but exact failed at approach step 98 before MoveIt generated a path.
The official model and Action Contract command bound for
`right_wrist_rotate` is `3.14158`, whereas the Gym/MJCF observed joint state may
reach mathematical pi, leaving only about `1.27e-5` rad of representation
mismatch; the old sidecar returned `start_state_out_of_bounds`.
The same run also exposed that the host installation lacked the official
Interbotix ament package and meshes. MoveIt logged no collision geometry, while
the old identity protocol still returned `ok`. The plan `022` report remains
immutable, but it is infrastructure-incomplete and is neither planner
acceptance nor evidence that collision-checked RRTConnect failed.

Plan `023` changes only this official-runtime boundary. It installs the pinned
Interbotix commit and validates every composed `package://` resource, requires
the sidecar to report positive collision-link and collision-shape counts, and
fails at startup when no collision geometry is loaded. At request ingress it
uses MoveIt's own `RobotState::enforceBounds` only when an arm-joint violation
is at most `0.00002` rad, reports every reconciled joint/delta, and rejects any
larger violation. The bounded-waypoint sampler also measures the command from
the original observed state, shortening the first segment when necessary so
the representation reconciliation never consumes extra Action Contract motion
budget. OMPL, RRTConnect, LMA, the Action Contract, teacher, Mink, 1 mm / 3 mm
pose gates, seeds and label boundaries are unchanged. Its authority is
`reports/training/m2-smolvla-moveit-runtime-boundary-preregistration-2026-08-15.md`
and its JSON companion; train-only exact must pass before tuning can even be
reviewed.

Athena plan `023` passed that runtime and protocol boundary, loaded 22 links
with 22 collision shapes, and reproduced reward-4 calibration in 294 steps.
Its exact evaluation still failed `0/1` at approach step 98. This time the
failure is valid algorithm/control evidence: the live `right_wrist_rotate`
observation was `0.005985603256225769` rad above the registered upper bound,
far beyond the `0.00002`-rad representation-only reconciliation allowance, so
the official sidecar correctly returned `start_state_out_of_bounds` without
planning. The exact report SHA-256 is
`ec0ed7a2e910eabbc9fefc3b9369a990b82d4ab37c588055f085a53e73af1234`.
Do not widen the reconciliation allowance: the next hypothesis must keep the
closed-loop state inside a registered joint-limit margin or otherwise change
the joint-limit-aware feedback itself. Tuning, development, collection,
policy-Gate, validation, hidden and recovery-label gates remain sealed.

Plan `024` applied the next single axis: the solver-local joint ranges used by
upstream Mink `ConfigurationLimit` were inset by `0.01` rad for the twelve arm
joints before any command was generated. Exact removed the arm start-bound
failure but stopped on the separate Gym-open `0.058` m versus official finger
upper `0.057` m representation mismatch. Plan `025` added only a bounded
`0.001`-m finger adapter; it then executed 21 official RRTConnect waypoints
before the live `right_forearm_roll` state exceeded its physical bound by
`0.0005873297882081907` rad at step 168. Both results are immutable negative
evidence.

Plan `026` preregisters the measured next axis. It carries twelve native
`moveit_msgs/JointConstraint` entries in MoveIt's `MotionPlanRequest` with the
same `0.01`-rad margin, applies them to LMA validity and the official OMPL
request, and independently rejects any start, goal, returned trajectory
waypoint or bounded execution waypoint outside that margin. The new image
compiled, Ruff and 37 focused tests passed, five-sample Gym/MoveIt FK parity
remained exact within `3.188872858294072e-16` m, an eight-waypoint direct smoke
maintained at least `0.25577550429170226` rad of physical path margin, and a
`0.00658000000000003`-rad-margin start was rejected. The authority is
`reports/training/m2-smolvla-moveit-joint-path-margin-preregistration-2026-08-15.md`
and its JSON companion.

Athena then executed plan `026` train-only exact. Calibration again reached
reward `4` in 294 steps, but exact failed `0/1` at approach step 113 after
three bounded RRTConnect waypoints. Successful official paths retained at
least `0.030544891433715637` rad of physical margin and bounded next waypoints
retained at least `0.25811082074868175` rad. Before the fourth fallback, the
later observed start state retained only `0.00961627014160138` rad, a
`0.0003837298583986206`-rad shortfall from the registered margin, so the
sidecar correctly returned `start_state_outside_joint_path_margin` before
IK/OMPL. This proves planned-path constraints are active and moves the next
diagnosis to commanded-versus-observed margin evolution and execution/control
feedback. The authority is
`reports/training/m2-smolvla-athena-plan026-exact-audit-2026-08-15.md` and its
JSON companion. Later seeds and labels remain unauthorized; neither pose gates
nor the joint margin may be relaxed.

Plan `027` preregistered the evidence-only boundary needed to resolve that
ambiguity. On every executed exact-control step it records the twelve arm
joints in the pre-step observation, absolute command and next observation,
plus per-joint lower/upper margins, tracking error, same-direction overshoot
past the command and command-to-observation margin loss. The diagnostic is
explicitly forbidden from affecting action selection; the teacher, Mink,
MoveIt/LMA/RRTConnect, controller adapter, `0.001` / `0.003` pose gates,
physical `0.01`-rad margin and all seeds are unchanged. Its authority is
`reports/training/m2-smolvla-execution-margin-diagnostic-preregistration-2026-08-15.md`
and its JSON companion. In the existing Linux image with networking disabled
and a read-only repository mount, Ruff and all 39 focused
geometry/Mink/MoveIt/Gym tests passed.

Athena then ran plan `027` train-only exact. Calibration again reached reward
`4` in 294 steps, but exact failed `0/1` at approach step 140. There were zero
commanded-margin breaches: the minimum command margin was
`0.013024463729858216` rad. At step 139, however, the safe
`left_wrist_rotate` command retained `0.015719308929443184` rad and the next
observed state retained only `0.007716312484741028` rad after a measured
`0.008002996444702148`-rad overshoot toward the upper bound. The next MoveIt
request correctly returned `start_state_outside_joint_path_margin`. This
locates the next repair axis at measured execution reserve or closed-loop
feedback; no later seed or label gate is open, and neither the physical margin
nor pose gates may be relaxed. The authority is
`reports/training/m2-smolvla-athena-plan027-exact-audit-2026-08-15.md` and its
JSON companion.

Plan `028` preregisters the resulting standard robust constraint-tightening
axis. The maximum same-direction overshoot across every arm joint and executed
Plan `027` step was `0.03540462255477905` rad. Added to the unchanged physical
`0.01`-rad margin, it yields a `0.04540462255477905`-rad command margin applied
uniformly through upstream Mink `ConfigurationLimit` and official MoveIt
`JointConstraint`; no new IK or path planner is introduced. Commanded and
observed physical-margin breaches are both fail-closed acceptance criteria.
The bound is train-only evidence, not a development guarantee, so only exact
episode 2 / seed 10 may run next. The authority is
`reports/training/m2-smolvla-execution-reserve-preregistration-2026-08-15.md`
and its JSON companion.

Athena then ran plan `028` train-only exact. Calibration again reached reward
`4` in 294 steps, while exact failed `0/1` in `approach` at step 106. The
tightened bound did its safety job: commanded and observed physical-margin
breaches were both zero, the minimum command margin was
`0.04555977828979474` rad and the minimum observed margin was
`0.039528980331420716` rad. The latter remained outside the physical
`0.01`-rad band but fell short of the `0.04540462255477905`-rad tightened set,
so the next official MoveIt request correctly returned
`start_state_outside_joint_path_margin`. This proves a recursive-feasibility
gap rather than grounds for widening the start tolerance. Any next plan must
add a fail-closed retreat / backup feedback phase that returns physically safe
observations to the tightened set before pose tracking or path planning. The
authority is
`reports/training/m2-smolvla-athena-plan028-exact-audit-2026-08-15.md` and its
JSON companion. Tuning, development, collection, policy-Gate, validation,
hidden and label gates remain sealed.

Plan `029` preregisters that backup path without introducing a custom motion
planner. The sidecar now loads MoveIt 2 Humble's official
`default_planner_request_adapters/FixStartStatePathConstraints` through
`planning_pipeline::PlanningPipeline`, while retaining official LMA IK, OMPL
`RRTConnect`, the physical `0.01`-rad margin and the tightened
`0.04540462255477905`-rad `JointConstraint`. The official adapter may act only
when the start remains physically safe but violates the tightened path
constraint. Its adapter-added prefix indices must be a contiguous prefix; every
prefix waypoint is collision-free, physically safe and monotonic for each
violating joint margin; all non-prefix waypoints satisfy the tightened
constraint; and the bounded execution waypoint must make positive margin
progress. A start inside the physical margin still fails closed.

In the network-disabled official MoveIt image, the registered plugin loaded and
the plan `029` C++ sidecar passed normal, recovery and physical-negative local
requests. On Athena, however, its remote static attempt failed before exact:
MoveIt's official adapter emitted a `37`-waypoint prefix, and Rosetta rejected a
`right_wrist_rotate` margin change from about `3.13601` to `3.00571` rad at
prefix waypoint `25` as non-monotonic even though both states were far inside
the `0.04540462255477905`-rad tightened set. The same attempt had already passed
`52` tests and exact FK parity. No exact episode, later seed or label gate was
opened. The durable attempt identity is `athena-plan029-static-001`, with its
direct-smoke and execution-log hashes bound by plan `030`.

Plan `030` changes only that Rosetta-side validator. A violating joint must
improve monotonically while it remains below the tightened command margin; once
it first enters the tightened set, subsequent official-prefix states may move
non-monotonically but may not re-enter the violating region. Collision checks,
the physical `0.01`-rad prefix floor, positive bounded first-step progress and
the tightened non-prefix suffix remain mandatory. Local image
`humble-2.5.9-start-path-constraints-002` then accepted the same recovery with
`37` prefix waypoints, a minimum prefix margin of `0.01998734641020672` rad,
minimum constrained-suffix margin `0.051881707845083724` rad and positive
first-step progress `0.1108319211228852` rad. Five-sample MoveIt/Gym FK parity
again retained maximum position error `3.188872858294072e-16` m and zero
orientation error, and `54` focused tests passed.

Athena then reproduced calibration reward `4` in `294` steps and ran Plan `030`
train-only exact episode 2 / seed 10. Exact failed `0/1` after all `500` steps,
with reward `0`, final phase `orient`, 131 planner attempts and 23 official
start-state recovery events. The validator and safety contract held: teacher,
IK, adapter-clip, joint-projection, commanded-margin and observed-margin
failures were all zero; minimum commanded, constrained-path and observed
margins were `0.04540456779479962`, `0.045446080555739954` and
`0.01990832336425763` rad respectively. The failure boundary is now safe but
non-progressing orient feedback rather than joint-limit rejection. Its authority
is `reports/training/m2-smolvla-athena-plan030-exact-audit-2026-08-15.md` and
its JSON companion. Any next plan must diagnose that phase locally and
preregister one controlled axis. Tuning, development, collection, policy-Gate,
validation, hidden and label gates remain sealed.

Plan `031` diagnoses the execution boundary rather than inventing another
planner. The pinned sidecar already returned complete collision-checked OMPL
trajectories, but the evaluator discarded each path after its bounded first
command and replanned from a new redundant LMA IK solution on the next fallback
step. MoveIt 2.5.9 Hybrid Planning's official `SimpleSampler` instead retains a
reference trajectory, advances one waypoint when the current bimanual joint
state is within the upstream `0.2`-rad L1 tolerance, and otherwise continues to
forward that waypoint; `ForwardTrajectory` sends the sampled waypoint toward
the controller. Plan031 ports that state machine into the existing sidecar
client/evaluator boundary and replaces the reference only on teacher phase
change. It does not change RRTConnect, LMA, the official start-state adapter,
pose thresholds, Action Contract delta, collision resources, joint margins,
seed identity or label authority.

The network-disabled official image returned the requested full normal path in
8 waypoints with maximum waypoint joint delta `0.0993591379230041` rad and the
full recovery path in 41 waypoints with maximum delta
`0.11261942148813398` rad, both below the frozen
`0.23561944901923448`-rad command delta. Ruff and 57 focused Linux-container
tests passed. The authority is
`reports/training/m2-smolvla-moveit-hybrid-trajectory-execution-preregistration-2026-08-15.md`
and its JSON companion. These are local preregistration checks, not exact-pass
evidence; only episode 2 / seed 10 may run next after a new content-addressed
workspace and verified shutdown watchdog.

Athena's remote Plan031 static attempt then passed Ruff, 60 focused tests,
five-sample MoveIt/Gym FK parity and the actual pinned-sidecar trajectory client.
The exact attempt completed calibration but stopped before planner creation with
`KeyError: 'output'`; Plan031 had omitted its required output and stop-condition
tail, so it produced no exact report or algorithm evidence. Plan032 is a
schema-only repair: it restores that frozen tail and adds pre-calibration output
contract validation without changing planner, teacher, margins, pose gates,
seeds or labels. Its authority is
`reports/training/m2-smolvla-plan032-output-schema-repair-preregistration-2026-08-15.md`
and its JSON companion.

Plan032 remote static passed Ruff, 61 focused tests, FK parity and the actual
sidecar full-trajectory client. Its train-only exact then made one global plan,
reused that reference 402 times and advanced through all 15 later waypoints,
eliminating Plan030's 131-plan branch switching. It nevertheless exhausted 500
steps in `approach`: at the final trace, joint-space L1 distance to the last
reference remained `0.03446431288757733` rad and task-space position error
remained `0.02851971797645092` m. All teacher/IK/clip/margin/collision safety
counts remained zero. The authority is
`reports/training/m2-smolvla-athena-plan032-exact-audit-2026-08-15.md` and its
JSON companion. This moves the next local single-axis diagnosis to
final-waypoint/controller feedback; no threshold or later gate is open.

Plan `033` preregisters that controller boundary without changing the global
planner or protected Gym adapter. At the final retained waypoint and only
after the existing `0.2`-rad `SimpleSampler` L1 handoff condition, it evaluates
the official MuJoCo static force balance. For each direct one-DOF arm actuator,
the affine force law is inverted so the position reference supplies
`qfrc_bias - qfrc_passive`; unsupported transmission, gain, bias, activation,
force-range or control-range identities fail closed. The correction is capped
at `0.05` rad, must remain inside the existing
`0.04540462255477905`-rad command margin, is latched for that teacher phase and
cannot trigger a new global plan. This is one low-level execution axis, not a
new IK or search algorithm. The 1 mm / 3 mm IK gates, 12 mm approach gate,
Action Contract and every seed/label seal remain unchanged. Its authority is
`reports/training/m2-smolvla-mujoco-position-feedforward-preregistration-2026-08-15.md`
and its JSON companion. The Plan033 Ruff and 67-test focused suite passed in a
network-disabled Linux container with a read-only repository mount. The
official sidecar and five-sample FK parity then passed locally, but train-only
exact failed safely at step 127 after reaching waypoint 15 within the handoff
tolerance. The terminal command was not executed: MuJoCo 3.8.1 exposed the
official actuator moment map as CSR values plus `moment_rownnz`,
`moment_rowadr` and `moment_colind`, while Plan033 required a dense `nu x nv`
array. Its immutable authority is
`reports/training/m2-smolvla-athena-plan033-local-exact-audit-2026-08-15.md`
and its JSON companion.

Plan `034` repairs only that versioned storage adapter. It validates CSR row
bounds and unique columns, requires the same one nonzero direct-DOF moment, and
retains the dense unit boundary. It does not change the affine force balance,
terminal handoff, correction bound, command margin, planner, teacher, Action
Contract, seeds or labels. Ruff and 69 focused tests passed in the same
network-disabled, read-only Linux boundary; the official sidecar remained
22/22 and FK parity remained `3.188872858294072e-16` m / `0.0` rad. Its
authority is
`reports/training/m2-smolvla-mujoco-sparse-actuator-moment-repair-preregistration-2026-08-15.md`
and its JSON companion. Train-only exact has not run.

Plan034 did pass the original CSR failure, but then raised before it could
write `exact.json`: terminal feedforward passed Action Contract/MoveIt name
`left_waist` to Gym MuJoCo, whose registered joint name is
`vx300s_left/waist`. Its authority is
`reports/training/m2-smolvla-athena-plan034-local-exact-audit-2026-08-15.md`
and its JSON companion. Plan `035` changes only this namespace adapter, using
the evaluator's existing ordered `LEFT_JOINTS` plus `RIGHT_JOINTS` mapping for
the same 12 action dimensions. Ruff and 70 focused tests passed. Its authority
is
`reports/training/m2-smolvla-gym-joint-name-adapter-repair-preregistration-2026-08-15.md`
and its JSON companion. It is the third and final local exact attempt in the
current authorization boundary; no fourth attempt may be inferred. Plan035
then completed all 500 steps with one global plan, all 28 waypoint advances,
one terminal-control activation and 347 compensated commands. The maximum
correction was `0.0169158762352822` rad and the minimum command margin was
`0.84494201442145` rad; no IK, clip, margin or collision failure occurred.
Nevertheless, the final position error was `0.026163499802350998` m, still
above the unchanged `0.012` m approach gate. Static feedforward reduced the
Plan032 residual by only `0.00235621817409992` m (`8.26%`). Its authority is
`reports/training/m2-smolvla-athena-plan035-local-exact-audit-2026-08-15.md`
and its JSON companion. Exact remains failed and execution stops at this
three-attempt boundary.

The run also exposed three engineering defects that are distinct from the Gate
4 research failure: new formal plans store normalization identity under
`prerequisites`, the historical evaluator assumes tokenizer files live inside
every policy directory even when the base policy uses its pinned VLM dependency,
and an unset runtime-root environment variable can otherwise resolve to the
current working directory. Completed Way compatibility wrappers remain frozen
for provenance. Future plans use the shared version-2 compatibility boundary,
which reconciles only identical normalization views, validates every tokenizer
file against its immutable manifest, routes ignored `runs/` evidence through
the durable run root and fails closed on missing roots. It also requires an
accepted dedicated CUDA Graph smoke before `compile_mode=reduce-overhead` may
authorize optimizer work.

## 9. Current repair routing map

The detailed evidence and acceptance criteria live in the audit. Use this table
to find the owning layer before editing:

| Finding | Owning layer / entry point | First required evidence |
|---|---|---|
| T1 executed-horizon loss mismatch | `horizon_loss.py` plus `run/train_smolvla_horizon_loss_formal.py`; the Zen two-arm campaign rejected `first_action_only` at batch 64 / 316 updates (no offline gain, no closed-loop change), and the Aster batch-8 offline gain does not transfer across regimes — the axis is closed unless a longer-schedule / smaller-batch replication is separately preregistered | selected-valid reduction tests, exact upstream SHA, preflight and two-step optimizer smoke |
| T2 no recovery distribution | `geometry_teacher.py`, upstream Mink adapter, official MoveIt/OMPL sidecar, MuJoCo position-feedforward boundary, staged evaluator and recovery-data contract | preserve plans `022`--`054`; Plan053 is safe but horizon-exhausted and Plan054 failed grasp drift before its new event locally and on Athena; no further planner plan is authorized, and later seeds and labels remain sealed |
| T3 validation noise mismatch | `scripts/evaluate_smolvla_action_repair_validation.py` and new evaluation config | fixed Gaussian seed ensemble matching deployment |
| T4 state-dominant shortcut | `scripts/diagnose_smolvla_aster_modalities.py` plus new single-axis configs | per-module gradients and controlled image/history ablations |
| T6 periodic gripper latent | `src/rosetta_reality/vla/processor.py` | internal-support rate plus endpoint and standard-bound tests |
| T7 formal resume gap | formal runner/trainer and `checkpoint_memory.py` | uninterrupted versus stop/resume parity |
| T9 checkpoint/log mismatch | formal plan validation and checkpoint writer | exact same-step metric/resource snapshot |
| T10 missing failure trace | `scripts/smolvla_action_repair_sim_gate.py` and `src/rosetta_reality/eval/diagnostics.py` | object/EEF/reward/raw-action first-deviation window |
| O2 clipping observability | local instrumentation around pinned LeRobot trainer | pre/post-clip global and per-module norms |
| O3 scheduler peak semantics | local scheduler contract/extension, not dependency-cache edits | exact LR sequence around warmup and final step |

Do not combine these repairs into one furnace. Instrumentation and resume parity
may share a no-training implementation change, but each learning hypothesis
needs its own registered comparison.

## 10. Next safe work sequence

The current next sequence after the Zen two-arm Gate 4 failure is:

1. completed 2026-08-28: both selected Zen deploy artifacts transferred to the
   local artifact root (SHA256-verified; the AutoDL instance was shut down
   again under the registered procedure and must not be released) and the
   preregistered Zen first-deviation trace executed locally
   (`reports/training/m2-smolvla-zen-first-deviation-trace-2026-08-28.md`) —
   divergence again begins at step zero, earlier than Aster;
2. preserve the completed first-deviation traces (Aster and Zen) as the
   immutable comparison baselines and do not reinterpret their time-indexed
   expert actions as recovery labels;
3. add per-module gradient diagnostics and test whether stronger visual
   conditioning changes the now-confirmed state-dominant shortcut;
4. add exact checkpoint metrics, pre/post-clip and per-module optimizer
   diagnostics;
5. add formal resume parity;
6. align validation noise with deployment;
7. test gripper internal-support handling;
8. preserve Way and both Zen arms as completed negative evidence and do not
   continue them in place or reuse any optimizer/checkpoint for a different
   hypothesis;
9. preserve object-geometry plans `003`--`054` as immutable evidence; Plan053
   is the latest safe active-set result and Plan054 is a rejected target-budget
   axis because it failed grasp drift before exercising that event; do not
   create Plan055 under the current scope lock, and do not widen the 0.01-rad
   physical margin, 1 mm / 3 mm pose gates or representation-only
   reconciliation gate;
10. recovery-distribution data (T2) remains the only untried primary axis, but
    it still requires a state-conditioned teacher that passes its own gate
    before any collection or furnace is preregistered;
11. only then consider history, image augmentation, differential LR, weight
    decay, EMA or backbone adaptation as separate axes.

Before optimizer work, a new plan must freeze the hypothesis, data/model/code
identity, resource budget, optimizer/scheduler contract, validation protocol,
Gate 3/4 protocol and stop conditions. Reuse Faust only as a read-only control.

## 11. Common wrong turns

- Do not treat Qwen frozen-feature action-head work as the current VLA.
- Do not initialize SmolVLA from Odyssey, Don Quixote, Moby Dick or Faust unless
  a new controlled warm-start experiment explicitly authorizes it.
- Do not assume `src/rosetta_reality/train/losses.py` changes the SmolVLA flow
  loss.
- Do not call a raw-loss temporal mask a single-axis experiment unless its
  reduction denominator is normalized over the selected valid entries.
- Do not infer compatibility from action tensor shape.
- Do not treat offline MAE, finite loss, a positive reward or Gate 3 as Gate 4
  success.
- Do not generalize the Aster batch-8 offline first-action gain across training
  regimes; the Zen batch-64 two-arm campaign rejected the same temporal-weight
  axis at its scale.
- Do not use time-indexed expert actions as recovery labels after deviation.
- Do not treat the exact-seed retrieval control as a cross-pose recovery oracle;
  its two registered seed-1900 retargeting attempts failed at reward 0.
- Do not select on hidden-test episodes.
- Do not omit processor state from checkpoint export/reload.
- Do not run ML, data, training, evaluation or simulation from native Windows
  Python or the WSL host Python; use the Docker path launched from WSL Bash.
- Do not edit ignored evidence to make a run pass.
- Do not change multiple learning axes in one comparison.

## 12. Verification entry points

Run these from the repository root in WSL Bash through the pinned Docker path:

```bash
scripts/run_m2_container.sh vla-xpu \
  python -m pytest -q \
  tests/test_smolvla_action_repair.py \
  tests/test_smolvla_faust_protocol.py \
  tests/test_smolvla_formal_protocol.py \
  tests/test_smolvla_training_plan_schema.py \
  tests/test_smolvla_training_features.py \
  tests/test_smolvla_training_launch.py

scripts/run_m2_container.sh vla-xpu \
  python -m ruff check \
  src/rosetta_reality/vla \
  src/rosetta_reality/vla/training \
  scripts/run_smolvla_v2.py \
  scripts/train_smolvla_v2.py \
  scripts/run_smolvla_action_repair_formal.py \
  scripts/train_smolvla_action_repair_formal.py \
  scripts/evaluate_smolvla_action_repair_validation.py \
  scripts/export_smolvla_action_repair.py \
  scripts/smolvla_action_repair_sim_gate.py
```

These are code/protocol checks, not authorization to launch another formal run.

## 13. Architecture update rule

Update this document in the same change whenever any of the following changes:

- component ownership or a major entry point;
- dataset/model/Action Contract boundary;
- processor or action representation;
- trainer, loss, optimizer, scheduler or formal resume architecture;
- export/reload or closed-loop control flow;
- the current audit source or Gate 4 status;
- the next registered repair stage.

Keep dated audits and machine evidence append-only. If the stable path ever
changes, update `AGENTS.md`, `README.md` and `docs/architecture.md` together.
Tracked architecture and handoff files must contain only repository-relative
paths.
