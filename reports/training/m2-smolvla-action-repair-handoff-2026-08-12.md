# M2 SmolVLA action-repair reference — 2026-08-12

> Current navigation source: read `docs/m2-smolvla-architecture.md`, then
> `reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.md`
> and its JSON companion before using this historical handoff. Sections 2, 7,
> 8 and 9 below describe the initial no-optimizer boundary and are intentionally
> preserved as provenance; Faust training and simulation were completed later.

## 1. Purpose and authority

This is the primary handoff for the current SmolVLA closed-loop action repair.
An AI resuming this work should read, in order:

1. `AGENTS.md` for repository, runtime, training and safety rules;
2. this document for the current diagnosis and evidence boundary;
3. `configs/vla/smolvla_450m_aloha_insertion_action_repair_001.yaml`;
4. `configs/sim/aloha_insertion_smolvla.yaml`;
5. `src/rosetta_reality/vla/action_space.py` and
   `src/rosetta_reality/vla/processor.py`;
6. the evidence files listed in section 8.

This document does not supersede the Action Contract or executable checks. If
prose conflicts with a checksum-pinned config, code assertion or evidence file,
stop and resolve the identity mismatch. Do not silently choose one.

Current branch at diagnosis time: `codex/er-vla-smolvla-pipeline`.
Current base Git revision: `0f705089f8263e2da5ccf29faf6069014e96a912`.
The worktree is intentionally dirty and contains pre-existing user work. Do not
clean, reset, delete, commit or push it without explicit authorization.

## 2. Current status in one paragraph

The prior three formal checkpoints remain immutable negative evidence; they are
not initialization for the repair. Data is reaching the model and vision has a
measurable effect, but the historical pipeline learned targets that included
tolerated gripper overshoot and did not expose an explicit dataset-to-model
representation boundary. The repair now projects raw standard-ALOHA targets to
the physical contract, maps raw state/actions into pi-Aloha representation,
computes train-only statistics in that representation, and applies the exact
inverse after action unnormalization. A real batch-1 no-optimizer forward passes.
No repair optimizer run, repair checkpoint, repair overfit acceptance, Gate 3 or
Gate 4 success existed when this reference was first written.

## 3. What was actually wrong

### 3.1 Raw target overshoot was included in normalization

The 20,000 train rows contain 280,000 action elements. Before projection:

- left gripper exceeded `[0, 1]` on 686 target elements;
- right gripper exceeded `[0, 1]` on 5,801 target elements;
- 6,487 elements required projection, or `0.023167857142857143` of all
  train action elements;
- maximum observed overshoot was `0.1624636650` for the left gripper and
  `0.0649571121` for the right gripper, both inside the registered `0.20`
  source tolerance.

These values are legal source-data tolerances, not legal controller targets.
They must be projected before statistics are computed. Clipping only after
decoder output hides training-target inconsistency and is not a model-quality
fix.

### 3.2 The tempting policy-level adapter is at the wrong boundary

The pinned base checkpoint has `adapt_to_pi_aloha=false`. Simply setting it to
`true` failed the real no-optimizer forward because the pinned policy calls its
state adapter on `[B, T, D]` before `prepare_state`, while that adapter indexes
the second axis as if state were `[B, D]`. The observed state shape was
`[1, 1, 14]`, producing an out-of-bounds index.

There is also a representation-order problem: the policy-level adapter executes
inside model forward after the normalizer. Its nonlinear gripper formula would
therefore consume z-scored values rather than raw standard-ALOHA gripper values.
The final repair must not enable that path or squeeze away the observation-history
axis merely to suppress the error.

### 3.3 The model uses vision, but state is the stronger shortcut

Moby Dick step 1260 was evaluated teacher-forced on 20 fixed validation contexts:

| condition | chunk MAE | fixed flow loss |
| --- | ---: | ---: |
| normal | `0.0478181764` | `0.1211501330` |
| cross-episode image shuffle | `0.0636706054` | `0.1843460217` |
| zero image | `0.0743345544` | `0.1920972273` |
| cross-episode state shuffle | `0.1021955982` | `0.8398324069` |

Image shuffle raises MAE by about 33%, so images are present and used. State
shuffle raises MAE by about 114%, showing substantially stronger state
dependence. This is a diagnostic result, not a claim that visual grounding is
adequate for closed loop.

The normal teacher-forced right-gripper prediction has:

- MAE `0.0490421468` and first-action MAE `0.0371931849`;
- strict contract violation rate `0.223`, entirely below the lower limit;
- predicted minimum `-0.0760045946`;
- target projection rate `0.234` on these validation chunks.

The corresponding left-gripper strict violation rate is only `0.002`. The
asymmetry is why aggregate validation MAE is not an adequate acceptance metric.

## 4. Final representation contract

The only registered repair chain is:

```text
standard-ALOHA raw state/action
        |
        | target only: reject beyond source tolerance
        | target only: project to physical Action Contract
        v
Rosetta raw-feature pi-Aloha adapter
  - flip joints 1, 2, 8, 9 on the last dimension
  - apply pinned gripper formulas at dimensions 6 and 13
        |
        v
train-only mean/std computed after representation adaptation
        |
        v
SmolVLA forward with policy.adapt_to_pi_aloha=false
        |
        v
action unnormalization in pi-Aloha representation
        |
        v
Rosetta pi-Aloha -> standard-ALOHA inverse
        |
        v
final Action Contract clamp -> simulator adapter
```

The final clamp is a safety boundary, not permission to accept a policy with a
high raw violation rate. Training, reload, evaluation, export and simulation
must serialize and verify the same pre/post processor order.

## 5. Implemented repair components

- `src/rosetta_reality/vla/action_space.py`
  - checksum-pinned experiment overlay loading;
  - explicit representation identity;
  - rejects policy-level double adaptation.
- `src/rosetta_reality/vla/processor.py`
  - create/save/reload-safe target projection step;
  - last-dimension raw-feature pi-Aloha state/action transform;
  - symmetric output inverse and final physical clamp;
  - exact ordering and idempotence checks.
- `scripts/prepare_smolvla_train_stats.py`
  - loads train episodes only;
  - projects standard-space target overshoot;
  - transforms raw representation before computing statistics;
  - creates a checksum-validated immutable dataset view.
- `scripts/diagnose_smolvla_action_space.py`
  - loads no model weights or dataset rows;
  - checks formula parity with the pinned upstream implementation;
  - checks standard-space round trip.
- `scripts/smolvla_forward_check.py`
  - installs the complete repair boundary;
  - checks real raw/preprocessed/padded shapes and finite no-gradient loss.
- `src/rosetta_reality/eval/diagnostics.py`
  - per-dimension and arm/gripper group metrics;
  - raw violation, target projection and gripper open/close metrics.
- `scripts/diagnose_smolvla_modalities.py`
  - fixed-context teacher forcing;
  - normal, image shuffle, zero image and state shuffle comparisons;
  - disables performance-only `torch.compile` without modifying the historical
    checkpoint or historical evaluator.
- `scripts/run_smolvla_action_repair.py` and
  `scripts/train_smolvla_action_repair.py`
  - separate repair entrypoints;
  - optimizer remains fail-closed until explicit launcher authorization and
    prerequisite validation.

Historical hash-bound scripts and reports must not be edited to retrofit this
repair. New training/evaluation/export/simulation evidence belongs under the
new repair experiment identity.

## 6. Failed approaches that must not be repeated

1. Do not set only `policy.adapt_to_pi_aloha=true`. It failed on `[B,1,14]`
   state and applies nonlinear gripper mapping after normalization.
2. Do not `squeeze(1)` merely to satisfy the upstream adapter. It would conceal
   the contract mismatch and is unsafe if observation history later grows.
3. Do not compute statistics on unprojected standard-space targets.
4. Do not treat post-decoder clamp as proof that the model learned legal action.
5. Do not use time-indexed expert actions as recovery labels after closed-loop
   deviation.
6. Do not run a fourth historical furnace or initialize repair training from a
   historical furnace checkpoint.
7. Do not enable checkpoint `compile_model=true` in the modality diagnostic.
   Under the hardened container `/tmp` is `noexec`; Triton cold compilation
   failed before inference. The diagnostic intentionally loads identical
   weights with compile disabled.

## 7. Acceptance boundary and next work

The user explicitly authorized simulated training and other in-scope repair
changes on 2026-08-12. This authorizes local offline smoke/overfit and simulation
within repository rules. It does not authorize commit, push, Hub upload, hidden
test use, destructive cleanup or physical-robot control.

Before any full repair training:

1. Run the repair optimizer smoke at batch 1 and minimal steps.
2. Verify finite loss/gradients, processor serialization, checkpoint creation,
   independent reload and no hidden-test exposure.
3. Run fixed-episode small-data overfit.
4. Compare per-dimension and especially right-gripper teacher-forced metrics
   against both the repair base and historical Moby diagnostic.
5. Stop if right-gripper raw violation regresses, image shuffle has no material
   effect, processor identity changes, or source values exceed tolerance.
6. Only after overfit acceptance, preregister a bounded formal repair run.
7. Run Gate 3 short closed loop, then the CPU-light Gate 4 policy. Offline MAE
   alone cannot promote a checkpoint.

Hidden-test episodes `[31, 6, 1, 24, 5]` remain sealed until validation-only
selection and development simulation acceptance.

## 8. Canonical evidence and checksums

Code/config identities at the completed no-optimizer boundary:

| path | SHA-256 |
| --- | --- |
| `configs/vla/smolvla_450m_aloha_insertion_action_repair_001.yaml` | `3ab3e213d4fd4197e4bd92550a3b4193d89814de68338bcfdbb31b0b2eb1ce03` |
| `configs/vla/smolvla_450m_aloha_insertion.yaml` | `7306014717eb3fe5a5997a92406ec18cb251f1c17cc9d46ed74d75ac71d8b19f` |
| `configs/sim/aloha_insertion_smolvla.yaml` | `fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62` |
| `src/rosetta_reality/vla/action_space.py` | `f88dd5534449655b9e7127ef6c98656ec2f337d4ff45c63f0fec4062c795dde3` |
| `src/rosetta_reality/vla/processor.py` | `977792d923c099c5759c7b6d5fa5628da0c8c2dae6400fefae7fa91acafc7453` |
| `scripts/prepare_smolvla_train_stats.py` | `7dc6018e7ff6722ffe4f17b213b4b294700e87c1b25ac847643bb65b3c26596c` |
| `scripts/diagnose_smolvla_action_space.py` | `8e3bf583455d89983a771914d477bdf7947a863cff8a171fbad49b4eed8c360d` |
| `scripts/diagnose_smolvla_modalities.py` | `192cf88f35b2f2dc0e6838ce4514568612a90533ee80e85ef94317b4e6b7e390` |
| `scripts/smolvla_forward_check.py` | `ef585b3940acba34c87bd11ba4dde5176948e673bdbe7de6d2dd351e3dc82a33` |
| `tests/test_smolvla_action_repair.py` | `7b6a1db0743943c893daee6d5b805c2d7ec1c76d3b4b5cde2e9e445b56232d83` |

Runtime evidence:

| path | SHA-256 | meaning |
| --- | --- | --- |
| `runs/m2-smolvla450m-aloha-insertion-action-repair-001/normalization/train-only-5144bffdf1530aef.json` | `501d0c35138e21edda9e3272ff66d78656cb6e5ba04b45573c6d2f55327fad54` | transformed train-only statistics and projection counts |
| `runs/m2-smolvla450m-aloha-insertion-action-repair-001/diagnostics/action-space-a89c584397e95737.json` | `dfed04668e9a5fca0ffcc62c8969b75ce688a20007d2e61294f020393838b2dc` | upstream formula parity `0`, round-trip error `5.26e-17` |
| `runs/m2-smolvla450m-aloha-insertion-action-repair-001/preflight/m2-smolvla-action-repair-preflight-003.json` | `34880bc65f6adb7e907797cf7889a805cc0e8962959e22cdb244b94a2f66e860` | real batch-1, no-optimizer forward, finite loss `1.9002937` |
| `runs/m2-smolvla450m-aloha-insertion-001/diagnostics/teacher-forced-modalities-step001260-d54229e9b3c4a230.json` | `4cb0e8d2fb04f7419fd7c6e3b029463aff44d9d6ed78ce9aa73243197206778c` | historical Moby per-dimension and modality diagnosis |

The first three repair evidence files report `hidden_test_loaded=false` and
`optimizer_created=false`. The Moby modality report is explicitly non-gating
historical diagnosis.

## 9. Reproducible read-only/no-optimizer commands

Run from the repository root inside WSL Bash. Do not substitute Windows Python
or WSL host Python for the Docker environment.

```bash
./scripts/run_m2_container.sh vla-cpu \
  python scripts/prepare_smolvla_train_stats.py \
  --config configs/vla/smolvla_450m_aloha_insertion_action_repair_001.yaml

./scripts/run_m2_container.sh vla-cpu \
  python scripts/diagnose_smolvla_action_space.py \
  --config configs/vla/smolvla_450m_aloha_insertion_action_repair_001.yaml \
  --normalization-report \
  runs/m2-smolvla450m-aloha-insertion-action-repair-001/normalization/train-only-5144bffdf1530aef.json

./scripts/run_m2_container.sh vla-cpu \
  python scripts/run_smolvla_action_repair.py preflight \
  --run-name m2-smolvla-action-repair-preflight-new \
  --normalization-report \
  runs/m2-smolvla450m-aloha-insertion-action-repair-001/normalization/train-only-5144bffdf1530aef.json \
  --action-space-report \
  runs/m2-smolvla450m-aloha-insertion-action-repair-001/diagnostics/action-space-a89c584397e95737.json
```

Create a new run name for create-only evidence. Existing evidence must never be
overwritten or deleted.

## 10. Primary upstream references

- SmolVLA paper: <https://arxiv.org/abs/2506.01844>
- Pinned LeRobot SmolVLA configuration:
  <https://github.com/huggingface/lerobot/blob/c903b114a90e703b3f7d0c46cb38727c328c55ff/src/lerobot/policies/smolvla/configuration_smolvla.py>
- Pinned LeRobot SmolVLA implementation and ALOHA formulas:
  <https://github.com/huggingface/lerobot/blob/c903b114a90e703b3f7d0c46cb38727c328c55ff/src/lerobot/policies/smolvla/modeling_smolvla.py>
- Pinned base checkpoint configuration:
  <https://huggingface.co/lerobot/smolvla_base/blob/c83c3163b8ca9b7e67c509fffd9121e66cb96205/config.json>
- LeRobot processor pipeline source at the pinned code revision:
  <https://github.com/huggingface/lerobot/blob/c903b114a90e703b3f7d0c46cb38727c328c55ff/src/lerobot/processor/pipeline.py>
