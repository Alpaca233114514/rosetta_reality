# M2 SmolVLA Zen module-gradient diagnostic preregistration — 2026-08-28

## 1. Authority and scope

This preregisters the per-module gradient diagnostic named as the next local
item in `docs/m2-smolvla-architecture.md` section 10 (finding T4, the
state-dominant shortcut). It is a **non-gating local diagnostic**: no
optimizer, no weight update, no training, no hidden-test access, no
closed-loop claim. It executes entirely in the local WSL
`vla-sim-xpu` container against the two already-transferred Zen deploy
artifacts. The "stronger visual conditioning" axis itself remains a separate
future single-axis registration informed by these results.

## 2. Question and hypothesis structure

The Aster-era teacher-forced modality diagnostic established output-level
state dominance (state shuffle moved first-action MAE 8.4x more than image
shuffle). The open question is whether the **learning signal itself** is
state-dominated: measured at the registered validation sample identity, do
the gradients flowing into the trainable modules (action expert, state
projector, action I/O projections) change massively when the state input is
shuffled across episodes, while barely changing when images are shuffled or
zeroed?

Either answer is recorded as-is. This is descriptive evidence for axis
design, not acceptance of any policy.

## 3. Frozen identities

| Field | Value |
|---|---|
| artifacts | the two local Zen deploy artifacts; uniform manifest sha256 `ecc73b9e…`, firstaction manifest sha256 `d6b2a7ff…` (verified against manifest files at run time) |
| samples | the registered validation protocol of the Zen plan: episodes `[22, 13, 7, 33, 45]`, frame offsets `[0]`, 5 samples — never train-gradient episodes, never hidden test |
| conditions | `normal`, `image_shuffle` (cross-episode derangement), `state_shuffle` (cross-episode), `image_zero` (black images) |
| shuffle | `cross_episode_same_frame_offset_derangement`, seed `20260812` (the registered modality-diagnostic seed) |
| loss point | teacher-forced flow-matching loss with **zeros noise** and fixed **flow time `0.5`** (the registered historical validation flow time) |
| adaptation rebuild | `freeze_vision_encoder=true`, `train_expert_only=true`, `train_state_proj=true` — the registered shared contract; fail-closed verification that vision/language groups contain zero trainable parameters and expert `lm_head` stays frozen |
| gradient groups | first-match-wins over `named_parameters()`: `vision_encoder` (`vlm_with_expert.vlm.vision_model.*`), `language_model` (remaining `vlm_with_expert.vlm.*`), `action_expert` (`vlm_with_expert.lm_expert.*`), `state_projector` (`state_proj.*`), `action_io_projections` (`action_in_proj.*`, `action_out_proj.*`, `action_time_mlp_in.*`, `action_time_mlp_out.*`); any unmatched parameter fails closed |
| metric | per-group gradient L2 (sqrt of summed squared grads over the group) per sample; report mean/max over the 5 samples per condition, plus perturbed/normal ratios |
| dataset | `lerobot/aloha_sim_insertion_human` revision `cc571a3c…`, checksum-validated local cache; view restricted to the validation episodes |
| boundary | artifact processors + `ensure_smolvla_action_boundary` with the artifact's action space; Action Contract sha `fc71a043…` |
| runtime | local WSL Docker `vla-sim-xpu`, image `sha256:f4a71c40…`, memory `6g`, networking disabled, XPU, bf16 autocast, policy in `eval` mode |
| script | `scripts/diagnose_smolvla_zen_module_gradients.py`, sha256 `8558f5a5884c2344081bb624430dc157048483d5139091857c5648698ed14ec6` |
| runner | `scripts/run_zen_module_gradients.sh`, sha256 `30c104c2c1659028368263d801fafd2c99dcb46ff0aa08f73f1390f2f19bd8f1` |

## 4. Acceptance criteria (of the diagnostic run itself)

1. both artifacts produce complete
   `zen-module-gradients-<arm>-<hash>.json` reports with all-finite losses and
   gradients;
2. the freeze verification passes (vision/language groups contain zero
   trainable parameters; expert `lm_head` frozen);
3. the group contract covers every policy parameter (fail closed otherwise);
4. hidden-test episodes never loaded; no optimizer created; zero weight
   updates.

## 5. Stop conditions

- any identity/checksum/freeze/group mismatch (fail closed, no fallback);
- non-finite loss or gradient;
- container OOM or XPU failure (report and stop; no automatic retry);
- no re-run with changed samples, seeds, conditions or artifacts under this
  registration.

## 6. Expected evidence

- `runs/<experiment_id>/diagnostics/zen-module-gradients-{firstaction,uniform}-<hash>.json`
- `runs/<experiment_id>/orchestration/zen-module-gradients-001.{log,status}`
- a completion report
  `reports/training/m2-smolvla-zen-module-gradient-diagnostic-2026-08-28.{md,json}`

## 7. What not to conclude

- frozen-module gradients are expected to be exactly zero; this verifies the
  freeze, it does not measure visual signal use;
- the zeros-noise / fixed-time loss point is a registered deterministic
  probe, not the full training loss distribution;
- results do not gate M2, do not rank the arms as policies, and do not
  authorize the visual-conditioning training axis by themselves.
