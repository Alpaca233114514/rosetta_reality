# M2 first-step vision-representation diagnostic (2026-09-03)

Closes the feed-vs-absorption-vs-representation question behind the two
recorded closed-loop failure modes (step-zero deviation; no recovery), and
fixes the direction of the policy-line redo. JSON companion:
`m2-smolvla-t2-first-step-vision-representation-diagnostic-2026-09-03.json`.

All four measurements are read-only diagnostics (no gate evidence written,
no weights updated, only the already-opened seeds/episodes touched: the
45 non-hidden dataset episodes and the recorded artifacts).

## 1. Measurements

### 1.1 Teacher step-0 self-deviation (teacher line, own bug)

`scripts/diagnose_teacher_first_step.py`, seed-10 calibration replay: the
registered geometric teacher's first command sits **0.117 rad mean-abs from
its own current state** (expert-relative action MAE 0.117 vs state-relative
0.019). Root cause: the OPEN-phase "hold home" target was re-solved
through the constrained QP, whose redundancy resolution lands on a
different arm configuration; the state then converges there during the
opening phase (state-vs-expert MAE 0.019 -> 0.094 over 11 steps). Every
teacher episode of every candidate inherited the wrong configuration
manifold. Fix implemented and tested the same day (zero-delta hold with an
explicit `hold_enabled` switch; 22 focused tests pass).

### 1.2 Dataset frame-0 degeneracy (the forced ambiguity)

`scripts/diagnose_first_action_spread.py`: all 50 episodes share
bit-identical frame-0 states (pairwise max difference 0.000000) while
their expert first actions spread **MAD 0.0134 / pairwise 0.0185**. At
frame 0 the image is the only signal identifying which trajectory to
follow. The policies' recorded step-0 action MAE (Aster 0.020, Zen 0.033)
sits at/above this mean-predictor floor — quantitatively closing the
"vision ignored -> first action is the episode mean" mechanism.

### 1.3 Frame-0 functional probe (deployed artifacts ignore image content)

`scripts/diagnose_frame0_vision_probe.py`: 45 non-hidden episodes, frame 0,
zero noise, deterministic flow, identical state with each episode's own
image —

| artifact | output spread MAD | MAE vs episode expert | output-vs-expert per-dim corr |
|---|---|---|---|
| vcdropout step 237 (gradient gate PASSED) | 0.0120 | 0.0299 | mean −0.05 |
| Zen-uniform step 316 (baseline) | 0.0075 | 0.0206 | mean +0.01 |

Both outputs are uncorrelated with the episode-correct first action and
both are worse than the mean predictor (0.0134). Retroactive
methodological finding: the frozen gradient-gate criterion
(`image_sensitivity >= 0.1`) was satisfied by image-dependent noise, not
image-aligned behaviour — sensitivity is not correctness, which explains
the vcdropout paradox (gradient gate PASS, Gate 4 still 0/5).

### 1.4 Frozen-feature linear probe (representation verdict)

`scripts/diagnose_frozen_feature_probe.py`: ridge regression (5-fold CV,
alpha swept) from mean-pooled frozen visual representations to the
episode-specific first action, vcdropout artifact —

| module (frozen, trainable=0) | dim | cv MAE | cv R² |
|---|---|---|---|
| vision tower (`vlm.model.vision_model`) | 768 | 0.0153 | −0.182 |
| connector (last frozen rep the expert consumes) | 960 | 0.0157 | −0.228 |

Both are above the registered floor (0.0134; same-data CV mean-predictor
0.0136) with negative R² — the ridge collapses to the constant predictor;
the signal is absent, not weak. Tower and connector agree (no
"present-in-tower, lost-in-connector" split). Registered boundary: the
probe covers mean-pooled LINEAR decodability only; non-linear/token-level
decodability is untested.

## 2. Verdict

1. **Feed: intact** (images real, pairwise max difference 223/255; input
   path live, image_zero effects recorded).
2. **Function: vision content unused** at the only frames where it is the
   sole disambiguating signal — on both deployed artifacts, including the
   one whose training forced gradient-level image sensitivity.
3. **Representation: limited** — the frozen mean-pooled visual features do
   not linearly decode the episode-discriminative action variation. The
   closed-loop failure cannot be explained by optimization/training
   insufficiency alone; the trainable head had nothing decodable to absorb.

Therefore the policy line's blocking problem is **representational**:
the frozen vision stack does not encode the object-layout signal that the
first action depends on.

## 3. Consequence (redo direction, per the standing user authorization)

- **Primary redo axis: unfreeze/adapt the vision encoder** as a new
  single-axis plan through the version-2 harness — copying the existing
  hash-bound plan template (e.g.
  `configs/vla/smolvla_450m_aloha_insertion_vcdropout_cuda_b64_001.yaml`)
  with the one registered change (`freeze_vision_encoder: false`, with the
  associated adaptation/batch/resource consequences), through the full
  registered chain (tiny smoke -> small-data overfit -> formal ->
  selection -> export/reload -> Gate 3 -> Gate 4).
- **Gate the redo on alignment, not sensitivity**: add the frame-0
  functional-probe correlation (output-vs-expert, 45 episodes) and/or the
  linear-probe floor comparison as frozen acceptance criteria — the
  vcdropout result proved a sensitivity-only gate is gameable by noise.
- **The teacher line stays necessary and is now unblocked-for-retest**:
  the zero-delta hold fix removes the 0.117-rad yank that contaminated
  every teacher trajectory and the earlier feasibility verdicts; the
  candidate-004 retest (live-geometry composition + hold fidelity) is the
  next registered step for the recovery-data axis. Both axes are required
  for a finished product: representation to stop guessing the mean at
  step 0, recovery supervision to return from deviation.
- Compute note: the redo furnace (vision unfreeze) needs AutoDL
  authorization and a resource re-budget (trainable parameters grow by
  ~100M); nothing is launched by this diagnostic.

## 4. Boundaries

No gate thresholds were modified; no gate evidence written; the hidden
test stays sealed; no collection or furnace authorized; no AutoDL, SSH,
download or external service used; the frozen observation contract and
Action Contract unchanged.
