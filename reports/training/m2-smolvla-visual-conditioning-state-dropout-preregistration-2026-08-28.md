# M2 SmolVLA visual-conditioning state-dropout preregistration — 2026-08-28

## 1. Decision and authority boundary

The next T4 axis is registered as **sample-wise whole-state dropout in
train-normalized space**.  It is an experimental visual-conditioning
intervention, not a recovery oracle and not evidence that the vision encoder
should be unfrozen.  This document authorizes the local implementation and
static/container verification completed below.  It does **not** authorize an
optimizer smoke, a formal AutoDL run, Gate 3/4, hidden-test access or release of
the stopped AutoDL instance.

Current causal evidence is the completed Zen module-gradient diagnostic at
validation frame offset 250.  The offset-0 protocol is forbidden for state
sensitivity because all 50 episode states are bit-identical there.

## 2. Single-axis hypothesis

Against the immutable Zen-uniform control, replacing the complete normalized
`observation.state` vector with normalized zero for exactly half of each
training batch will force the already-trainable action expert to use more of
the frozen visual representation.  Everything else remains fixed: fresh pinned
base, uniform flow loss, 20,224 exposures, batch 64, 316 optimizer updates,
dataset/split/order seed, optimizer, scheduler, action boundary, frozen vision
and language modules, clean validation and clean deployment state.

This is deliberately tested before partial vision unfreezing.  Unfreezing would
also change the trainable parameter set, memory budget and optimizer dynamics;
those are separate axes.

## 3. Treatment contract

| Field | Registered value |
|---|---|
| profile | `samplewise_normalized_state_dropout` |
| input | train-normalized `observation.state` |
| probability | `0.5` |
| granularity | complete sample state, never per coordinate |
| replacement | normalized zero (train mean), no retained-state rescaling |
| RNG | dedicated CPU generator, seed `20260828` |
| global model/dataloader RNG | must remain bit-identical to control |
| target | unchanged absolute expert action |
| validation/deployment | no dropout |
| minimum optimizer batch | 2; degenerate all-kept/all-dropped masks fail closed |
| resume | forbidden until the feature generator is checkpointed and T7 parity passes |

The dedicated generator is a causal-control requirement: consuming the global
RNG would also shift SmolVLA flow noise and would invalidate the single-axis
claim.

## 4. Immutable control and target metric

Control artifact:
`m2-smolvla450m-zen-cuda-b64-uniform-001-step0316-deploy-001`.
Control evidence:
`reports/training/m2-smolvla-zen-module-gradient-diagnostic-2026-08-28.json`
(SHA-256 `5ee9824e8bea8031094a7d650a9f9e594a5b55f96470f9ba7eda5065dc38190c`).

The candidate must use the exact offset-250, five-validation-episode,
cross-episode derangement, zero-noise, flow-time-0.5 protocol.  For the three
trainable groups (`action_expert`, `state_projector`,
`action_io_projections`), define:

```text
state_sensitivity = mean(abs(log(state_shuffle_gradient_ratio)))
image_sensitivity = mean(abs(log(image_shuffle_gradient_ratio)))
state_dominance_score = state_sensitivity - image_sensitivity
```

The Zen-uniform baseline is approximately `0.953`, `0.047` and `0.905`.
The candidate gradient gate passes only if all of the following hold:

1. freeze/trainable grouping is exact and every trainable-group normal gradient
   is finite and nonzero;
2. sample state/image diversity guards pass at nonzero offset;
3. normal mean flow loss is at most `0.22005` (1.5x the uniform control's
   `0.1467`), preventing an input-destruction "win";
4. `state_sensitivity <= 0.70`;
5. `image_sensitivity >= 0.10`;
6. `state_dominance_score <= 0.453` (at least 50% below the control score).

This is a pre-Gate diagnostic gate, not M2 acceptance.  A pass only permits a
separately registered Gate 3/4 comparison.  Any failed criterion stops the
axis; thresholds may not be relaxed after seeing the candidate.

## 5. Implementation and verification state

New implementation:
`src/rosetta_reality/vla/visual_conditioning.py`.  The v2 registry exposes it
only as `state_conditioning_dropout`; the plan schema requires an explicit
`visual_conditioning_contract`, and public Trackio identity fields record the
profile, probability, granularity and training-only boundary.

The completed no-weight/no-data Linux-container checks are:

- 50 focused pytest cases passed;
- Ruff passed on the implementation, v2 registry/schema and focused tests;
- the only pytest warning was inability to write `.pytest_cache` on the
  intentionally read-only repository mount.

Implementation hashes at this registration:

| Path | SHA-256 |
|---|---|
| `src/rosetta_reality/vla/visual_conditioning.py` | `8113fb1bb0c0dcf99111b970d926fd39c3235ebe6b356e35441903463e6c369c` |
| `src/rosetta_reality/vla/training/features.py` | `cee1769c122b42de477b58cd1039c783798fd53915987cbc7ce5e9761e254055` |
| `src/rosetta_reality/vla/training/plan.py` | `06bd1d6dc8728e6c96ac18cf451e2577dead5ed1f7fac2ac5acc79cc25f99d9b` |
| `tests/test_smolvla_visual_conditioning.py` | `f281e929280df202bd3e00e35435c56d790f8654cc384fbbc83235e12acfb132` |

## 6. Required next gate before any optimizer work

Before creating an executable formal plan, add and verify a create-only v2
post-training path that binds selection/export to the candidate plan, performs
exact independent reload, and runs the metric gate above.  Then freeze a new
hash-bound v2 plan plus no-optimizer preflight and two-step CUDA smoke.  No Zen
checkpoint or optimizer state may be reused; no hidden-test or recovery-label
boundary opens.

