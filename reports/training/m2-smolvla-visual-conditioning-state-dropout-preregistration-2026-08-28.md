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

## 7. Post-training path completion (append-only amendment, 2026-08-28)

The two implementation gaps named in section 6 now exist and are verified:

| Path | Owns |
|---|---|
| `scripts/smolvla_vcdropout_protocol.py` | frozen candidate identity (`m2-smolvla450m-vcdropout-001`), the section-3 treatment contract, the section-4 control-evidence pins, the offset-250 gate protocol and the six preregistered criteria as executable logic |
| `scripts/smolvla_vcdropout_validate.py` | candidate-bound fixed-validation wrapper around the frozen `evaluate_smolvla_validation` engine (validation inputs stay clean; the dropout is never installed here) |
| `scripts/select_smolvla_vcdropout_checkpoint.py` | validation-only selection with earlier-checkpoint tie-break |
| `scripts/export_smolvla_vcdropout.py` | deploy artifact export plus exact independent reload through a fresh validation process |
| `scripts/gate_smolvla_vcdropout_visual_conditioning.py` | the executable offset-250 gradient gate: consumes a verified, exactly-reloaded candidate artifact bound to the validated plan, reuses the completed Zen diagnostic's measurement helpers read-only, and reports pass/fail per criterion create-only |
| `tests/test_smolvla_vcdropout_protocol.py` | 14 focused protocol tests: frozen thresholds, control-baseline reproduction, fail-closed criteria math, single-axis stack, treatment/contract/inventory drift |

Verification: the 14 new tests plus the schema, visual-conditioning and
feature-registry suites (65 tests total) and Ruff pass in the pinned read-only
Linux container; no weights, no dataset rows, no optimizer.

Implementation hashes at this amendment:

| Path | SHA-256 |
|---|---|
| `scripts/smolvla_vcdropout_protocol.py` | `3967c6192f74fac6d5aa7d9891d25d5f8eaed4aec02a59e6833522cfff1b2cbc` |
| `scripts/smolvla_vcdropout_validate.py` | `cf319d3e407fcabebdb2aa3ba0453131efff9ae890734cc6d872bea10918276c` |
| `scripts/select_smolvla_vcdropout_checkpoint.py` | `4cce7bbbdc77b901e1be59ea6f14ed1ed149f6d23029573cbde1848f0040e4e6` |
| `scripts/export_smolvla_vcdropout.py` | `debe434d71a5b9b4ccffa7086570e8253b416e8a300c92104c249b02ef721bbb` |
| `scripts/gate_smolvla_vcdropout_visual_conditioning.py` | `051c98825c1f04b865f30b8e0f9f79880bfff07926134c77532cafa390cc26e1` |
| `tests/test_smolvla_vcdropout_protocol.py` | `dfd28ea00c09059210c08fc78f27a0edb0351526564487fa876ff651be357f11` |

Schema amendment recorded in the same session: `state_robustness_jitter` and
`state_conditioning_dropout` are now mutually exclusive in the version-2 plan
schema (both patch the pinned policy forward and mutate the same training-time
state), changing `src/rosetta_reality/vla/training/plan.py` to SHA-256
`7fd3d0c45c29c5cc2c4b08b4e1b6d3666519f6a1bb53fd84f12260f3674a3c63` and
`tests/test_smolvla_training_plan_schema.py` to
`c6bb82b6c20f562fef0781605119a59543cf66cfff83a01f57ea8d380db62807`; the
treatment semantics of section 3 are unchanged.

Still not created and not authorized by this amendment: the executable formal
plan YAML, the no-optimizer preflight, the two-step CUDA optimizer smoke and
any training, Gate 3/4 or hidden-test access.


## Amendment 2026-08-28 (furnace attempt 1 instrumentation repair)

Attempt 1 of the authorized AutoDL ladder (`m2-smolvla450m-vcdropout-cuda-b64-001`,
workspace archive SHA-256 `7f0be965df5299a84a2ecfeee7e40b7970bc10502b55fc2b9c13a699a7bc13ad`)
passed doctor, benchmark and the no-optimizer preflight, then crashed the
two-step optimizer smoke at step 0 before any optimizer state or checkpoint was
written. Two defects in the O2/T9 instrumentation features caused it:

1. `checkpoint_metric_snapshot` read `float(tracker.loss)` etc., but the pinned
   trainer routes every `train_metrics` assignment through
   `MetricsTracker.__setattr__` into `AverageMeter.update`, so each step value
   lives in `.val`. The crash (`TypeError: float() argument must be a string or
   a real number, not 'AverageMeter'`) stopped the run fail-closed.
2. The same crash exposed a silent defect: `gradient_clip_diagnostics` iterated
   the `policy.parameters()` generator for its pre-clip norms, so the original
   clip path received an exhausted iterator and accelerate logged "no gradient
   clipping will occur" — the registered `grad_clip_norm = 10` contract would
   have been silently disabled for the whole run.

Repair (no learning-semantics change; treatment, thresholds, optimizer,
scheduler, data identity and all six gate criteria unchanged):

- `src/rosetta_reality/vla/training/features.py` now reads the step value
  through a `_meter_value` helper (AverageMeter `.val`, plain floats still
  supported) and materializes the clip parameters once per call so the
  diagnostics passes and the original clip observe the same live tensors.
- `tests/test_smolvla_optimizer_diagnostics.py` replaces the float-valued test
  double with a pinned-shape `AverageMeter` tracker, passes a parameters
  generator to the clip wrapper, and adds an assertion that clipping actually
  applies below the current norm (both regressions are now encoded).

Focused container verification re-run after the repair: 78 tests across the
schema, features, launch, protocol and visual-conditioning suites plus Ruff
pass; no weights, no dataset rows, no optimizer.

Implementation hashes at this amendment:

| Path | SHA-256 |
|---|---|
| `configs/vla/smolvla_450m_aloha_insertion_vcdropout_cuda_b64_001.yaml` | `fd6784ae221eb96c9ac36531e9fa14d73182f243a3a1936c660ddb86a9509e39` |
| `src/rosetta_reality/vla/training/features.py` | `e0ded450340be7f6808d8f764122dd52e8adf3a964b07f00f0f3ea3dd5fdbe53` |
| `tests/test_smolvla_optimizer_diagnostics.py` | `eb84a14edefdf0f8dcf521bac1a4369e3b83bba0e4c4594c6e9c697d82f327eb` |

Attempt 1 durable artifacts are archived (not deleted) under the durable
orchestration area as `vcd-attempt1-archive/`; no run identity was consumed by
the failed smoke, so the frozen run names are unchanged. The frozen gradient
gate, its six criteria, and the post-export boundary (stop after export, local
XPU gate, separately registered Gate 3/4) are unchanged.
