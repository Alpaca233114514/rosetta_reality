# M2 SmolVLA Zen module-gradient diagnostic — 2026-08-28

## 1. Authority and scope

Completion report for the per-module gradient diagnostic, covering both
registrations:
`reports/training/m2-smolvla-zen-module-gradient-preregistration-2026-08-28.md`
(registration 001, frame offset 0, executed as orchestration `003` after two
fail-closed implementation repairs) and
`reports/training/m2-smolvla-zen-module-gradient-offset250-preregistration-2026-08-28.md`
(registration 002, frame offset 250, orchestration `005` after one more
fail-closed repair). Non-gating local diagnostics: no optimizer, no weight
update, no hidden-test access, no closed-loop claim. Executed against the two
locally transferred Zen deploy artifacts in the pinned `vla-sim-xpu` container.

## 2. Result — the learning signal is state-dominated and nearly image-blind

At frame offset 250 (mid-trajectory, where the five validation episodes'
states genuinely differ pairwise by up to `0.8384` rad), the teacher-forced
flow-matching loss probe (zeros noise, fixed time `0.5`) gives:

| Arm | condition | mean loss | expert grad | state-proj grad | action-I/O grad |
|---|---|---|---|---|---|
| firstaction | normal | `1.0902` | 14.32 | 2.63 | 13.38 |
| firstaction | state_shuffle | `2.2468` (×2.06) | **×1.547** | **×1.878** | **×1.551** |
| firstaction | image_shuffle | `1.0373` (×0.95) | ×0.989 | ×1.000 | ×0.978 |
| firstaction | image_zero | `1.0477` (×0.96) | ×1.046 | ×1.076 | ×1.008 |
| uniform | normal | `0.1467` | 5.04 | 0.52 | 4.14 |
| uniform | state_shuffle | `0.6485` (×4.42) | **×2.387** | **×3.378** | **×2.162** |
| uniform | image_shuffle | `0.1410` (×0.96) | ×0.972 | ×0.918 | ×0.972 |
| uniform | image_zero | `0.2456` (×1.68) | ×1.465 | ×1.272 | ×1.403 |

(gradient columns at `normal` are per-group mean gradient L2; the other rows
are ratios versus `normal`.)

**Reading.** Shuffling the state across episodes multiplies the loss by
2.1–4.4× and the gradients flowing into every trainable module by 1.5–3.4×,
while shuffling the images across episodes changes the same gradients by at
most ~3% (firstaction) / ~8% (uniform). Zeroing the images entirely moves the
gradients by at most ~8% (firstaction) / ~47% (uniform) and never approaches
the state effect. This is the gradient-level confirmation of finding T4: the
supervision signal itself — not just the trained model's outputs — is
state-dominated and nearly image-blind for both Zen arms. Any future
"stronger visual conditioning" axis now has a concrete, cheap, local metric
to move: these per-group gradient ratios.

The absolute-loss difference between arms at the probe (firstaction `1.09` vs
uniform `0.15`) reflects that the probe evaluates the standard uniform flow
loss while the firstaction arm was trained under the first-action weighting;
ratios, not absolute losses, are the diagnostic metric.

## 3. Result — registration 001 (offset 0) and the protocol discovery

At the registered validation offset 0, `state_shuffle` reproduced `normal`
**bit-exactly** (identical loss and gradients for both arms). Root cause,
verified directly against the raw parquet columns of the pinned dataset: at
frame 0 all 50 episodes share the identical home-pose state
(`[0, -0.96, 1.16, 0, -0.3, 0, 0]` per arm, pairwise max difference `0.0000`);
episode-to-episode variation at frame 0 lives only in the randomized object
poses (images differ by up to `223` intensity levels). The offset-0 probe
therefore cannot measure state sensitivity, and its image conditions measure
the shared-home-pose point only: firstaction's gradients were essentially
image-insensitive there too (image_shuffle ×0.998–1.014), while uniform showed
more image response (image_shuffle ×1.18–1.23, image_zero ×2.05–2.48).

Correction note: two intermediate one-off probe scripts used during
investigation initially reported "states identical at all offsets"; that was a
bug in the probes themselves (pairwise comparison over only the first five
rows of a 25-row sample stack, i.e. within one episode). The final truth,
established by per-episode values from both the raw parquet and the dataset
view, is: identical only at frame 0; differing by 0.36–0.84 rad at frame 250.
The diagnostics themselves were never affected — the 002 registration added
fail-closed degeneracy guards plus recorded pairwise differences so a no-op
condition can never silently recur.

## 4. Freeze verification

All runs verified the registered adaptation rebuild exactly:
`vision_encoder` and `language_model` groups contain zero trainable
parameters and produce exactly zero gradients; the expert `lm_head` stays
frozen; `state_projector`, `action_expert` and `action_io_projections` are
trainable. The parameter grouping (five groups, substring alternatives,
fail-closed on unmatched) covered every policy parameter.

## 5. Implementation-defect ledger (all fail-closed, all evidence retained)

1. orchestration `001`: fixed-prefix grouping missed the runtime `model.`
   parameter prefix — aborted before any gradient;
2. orchestration `002`: the action-I/O group required all four markers in one
   name (`all` where any-of-alternatives was meant) — aborted with the eight
   action-projection parameters unmatched;
3. orchestration `004`: the offset list was built per-episode, making the
   sample selector return the 5×5 cartesian product — aborted by the
   sample-count guard.

Each repair was committed with an amended registration before the re-run; no
sample, seed, condition or artifact changed across repairs.

## 6. Evidence ledger

- offset 250: `runs/…/diagnostics/zen-module-gradients-cuda-146acab3e4295d47.json`
  (firstaction), `zen-module-gradients-cuda-ae499e966f438a31.json` (uniform)
- offset 0: `runs/…/diagnostics/zen-module-gradients-cuda-c91c0de4d944515f.json`
  (firstaction), `zen-module-gradients-cuda-b8fbf339786a4eef.json` (uniform)
- orchestration `003` (success, offset 0), `005` (success, offset 250),
  `001`/`002`/`004` (fail-closed attempts, retained)

## 7. What not to conclude

- non-gating; no arm ranking as policies; no M2 acceptance;
- the zeros-noise / fixed-time loss at two frame offsets is a probe, not the
  training distribution;
- these ratios do not by themselves authorize the visual-conditioning
  training axis; that axis still needs its own single-axis preregistration,
  for which these numbers define the target metric.
