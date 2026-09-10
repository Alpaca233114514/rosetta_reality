# Visual-pair objective — lightweight local result, 2026-09-09

The experimental arithmetic prototype passed **20 CPU tensor tests in 5.18 s**.
Ruff check and format check passed for its implementation and tests. No model
was loaded or called, no real sample was loaded, and no optimizer, SSH,
download or accelerator computation was used. This follows the user's updated
instruction to prioritize model visual utilization while avoiding heavy work.

Design frozen before checks:
`reports/training/m2-smolvla-visual-pair-objective-design-2026-09-09.md`.
Implementation: `src/rosetta_reality/vla/visual_pair_loss.py`.
Tests: `tests/test_smolvla_visual_pair_loss.py`.
Evidence: `runs/smolvla-visual-repair-20260909-002/`.

## What the checks establish

For a synthetic pair with equal state/instruction/pure-noise input and distinct
expert actions, a state-only prediction leaves positive paired error. Changing
its common state bias has zero derivative for that paired term. Adding a toy
visual feature gives a gradient toward the correct action difference; reversing
its direction increases loss, and exact target velocities make both terms zero.
A common prediction offset is caught by the anchor term even if the predicted
difference is correct.

The tests also reject partial-noise action-input shortcuts and differences in
nonvisual conditioning, reject out-of-scope episode IDs and repeated images,
exclude padding from supervision, preserve velocity gradients without training
the labels/noise, and stop on non-finite arithmetic. The explicit zero-weight
control preserves the same positive-target anchor loss.

These are algebraic checks on four-scalar action tensors. The toy visual
feature was constructed from toy labels for an exact test; no empirical claim
about actual images, attention, learned representations or task success follows.

## What is actually new, and what remains uncertain

With equally valid coordinates, let `e_left` and `e_right` be the two velocity
errors, `c = (e_left + e_right)/2` and `d = e_left - e_right`. Then:

```text
ordinary paired MSE = mean(c^2 + d^2/4)
proposed objective  = mean(c^2 + (1/4 + pair_weight)*d^2)
```

The additional term reweights within-pair error differences; it does not create
new information beyond the two correct labels. The crucial constraint is the
verified pair construction and identical full-noise/nonvisual inputs. Ordinary
anchored training on those pairs is therefore the necessary zero-weight control.
This is a candidate to investigate, not evidence that another loss term fixes
the real model. Increasing noisy or uninformative target differences could
instead amplify bad supervision.

The next local implementation step is a train-only pair builder with actual
sample/processor/normalization/Action Contract identities, plus a model-facing
integration that verifies both forwards are comparable. Neither is wired into
the trainer in this phase. The builder must fail on absent exact matches rather
than quietly relaxing state equality or inventing image-action pairs. Actual
pair availability, full-model gradient behavior and trained-policy improvement
remain **not measured**. Formal training and large inference remain deferred.

## Runtime and retained state

The existing digest-pinned Linux container ran through WSL Bash with one CPU,
1 GiB memory and memory+swap, no network, no GPU device/library passthrough and
no model/dataset mounts. PyTorch emitted `UserWarning: XPU device count is zero!`
during scalar backward, consistent with the CPU-only configuration; all tests
passed. The warning is retained in `unit-tests-001.log`.

Source snapshots, pre-check hashes, the exact container command, test log and
JUnit results are retained. The JSON companion binds their identities.
Historical inference results and registrations are unchanged. The cache wrapper
was not expanded. No training config, existing trainer feature, optimizer or
Gate protocol changed; no commit or push was made. M2 is still incomplete.
