# Experimental visual-pair objective — local design, 2026-09-09

Status: implementation prototype, not a training plan or learned-model fix.
The user asked to focus on visual utilization, continue local edits and avoid
heavy work. This phase permits source inspection and tiny CPU tensor checks.
No model forward, dataset read, optimizer, accelerator, SSH, download or formal
run is scheduled. The earlier SSH question is deferred by the newer instruction.

## Evidence and hypothesis

The September 9 comparison found greater image sensitivity but worse
correct-image joint MAE for vcdropout than Zen in all 16 groups. The vfunfreeze
closure separately demonstrated visual-weight updates without task success.
Neither implies a disconnected image path. Historical evidence is unchanged.

Static inspection of the pinned implementation confirms that the flow loss
uses `x_t = t*z + (1-t)*a`, targets `u_t = z-a`, and ordinary velocity MSE.
Images, language and state enter the prefix, and expert queries can attend to
visual context. The inspected image encoder/training forward does not detach
that path. The existing scaled Beta(1.5, 1.0) time sampler leaves some target
action information in `x_t` whenever `t < 1`.

This is expected flow-matching behavior, not an upstream bug or proof of the
real failure's cause. It motivates an auxiliary task in which the image is
the only changing conditioning input and the required action difference is
known, instead of treating greater image sensitivity as the objective.

| Pinned source | SHA-256 |
|---|---|
| `lerobot/policies/smolvla/modeling_smolvla.py` | `37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d` |
| `lerobot/policies/smolvla/smolvlm_with_expert.py` | `996d3b0c713c0ed42b383aa2cf89b2e6f9868e337747c480a64adaecdc1073cf` |

## Proposed objective

Pair verified train observations from distinct episodes at the same frame
offset, with exactly equal state and instruction, different images, and
sufficiently different expert actions on shared valid coordinates. Earlier
evidence suggests reset frames as a candidate pool; actual pair availability
and label quality are not measured here. Do not silently use approximately
matched poses or incompatible labels if exact matches are scarce.

Both observations receive the same independently sampled Gaussian noise `z`
at **t = 1**. The action input is then identical and contains no per-sample
target difference. Actions must be projected through the Action Contract and
normalized with the same train-only statistics:

```text
u_left = z - a_left; u_right = z - a_right
desired velocity difference = a_right - a_left

anchor = mean_valid((v - u)^2)
paired = mean_shared_valid(((v_left - v_right) - (a_right - a_left))^2)
loss = anchor + pair_weight * paired
```

Normalize each pair by its valid count, then average pairs. For a deterministic
state/noisy-action-only predictor, equal nonvisual inputs produce equal outputs;
it cannot eliminate the paired error for differing targets. The anchor term
penalizes common prediction error that a difference-only term cannot detect.
Both branches have correct positive targets. This is supervised action-difference
learning, without deliberately worsening a wrong-image prediction.

`src/rosetta_reality/vla/visual_pair_loss.py` implements the arithmetic and
its input contract. It rejects state/language/noise mismatches, partial noise,
non-training identities, different offsets, repeated images, absent shared
valid coordinates, insufficient target contrast and non-finite results. Labels
and supplied noise are detached; velocity gradients remain available.
It does not register a trainer feature or change a completed plan/config.

## Limits and next integration gate

- The helper cannot prove pixel/label provenance from supplied identifiers.
  A future train-only pair builder must bind actual sample, processor,
  normalization and Action Contract identities. Different hashes alone do not
  prove useful object-placement variation.
- Both forwards need identical nonvisual inputs and deterministic stochastic
  behavior. Independent dropout masks would confound the pair.
- Full-noise alignment may not survive the denoising trajectory or closed-loop
  distribution. Acceptance still requires correct-image absolute accuracy,
  matched-noise paired gain across phases, and the frozen task protocol.
- Both future comparison arms must use identical pairs, noise/time schedule,
  exposures, optimizer and model scope. Compare `pair_weight=0` with one
  preregistered positive weight. Changing both sampling and loss against
  historical Zen would not isolate a single variable.
- No state ablation, unfreezing or loss strength is tuned from validation or
  hidden samples. This prototype does not authorize real training.

## Lightweight verification scope fixed before execution

Use generated CPU tensors in the existing image
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`,
through WSL Bash, with one CPU, 1 GiB memory/swap, network disabled, no GPU
device/library passthrough and no model/dataset mounts. Each test action tensor
has four scalars. There is no model object or optimizer.

Checks cover a state-only solution, visual gradient direction, a reversed-vision
negative control, the common-error anchor, zero-weight control, a partial-noise
action-input shortcut, input/identity boundaries, padding, detached supervision
and finite arithmetic. The synthetic image feature is constructed from toy
labels for exact algebra; it is not real image-understanding evidence.

Stop on failed assertions or runtime/resource errors and retain the log before
amending the prototype. No failure authorizes a model run or OOM retry.
Results will be recorded separately after these checks.
