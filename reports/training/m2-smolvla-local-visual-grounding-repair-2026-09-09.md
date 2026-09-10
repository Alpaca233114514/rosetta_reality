# Local visual-grounding repair and inference reuse — 2026-09-09

Local code repairs and bounded inference checks completed on branch
`codex/smolvla-visual-conditioning-repair-20260909`. Learned visual grounding
is **not resolved**. This work does not establish a training improvement or
change any historical Gate result. The latest vfunfreeze-003 export is absent
from the local artifact store; its new temporal diagnosis is **not measured**.

The two existing local controls do use image information. On this small train
slice, vcdropout is more image-sensitive than Zen in all 16 offset/noise groups,
but its correct-image joint MAE is worse in all 16 groups. Gripper MAE is worse
in 15 of 16 groups. Increasing image sensitivity alone is therefore an
insufficient repair target for these controls. This is a comparison of two
selected artifacts, not a controlled causal attribution to their training axis.

## Concrete repairs

- `src/rosetta_reality/vla/vision_front_end.py` now checks the complete
  non-visual VLM complement before enabling the declared tower/connector
  scope. The old name filter missed `language_model`, token embeddings and
  `vlm.lm_head`, and could report a frozen language model vacuously. Empty
  scopes, missing complements and parameter aliases fail before mutation.
- The same module checks both checkpoint wrappers before installing either.
  A connector conflict no longer leaves the tower partially wrapped.
- `src/rosetta_reality/vla/visual_grounding.py` and
  `scripts/diagnose_visual_grounding.py` add train-only temporal paired
  inference, with per-call image/range/mask checks, matched noise, sample and
  artifact identity checks, unit-separated gains and a persistence baseline.
- `src/rosetta_reality/vla/inference_image_cache.py` and
  `scripts/diagnose_visual_grounding_cached.py` provide an opt-in bounded
  embedding cache for this frozen diagnostic, with exact first-action parity
  against a completed uncached reference. The cache is not connected to the
  trainer, export, selection or Gate execution paths.

The retained pre-edit scope implementation reproduces two defects: an unfrozen
VLM output head is accepted while `language_model_stays_frozen=true`; an
out-of-scope alias raises only after trainability flags have changed. The
repaired implementation rejects both without changing incoming flags. This
synthetic evidence does not invalidate the historical independent verification
that vfunfreeze updated 198 visual tensors and no language tensors. Future
training must bind the repaired source hashes in a new authorized plan.

## Registered local evidence

Plans, each written before its corresponding real inference:

- `reports/training/m2-smolvla-local-visual-grounding-preregistration-2026-09-09.json`
- `reports/training/m2-smolvla-local-image-cache-preregistration-2026-09-09.json`

Five train episodes: 49, 4, 23, 43, 21. Four offsets: 0, 100, 250, 450.
Each correct/cyclic-mismatched image pair keeps destination state, task,
target and diffusion noise fixed. Noise settings are zero plus Gaussian seeds
20260905, 20260906 and 20260907. No validation or hidden samples, optimizer,
training, simulation, downloads or SSH were used. Actual 20-frame decoding
was checked against filtered Arrow rows before weights were loaded.

The existing Linux/XPU image was pinned to
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`.
Execution used WSL Bash into Docker, networking disabled, batch one, memory
and memory+swap both capped at 6 GB, and two CPUs. Environment preflight found
Python 3.12.3, PyTorch 2.11.0+xpu and Intel Graphics XPU. Docker startup
required reversible preservation of stale runtime socket directories; no
retained Docker data disk, model cache or checkpoint was deleted.

Cache inspection validated nine files totaling 91,364,026 bytes. Ingress
checks passed on every sample: one real RGB camera and two masked placeholders,
each prepared as `[1, 3, 512, 512]`, with masks `[true, false, false]` and
finite SigLIP inputs in `[-1, 1]`.

The full JSON companion includes every offset/noise result and physical-unit
group, including negative paired gains. Positive gain means that the correct
image has lower MAE than the swapped image for the same destination and noise.
The following table is only the zero-noise slice; it is not a ranking gate.

| Artifact | Offset | Joint MAE (rad) | Joint paired gain (rad) | Gripper MAE (normalized) | Gripper paired gain |
|---|---:|---:|---:|---:|---:|
| zen | 0 | 0.014260 | 0.000416 | 0.012345 | -0.000045 |
| zen | 100 | 0.034804 | 0.007026 | 0.065665 | -0.003542 |
| zen | 250 | 0.042180 | -0.000395 | 0.040671 | 0.017592 |
| zen | 450 | 0.020384 | 0.004179 | 0.008379 | -0.001795 |
| vcdropout | 0 | 0.019073 | 0.001456 | 0.041167 | 0.000167 |
| vcdropout | 100 | 0.039692 | 0.031594 | 0.110802 | -0.010526 |
| vcdropout | 250 | 0.052743 | 0.008713 | 0.048933 | 0.044797 |
| vcdropout | 450 | 0.035925 | 0.009218 | 0.010732 | 0.001669 |

At frame 100 the mixed-unit persistence baseline MAE is 0.006777, versus
0.039213 for Zen and 0.049851 for vcdropout with zero noise. This highlights
low-action-change frames and the need to measure beyond static frame-zero
selection. It is not evidence that a persistence controller can solve insertion.
Frame-zero states are identical across these episodes, making state shuffling
degenerate there. No separate state-shuffle intervention was run.

## Measured acceleration

The unchanged Zen diagnostic was repeated with the image cache:

| Measurement | Uncached | Cached |
|---|---:|---:|
| Policy forward calls | 160 | 160 |
| Image encoder calls | 480 | 21 |
| Cache hits | 0 | 459 |
| Timed model-loading/inference stage | 213.605 s | 153.598 s |

All 2,240 recorded first-action scalars (160 calls × 14 dimensions) match
exactly; maximum absolute difference is **0**. Cache peak is **2,580,480 bytes**
against its 64 MiB budget. Actual placeholder embeddings are reused as-is.
There is no zero-substitution or mask change. The observed speedup is **1.391×**
(**28.1%** less elapsed time). One sequential comparison includes model load
and host variation; it does not establish steady-state throughput. Full chunks
were checked for finiteness, but exact parity covers recorded first actions.

This supports using the cache for further preregistered frozen diagnostics.
It provides no training-speed or closed-loop performance claim. A trainable
visual front end must continue using the native online processor and cannot
reuse these frozen embeddings for fine-tuning.

## Verification and retained failures

- 68 focused synthetic tests passed in 14.02 s in the pinned container.
- One real-cache data test passed in 11.61 s, checking all 20 registered frames.
- Retained-code before/after reproduction passed after its import entrypoint
  was corrected. The initial `ModuleNotFoundError: No module named
  'rosetta_reality.vla.vision_front_end'` remains in the first log; the 68 tests
  had passed before that separate script failed.
- Ruff check and format check passed for all nine affected Python files.
- 320 uncached real-model forwards completed, followed by 160 cached forwards.
  Zen took 213.605 s and vcdropout 190.586 s in their timed stages.
- TorchCodec could not load its shared libraries. The existing LeRobot runtime
  selected PyAV and the data test and all inference checks passed. Full errors
  remain in the logs; dependencies were not changed.

Evidence root: `runs/smolvla-visual-repair-20260909-001/`. The JSON companion
binds the raw reports, logs, registrations and source identities by SHA-256.
Original working-tree edits and the staged patch were preserved; no commit
or push was made. Existing historical reports and hash-bound plans were not
rewritten.

## Next efficient step and limits

Use the same bounded, cached temporal diagnostic on the actual vfunfreeze-003
export before proposing another furnace. The required artifact is
`m2-smolvla450m-vfunfreeze-cuda-b32-003-step0632-deploy-001`. Its September 6
closure records a remote durable location and a deliberate no-transfer traffic
agreement; a current local file search found no matching export. Continuing
that comparison requires an existing local copy or explicit authorization to
use SSH to verify and retrieve only that export and its manifests. Compute can
remain local. No remote state or availability has been assumed.

Any follow-up must register the new artifact checksum and output identity;
the completed Zen reference cannot be used as a parity baseline for different
weights. Further training would need a separate single-axis plan and real
preflight. Phase/noise-aware diagnostics inform that plan without modifying
historical selection rules or treating the train slice as validation.

The slice is small and in training; cross-episode swaps away from reset can
conflict with robot pose. Image sensitivity, target alignment and task success
remain separate claims. M2 remains incomplete. Latest-candidate temporal
grounding and learned-policy improvement are **not measured** in this work.

