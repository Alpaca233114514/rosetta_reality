# GPU audit 006: actual device-specific image execution

Run 005 passed native temporal state/action ingress checks but stopped before
optimizer because its image reference used CPU float division while the pinned
native make_dataset returns uint8, Accelerate moves the batch to CUDA, and the
native loop converts to float and divides by 255 on CUDA. Its complete failure
bundle is retained. Successful optimizer steps remain zero across prior audits.

Independent probes establish exact raw-cache/train-view state, action and video
agreement at all eight samples; the saved native processor preserves images
exactly on CPU. Comparing CPU versus CUDA scaling of those same bytes yields
maximum absolute float32 difference 5.960464477539063e-08; all original pixel
bytes roundtrip exactly. This is an execution-device rounding effect, not proof
of wrong frames. Do not assume its model effect is negligible without measuring.

Run 006 compares actual ingress against independent raw uint8 normalized with
the same CUDA operation, keeping zero tolerance and explicit byte roundtrip.
It remeasures native versus masked-camera-skip exact loss/all gradients/action
on native CUDA-scaled inputs. Three additional no-optimizer forward/backward
and denoising comparisons measure CPU-scaling effects on loss, gradient hashes
and full action error separately, without weakening the native/skip assertions.
Reload input identity explicitly includes the CUDA scaling recipe.

The same eight-frame schedule, two total optimizer updates, original noise,
fixed Iris control endpoint and Gate thresholds, 2700-second deadline and all
memory limits apply. No formal training, selection or hidden cohort is added.
