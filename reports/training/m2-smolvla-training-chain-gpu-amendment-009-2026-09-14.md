# Canonical pixel repair: bounded run 007

The user asked to repair execution artifacts and rerun affected measurements.
Run 006 completed two real native updates and independent full-array reload;
it also showed that legacy CUDA division versus CPU division changes model
loss/gradients/actions away from frame zero. Old evidence is retained.

Add opt-in v2 feature canonical_image_scaling. It converts native uint8 camera
batches with one float32 lookup table defined from all 256 integer/255 values.
The native loop's uint8-only scaling branch then correctly skips redivision.
Frame metadata and input bytes remain unmodified; float/invalid input fails
closed. This makes training agree exactly with canonical CPU inference input.
Historical plans and installed upstream package files remain unchanged.

New run 007 starts again from the pinned base, with exactly two new diagnostic
updates (four successful updates across the two separate diagnostics if it
passes). No resume, formal furnace or selection. Rerun all affected native/skip
loss/gradient/action comparisons, REQUIRE CPU/CUDA canonical model equivalence,
actual eight-frame optimizer ingress, saved-processor reload, and provenance.
The checkpoint receives an explicit image-recipe sidecar before inventory.

The fixed Iris control Gate 3/4 remeasurement is already running under run 006.
Its immutable endpoint and CPU input recipe are unchanged by this opt-in training
feature; it evaluates the existing 1280-step control, not either two-step policy.
Do not repeat the identical Gate protocol solely for a training-only extension,
and never attribute that Gate outcome to the new smoke checkpoint. Run 007's
stage list therefore ends at verify-reload. All 2700-second deadline, memory,
no-download, evidence preservation and protected shutdown bounds remain.

Validate all 256 pixel values on CPU and CUDA, noncontiguous RGB inputs, byte
preservation, no double scaling, metadata identity and feature restoration.
CPU-only checks may run alongside the existing Gate; CUDA tests and the new
model run start only after the Gate worker exits.
