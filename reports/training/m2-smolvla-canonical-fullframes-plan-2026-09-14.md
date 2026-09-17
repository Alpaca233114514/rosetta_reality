# Canonical full-frame baseline 001

User explicitly authorized one new fresh furnace using the repaired training chain.
Run: `canonical-fullframes-20260914-001`. No resume, downloads, dependency changes,
hidden data, automatic retry, deletion, commit or push. Existing GPU worker only.

Fresh pinned SmolVLA base and unchanged Action Contract, train-only normalization,
native uniform flow loss, frozen VLM, trainable expert/projections, AdamW 1e-4,
betas 0.9/0.95, epsilon 1e-8, weight decay 1e-10, global clip 10, BF16, batch 4.
Keep warmup 16; explicitly extend cosine decay to 5000 updates and 2.5e-6 endpoint.
All 40 registered training episodes, all 500 frames each, appear exactly once in
a sealed Python Random(20260809) shuffled schedule: 20000 exposures and unique
inputs, 5000 successful updates. This new coverage/budget baseline is not an
isolated causal comparison with historical frame-zero work.

Reuse the tested bounded temporal interface (`run_smolvla_v2.py smoke`) as Hestia
did, with an explicit 5000-update permit; the CLI mode name does not imply a
two-step experiment. The native loop/optimizer remain unchanged. The plan's
active smoke block and training block both declare the full budget; no formal
resume or validation-only selection is claimed. Existing formal-mode gates
are unchanged. Fixed endpoint 5000, no checkpoint search.

Canonical pixel feature and native observer were verified in run 007 (105 tests,
real CPU/CUDA model equality, two actual updates and independent reload). New
composition must pass regression, doctor, benchmark, batch-1 no-optimizer forward
and fresh native/skip/canonical parity before updates. Every actual batch retains
its incoming raw tensor reference for policy-ingress image/state/action/padding
comparison, with successful update counting and all finite gradients checked.
This reference proves processor execution, not independent video physical timing.

Save complete checkpoints at 2500 and 5000; retain both. Minimum initial free
disk 5 GiB for two approximately 1.50 GiB recovery states, one transient save
and headroom; no full-copy export on this disk. CUDA allocated/RSS <=8 GiB,
reserved <=10 GiB. One 7200-second deadline (expected 30–90 minutes including
checks), followed by a 300-second result-transfer grace and protected shutdown.
Stop on nonfinite, identity/coverage/input mismatch, OOM or resource failure.
Sample stable training at five-minute intervals; save exact checkpoint metrics.

Initial new endpoint reload uses eight temporal train samples and two fixed
noise conditions in separate model processes; preserve all full arrays. This
is bounded serialization verification, not developmental quality. Offline
train/dev evaluation and the unchanged Gate 3/4 must use a separately sealed
post-training endpoint after completion and backup. No M2 claim from training.
