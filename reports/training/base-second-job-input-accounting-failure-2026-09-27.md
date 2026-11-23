# Base second-job input accounting failure — 2026-09-27

Execution identity `three-round-base-20260927-004` completed `init` and `doctor`
with return code 0. The actual CUDA doctor and simulator supplement report
`passed`. It then exited during `prepare` with the supervisor's `OSError`,
without reaching artifact `check` or policy `measure`. Exp1 remains unmeasured;
no task outcome can be assigned to any of the five base scenes.

The accounting failure is supported by the actual prepared artifact and source
assertion. `base-artifact/pretrained_model/model.safetensors` is a symlink to the
previously verified pinned-base cache. Its dereferenced size is **906,712,520
bytes**, while the entire job's physical `du` observation is only **92 KiB**.
The registered generated-output cap is 256 MiB (268,435,456 bytes).
`scripts/run_round_experiment_job.py` uses `p.stat().st_size` on files under the
job, following that cached-input symlink. This necessarily counts the existing
base payload as new output and exceeds the cap immediately after preparation.
The prepared manifest exists, but `prepare.log` is empty and there is no
`prepare-exited.json`; the supervisor stopped the owned process before advancing.

The repair must preserve the output safety cap. A separate, create-only,
hash-verified input-artifact directory outside the job's generated-output tree
avoids classifying existing input weights as new evidence. It must remain fully
SHA-bound and may not overwrite the old artifact or cached weights. Alternatively
any cache accounting exclusion requires explicit pre-existing-input identity;
blanket symlink skipping or increasing the evidence cap is not an equivalent
repair. The next execution must use a fresh job identity, preserving 004.

The base reviewer independently received the closed job's ordinary files only,
excluding all artifact symlinks and their targets. All **24/24** small returned
files match fresh remote per-file SHA256 observations. Small-archive SHA256 is
`bcc1a54c0ccf8cb83a5c53710510ef37e30e1c75806cfdc8e68626e24d1b2bf9`.
Evidence is `.work/base-run-004-independent-received/extracted/`.

Worker and guard receipts both report `owned_group_empty` at the prepare stage
and `shutdown_requested=false`; the guard closed after observing worker exit.
No policy episode or optimizer update ran. These are job lifecycle receipts,
not independent platform power/billing-state proof, and do not authorize power-off
while the requested complete experiment round remains unfinished.
