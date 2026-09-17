# Repaired training-chain CUDA verification — 2026-09-14

The user reopened the original GPU SSH window for the pending engineering tests.
Instance `44db45aec7-cd880eb6`, port 20497, RTX 4090 D GPU UUID
`GPU-746698db-880c-bc00-8da0-880cb3fc5b78` is idle at admission.
Use a new staged source identity, existing pinned environment/cache and
`configs/runtime/autodl_rtx4090.yaml` through `scripts/run_autodl.sh`.
No download, dependency change, formal furnace, hidden cohort, commit or push.

## Fixed scope

- Run `training-chain-gpu-audit-20260914-001`; seal local source/config files in
  its template before staging. Verify sources and GPU identity before dispatch.
- One 2700-second worker deadline, 300-second evidence-transfer grace, protected
  shutdown with the existing checked wrapper; retain checkpoints and all failures.
- Worker tree RSS and CUDA allocated <=8 GiB; CUDA reserved <=10 GiB. Require
  at least 4 GiB free durable disk. Stop on identity drift, OOM, nonfinite values,
  wrong input frames, failed numerical equality or missing reload evidence.
- Validate newly changed reload schema/tests and runner syntax before model work;
  reuse the prior 599 CPU checks only for source files that remain unchanged.
- Doctor and benchmark, then a batch-1 native no-optimizer preflight.
- Compare unchanged-base native versus repaired masked-camera skipping at train
  episode 49, frames 0/249/499, matching RNG/autocast. Require exact scalar loss,
  every present gradient tensor and complete predicted chunk; parameter hashes
  must remain unchanged. No optimizer in this comparison.
- Exactly TWO real native optimizer updates, batch 4, using the current v2 CLI
  and temporal feature. Samples in order: `(49,0),(4,0),(49,249),(4,249),
  (49,450),(4,450),(49,499),(4,499)`. Keep original base, AdamW, matched original
  scheduler recipe and all train-only normalization statistics. This is a smoke
  diagnostic, not an equal-budget learning comparison or formal training.
- Observe actual policy forward ingress, compare normalized targets/state with
  independent raw transformations, verify full images and tail masks, successful
  optimizer counts, immutable frozen tensors and step-2 checkpoint metrics.
- In two independent processes, load the new smoke checkpoint AND its saved
  processors; collect every full action/noise tensor for all eight samples under
  zero noise and seed 20260905, retaining 14 output and 32 internal noise dimensions.
  Require exact full-array equality. This does not implement formal resume.
- After those gates pass, remeasure the existing fixed Iris CONTROL endpoint
  (1280 updates, already independently backed up), not the two-step smoke policy.
  Use the unchanged Gate engine, fresh suffix 461, original thresholds, Gate 3
  followed by Gate 4 seeds 1000--1004 with 500 steps each. Record native-to-adapter
  full-chunk parity and unchanged weights. Preserve a measured Gate failure as a
  negative model result, distinct from an infrastructure exception. No reselection.

## Additional confirmed defect

Before modification, the new saved-array verifier rejected retained REAL arrays
with actions `[1,50,14]` and noise `[1,50,32]`. The read-only counterexample
returned `Full-chunk array shapes disagree`, with zero forwards/updates.
Add explicit `noise_action_dim` to the identity for unequal dimensions; preserve
equal-dimension v1 bundles and strict finite/shape/checksum checks. Never truncate
noise to force equality. Tests reject omitted/truncated/malformed noise dimensions.

Passing these bounded checks cannot prove that the entire implementation is free
of defects. Old frame-0 coverage remains a registered small-sample design, not a
proven accidental loss or unique cause of historical Gate failures. Formal resume,
physical image/label semantics and complete production-data coverage remain separate.
