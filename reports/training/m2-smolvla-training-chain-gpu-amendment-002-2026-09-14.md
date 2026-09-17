# GPU audit 002: explicit native batch ingress

Run 001 passed doctor, benchmark and batch-1 native preflight, then the NEW audit
collector supplied an unbatched `[50,14]` action to a policy expecting
`[batch,50,14]`. The saved AddBatchDimension processor handles observations but
does not repair this action shape. The failure was in the new diagnostic harness,
before an optimizer update; it is not evidence of a historical training bug.

All 15 failure-bundle members were recovered and hash-verified. Run 001 remains
immutable and no optimizer state is reused. The user explicitly requested
continuation. New run `training-chain-gpu-audit-20260914-002` applies native
`default_collate([sample])` BEFORE the saved processor and asserts full action
and padding batch shapes. No loss, gradient, RNG, inference or Gate threshold changes.

All other bounds and tests remain as registered in
`m2-smolvla-training-chain-gpu-plan-2026-09-14.md`: exactly two total real smoke
updates in this new run (prior run zero), 2700-second deadline, 300-second handoff,
8 GiB process-tree RSS/allocated and 10 GiB reserved CUDA memory, fixed Iris
control Gate 3/4 only after earlier checks pass. A new source/template identity
is mandatory. Preserve the first failure and protected shutdown lifecycle.
