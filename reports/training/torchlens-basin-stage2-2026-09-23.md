# Basin TorchLens isolated bridge — stage 2

Five-minute bounded stage on 2026-09-23: implementation added; runtime validation
blocked before any model or test executed. This is not an accepted migration.

## Implemented

- Synthetic CLI adds explicit `--backend basin-torchlens`; original defaults remain.
- `scripts/basin_torchlens_worker.py` imports Basin only inside the disposable
  integration worker. The parent and each project's core remain independent.
- Basin owns actual capture; raw output is retained in `observed/basin-raw/`.
  Rosetta records independent input/output/state/RNG and binds raw file SHA in
  the supervisor receipt. Strict mismatch still exports incomplete evidence and
  exits nonzero. No post-execution RNG restoration or relaxed comparison added.
- Three bridge test cases cover actual supervision plus independent six-activation
  comparison and import/query, forward exception, and nonfinite output.

## Verification boundary

Passed: AST parsing of all four touched Python files through read-only WSL,
and `git diff --check`. No torch/model import was used for static verification.

Not run: new bridge tests. Docker was stopped at entry; its saved configuration
still pointed to the existing D-drive data disk. Startup failed on a stale
Ingest socket before image inspection. A bounded recovery attempt refused to
move the runtime directory because its contents differed from the previous
allowlist; no directory was moved or deleted. Docker/backend processes were
stopped during that attempt. No VHDX or settings were changed in this stage.

The earlier stage's 36/10 passing counts do not validate this new bridge.
No new model, training, Gate, commit, push or existing-store append occurred.

## Resume

After separately restoring Docker, run:

```bash
bash scripts/run_torchlens_synthetic.sh .cache/basin-torchlens-stage2-001 \
  -m pytest -q -p no:cacheprovider tests/test_basin_torchlens_bridge.py \
  --basetemp=/output/pytest
```

The prior attempts failed before creating that output directory. Check it is
still absent before reuse; otherwise choose a new identity. Complete targeted
runtime checks and lint before calling the bridge accepted. Real SmolVLA and
training remain outside this phase.
