# TorchLens / Basin — stage 1, 2026-09-23

## Delivered

Basin owns the optional `basin/torchlens_backend.py` API `capture_synthetic`.
It performs actual TorchLens capture and emits the existing native envelope;
Basin import, query, evidence lookup and SHA verification work without changing
the core CLI/MCP protocol. Optional dependencies are hash-pinned separately.
Core Basin imports remain free of torch/TorchLens; optional tests are isolated
under `optional_tests/`. Rosetta supplies the bounded container runner and
independent acceptance experiments. Its trainer and processors are unchanged.

Both repositories remain on existing feature branches. No commit or push was
performed, no existing history was appended, and prior dirty work was preserved.
Moving the Rosetta collector to use Basin's backend is a later stage; this stage
establishes the working Basin backend and its evidence interface.

## Actual verification

- Basin core plus new optional backend: **36 passed**, 18.57 seconds.
- Rosetta collector, timeout, layered acceptance and Basin CLI/MCP contracts:
  **10 passed**, 61.38 seconds. These tests include expected strict rejection.
- Actual supervisor produced an incomplete native envelope and nonzero exit
  for strict parity failure; it did not declare acceptance.
- Fixed existing image, 2 CPU, 4 GiB, no extra swap, no network, 180-second cap.
  No real model weights, dataset, CUDA inference, training or rollout executed.

## Acceptance rationale and measured results

Three synthetic cases each used two plain runs, TorchLens and an independent
PyTorch-hook reference. Model seed 1729, forward seed 2718; no post-execution
RNG restoration. The hook reference preserved full output/state/RNG in all cases.

| Model randomness | Full output/state | Six selected activations | Python RNG after | Interpretation |
|---|---|---|---|---|
| torch | exact | exact | changed | usable for this isolated fixed-seed case |
| NumPy + torch | exact | exact | changed | usable for this isolated fixed-seed case |
| Python random + torch | output differs | differs | changed | capture not equivalent |

Torch/NumPy RNG after execution matched in all three cases. Strict transparency
failed in all three. Default TorchLens seed behavior additionally changed output;
the failure evidence is retained. Thus RNG checking is justified for transparent
integration, but its failure alone does not invalidate every isolated diagnostic.
Graph completeness remains unverified and upstream provenance warnings are saved.

Basin `status=complete` means capture finished, **not** equivalence: its backend
sets `consistency=not_assessed` and `live_integration_accepted=false`. Independent
comparison is mandatory before using a capture as equivalent model evidence.

## Evidence and limits

- `.cache/basin-torchlens-stage1-final/`: full Basin regression and retained capture/history.
- `.cache/torchlens-tests-007/`: ten collector/contract checks.
- `.cache/torchlens-acceptance-001/audit/acceptance-audit.json`: three-case audit with file hashes.
- `.cache/torchlens-capture-001/capture/`: actual supervisor rejection and Basin envelope.
- `.cache/torchlens-tests-001` through `006`: preserved dependency/API/parity failures and fixes.
- `configs/diagnostics/`: reviewed dependency provenance; `requirements/torchlens-diagnostics.txt`.
- `docs/torchlens-basin-diagnostics.md`: runtime, interface and replacement boundaries.

Docker's existing data directory was restored to the user's D drive after
explicit approval. Broken runtime socket directories were preserved by reversible
renaming; neither VHDX was deleted or overwritten. No image was downloaded.
This is infrastructure recovery, not an experiment result.

Real SmolVLA/CUDA compatibility, arbitrary RNG-state preservation, generic live
capture, full graph correctness and M2 success remain unverified. Stop after this
stage as requested; no further migration or external compute is launched.
