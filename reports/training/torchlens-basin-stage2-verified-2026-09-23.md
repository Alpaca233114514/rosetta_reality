# Basin TorchLens bridge — stage 2 verification

The isolated synthetic bridge is verified. This closes the runtime-validation
gap recorded in `torchlens-basin-stage2-2026-09-23.md`; that earlier blocked
record is retained unchanged. It does not accept transparent live integration.

## Measured result

- `tests/test_basin_torchlens_bridge.py`: **4 passed in 22.38 seconds**.
- Ruff on the four relevant Python files: passed; `git diff --check`: passed.
- Rosetta's explicit `--backend basin-torchlens` child called Basin's real
  optional backend. The parent/core imports remain independent; lazy-import
  verification confirms importing the bridge loads neither Basin nor torch.
- Full output/state and six selected activations match their independent
  controls in the registered normal synthetic case. The independent hook
  reference is transparent. Python RNG still differs after capture, so strict
  acceptance remains false, supervisor exits nonzero and final status is incomplete.
- Raw Basin capture SHA is bound in the supervisor receipt. The final envelope
  was imported into a new disposable history and verified/queried successfully,
  retaining incomplete status and an empty Gate claim.

## Failure-state correction

The first integration run passed two tests but exposed an incorrect assertion
in the nonfinite case. It required Basin itself to fail when the model introduced
infinity *after* the selected head output. Basin had correctly captured finite
selected layers; Rosetta then correctly rejected the nonfinite complete output.

The final contract distinguishes these scopes. A forward exception retains both
backend and enclosing-worker incomplete records. A downstream nonfinite result
retains the completed, explicitly unassessed layer capture plus the enclosing
worker's incomplete failure record. No final successful worker artifact is emitted.
Failure tests now check the expected exception type and diagnostic text, preventing
dependency/import errors from being mistaken for successful negative tests.

## Evidence

- `.cache/basin-torchlens-stage2-verified-001/`: preserved first integration result.
- `.cache/basin-torchlens-stage2-verified-002/`: final logs, native envelopes,
  raw capture, independent activation assessment, test history and source hashes.
- `.cache/basin-torchlens-stage2-lint-003/`: final static check.
- The JSON companion binds current implementation and verification artifacts.

The existing D-drive image was available after the user started Docker. This
stage made no Docker settings changes and downloaded no images. Execution used
the pinned existing image, offline reviewed wheel overlay, 2 CPU, 4 GiB and the
180-second cap. No real weights/data, CUDA inference, training or simulation ran.
No historical store was appended; no commit or push was performed. Basin source
and the existing trainer/processors were not changed in this verification stage.

The bridge remains explicit and synthetic-only. Original diagnostic defaults
remain available. Python-random model computation and graph completeness retain
the stage-1 limitations; successful evidence transport is not model/Gate success.
