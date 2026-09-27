# TorchLens → Basin bounded diagnostics

This is an opt-in, **synthetic CPU only** collector. It does not change the
fixed LeRobot revision `c903b114a90e703b3f7d0c46cb38727c328c55ff`, SmolVLA,
processors, training entrypoints, action contracts or Gate results.

Stage 2 adds explicit `--backend basin-torchlens` to the synthetic CLI. The
supervisor stays dependency-light; only its disposable integration worker imports
Basin's optional backend and passes the constructed synthetic model. Neither
project's core imports the other. The original `torchlens` backend remains the
historical default/control. Parent/worker and Basin-store boundaries remain JSON.
Raw Basin capture is retained under `observed/basin-raw/`, separately from the
Rosetta output/state/RNG comparison and final complete/incomplete envelope.

```bash
bash scripts/run_torchlens_synthetic.sh .cache/basin-bridge-new \
  scripts/collect_torchlens_synthetic.py --backend basin-torchlens \
  --output /output/capture --run-id basin-bridge-new
```

Strict RNG mismatch still exits nonzero and preserves an incomplete final
envelope. Do not import raw `status=complete` as equivalent-execution proof;
it only means Basin finished capture. A real-model path is not exposed.
Selected-layer completion is also distinct from full-output validity: if a
nonfinite value is introduced after the selected head layer, Basin's finite
layer capture remains complete while Rosetta's full-output check rejects the
worker. Both pieces of evidence remain; no successful layer capture overrides
the failed model-output validation.

**Measured compatibility result:** TorchLens 2.23.0 trace fails strict RNG
parity. Explicit identical forward seeds recover exact output/state and
torch/NumPy RNG, but Python RNG still advances during trace bookkeeping.
The collector rejects acceptance and exports an **incomplete** Basin unit.
Tests proving this rejection are not a successful compatibility result.

## Why acceptance is layered

Strict transparency is necessary before replacing a collector inside an existing
training/rollout process: changing any global RNG can change later sampling,
augmentation or model calls. It is deliberately stricter than a single output
comparison. It is not a universal definition of useful offline instrumentation.

For a disposable, fixed-input/fixed-seed process, Python RNG changes that happen
only after computation cannot escape to the parent process. Such a capture may
still contain useful evidence, but this must be measured rather than assumed.
`scripts/audit_torchlens_acceptance.py` therefore runs three cases (torch noise,
Python noise and NumPy noise), each with two plain arms, TorchLens and an
independent PyTorch module-hook reference. It checks six selected module outputs
including both shared-module calls, alongside full output/state and RNG.

The audit reports `isolated_case_evidence_usable` independently from
`strict_transparency.passed`. Its own `status=complete` means the audit finished,
not that TorchLens passed. Live integration remains unaccepted in all cases.
Graph completeness is a separate, unverified property; warnings are retained.
Changing acceptance scope never rewrites the original strict-parity evidence.

```bash
bash scripts/run_torchlens_synthetic.sh .cache/torchlens-acceptance-001 \
  scripts/audit_torchlens_acceptance.py --output /output/audit
```

## Dependency and runtime boundary

`requirements/torchlens-diagnostics.txt` pins the official TorchLens 2.23.0
wheel and its PyPI SHA256. The wheel declares Apache-2.0 and Python >=3.9;
base dependencies are torch >=2.4, numpy, packaging, safetensors >=0.4, tqdm,
graphviz, typing_extensions >=4.0 and pillow >=9. No extras are requested.
The reviewed wheel has no `.pth` or entry-point installation hooks. The fixed
image lacks the graphviz Python package, so the same lock includes graphviz
0.21 (MIT, no base dependencies). No system Graphviz executable is installed;
this stage exports graph metadata, not rendered diagrams. Installation
uses `--no-index --no-deps --require-hashes --target /tmp/overlay` inside the
disposable container; it cannot resolve or upgrade the training environment.
Missing prerequisites must be reported, not silently installed.

The inspected wheel warns that TorchLens wraps global torch functions. Each
control/repeat/observed arm therefore uses a fresh subprocess; nothing imports
TorchLens into the trainer. A predicate selects module outputs in a single
capture. The legacy selective-layer list API can run two forwards and is not
used. A mutable buffer independently rejects additional forward calls.

The WSL runner uses a fixed existing image, `--pull never`, no network, a
read-only source and Basin mount, 2 CPU, 4 GiB memory with no additional swap,
and a 180-second outer timeout. The collector checks effective cgroup limits
before importing ML dependencies. No model, dataset or live policy is loaded.

## Run and inspect

Place the hash-verified official wheel under `.work/torchlens/wheels/` (ignored
by the existing `wheels/` rule). From WSL Bash in the repository:

```bash
bash scripts/run_torchlens_synthetic.sh .cache/torchlens-capture-001 \
  scripts/collect_torchlens_synthetic.py --output /output/capture \
  --run-id torchlens-synthetic-001

bash scripts/run_torchlens_synthetic.sh .cache/torchlens-tests-001 \
  -m pytest -q -p no:cacheprovider tests/test_torchlens_capture.py \
  --basetemp=/output/pytest
```

Output directories must be new. A failed worker retains logs and an incomplete
`failure.json`. The supervisor retains each arm and its `receipt.json`; external
timeout/OOM is additionally recorded by the shell's `container-exit.json`.
After completed arms, `basin-native.json` preserves either a complete accepted
capture or an incomplete parity failure. Failure exits nonzero; downstream users
must check both `status` and `parameters.consistency.passed`. Comparison covers
complete nested outputs (container type, tensor shape/dtype/bytes), inputs,
parameters/buffers and Python/NumPy/CPU torch RNG before/after execution.
Both control arms and TorchLens explicitly use forward seed 2718 after model
construction (seed 1729). No RNG is restored after execution. Default TorchLens
seed behavior was also measured and failed output plus RNG parity; its evidence
is retained separately. Neither mode is suitable for transparent live wrapping.

## Basin interface

The collector emits Basin's existing `native envelope v1`. `source_run` groups
the arms; evidence is `synthetic`; `gate` stays empty. Step 0 denotes this one
forward, not a training update. Stages include module path, call index and op
index. Graph parents, shape and dtype exist for unsaved ops; their finite status
is null, not guessed. Selected finite tensors retain exact values and hashes.
Nested final output is a separate `model:output/call:1` event.

In the same bounded container, with Basin source on PYTHONPATH, import into a
**new disposable store**, never the existing canonical history:

```bash
python -m basin --store /output/new-history import-native \
  /input/capture/basin-native.json --id torchlens-synthetic-001
python -m basin --store /output/new-history verify torchlens-synthetic-001
python -m basin --store /output/new-history events torchlens-synthetic-001 --step 0
```

The JSON file boundary is the production interface. Only integration tests
import Basin directly. Existing CLI/Python/MCP tools can query imported units.
Basin's import hash means byte integrity at import, not independent scientific
certification; the parity fields are preserved collector evidence.

Each JSON is capped at 16 MiB, each selected tensor at 4096 elements and graph
export at 4096 ops. Overflow fails without truncation. The cgroup and timeout
bound upstream capture itself. Missing observations remain absent/null.

## Replacement decisions

| Existing responsibility | TorchLens candidate | Adoption boundary |
|---|---|---|
| Generic graph discovery and module metadata | Captured op DAG | New synthetic entry only |
| Generic activation capture | Selected module-output predicate | Requires exact parity |
| Repeated modules and nested output | Call-aware ops and full output evidence | Contract tests required |
| Input/sample ledger and completed optimizer updates | Not replaced | Existing observer retained |
| Action conversion and normalization | Not replaced | Existing processors retained |
| Noise, rollout physics and Gate evidence | Not replaced | Existing collectors retained |
| Artifact/checkpoint independent reload | Not replaced | Existing reload protocol retained |

No historical collector is removed. A real SmolVLA/CUDA trial needs a separate
registered checkpoint/input/noise/resource identity and execution authorization.
Synthetic parity does not establish that live integration is safe or that M2
has succeeded. See the dated implementation report for actual verification.
