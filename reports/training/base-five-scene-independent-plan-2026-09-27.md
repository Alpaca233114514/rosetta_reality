# Base independent five-scene measurement — 2026-09-27

This new measurement implements the user's latest instruction to complete
Exp1 independently of task-success prerequisites. It measures the pinned base
under the saved canonical ALOHA interface, with zero optimizer updates. It does
not change the formal Gate engine or manufacture a passed Gate3. The historical
base Gate3 remains failed (9 joint-limit counts and one cross-arm contact), and
historical formal Gate4 remains not measured. The existing trained result is
read-only context and is not rerun.

The executable registration is
`configs/diagnostics/base-five-scene-independent-20260927-002.json`.
The earlier 001 registration is retained as a preparation draft and must not run.
The entrypoint is `scripts/diagnose_smolvla_base_five_scene.py`.

The seeds are 1000–1004, each with the same policy-noise seed and at most 500
environment steps. Instruction, seeded Gaussian generation, BF16 CUDA loader,
saved config/processors/train-only normalization, bounded gripper decoding,
Action Contract projection and receding-horizon execution of slot0 retain the
original protocol. The pinned model revision is
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`, weight SHA256 is
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
Historical interface manifest SHA256 is
`6c35ef6e294ba50dae1e1961b042eaf1f42e988d48f876009da122be3d61cc77`.

Each episode ends immediately after an unexpected contact or physical joint
limit violation, then the next independent seed runs. An unsafe reset ends that
episode before any command. Task failure also does not skip later seeds.
Nonfinite observations/actions/rewards, shape or identity failures, extra
adapter clipping, infrastructure faults and resource failures stop the batch.
No success is imputed for an unfinished seed; partial trace files remain.
Since safety-censored episodes differ from the complete formal Gate4 protocol,
`original_gate4_protocol_equivalent=false` and `formal_gate_result=null` always.
Measured task-success fractions are diagnostic descriptions and cannot qualify
the base as safe or accepted for M2.

The process has an 1800-second work limit, 12 GiB RSS, 8/10 GiB CUDA allocated/
reserved caps and 256 MiB evidence cap. The integration supervisor must launch it
as owned work under a separate bounded work-only guard (suggested cap 2100
seconds, including final tensor hash capture), preserve output, and enforce
global experiment closeout. This entrypoint contains no power-off path. It must
not schedule a platform shutdown on Exp1 completion or failure.

The live AutoDL RTX4090D runtime and payload still require current verification.
Only the registered AutoDL platform-container exception is allowed; no nested
Docker. Use `configs/runtime/autodl_rtx4090.yaml` and `scripts/run_autodl.sh` to
record runtime identity. The historical loader supplies library/GPU metadata.
Library/runtime doctor checks remain the integration supervisor's responsibility.
No new dependency, download, model execution or simulator execution has been
performed by the implementation subtask.

Preparation first verifies all source payload SHA256s, then creates a new
artifact directory using symlinks. It preserves the exact historical manifest
bytes, with metadata from the historical received evidence and processor payload
from the current canonical artifact. Missing or mismatched files stop preparation;
the manifest is never edited to accommodate a changed source.

```bash
# Run in the registered Linux runtime, not Windows Python.
python -m unittest discover -s tests -p test_base_five_scene_diagnostic.py -v

# Set these to verified runtime paths; every destination must be new.
python scripts/diagnose_smolvla_base_five_scene.py \
  --prepare-artifact --artifact "$NEW_BASE_ARTIFACT" \
  --canonical-artifact "$VERIFIED_CANONICAL_ARTIFACT" \
  --base-weights "$PINNED_BASE_WEIGHTS"

python scripts/diagnose_smolvla_base_five_scene.py \
  --plan configs/diagnostics/base-five-scene-independent-20260927-002.json \
  --artifact "$NEW_BASE_ARTIFACT" --check-only

# Supervised, offline CUDA measurement; this command never shuts down.
python scripts/diagnose_smolvla_base_five_scene.py \
  --plan configs/diagnostics/base-five-scene-independent-20260927-002.json \
  --artifact "$NEW_BASE_ARTIFACT" --output "$NEW_BASE_DIAGNOSTIC_OUTPUT"
```

Stage the historical metadata directory
`runs/prepost-gate-received-20260924-001/prepost-gate4-20260924-001/artifacts/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/prepost-base-aloha-interface-001`
with the executable sources. The helper needs its manifest, config,
normalization and Action Contract JSON files; model and processor payload stay
in the verified remote cache. Source identity includes the diagnostic, tests,
original simulation engine, CUDA wrapper, adapter, action-space and processor.

Output includes per-seed JSON and JSONL, a complete denominator summary,
partial/unmeasured seed distinctions, resource/identity exceptions, per-step
executed and preprojection actions, robot states, rewards, task success, safety
counts/contact pairs and latency. Full policy parameter/buffer content hashes
are captured before and after to verify zero-update integrity. This trace does
not capture every image/noise/internal activation or full MuJoCo integration
state, and cannot establish exact paired trajectory equivalence with the old
trained run. Observer transparency and remote test execution remain unverified.

Ten tests cover independent task/safety continuation, infrastructure stops,
create-only behavior, formal-result separation, artifact identity enforcement,
one-step joint-limit/contact stops, reset stops, pre-step nonfinite refusal,
registered action projection and reward-versus-success separation. The tests
use fake physics with torch tensors; they do not construct a model or real
simulator. Execute them in the authorized Linux ML runtime before dispatch.
