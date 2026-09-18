# SmolVLA post-training selection and tensor evidence v2

The frozen Zen/vcdropout/vfunfreeze selectors and exporters remain historical
provenance. New work uses the explicitly versioned entries below. These tools
perform no policy training, inference, checkpoint search outside a registered
grid, or Gate evaluation. Their output is not M2 acceptance.

## Selection

`scripts/select_smolvla_checkpoint_v2.py` consumes a new v2 training plan and a
SHA-bound receipt of all registered validation reports. The plan must bind the
current v2 core plus this selector and
`src/rosetta_reality/vla/checkpoint_selection.py` in `implementation_files`.
It also declares:

```yaml
selection_contract:
  primary_metric: first_action_mae
  tie_break: earlier_checkpoint
  base_role: reference_only
  fixed_input:
    noise: zeros
    flow_time: 0.5
```

The receipt contains `plan_sha256` and a `reports` list. Each item has a
repository-relative `path`, the actual report `sha256`, and `source` with
`kind` (`base` or `checkpoint`), integer `step` (base is 0), and `model_sha256`
from the retained checkpoint identity. It must cover base plus every step in
`training.checkpoint_steps`, exactly once.

Run through the existing offline WSL/Docker boundary:

```text
python scripts/select_smolvla_checkpoint_v2.py --plan NEW_PLAN.yaml
  --inputs INPUT_RECEIPT.json --inputs-sha256 RECEIPT_SHA256 --output NEW_SELECTION.json
```

The source report must have native `smolvla_fixed_validation` fields. The
selector checks parent/plan/model/data/normalization/Action Contract identities,
the exact registered validation episodes and offsets, materialized episodes,
sample count, noise, and no optimizer/gradients/hidden access. Every used metric
must be finite and nonnegative. Fractional/string/bool checkpoint IDs, duplicate
or missing reports and a different checkpoint's model digest are rejected.

Selection ranks checkpoints by `(first_action_mae, step)`. Secondary metrics do
not secretly break ties. The base is only a reference: a checkpoint worse than
base remains explicitly labeled `improves_over_base=false`. If base MAE is zero,
fractional improvement is `null` with `zero_base_metric`, never NaN or infinity.
Safety rates are retained as diagnostics; selection alone does not pass a
safety or task-success gate. Outputs are create-only and have a new v2 stage;
do not feed them into a frozen historical exporter by relabeling their schema.

## Full-array evidence

`src/rosetta_reality/vla/reload_evidence.py` supplies `write_bundle` and
`compare_bundles`. The collector calls `write_bundle` once in each actual
source/reload invocation with:

- `normalized_actions`, `standard_actions`, `noise`: complete floating arrays
  shaped `[sample, horizon, dimension]`, retaining every chunk slot;
- `sample_identities`: integer `[sample, 2]` episode/frame pairs;
- `valid_mask`: boolean `[sample, horizon]`, with valid entries preceding tail padding;
- `identity`: SHA-256 fields `plan_sha256`, `model_sha256`, `processor_sha256`,
  `action_contract_sha256`, `dataset_manifest_sha256`, `input_sha256`,
  `inference_recipe_sha256`, plus positive integer `sample_count`, `chunk_size`,
  `normalized_action_dim`, `standard_action_dim` from the registered contract.

`write_bundle` records a process-lifetime UUID and the full NPZ checksum. Caller
identities must be derived from the actual model/processor/inputs; they must not
be invented or inferred from matching shapes. The action contract digest binds
units, coordinate ordering and gripper semantics, separately from the array sizes.

```text
python scripts/verify_smolvla_reload_v2.py --first FIRST_BUNDLE --second RELOAD_BUNDLE
  --identity EXPECTED_IDENTITY.json --identity-sha256 IDENTITY_SHA256 --output NEW_PROOF.json
```

The verifier requires matching expected identities and complete registered
shapes, independent collection processes, valid checksums, identical dtypes,
and every array element equal. It rejects scalar-only reports and truncated
first-action bundles. A tensor mismatch saves negative evidence and returns 4;
existing output is never overwritten.

The proof's scope is **saved full arrays from distinct collection processes**.
It explicitly does not prove model execution or formal resume. An exporter must
also bind the actual independent policy/processor loading and execution evidence
before claiming model reload. Existing historical claims based only on seven
equal metrics cannot be upgraded by passing those metrics to this verifier.
No old artifact or historical selection is automatically regenerated.

## Historical protocol tests

Native SmolVLA noise can retain 32 internal dimensions while returned normalized
actions contain only 14 effective dimensions. Such collectors must explicitly
register `noise_action_dim: 32`; the verifier checks every noise entry separately
from `normalized_action_dim`. Existing equal-dimension bundles remain supported
when this optional field is absent. Never truncate noise to match output actions.

`tests/fixtures/smolvla_historical_sources/` contains ten exact, SHA-verified source
snapshots. Tests use them as data files in a temporary filesystem and continue to
execute the production hash validators. Tampered snapshot bytes and incompatible
current source must still be rejected. The fixtures are never executed as code.

Zen and vfunfreeze protocol fixtures are complete. Two original vcdropout dirty
source versions are absent locally. Its gate unit tests use an explicitly synthetic
temporary plan with real current hashes; a separate historical-plan rejection test
remains. This is not historical vcdropout reproduction or permission to resume it.

Formal resume remains disabled. These engineering repairs do not authorize a new
furnace, CUDA run, model download, or changed learning/selection experiment.
