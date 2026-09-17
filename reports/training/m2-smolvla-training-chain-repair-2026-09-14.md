# SmolVLA training-chain follow-up repair — 2026-09-14

## Result

The identified follow-up code defects are repaired locally. The final related
regression is **569 passed, 0 failed, 1 skipped, 2 deselected**. A separate cached
dataset audit decoded all **22,500 nonhidden frame samples** and compared all
**22,500 complete 50-by-14 action chunks** with independently read raw labels
(15,750,000 scalar comparisons, including registered tail padding). No hidden
rows were materialized. This is bounded engineering verification, not a claim
that the full training/deployment system has no remaining defects.

This report supplements `m2-smolvla-training-chain-rescan-2026-09-14.{md,json}`;
that earlier evidence remains unchanged. The earlier six repaired contract gaps
and 16 reproduced counterexamples are included in the final regression.

## Repairs

1. `scripts/select_smolvla_checkpoint_v2.py` and
   `src/rosetta_reality/vla/checkpoint_selection.py` implement the declared
   earlier-checkpoint tie break. Tests retain counterexamples against unchanged
   historical selectors, which chose later checkpoints on ties. The new selector
   verifies source/report/plan identities, the complete registered checkpoint
   grid, exact validation cohort and materialization, fixed inference input,
   finite metrics, and no hidden access or optimizer activity. The base is a
   reference only; negative improvement and a zero base denominator are explicit.
   Safety metrics remain diagnostics, not task acceptance.
2. `src/rosetta_reality/vla/reload_evidence.py` and
   `scripts/verify_smolvla_reload_v2.py` compare complete registered action/noise
   tensors, sample identities and padding masks. They require checksums, complete
   shapes, matching expected identities/dtypes, and distinct collection processes.
   Equal means or equal first actions cannot hide a changed later chunk slot.
   The verifier explicitly does **not** prove model execution or formal resume.
3. The eight historical-source test failures were repaired at the fixture
   boundary. Ten exact source snapshots are stored as non-executable test data
   and checked byte-for-byte by the production validator. Tampering and current
   source drift still fail. Zen/vfunfreeze positive fixtures are complete.
   Vcdropout gate unit tests use a clearly synthetic temporary plan with real
   current hashes; the actual historical plan still rejects incompatible code.
   No historical YAML, frozen selector/exporter, or production hash check was
   rewritten or bypassed.
4. `scripts/audit_smolvla_dataset.py --all-images` extends the earlier 405-sample
   audit to every train/development frame and its complete target chunk. This
   measures audit coverage, not historical optimizer input coverage. Historical
   5,120 exposures still mean 40 unique frame-0 inputs in the registered design.

The interface and new-plan requirements are documented in
`docs/m2-smolvla-posttrain-v2.md`. Old selected artifacts are not regenerated.

## Verification evidence

- Runtime: WSL-launched, offline Linux Docker; immutable image
  `sha256:2696f8e0430050951ffdd16721e41af19473bd741de4c5547cfb330d6f08580b`,
  two CPUs and 6 GiB memory/swap ceiling. No package/model/data download.
- Final regression: `runs/training-chain-repair-20260914-001/regression-final.xml`;
  569 passed, 0 failed, one normalization CUDA test skipped without CUDA, two
  data-marked tests deselected by the explicit `not data` selection. The separate
  full cached-data audit below is not a claim those two tests ran.
- Data: `runs/training-chain-repair-20260914-001/all-images/result.json` and
  `decoded-samples.json`; 20,000 train plus 2,500 development rows, 22,500 decoded
  samples and complete action chunks, hidden rows zero, model not loaded,
  optimizer steps zero.
- Earlier test attempts remain in `selection-history-first.xml`,
  `repair-integration.xml` and `selection-reload-second.xml` under the same run
  directory. Intermediate fixture/import failures were corrected before the
  final suite; those results are retained rather than overwritten.
- Ruff lint/format and `git diff --check` passed. The companion JSON records
  current source hashes, evidence hashes and exact commands. The final dataset
  script differs from the executed full-data version only by redundant loop
  parentheses removed during formatting; no behavioral change followed that run.

## Remaining boundaries

The saved-array verifier is tested with synthetic arrays from real separate
processes. Actual 450M policy forward/backward, independent model/processor
reload, formal resume, CUDA execution and new Gate 3/4 were not run in this
repair. Independent video PTS and physical image/state/action semantics were
not measured. All frame samples decoding successfully is narrower than that
semantic claim. Formal resume remains disabled.

Two vcdropout dirty source versions were not found in local Git or retained
copies: `training/features.py` SHA
`e0ded450340be7f6808d8f764122dd52e8adf3a964b07f00f0f3ea3dd5fdbe53`, and
`training/plan.py` SHA
`efc70554855d6cfb63c78d180f485d82a94ee7ca3bb053d4e29136e29ddd09d7`.
Historical reproduction remains incomplete; synthetic tests do not fill that gap.

No formal training, remote GPU activity, commit or push was performed. Existing
dirty work and failed experiment evidence were preserved. Iris Gate 3/4 results
and M2's incomplete status are unchanged.
