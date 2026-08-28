# M2 SmolVLA Zen module-gradient diagnostic, registration 002 (frame offset 250) — 2026-08-28

## 1. Authority and scope

Follow-up registration to
`reports/training/m2-smolvla-zen-module-gradient-preregistration-2026-08-28.md`
(registration 001). Same non-gating local diagnostic, same artifacts,
conditions, groups, loss probe and runtime. The only protocol change is the
sample offset, motivated by a discovery made by the completed 001 execution
(orchestration `003`): the registered validation protocol samples frame
offset `0`, and at frame 0 **all episodes of the dataset share the identical
home-pose robot state** (verified directly against the raw parquet columns:
50/50 episodes, pairwise state max difference `0.0000`), so the
`state_shuffle` condition of registration 001 was a provable no-op — it
swapped bit-identical states. The 001 result stands as evidence about the
frame-0 probe point; it cannot answer the state-versus-image gradient
question. Registration 002 moves the probe to mid-trajectory frames where
states genuinely differ.

Verified sample facts (raw parquet and the restricted dataset view agree):
at frame 250 the five validation episodes' states differ pairwise by
`0.36–0.84` rad; images differ; actions differ.

## 2. Protocol delta from registration 001

| Field | Registration 001 | Registration 002 |
|---|---|---|
| frame offset | `0` (registered validation protocol) | `250` for all five validation episodes |
| degeneracy guards | none | fail closed when offset `> 0` samples share identical states; fail closed when frame-0 samples differ; fail closed when images are identical across samples |
| recorded sample facts | — | pairwise state/image max differences inside the diagnostics JSON |

Everything else (episodes `[22, 13, 7, 33, 45]`, conditions, shuffle seed
`20260812`, zeros noise, flow time `0.5`, groups, adaptation rebuild,
artifacts, container) is unchanged from registration 001 and its two recorded
implementation amendments.

## 3. Frozen identities

- script `scripts/diagnose_smolvla_zen_module_gradients.py`, sha256
  `f6a7a182a65f458bf460a0a242ba6a4e92e7d2ce0753fc3be74232b284a1c742`
  (adds `--frame-offset`, the degeneracy guards and the recorded pairwise
  sample differences; the offset-`0` default reproduces registration 001
  exactly);
- runner `scripts/run_zen_module_gradients.sh`, sha256
  `2bef6e46dc4d43e441ebc60be5e9205702fdd4206910f4e47d9cc73e3627d4a8`
  (orchestration suffix `004`, `--frame-offset 250`);
- artifacts, dataset revision, contract, groups and runtime exactly as in
  registration 001.

## 4. Acceptance, stop conditions, evidence

Unchanged from registration 001, plus: the run fails closed if the selected
samples do not genuinely differ in state. Expected evidence:
`runs/<experiment_id>/diagnostics/zen-module-gradients-{firstaction,uniform}-<hash>.json`
(their `protocol.frame_offset` is `250` and their recorded pairwise state
difference is positive), orchestration `zen-module-gradients-004.{log,status}`,
and the combined completion report
`reports/training/m2-smolvla-zen-module-gradient-diagnostic-2026-08-28.{md,json}`
covering both executions.

## 5. What not to conclude

Unchanged: non-gating, no arm ranking, no visual-conditioning authorization.
Additionally: frame-250 mid-trajectory teacher forcing is one probe point,
not the full training distribution; the 001 frame-0 result (near-total image
insensitivity of the gradient at the shared home pose) remains valid for
what it measures.
