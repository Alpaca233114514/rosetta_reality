# Canonical step-5000 traced G3/G4: measured negative result

`canonical-fullframes-posttrain-20260916-005` completed normally with **G3 passed,
G4 failed 0/5**. M2 remains incomplete. This is a measured closed-loop failure,
not a launcher crash or an unfinished evaluation.

## Execution and verification

- Current source was staged into workspace `20260916T100517Z-95cf9cf9483b-6bb3da3027df`.
- 84 CPU checks, CUDA doctor and artifact preparation passed.
- Two independent model-loading processes each performed 16 forwards; complete
  arrays matched one another and the retained native endpoint exactly.
- Both Gate parameter checks report unchanged weights and zero optimizer steps.
- Worker duration was 1135.6245319843292 seconds, with `error: null`.
- The result archive is 25,692,160 bytes; all 71 delivered files, totaling
  25,547,140 bytes, passed size and SHA256 checks after safe local extraction.
- Archive SHA256: `223547e6d25b9b267831db3102478e99ff4068009c6fa8a42d570617373816de`.
- Handoff manifest SHA256: `f7f8d99818321d15f4b502612e453becfb5efd98b38c566abead75dfd9fd4f08`.

The fixed weight SHA remains
`d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef`.
The saved processors, first-action execution, noise seeds and acceptance
thresholds retain the registered scientific protocol. No training, checkpoint
selection or action correction was performed.

## Gate 4

| Seed | Steps | Maximum reward | Success | Joint-limit counts | Unexpected collision counts |
| --- | ---: | ---: | --- | ---: | ---: |
| 1000 | 500 | 0 | false | 0 | 0 |
| 1001 | 500 | 0 | false | 1 | 0 |
| 1002 | 500 | 2 | false | 8 | 38 |
| 1003 | 500 | 0 | false | 18 | 15 |
| 1004 | 500 | 0 | false | 0 | 7 |

The failed acceptance criteria are task success, physical joint limits and
unexpected collisions. Invalid-action and command-action limit violation rates
are zero. The aggregate 27 joint-limit and 60 collision counts are per-step
counts, not counts of independent incidents. A maximum reward of 2 does not
meet task success.

All five complete G4 traces are retained, together with the G3 trace. They
record internal gripper support, action predictions, executed transitions,
physical snapshots and reward events. Persisted-event verification runs within
the traced wrapper; retrieval separately verifies all delivered file hashes.
The traces do not include a measured expert-deviation reference.

## Interpretation and limits

The unchanged model still fails the closed-loop acceptance protocol. This run
does not establish a visual repair or a unique cause. Its reward and physical
violation totals differ from the previous attempt-004 result despite retaining
the checkpoint. No real CUDA traced/untraced paired rollout was performed, so
the difference must not be attributed to instrumentation or claimed as policy
improvement. Reported latency includes trace overhead.

SSH retrieval succeeded and verified completed results. It did not verify
current platform power or billing state, shut down the instance, release it,
delete remote files or rerun the experiment.

## Evidence

- Local package root: `runs/canonical-posttrain-received-20260916-005-ssh/`.
- Verified evidence root: `runs/canonical-posttrain-received-20260916-005-ssh/verified/`.
- `transfer-receipt.json`, `worker-exited.json`, `artifact-reload-proof.json`.
- `gate3-parameter-check.json`, `gate4-parameter-check.json`.
- `results/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/gates/gate4-smolvla-sim-471.json`.
- `traces/gate4-1000/` through `traces/gate4-1004/`.

These last four entries are relative to the verified evidence root. The original
dispatch record remains `reports/training/m2-smolvla-traced-gate-dispatch-2026-09-16.md`.
