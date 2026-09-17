# Canonical full-frame endpoint: Gate 3 passed, Gate 4 failed

Attempt `canonical-fullframes-posttrain-20260915-004` completed the registered
step-5000 evaluation without a runtime exception or optimizer update. Gate 4
measured **0/5 task success**. M2 remains incomplete.

| Environment / policy seed | Success | Steps | Joint-limit count | Unexpected-contact count |
| --- | --- | ---: | ---: | ---: |
| 1000 | false | 500 | 0 | 0 |
| 1001 | false | 500 | 1 | 0 |
| 1002 | false | 500 | 0 | 16 |
| 1003 | false | 500 | 18 | 15 |
| 1004 | false | 500 | 0 | 7 |
| Total | 0/5 | 2500 | 19 | 38 |

All episodes reached the registered 500-step horizon; maximum reward was zero in
every episode. Task success, physical joint limits and unexpected contacts failed
acceptance. Finite actions and policy/executed action-contract checks passed.
Joint/contact counts accumulate across simulation steps; they are not counts of
distinct accidents. The unexpected pairs were gripper fingers against the table.
Mean policy inference was 0.20254 seconds and mean simulation step 0.008438 seconds.
Complete post-training worker duration was 1093.035 seconds, about 18.2 minutes.

## Execution and provenance

The source is `canonical-fullframes-20260914-001`: 5000 fresh updates at batch 4
over the registered 20000 unique training input frames. This evaluation kept its
fixed final checkpoint; it did not select another checkpoint or change the five
Gate seeds, 0.2 success threshold, action contract or Gate engine.

The primary agent repaired the post-training adapter and verified 41 CPU tests and
1084 source/evidence files before Luna executed 004. Earlier attempts 002 and 003
failed engineering preconditions and remain preserved; neither is a Gate 4 result.
004's Gate 3 and Gate 4 share the exact code identity, simulation-plan SHA and
artifact-manifest SHA. The parent independently checked these relationships and
recomputed the per-episode aggregate counts after retrieval.

Two distinct processes, PIDs 1716 and 14057, loaded the model and saved processors
and each collected 16 registered forwards. Complete action/noise/sample/mask arrays
matched each other and the recovered original native endpoint exactly. Native and
simulation adapters also produced equal complete chunks. Gate 3 and Gate 4 parameter
digests remained unchanged. These checks establish equality on the tested paths
and inputs, not the absence of every possible implementation defect.

The returned package contains 47 files / 1,109,379 bytes; both Luna and the parent
checked all file SHA values. Manifest SHA:
`fe1bcd4651336ac20b76e43f6bffdcccf7b449cdc9bd8e2f6d55889a60540b5c`.
Archive SHA:
`9a93cdf5566289321926cdfacf69cea7375ab64c9850773a34b2fa79a3608eb1`.
The matching receipt was confirmed on the worker. Source checkpoints remain backed
up and were not deleted or recopied for this evaluation.

## Offline interpretation and remaining limits

Zero-noise action MAE was 0.0292088327 on 160 training observations and 0.0435541757
on 20 development observations. These values also exactly match attempt 003.
Observations use offsets 0/125/250/375; they do not measure all training frames or
all deployment states. The baseline is a single global mean over 20000 training
actions. Results mix temporal offsets and dimensions and retain actual state input,
so they cannot establish visual generalization or be directly compared with Hestia
frame-zero per-slot/group mean-median baselines. Offline legality measures postprocessed
actions; Gate 4 separately retained unprojected diagnostics.

The evidence is limited to five fixed simulated development seeds. Hidden tests,
physical robots, broader deployment distributions, formal resume and a new local
model reload were not tested. No simulation recordings were registered. Full-frame
training coverage and sampled reload/adapter equality did not yield closed-loop
task success; the unique cause of the model's remaining failure is not identified.

## Evidence

- Machine-readable result: `reports/training/m2-smolvla-canonical-fullframes-gate34-result-2026-09-15.json`.
- Verified package: `runs/canonical-posttrain-received-20260915-004/verified/`.
- Native Gate reports: the package's `results/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/gates/gate{3,4}-smolvla-sim-471.json`.
- Reload, bridge, parameter checks, original source seal, offline metrics, worker exit and receipt are retained in that package.

Luna confirmed the original instance shut down by refreshing the Chrome platform
page after the remote receipt was confirmed. This closure does not rely on SSH loss.
