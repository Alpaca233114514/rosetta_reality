# Iris 002 Gate acceptance closure: both Gate 3 passed, both Gate 4 failed

Acceptance is complete for the two fixed 1280-step Iris 002 checkpoints.
Both passed Gate 3; both failed Gate 4 with **0/5 task successes and maximum
reward 0 on every seed**. The only failed Gate 4 criterion is
`minimum_task_success_rate`. M2 remains incomplete. No optimizer update occurred.

| Arm | Gate 3 | Gate 4 | Successes | Joint-limit violations | Unexpected collisions | Invalid action rate |
|---|---|---|---:|---:|---:|---:|
| Control, lambda 0, suffix 451 | passed, 20 steps | failed | 0/5 | 0 | 0 | 0 |
| Treatment, lambda 0.01, suffix 452 | passed, 20 steps | failed | 0/5 | 0 | 0 | 0 |

All ten Gate 4 episodes used the original environment/noise seeds 1000–1004 and
ran 500 actions each. The unchanged protocol requires success rate >=0.2. Raw,
projected and executed action-limit violation rates were all zero in both arms.
The mean command smoothness L2 was 0.070272 / 0.070762; mean policy inference
latency was 0.228900 / 0.236718 seconds and mean simulation-step latency was
0.008701 / 0.008733 seconds (control / treatment). These latency measurements
do not establish real-time deployment suitability.

The earlier offline result remains unchanged: train40 fit passed 4/4 in both
arms; treatment improved full development left-joint MAE/MSE by about 1.04%/2.21%,
but still failed constant-baseline/generalization acceptance. The new closed-loop
measurement shows no task-success improvement under this registered protocol.
Neither clean safety metrics nor Gate 3 constitutes insertion-task success.

## Executed identity and boundaries

- Task: `iris-002-gate34-20260913-003`, source commit `5683e03`.
- Workspace: `20260913T153301Z-5683e03803e1-96bdb28d24e2`; archive SHA256
  `96bdb28d24e2bf345d896510cf51d2d73b8bfaf37a278651f6db3693efc96e72`.
- Template SHA256:
  `08e30497289d87704d96d9f1427512e9768d8b6e473b6c83cea7ae082d5a16a3`.
- Original `scripts/smolvla_sim_gate.py` engine and its thresholds, one-action
  receding horizon, BF16, seeded noise, physics and collision policy were retained.
- The native saved processors were loaded without statistics overrides. Each
  Gate invocation passed exact raw/projected chunk equality against direct native
  inference on a simulator reset. All parameter-before/after checks passed.
- Total: 5040 closed-loop actions/policy inferences, eight additional bridge
  inferences, zero optimizer steps; worker time 1408.967 seconds (23.48 minutes).
- Gate 4 used its corresponding passed Gate 3, identical artifact, plan and
  workspace identities. Hidden test data were not loaded. No checkpoint search,
  seed replacement, threshold relaxation or rerun inside a completed job occurred.

## Preserved engineering failures

The first two newly registered Gate attempts stopped before any closed-loop
action. Attempt 001 exposed a data-root/output-root namespace collision; attempt
002 exposed misuse of the upstream config's 6-dimensional state placeholder
instead of the actual 14-dimensional dataset contract. Both were wrapper defects,
not measured Gate failures. Their 21-file evidence packages and original source
snapshots were preserved, verified and shut down before the next fresh identity.

Corrections and their local regression checks are documented in
`m2-smolvla-iris-gate34-recovery-002-2026-09-13.md` and
`m2-smolvla-iris-gate34-recovery-003-2026-09-13.md`. The final relevant suite had
52 passing tests plus Ruff/format/shell checks. The unrelated historical vcdropout
source-pin failures remain recorded; old evidence and pins were not rewritten.

## Recovery and independent verification

All 46 files (462,808 payload bytes) were recovered to
`runs/iris-gate-recovered-20260913-003/verified/`. The archive is 563,200 bytes;
SHA256 `f112e70e92cc1164307d062a61a196a33a0ea804517fa3436b0ac2cc72439462`.
Manifest SHA256:
`76b4678ffb0a8003c1a766f393c48789af95a3e365bbe306c3c774f2ef089ee7`.
Every file's type, size, identity and SHA was checked before sending the receipt.
The original training exports remain backed up locally from the previous 125-file
recovery; Gate artifacts only add metadata and immutable references to those weights.

`iris-gate-validation-20260914/verify_gate_results.py` independently checked the
four Gate reports and ten separate episode records in the fixed offline Linux
Docker image: exact source-checkpoint identity, artifact/plan/Gate-3/code hashes,
fixed seeds and horizons, episode-to-summary equality, aggregate success/safety
arithmetic, bridge proofs and unchanged-parameter records all passed. It loaded no
model weights or datasets and did not rerun simulation.

The parent directly monitored the run and completed transfer. After the verified
receipt, protected shutdown executed; a refreshed live AutoDL console confirmed
the exact registered instance powered off. The instance was not released. This
task's post-transfer shutdown request is not part of the pre-shutdown archive;
retrieve it only in a future authorized window, not by reopening solely for it.

This closes the requested acceptance. Preserve both negative arms and their
checkpoints. Further training or a changed experimental axis requires a new
registration; there is no automatic promotion or retraining from these results.
