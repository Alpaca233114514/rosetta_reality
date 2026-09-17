# Authorized traced G3/G4 repeat — dispatch

The user authorized the available AutoDL GPU to run G4 with the current repaired
code and requested detached execution rather than waiting for results.

- Execution: `canonical-fullframes-posttrain-20260916-005`.
- Workspace: `20260916T100517Z-95cf9cf9483b-6bb3da3027df`.
- Source archive SHA256: `6bb3da3027dfc66fe876551977cdbaedd2f217f877dc4386f7f4269db538c96e`.
- Template: `reports/training/canonical-posttrain-template-20260916-005.json`.
- Template SHA256: `7fcddfd71fbda8214906bf959bf8b37790f2e8b897eea64ef9523bc81ec21f8e`.
- 1088 source/prerequisite hashes verified before dispatch.
- Worker: tmux `canonical-gate-repeat-005`, started 2026-09-16 18:08:28 China time.
- Watchdog PID 1805, start ticks `4139527129` observed at startup.
- Startup: 84 CPU tests passed; CUDA doctor and endpoint preparation exited zero.
- First observed active stage: `collect-first`, PID 1855.
- Final startup check: collect-first, collect-reload and verify-reload exited zero; offline prerequisite stage active, PID 26549.

`scripts/run_canonical_gate_repeat.py` adds a fresh execution namespace and scoped
observation around the original rollout. It routes the existing supervisor's
subprocesses through that namespace, including the independent watchdog.
The immutable scientific authority remains attempt 004. Fixed step-5000 model
SHA is `d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef`.
No optimizer, checkpoint selection, action correction or threshold change is
introduced. G3 must pass before G4; G4 retains seeds 1000–1004 and 500 steps.
Trace files include internal gripper predictions, actual executed transitions
and independent persisted-event verification. Latencies include tracing overhead.

The original guarded lifecycle runs CPU checks, doctor, artifact preparation,
independent full-array reload/native equivalence, offline prerequisite checks,
G3 and G4. Existing resource limits, the one-hour work deadline, independent
70-minute protected shutdown and result packaging remain in force. Outputs
are under `runs/canonical-fullframes-posttrain-20260916-005/` in that workspace;
the sibling `.tar` and `handoff-manifest.json` are the recovery entrypoints.
Retrieval and checksum verification remain pending; no completed G4 result is
claimed. Prior measured G4 remains 0/5 and M2 remains incomplete.

The first dispatch preparation found a missing historical receipt in the new
source-only package. Its exact-SHA copy was restored from the retained prior
workspace before any worker launch. No previous experiment was overwritten.

The separately requested Sol medium cleanup and Google Drive migration must
preserve all live dependencies. Its receipts are under
`runs/remote-cleanup-20260916-001/` and
`runs/remote-drive-migration-20260916-001/`.
