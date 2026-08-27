# M2 SmolVLA Athena Plan054 workspace package repair — 2026-08-16

Status: **package repaired**. The frozen Plan054 workspace archive omitted five
hash-bound files under the ignored `runs/` tree, which made the remote evaluator
fail in plan hash binding before exact execution. A new content-addressed
workspace archive was created locally; the original archive and manifest remain
immutable and were not overwritten.

## Root cause

`/runs/` is ignored by `.gitignore`, but Plan054's hash-bound config and
evaluator reference five files under `runs/m2-smolvla-aloha-geometry-teacher-054/`:

- `local-exact-001/run-exact.sh`
- `local-exact-001/launch-exact.sh`
- `local-exact-001/start-exact.sh`
- `local-orientation-step-probe-002/generate_probe_requests.py`
- `local-orientation-step-probe-002/run-probe.sh`

The frozen archive was built from `git ls-files --cached --others
--exclude-standard`, so those ignored files were omitted. Athena attempt 002
reproduced the omission as a plan-hash-binding failure before the evaluator
started. The remote create-only supplement allowed the registered exact stage
to run, but the upload package itself remained defective.

## Repair

- original package:
  `athena-plan054-workspace-20260815T212909Z-5bd66d5e4bdc-e759e520fd2b.tar`
- original SHA-256:
  `e759e520fd2b07213195a45408b8ac4a67673cd958870b41d0263c795b5ad041`
- repaired package: `athena-plan054-workspace-20260816T060456Z-5bd66d5e4bdc-071094e4d8de.tar`
- repaired SHA-256: `071094e4d8deebf824c896866a47412a906680234baf84b1d0cfde89f8e78d36`
- repaired release id: `20260816T060456Z-5bd66d5e4bdc-071094e4d8de`
- member count: 554 = original 549 members plus 5 missing hash-bound files;
  no original member was removed.

Every `implementation_files` entry in Plan054 and both hardcoded evaluator
probe paths are now present in the repaired archive with the registered SHA-256
values.

No planner, teacher, adapter, simulator or evaluator code changed. No Plan055
was created and no gate was opened.

## Evidence

- `reports/training/m2-smolvla-athena-plan054-workspace-package-repair-2026-08-16.json`
- `reports/training/m2-smolvla-athena-plan054-workspace-package-repair-2026-08-16.md`
- `runs/m2-smolvla-aloha-geometry-teacher-054/deployment-remote-002/athena-plan054-workspace-20260816T060456Z-5bd66d5e4bdc-071094e4d8de.tar`
