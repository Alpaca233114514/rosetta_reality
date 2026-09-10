# Gate 4 seating feasibility — execution blocked, 2026-09-06

**No five-seed experiment has run. This is not a 0/5 result.**
`gating=false`; M2 remains incomplete and historical evidence is unchanged.

Prepared: preregistration Markdown/JSON; create-only diagnostic
`scripts/diagnose_gate4_seating_feasibility.py`; focused tests
`tests/test_gate4_seating_feasibility.py`; architecture boundary update.
The instrument directly reuses the old privileged control function and
restores the historical teacher setting `hold_enabled=false`. Its total
500-step budget includes teacher preparation. Reset parity tests compare
robot state, all camera tensors and object/contact snapshots at all five seeds.

## Blocker and evidence

Initial WSL invocation was denied by the execution sandbox (`E_ACCESSDENIED`).
Escalated read-only invocations then failed to start Bash, including `pwd`.
UTF-16-decoded original error:

> 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。
> 错误代码: Wsl/Service/0x8007274c

The WSL distribution inventory is readable, and Windows reports WslService,
vmcompute and hns running. This does not establish a working Linux runtime.
No WSL/Docker restart, service reconfiguration or AutoDL startup was attempted.
The expected image identity is frozen from the existing teacher-gate image
registration and remains **unverified live**. A mismatch must stop the probe.

Container pytest was attempted but could not start; Ruff and probe execution
remain pending. Static JSON/source-hash and whitespace checks are not runtime
test evidence. Bootstrap evidence is separate from the future create-only run:
`runs/m2-t2-seating-feasibility-gate4-2026-09-06-bootstrap-001/wsl-startup-failure.json`.

## Five-seed inventory

| Seed | Seatable | Steps | Seating residual (mm) | Failure phase |
|---|---|---|---|---|
| 1000 | not measured | not run | not measured | WSL startup, before simulation |
| 1001 | not measured | not run | not measured | WSL startup, before simulation |
| 1002 | not measured | not run | not measured | WSL startup, before simulation |
| 1003 | not measured | not run | not measured | WSL startup, before simulation |
| 1004 | not measured | not run | not measured | WSL startup, before simulation |

## Continuation and user decisions

After local WSL is restored, run from repository root inside WSL Bash:

```bash
bash scripts/run_m2_container.sh vla-sim-xpu python -m pytest -q \
  tests/test_gate4_seating_feasibility.py tests/test_geometric_teacher.py
bash scripts/run_m2_container.sh vla-sim-xpu python -m ruff check \
  scripts/diagnose_gate4_seating_feasibility.py tests/test_gate4_seating_feasibility.py
bash scripts/run_m2_container.sh vla-sim-xpu \
  python scripts/diagnose_gate4_seating_feasibility.py
```

Only launch the probe after both preceding commands pass. Preserve this blocked
report; append a completion amendment or a new result report after execution.
No automatic retry, controller tuning or seed substitution is registered.

There is currently no basis to choose a tolerance, pose-group or control-mode
change. If the valid probe later returns 0/5, inspect whether failures actually
reach INSERT and distinguish a controller failure from a persistent seating
residual. A demonstrated-seatable control group would give a stronger diagnostic
comparison than immediately relaxing success tolerance. Changing tolerance,
pose distribution or control mode remains the user's separately registered
decision. Even all-seatable would support a learning/interaction gap without
proving that more DAgger data is the sole remedy. No such change is made here.

No model weights, dataset rows or hidden episodes loaded; no training,
downloads, external writes, commit or push performed.

## Continuation amendment — 2026-09-06, 12:48 UTC

WSL Bash is now accessible. All 20 preregistered source hashes still match.
The existing launcher returned `error: Docker Desktop Linux engine is not ready`.
Docker Desktop was not running, so the installed application was started.
Its backend then crashed at 12:48:33 UTC while initializing the Inference
manager: the `dockerInference` socket could not be accessed. Error excerpts:
`The file cannot be accessed by the system` and
`The filename, directory name, or volume label syntax is incorrect`.
The two pending read-only CLI/control-shell calls were interrupted after the
backend crash was confirmed; no simulator or test process had started.

No socket deletion, factory reset, WSL restart or Docker settings change was
performed. The current blocker is Docker startup, superseding the WSL startup
blocker above. Pytest, Ruff, calibration and all five probe results remain
unavailable. Separate create-only evidence:
`runs/m2-t2-seating-feasibility-gate4-2026-09-06-bootstrap-002/docker-startup-failure.json`.
