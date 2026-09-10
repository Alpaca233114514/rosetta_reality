# Gate 4 seating-feasibility continuation — 2026-09-08

Docker Desktop Linux engine and the registered local image were recovered.
The required focused tests passed, but Ruff failed before probe execution.
**All five target seeds remain not measured. This is not a 0/5 result.**

This is an additional report. The September 6 blocked reports, bootstrap
evidence and registration are unchanged; their hashes were checked before and
after this continuation. All 20 registered source hashes still match.

## Recovery and verification

The initial WSL sandbox call returned
`Wsl/Service/CreateInstance/E_ACCESSDENIED`; formally escalated WSL Bash worked.
Docker Desktop reported `stopped`, with backend event-stream HTTP 500 errors.
The official Desktop restart restored a Linux API, but its default storage had
zero images and zero containers. A retained prior data disk was located.
With Desktop stopped, the current settings were backed up and only
`CustomWslDistroDir` was restored to the existing data-disk location.
Both data disks were retained without moving, overwriting or deleting them.

Startup then encountered inaccessible stale runtime sockets. Error excerpts:
`initializing Ingest server` / `sailor-ingest.sock` and
`initializing Secrets Engine` / `engine.sock`, each ending in
`The file cannot be accessed by the system.` Path-verified Docker processes
were stopped, and the affected runtime socket directories were renamed and
retained. No credential contents were read. Restart then recovered the prior
inventory: 45 images and 134 container records (not a running-container count).
No factory reset, Debian termination or security-setting change was performed.

The live Linux engine reports version `29.7.2`. The registered image matches:
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`.
The existing WSL Bash/container launcher retains network isolation, 6 GB memory
and memory+swap limits, and 2 CPUs. Environment check exited 0:
Python 3.12.3, PyTorch 2.11.0+xpu, XPU available with one Intel graphics device.

## Required execution sequence

1. `python scripts/check_env.py`: exit 0 in `vla-sim-xpu`.
2. `python -m pytest -q tests/test_gate4_seating_feasibility.py tests/test_geometric_teacher.py`:
   **38 passed in 8.15 seconds**, exit 0. One cache warning is caused by the
   intentionally read-only repository mount. Reset-parity testing does reset
   the target seeds; it does not execute target probe actions.
3. `python -m ruff check scripts/diagnose_gate4_seating_feasibility.py tests/test_gate4_seating_feasibility.py`:
   **exit 1**, one `I001` error at the import block starting on line 17.
   Original message: `Import block is un-sorted or un-formatted`.
   Ruff suggests a blank line after the `diagnose_g2_seating_feasibility` import.
4. Calibration, sanity control and five-seed probe: **not started**.

The registered continuation explicitly requires both pytest and Ruff to pass
before the probe. No auto-fix, source-hash change, alternate lint configuration,
probe retry or threshold change was applied.

## Per-seed inventory

| Seed | Seatable | Probe steps | Seating residual (mm) | Blocking phase |
|---|---|---|---|---|
| 1000 | not measured | not run | not measured | pre-probe Ruff gate |
| 1001 | not measured | not run | not measured | pre-probe Ruff gate |
| 1002 | not measured | not run | not measured | pre-probe Ruff gate |
| 1003 | not measured | not run | not measured | pre-probe Ruff gate |
| 1004 | not measured | not run | not measured | pre-probe Ruff gate |

Privileged-only minimum, whole-rollout minimum and final pre-step residuals
are all null in the JSON because no target rollout occurred.
This is inconclusive infrastructure/verification evidence, with no support
for either seatability or a protocol-envelope ceiling. Historical policy
failures and M2 non-acceptance remain unchanged.

The next continuation needs a recorded formatting-only source-identity
amendment that preserves the original registration, followed by the same
focused tests, Ruff and unchanged probe semantics. The original create-only
probe output directory is still absent.

## Evidence

- `runs/m2-t2-seating-feasibility-gate4-2026-09-08-bootstrap-001/preflight.json`
- `runs/m2-t2-seating-feasibility-gate4-2026-09-08-bootstrap-001/recovery-actions.json`
- `runs/m2-t2-seating-feasibility-gate4-2026-09-08-bootstrap-001/check-env.log`
- `runs/m2-t2-seating-feasibility-gate4-2026-09-08-bootstrap-001/focused-pytest.log`
- `runs/m2-t2-seating-feasibility-gate4-2026-09-08-bootstrap-001/ruff.log`
- `runs/m2-t2-seating-feasibility-gate4-2026-09-08-bootstrap-001/result.json`
- `reports/training/m2-smolvla-t2-seating-feasibility-gate4-continuation-2026-09-08.json`

No model or dataset was loaded; no hidden episodes, training, requested
downloads, Gate protocol changes, commits or pushes were performed.
