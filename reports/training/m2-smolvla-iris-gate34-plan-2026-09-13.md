# Iris 002 fixed-endpoint Gate 3/4 acceptance registration

User authorization: continue acceptance and Gates after the completed Iris 002
training/backup. The parent monitors directly; the Luna monitors remain stopped.
This task measures closed-loop outcomes of both fixed checkpoints. It does not
relabel the failed offline generalization acceptance or select a new checkpoint.

Identity: `iris-002-gate34-20260913-001`. Source training workspace
`20260913T135230Z-5d285f5c8297-3ba03ac32579`, run `iris-k-scene-20260913-002`.
Source handoff manifest SHA256:
`8f1f80d9676c2a8f25c587d9bf7462fe41778c6fa992681096f7284dbcb9812b`.
Both policies/processors are already fully backed up locally and SHA-verified.

## Fixed protocol

- No optimizer creation, backward, training, checkpoint search or hidden access.
  Load only the two registered final 1280-step artifacts.
- Preserve the original `scripts/smolvla_sim_gate.py` engine unchanged (SHA256
  `5b76127a2e2d0e0049181a1d0ab12297474cbc8eb433c4fcdb6466c35c53c5ae`).
- Gate 3: environment/noise seed 20260809, maximum 20 steps. Finite actions,
  contract compliance, joint limits, zero unexpected collisions and artifact
  reload checks must pass. Gate 4 requires the matching passed Gate 3 report,
  exact artifact, simulation plan and workspace identity.
- Gate 4: environment/noise seeds 1000–1004, maximum 500 steps per seed,
  minimum task success rate 0.2, zero unexpected collisions and unchanged
  engine acceptance criteria. No thresholds are relaxed.
- Receding horizon executes one action from each 50-action chunk; seeded
  standard-normal noise, BF16, existing bounded decoder/output projection,
  action contract, physics and collision policy are unchanged.
- Separate report suffixes: control 451, treatment 452. A failed Gate 3 stops
  Gate 4 for that arm; the other independently registered arm may still be
  evaluated. A runtime/identity failure stops the campaign. No automatic retry.

## Artifact and inference boundary

`scripts/iris_gate.py` verifies the complete original file manifest and the
actual off-worker backup receipt before creating small gate-facing metadata.
It references the immutable exported weights using a new directory link; no
weight data is changed, moved or duplicated. The manifest binds every policy
file and the exact seven-array independent CUDA reload proof from Iris 002.
The gate-facing endpoint record's `status=passed` means fixed-step identity
verification only; it explicitly retains `scientific_acceptance=negative_result`.

The wrapper reuses the proven CUDA predict/noise methods while loading the
native Iris policy and **saved** processor state without statistics overrides.
Each Gate first compares raw/projected chunks on an actual simulator reset
against direct native saved-processor inference (two no-optimizer forwards,
exact equality). It also hashes all parameters before/after Gate execution.
Train40 frame-zero inputs are read by the shared native loader for provenance;
development/hidden data are not needed for this Gate dispatch.

The historical vfunfreeze-only alignment permit is not claimed or forged; that
candidate wrapper's registration template supplies the shared simulation
protocol, while this separately authorized Iris registration binds Iris's own
checkpoint/backup/reload prerequisites. The generic Gate engine retains all
prior-failure, contract, artifact and Gate 3→4 checks.

## Runtime and lifecycle

Only the registered AutoDL CUDA worker may run this task, with the offline
runtime profile and no nested Docker. Prepare a new content-addressed workspace.
The supervisor and independent watchdog are detached; work limit 3600 seconds,
protected shutdown 4200 seconds, process-tree RSS <=8 GiB, CUDA allocated <=8 GiB
and reserved <=10 GiB. At most 5040 closed-loop actions plus eight bridge forwards
across the four Gate invocations; zero optimizer updates. No new downloads,
dependencies or storage deletion. Existing source artifacts and failure evidence
remain untouched. Only small metadata/results are written and recovered.

Stop on nonfinite/identity/budget/runtime failure. Preserve negative Gate reports.
Recover the bounded evidence archive and verify every file before sending the
receipt; the existing protected shutdown helper then shuts down, never releases,
the worker. Verify platform power state separately. Do not automatically launch
another experiment or retry a failed Gate.

## Local verification

`wsl.exe bash scripts/check_iris_gate.sh`: **49 passed**, Ruff passed, three Python
files formatted and all three launch/receive shells parsed individually. The
fixed offline Docker image and 2 GiB/2 CPU resource cap match the prior local
Iris checks. No local model weights or real datasets were loaded.

An initial broader run also exercised the unrelated old vcdropout-specific
wrapper suite: 49 passed, two failed because its historical plan pins
`scripts/train_smolvla_v2.py` to `888cca61...`, while the current unchanged file
is `f7aafb6a...`. Neither that old plan nor its tests/source pins were edited.
This is recorded as historical compatibility failure, not a passed check. The
Iris runtime uses its own valid source pins and the generic engine suite.

New template: `iris-gate-template-2026-09-13.json`, SHA256
`413fb2d5a96d893ca3103436bef9c69942c86fe65a410ce15fefb7fe7e3a1141`.
CUDA bridge and Gate outcomes remain unmeasured at registration. M2 is incomplete.
