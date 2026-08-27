# M2 SmolVLA Athena Plan054 remote exact audit — 2026-08-16

Status: **failed train-only exact** on Athena. The remote exact reproduced the
local Plan054 failure boundary and is preserved as immutable negative evidence.

## Identity

- run name: `Athena`
- host: `autodl-container-44db45aec7-cd880eb6`
- boot id: `c7f0d91f-cfd3-488a-9969-c70510f77465`
- branch: `codex/er-vla-smolvla-pipeline`
- git HEAD: `5bd66d5e4bdc0228dc1648825d8d3ec1236dc09f`
- plan: `configs/sim/aloha_insertion_geometry_teacher_054.yaml`
- plan SHA-256: `d8d68745ea7a22c59f924713c93f04bcacce4371fca820b6040645b913c5ef6d`
- workspace release: `20260815T212909Z-5bd66d5e4bdc-e759e520fd2b`
- workspace SHA-256: `e759e520fd2b07213195a45408b8ac4a67673cd958870b41d0263c795b5ad041`
- exact runner SHA-256: `08738df042bbfc4c05a336b7f8164531de510f535d2043c20199028d158c087d`
- exact report SHA-256: `b20e503fce4adf91d37ff55bdf81ce956d4e7e3c08a188902611294fb6ef2dc2`
- MoveIt stderr SHA-256: `f5366fa04059a56141f97f375d8decc1aff25805feb97bd2a69493e848efcadd`
- execution log SHA-256: `ce6472eda0fa4ba7f0a70b34da47775b5a5d269ae684ed2295b9667081875e19`
- pre-shutdown summary SHA-256: `d18583b5e5ded4ae0dfe0475d4120cf99756ec9ea8754a9def7c2b2fe7dbf60d`

## Remote prerequisites

- Read-only audit passed before the first create: Python 3.12.3, `pip check`
  passed, Gym-ALOHA 0.1.4, MuJoCo 3.8.1, Mink 1.2.0, qpsolvers 4.13.0,
  DAQP 0.8.7, ROS Humble and pinned Interbotix descriptions present.
- Mesh and URDF-source manifests matched Plan054 hashes before upload:
  `63edd159854e2eaa99bbe640c76b6b65e0e0ac517081689796d31c1db579e04d`
  and `c74d4712fe206303ef081a3f81c65c6aa2a8b1a0b29b3d52b174e9c6cbc8ccf1`.
- Two watchdogs were armed before any workspace write and verified from an
  independent SSH reconnect:
  - task-low watchdog, deadline `2026-08-16 08:20:00 +0800`, `sleep` uid 65534,
    nice 19;
  - hard watchdog, deadline `2026-08-16 09:40:00 +0800`, `sleep` uid 0,
    nice 19.
- Workspace, MoveIt runtime (`joint-margin-selection-006`), and the two
  hash-bound `runs/` probe/launcher file sets omitted from the frozen archive
  were all create-only uploads. No existing remote path was overwritten.
- No optimizer step, CUDA training, download, validation/hidden episode, later
  seed, recovery label, or nested Docker was executed.

## Attempt history

- `athena-plan054-exact-001`: stopped in runner preflight because
  `gym_aloha.__version__` does not exist; evaluator never started. Preserved.
- `athena-plan054-exact-002`: stopped at plan hash binding because the frozen
  workspace archive omitted ignored `runs/` files referenced by the config.
  Preserved.
- `athena-plan054-exact-003`: ran the registered train-only exact stage.

## Exact result

- calibration: episode `2`, seed `10`, reward `4`, `294` steps.
- exact: `0/1`, `423` steps, final phase `lift`.
- phase visits: `{open: 15, approach: 241, orient: 97, descend: 28, grasp: 7,
  lift: 34}`.
- teacher failure: observed object-to-end-effector transform exceeded the
  unchanged 45 mm grasp-drift limit.
- path planner: 6 attempts, 188 waypoints, 182 reference-reuse commands, 44
  waypoint advancements, 6 terminal-control activations, 5 completions.
- expanded orientation target budget trust-region events: **0**; all registered
  trust-region event counters: **0**.
- IK failures, adapter-clip failures, joint-limit projections, commanded and
  observed margin breaches, unexpected collisions: all **0**.

Plan054 remains rejected rather than repaired in place. The new
trust-region budget axis was not exercised before the grasp was lost.

## Gate decision

No Plan055. Tuning, development, collection, policy-Gate, validation/hidden,
recovery labels and CUDA training remain sealed. Exact remains failed.

## Evidence

- `runs/m2-smolvla-aloha-geometry-teacher-054/remote-athena-plan054-exact-003/`
- `reports/training/m2-smolvla-athena-plan054-remote-exact-audit-2026-08-16.md`
- `reports/training/m2-smolvla-athena-plan054-remote-exact-audit-2026-08-16.json`
