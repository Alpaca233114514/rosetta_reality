# Iris local furnace pipeline completion

Status: local implementation and synthetic checks passed. CUDA admission,
calibration, optimizer smoke, training, export and independent real checkpoint
reload have **not run**. M2 remains incomplete; Gate 3/4 are not measured.

This completes the launch pipeline left pending by
`m2-smolvla-iris-local-repair-2026-09-13.md`. The user has requested local bug
completion while the registered worker is shut down; this module made no SSH
connection. Reopen the registered worker only when the user supplies that window.

## Frozen experiment

- Run: `iris-k-scene-20260913-001`; two fresh pinned-base arms, lambda 0 and 0.01.
- Only intervention: image K scene variance at layers 1,3,5,7,9,11,13,15,
  native image tokens [0,64), preserving the full 241-token projection.
- Historical recipe: `iris-preparation-20260913/historical-main1280.yaml`, SHA256
  `70fb068fc732b02c4bfd3cf04e70e5465e3094e8d32750bb670009e2f2f2507b`.
- Train40 frame 0, batch 4, seed 20260809, historical AdamW and scheduler;
  each arm has a separate fresh 2-step smoke followed by a fresh 1280-step run.
  Native v2 bounded visual-overfit smoke mode is retained intentionally. This
  is not full-trajectory formal training. No smoke optimizer state is reused.
- Maximum optimizer updates: 2564. Save recovery checkpoints at 320/640/960/1280.
  Final step 1280 is fixed before evaluation; no dev checkpoint selection search.
- Fresh train40-only calibration is generated on CUDA and SHA-bound into four
  training plans before any optimizer step. The shared input reader loads labels;
  calibration statistics do not use them. The native loss/gradient zero-control
  check uses training labels and restores CPU/CUDA RNG. No dev or hidden calibration.
- Shared work budget 3600 seconds, protected shutdown at 4200 seconds. Initial
  free storage requirement 23 GiB; main admission requires 17 GiB remaining and
  both actual smoke timings. Allocated CUDA <=8 GiB, reserved <=10 GiB, process
  tree RSS <=8 GiB. Do not expand budgets to force admission.

## Executable ownership and stop behavior

`scripts/iris_protocol.py` owns create-only plans and prerequisite ordering;
`iris_calibration.py` owns fresh-base calibration and lambda-zero equivalence;
`iris_stage.py` dispatches doctor, real-data tests, benchmark, batch 1/4 forwards,
smokes, independent reload checks, main admission/training, export and analysis.
`run_iris_furnace.py` owns the detached supervisor and independent watchdog.
`iris_delivery.py` owns bounded regular-file archives, SHA verification and receipts.
`analyze_iris.py` owns paired offline acceptance and preserves negative results.

Each completed stage requires an exit-zero receipt before the next stage. Any
source/config identity drift, skipped required data test, nonfinite tensor,
missing gradient, memory/storage violation, changed frozen parameter, mismatched
checkpoint or reload, or insufficient time stops progression. No retry/resume.
The watchdog uses PID plus process start ticks; it does not terminate a completed
worker during result transfer. Shutdown still uses the existing guarded helper;
it never releases the instance. Before supervisor registration, a source/profile
failure can exit without installing a watchdog, so inspect launch status promptly.

Export loads saved processor state without normalization overrides. Independent
processes must reproduce all seven recorded arrays exactly. Analysis requires
train-fit acceptance, improvement in left-joint error across all four noise
conditions and leave-one-scene-out comparisons, no right-joint/gripper regression,
and beating train-only constant baselines. Even a passing offline result is not
closed-loop success. A scientific negative result is retained and transferred.

## Local validation

`wsl.exe bash scripts/check_iris_preparation.sh` completed with **117 passed**,
Ruff passed and all 15 Python files formatted. All three launch/receive shell
scripts passed individual `bash -n` parsing. The fixed offline Docker image was
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`.
Tests ran with 2 GiB RAM, 2 CPUs, network disabled and no model/data loading.
One warning reported zero XPU devices; these are CPU/synthetic checks, not CUDA.

Coverage includes native persisted-YAML resolution, no-overwrite plans,
regularizer loss/gradient behavior, checkpoint boundaries, scientific counterexamples,
archive traversal/symlink/duplicate rejection and round trip, stale PID rejection,
and completed-worker transfer protection. No local or remote training occurred.

## Sealed handoff

Template: `iris-furnace-template-2026-09-13.json`, SHA256
`307567ad0a942fc3bb137550eabdeb50ec896de63c75e7cbab9008b77bccfcc1`.
It binds this exact workspace source snapshot, including earlier uncommitted
diagnostic sources needed by the workspace. A Git commit alone does not recreate
that entire snapshot; use the content-addressed staging workflow and verify all
template seals. A later source edit requires a newly reviewed template identity.

After the user opens the registered SSH window:

1. Verify the registered worker and current storage/cache identities; stage a new
   content-addressed workspace with `scripts/stage_iris_preparation.sh`.
2. Invoke `scripts/launch_iris_from_wsl.sh` with that exact fresh workspace, the
   template's repository-relative path and its SHA above. Inspect supervisor,
   watchdog and first-stage output; do not treat a returned PID as training success.
3. Follow registered stage boundaries and the existing five-minute training status
   cadence. Verify first optimizer steps, finite loss/gradients, memory, checkpoint
   evidence and Trackio state. Any prerequisite failure stops the furnace.
4. At `package.json`, run `scripts/receive_iris_from_wsl.sh` for that workspace;
   full archive/file SHA verification precedes the receipt. If delivery fails,
   retain remote artifacts; do not release the instance. Confirm protected shutdown
   separately from request-file existence and retrieve the request at the next window.
5. Record actual results and commit the next completed module. No push or merge is
   authorized. Gate 3/4 remain separate work after successful artifact recovery.
