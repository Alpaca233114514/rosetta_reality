# Iris 002 completed: small offline gains, generalization acceptance failed

`iris-k-scene-20260913-002` completed all 27 registered stages without a runtime
error. Both fresh arms trained for 1280 updates after separate two-step smokes:
2564 optimizer updates total. Worker time was 2714.806 seconds (45.25 minutes).
The scientific result is **negative**: the regularizer has a small beneficial
effect, but does not repair development-scene generalization. M2 is incomplete;
no new Gate 3/4 was measured.

## Registered identity and intervention

Plan: `m2-smolvla-iris-plan-002-2026-09-13.md`. Source commit `5d285f5` includes
the metric-name repair `c308b8d`. Content-addressed workspace identity:
`20260913T135230Z-5d285f5c8297-3ba03ac32579`; archive SHA256
`3ba03ac32579b5c403014f32c69192aa2145f083fd4e943b666e9e089a41893a`.
Template SHA256:
`35516b4302714971db57ef50da3dc14b53ee48bf40240f517ac6db2633beb1b2`.

The single learning axis remained image K scene variance, lambda 0 versus 0.01.
Both arms started from pinned base, using train40 frame 0, batch 4 and the same
registered seed, optimizer, full-run scheduler and Action Contract. No old smoke
optimizer state was reused. This retains the native bounded visual-overfit launch
mode; it is not a full-trajectory formal-training claim. The fixed endpoint is
1280; no development checkpoint search or hidden-scene access occurred.

CUDA calibration and lambda-zero equivalence passed. Both two-step smokes and
their saved-processor independent reloads passed. The repaired `image_k_scene_*`
metrics logged successfully with the public sanitizer unchanged. Main training
completed and all 320/640/960/1280 recovery checkpoint inventories and same-step
metrics passed the registered checks. The two fixed endpoint exports were then
loaded in separate CUDA processes for evaluation and independent reload.

## Observed result

Both arms passed train-fit checks for all/joint/gripper groups in all four fixed
noise conditions (4/4 each). This does not imply development or closed-loop success.

The following means average the four fixed-noise measurements on five development
scenes over the full 50-action chunk. Joint MAE is in radians; gripper MAE uses
the registered normalized command scale.

| Metric | Control | Treatment | Relative reduction |
|---|---:|---:|---:|
| Left joint MAE | 0.061673 | 0.061034 | 1.04% |
| Left joint MSE | 0.009482 | 0.009272 | 2.21% |
| Right joint MAE | 0.062214 | 0.061971 | 0.39% |
| Left gripper MAE | 0.064954 | 0.064674 | 0.43% |
| Right gripper MAE | 0.067741 | 0.067238 | 0.74% |

Full-chunk left-joint MAE/MSE improve in all four noise conditions and all five
leave-one-scene-out comparisons. Full-chunk right-joint and gripper errors also
do not regress under the registered comparisons. However, every full-chunk metric
fails the requirement to beat both train-only mean/median constants. For example,
the train-mean left-joint MAE baseline is 0.047658, materially below treatment
0.061034. Some first-action metrics also regress. The full fixed acceptance fails;
these small gains do not identify a unique cause or justify claiming the problem
is solved. Development remains development, not an independent hidden test.

## Integrity, recovery and lifecycle

Each main arm's independent reload exactly matched seven arrays and 370,900
scalars, including normalized/physical predictions, targets, gripper internals,
noise and masks. Saved processor state was loaded without normalization overrides.
Frozen-parameter and checkpoint checks passed. Main evaluation used 720 policy
inferences across the two arms and their independent reloads.

Recovered archive: `runs/iris-recovered-20260913-002/archive.tar`, 2,414,888,960 bytes.
All 125 files (2,414,642,244 payload bytes), including both exported policies and
processors, passed type/file-set/size/SHA checks before the receipt was delivered.
Archive SHA256:
`0839eb1f2827ee8f69a0804c15391cfe12e43bd85daaea3ca16710ade32c24da`.
Manifest SHA256:
`8f1f80d9676c2a8f25c587d9bf7462fe41778c6fa992681096f7284dbcb9812b`.

Local verification ran in the fixed offline Linux Docker image, without loading
model weights or datasets. It independently recomputed 200 physical-error scalar
values, verified both main reload array sets exactly and confirmed all evaluated
slots valid. See `iris-002-validation-20260913/verify_results.py` and the companion
JSON. This was a numeric/byte verification, not a local policy forward or Gate run.
Full optimizer recovery checkpoints remain on the retained worker, with their
inventories recovered; they are not part of the selected-export local backup.

Luna's monitor failed to reliably execute its shell commands. The user instructed
the parent to take over; the parent performed five-minute checks through the rest
of training, then handled reload boundaries, delivery and shutdown confirmation.
The training supervisor and independent watchdog remained separate and active.

After the verified receipt, the guarded shutdown ran. A fresh AutoDL console
refresh confirmed the exact registered instance powered off; it was not released.
An attempted shutdown-request retrieval encountered SSH closing and left its empty
local receiver as failure evidence. The request itself remains pending for a future
authorized window; do not reopen solely to retrieve it. The earlier 001 request
was retrieved during the authorized 002 startup window; its original failure report
and template remain unchanged.

The run is closed. Preserve this negative result and the existing checkpoints;
do not automatically increase lambda, select another checkpoint, resume, or launch
another learning axis on the basis of training fit alone.
