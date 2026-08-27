# M2 SmolVLA Athena plan 026 exact audit (2026-08-15)

## Outcome

Athena plan `026` failed train-only exact `0/1`; tuning, development,
collection, policy-Gate, validation, hidden and recovery-label gates remain
sealed. No optimizer step or CUDA training ran.

The immutable exact report SHA-256 is
`f010221c3c94cf840b554a1dd646b5ecc6f3eb8fe47913fb03e22c40e9b2f3cd`.
Train episode 2 / seed 10 calibration still reproduced reward `4` in 294
steps. Exact ran 114 steps, stayed in `approach`, attempted official MoveIt
planning four times and executed three bounded RRTConnect waypoints.

## What plan 026 proved

The official MoveIt all-path constraint operated as registered. Successful
paths retained at least `0.030544891433715637` rad of physical joint margin,
and the bounded next waypoints retained at least `0.25811082074868175` rad.
There were no adapter clips, joint-limit projection events, collision-resource
identity failures or path-constraint identity failures.

At step 113, the next observed start state had only
`0.00961627014160138` rad of physical margin, a
`0.0003837298583986206`-rad shortfall from the registered `0.01` rad. The
official sidecar therefore stopped before IK/OMPL with
`start_state_outside_joint_path_margin`. The contemporaneous Mink weighted and
projected errors were `0.010972119914633108` and `0.010960835099750222`.

This separates the remaining failure from planned-path safety: accepted
MoveIt paths obeyed the margin, while a later observed simulator state entered
the protected band before the fourth fallback request. The next safe axis is
commanded-versus-observed joint-margin instrumentation and execution-feedback
or controller-tracking diagnosis. Neither the pose gates nor the `0.01`-rad
margin may be relaxed to hide this boundary.

## Runtime and static evidence

The content-addressed workspace archive SHA-256 was
`7394c4ea5d4cd577ae9996ae591e8beeb927c5f707923fcbc0da8300dfe98d8e`.
Safe extraction omitted only the repository's absolute local WSL
`checkpoints` symbolic link; it was not needed by geometry exact. The new C++
sidecar compiled to the preregistered
`e5da00ce9fe665d9d28edbd0bfa075df608418f2e5a471685b94136ff310cc6d`
binary. Ruff passed, all 37 focused tests passed, five-sample FK parity stayed
within `3.188872858294072e-16` m and `0` rad, and the direct positive/negative
joint-margin smoke passed.

The 16:30 hard watchdog and the independently verified 16:10 low-priority
watchdog were both armed. After the exact report, logs and 25-member evidence
bundle were transferred and hash-verified locally, a create-only manual
shutdown record was written and the instance was shut down early. The local
summary and evidence-bundle SHA-256 values are respectively
`194ff35c78c2cbe9d82b4e62568d69415a8b54618cf4b0055d9c44cab01aedb1`
and `f15afed9ce99f9077bff98853496ec0ec7bd374d985f03552bf8baa295354f82`.
