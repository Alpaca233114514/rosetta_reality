# M2 SmolVLA Athena plan 027 exact audit (2026-08-15)

## Outcome

Athena plan `027` failed train-only exact `0/1`; tuning, development,
collection, policy-Gate, validation, hidden and recovery-label gates remain
sealed. No optimizer step, CUDA training, model/data download or label write
ran.

The immutable exact report SHA-256 is
`7dd47ab9eada44ba8ca1105050ba6af784259f39bd55eb079f7025fe53999b73`.
Train episode 2 / seed 10 calibration again reached reward `4` in 294 steps.
Exact executed 141 steps, remained in `approach`, and stopped at step 140 with
`start_state_outside_joint_path_margin`.

## What the execution trace proved

The planner did not command an arm joint inside the registered `0.01`-rad
margin. There were zero commanded-margin breach events, and the smallest
commanded margin was `0.013024463729858216` rad at step 132 on
`left_wrist_rotate`.

The first and only observed breach occurred after step 139. The absolute
command for `left_wrist_rotate` retained `0.015719308929443184` rad to its
upper bound, but the observed controller state moved another
`0.008002996444702148` rad toward that bound and retained only
`0.007716312484741028` rad. The next planning request at step 140 correctly
failed closed because its observed start state was already inside the
registered path margin.

Across all twelve arm joints and all 140 executed command/observation pairs,
the maximum same-direction overshoot toward a joint limit was
`0.03540462255477905` rad. This is the uniform train-only disturbance bound
registered by the next constraint-tightening diagnostic; it is not evidence
about tuning or development seeds.

This resolves plan `026`'s ambiguity: the immediate boundary is
execution/controller tracking reserve between a safe command and the next
observation, not the official MoveIt path constraint or a command that first
crosses the margin. The physical `0.01`-rad margin and the frozen
`0.001` / `0.003`-m pose gates must not be relaxed. The next single-axis plan,
if separately preregistered, belongs at a measured execution reserve or
closed-loop feedback guard.

## Runtime and evidence

The content-addressed workspace archive SHA-256 was
`f2c4662a2dabff46e53a88ce1780679622473a298779d2e73679ed5d3c84e002`.
Remote static attempt `001` stopped before tests because the Python 3.12 venv
could not see Ubuntu's installed `python3-typeguard`. A task-scoped, no-network
copy of that official package repaired only the import boundary. Attempt `002`
then passed Ruff, all 39 focused tests, Gym/MoveIt FK parity within
`3.188872858294072e-16` m and `0` rad, and the positive/negative direct path
smoke. Exact attempt `003` stopped before the evaluator because two task parent
directories were absent; create-only attempt `004` produced the authoritative
report above.

The 18:00 hard watchdog and independently reconnected 16:40 low-priority
watchdog were live when the final summary was written. The superseded first
task-watchdog record was preserved. The 92-member evidence bundle was checked
for unsafe paths and hash-verified after transfer. Its SHA-256 is
`e6fca0350547889abb1f16a0130d9b099ab2495a78ed9b87201c0f39938c615e`;
the remote final-summary SHA-256 is
`34b181199aeedfdb71a3da903e98b61b9a9074bfcc37d6c5fc12fa121cb93906`.

After local validation, create-only manual shutdown record
`athena-plan027-complete-20260815T072222Z-4024` launched helper PID `4035`.
The helper called `/usr/bin/shutdown` with no arguments. A strict host-key SSH
reconnect then returned exit code `255` with `Connection reset`, confirming the
instance stopped accepting SSH well before the 18:00 hard deadline.
