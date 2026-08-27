# M2 SmolVLA Athena plan 028 exact audit (2026-08-15)

## Outcome

Athena plan `028` failed train-only exact `0/1`; tuning, development,
collection, policy-Gate, validation, hidden and recovery-label gates remain
sealed. No optimizer step, CUDA training, model/data download or label write
ran.

The immutable exact report SHA-256 is
`0848857b401a10635cd7d66da4dd7cd339f8417abd9982141669c01a05ccbb64`.
Train episode 2 / seed 10 calibration again reached reward `4` in 294 steps.
Exact recorded 107 steps, remained in `approach`, and stopped before executing
the step-106 command with `start_state_outside_joint_path_margin`.

## What the tightened constraint proved

The registered robust reserve was active in both upstream Mink
`ConfigurationLimit` and official MoveIt `JointConstraint`. The unchanged
physical margin was `0.01` rad, the train-only tracking reserve was
`0.03540462255477905` rad, and the resulting command margin was
`0.04540462255477905` rad.

There were zero commanded and zero observed physical-margin breach events.
The smallest commanded margin was `0.04555977828979474` rad at step 105 on
`right_wrist_rotate`, so every command satisfied the tightened bound. The next
observed state retained `0.039528980331420716` rad to the same lower bound,
which remained physically safe by `0.029528980331420716` rad but was
`0.005875642223358334` rad outside the tightened command set. The step-106
MoveIt request therefore failed closed instead of pretending that this
observed state satisfied its path constraint.

The failure had IK / projected errors of `0.011678277559489273` /
`0.011688468407996477`, above the unchanged `0.001` / `0.003` gates. Before
that failure, successful official paths retained at least
`0.052193536834716614` rad along the path and
`0.16251630740252576` rad at the bounded next waypoint. No start-bound
reconciliation, adapter clip or joint-limit projection was used.

Plan `028` therefore validates the safety effect of constraint tightening but
does not establish recursive feasibility. The same tightened set was used for
future commands and for the next observed planning start. Controller tracking
can leave that tightened set while the robot remains inside the physical safe
set, and there is no registered backup controller that returns such a state
to the tightened set. Simply widening the MoveIt start tolerance, lowering the
physical margin or relaxing the pose gates would hide this control-boundary
failure and is not allowed.

The next single-axis hypothesis, if separately preregistered, should add an
explicit joint-limit retreat / backup feedback phase for observations inside
the physical safe set but outside the tightened command set. It must move
monotonically back into the tightened set before resuming pose tracking or
MoveIt planning, retain the existing official Mink/MoveIt constraints for
normal motion, and prove physical-margin safety on every intermediate command
and observation.

## Runtime and evidence

The content-addressed workspace archive SHA-256 was
`574d6a7eae5fe42d21face90c9edd294c450858d0d950b502322f0507ff32171`.
Remote static attempt `001` failed before launch because the local PowerShell
wrapper quoted a create-only remote Python writer incorrectly; the empty
attempt was preserved. Create-only attempt `002` then passed Ruff, all 43
focused tests, Gym/MoveIt FK parity within `3.188872858294072e-16` m and `0`
rad, plan-boundary validation, and the tightened positive/negative path smoke.
The positive path retained `0.2557759341471746` rad, while a physically safe
`0.02`-rad start was correctly rejected from the
`0.04540462255477905`-rad tightened set.

The independently reconnected 18:00 hard watchdog and 17:20 low-priority
watchdog were live when the final summary was written. The latter ran at nice
`19` with idle I/O priority. The 59-member evidence bundle was checked for
unsafe paths and links before transfer and for unsafe paths again locally. Its
SHA-256 is
`2ed7a88d3746fb38d7e08be060935e4b2eeaef274587ee3c91e6695674fa706b`;
the remote final-summary SHA-256 is
`664a9039a0dcc743b3b70ca4947ddd4d7f5a2b70bdfa36ab657450870549a236`.

After report, summary and bundle hashes matched locally, create-only manual
shutdown record `manual-shutdown-athena-plan028-complete-001` launched helper
PID `3766`. The helper called `/usr/bin/shutdown` with no arguments after 30
seconds. A strict host-key SSH reconnect then returned exit code `255` with
`Connection reset`, confirming the instance stopped accepting SSH before both
watchdog deadlines.
