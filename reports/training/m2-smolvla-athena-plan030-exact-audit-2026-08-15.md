# Athena Plan 030 train-only exact audit

Date: 2026-08-15  
Status: **exact failed `0/1`; joint-limit safety held; all later gates remain sealed**

## Outcome

Athena reproduced the registered calibration control at reward `4` in `294`
steps, then ran only Plan 030's train episode `2`, simulator seed `10` exact.
The exact exhausted its `500`-step budget with reward `0`, success `false` and
final phase `orient`. It visited `open` for 15 steps, `approach` for 121 steps
and `orient` for 364 steps; neither object had grasp contact at the end.

The new official recovery boundary did not fail. Across 131 planner attempts
and 23 start-state recovery events, there were zero teacher, IK, adapter-clip,
commanded-margin, observed-margin or joint-limit-projection failures. Minimum
commanded margin was `0.04540456779479962` rad, minimum constrained-path margin
was `0.045446080555739954` rad and minimum observed physical margin was
`0.01990832336425763` rad, above the unchanged `0.01`-rad physical floor.
Therefore Plan 030 is safe under this exact but does not complete the task.

## Durable identities

- Plan SHA256: `763ba45ed8ca8d84120dae99ca1375d915427e77f290784437d291575ad25f4d`.
- Workspace release: `20260815T092123Z-5bd66d5e4bdc-357ee507c1ec`;
  archive SHA256
  `357ee507c1ecee7ca0386efc3e62b8c47362d085ae42d30c713e84177e319ceb`.
- Runtime image ID:
  `sha256:fbf64d43c017d51ebd00cee07aebeb9abb0d79b75e5c9cc477ba9a30acee8d44`;
  executable SHA256
  `900001236058db02e380db1838a0bca70fd07baad236e6c44cb0ed959b3fafaf`.
- Authoritative exact report SHA256:
  `aaa033c6a1740bac40d9589b01372c059e9d242099b2aa6a5ab4c4f8d1029fa3`.
- Local ignored evidence:
  `runs/m2-smolvla-aloha-geometry-teacher-030/remote-athena-plan030-20260815/`.

## Instance and dependency evidence

Remote static attempt `athena-plan030-static-001` passed Ruff, 54 focused
tests, exact FK parity, normal/recovery/physical-negative requests, and the
Plan 030 recovery smoke. Its recovery prefix contained 20 waypoints, retained
minimum physical margin `0.01998734641020672` rad and made positive first-step
margin progress `0.10163691515093953` rad.

The first exact attempt stopped before reading dataset rows because `pyarrow`
was absent and created no exact report. That failure remains immutable. The
image environment was then completed in place with `pyarrow==25.0.0`; its final
manifest SHA256 is
`7941a4fcb91d325adb097e0645b9b063085272c56bb4187d13d94d0bae634315`,
its sorted `pip freeze` SHA256 is
`fbbbc458458c41f472db0db95d69e1e285661237490320e8faf4d51642eb5168`,
and both import checks and `pip check` passed. The environment is on the system
image disk so it is eligible for the user's image save. No CUDA/driver package,
model or dataset was downloaded or changed.

## Research boundary

Plan 030 changes only the acceptance of the official MoveIt adapter's recovery
prefix. The result supports that validator and the robust joint-limit safety
contract, but it does not support task completion. The current failure boundary
is a safe but non-progressing `orient` phase, not another joint-limit rejection.
The next work must diagnose orient-phase object-geometry feedback and waypoint
execution locally, then preregister one new controlled axis. It must not relax
the frozen pose thresholds, physical margin or start-state tolerance.

Tuning seed 1900, development seeds 2000--2004, collection seeds 3000--3004,
policy Gate seeds 1000--1004, validation/hidden data and recovery-label writes
remain sealed. This run used zero optimizer steps and no CUDA training.

The JSON companion is the machine-readable authority for package versions,
attempt hashes, exact metrics and shutdown state. After final dependency,
report, process and GPU verification, create-only shutdown helper
`manual-shutdown-athena-plan030-complete-001` was armed with PID `7558` and
sleep child `7562`; it waits 60 seconds and then invokes `/usr/bin/shutdown`
with no arguments. The subsequent strict-host-key SSH check returned exit code
`255` with `Connection reset`, confirming that the instance stopped accepting
connections.
