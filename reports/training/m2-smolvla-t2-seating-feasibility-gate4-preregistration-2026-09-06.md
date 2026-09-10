# Gate 4 privileged seating feasibility preregistration — 2026-09-06

Identity: `m2-t2-seating-feasibility-gate4-2026-09-06`. Status: registered,
execution pending. `gating=false`. The JSON companion freezes source identities.

Hypothesis: a privileged live-geometry position controller may seat some of the
exact Gate 4 resets, separating demonstrated reachability from policy failure.
The user's authorization opens simulation seeds 1000–1004 for this diagnostic
only, overriding their sealed status in the older teacher-gate registration.
No dataset is read; hidden episodes [31,6,1,24,5], collection and development
groups remain sealed. No model, training, AutoDL, download or protocol edit.

## Frozen controller and differences from 1901

Reuse `_privileged_seater_step` directly from the September 3 instrument:
live peg-to-ideal-socket composition; live socket-to-left-site positional
offset and live left orientation; live right EEF hold; both grippers 0.09;
Mink default constrained IK, unshifted then the same eight ordered multistart
shifts when residual exceeds 5 mm; max arm delta 0.06 rad per step; unchanged
absolute-position Action Contract at 50 Hz. No search/tolerance retuning.

Calibration remains seed 1903, with the historical pre-terminal relative-pose
capture convention preserved (it is an approximate reference, not the exact
task-success geometry). A second seed-1903 sanity control hands over at INSERT
entry and must succeed with at least one privileged action before target seeds
run. Failure of either control stops execution and is inconclusive.

Explicit differences: target seeds are 1000–1004 instead of 1901; the **total**
budget from reset is 500 actions, including teacher preparation (not 500 extra
privileged actions, and not the old extra 60-action budget). Hand over at INSERT
stall as before. No handoff on early teacher refusal: record the earlier phase
as a controller limitation, not evidence of impossible seating. No extra reset
or step after termination/time limit. Current teacher source has a later HOLD
branch; explicitly `GeometricEscapeSettings(hold_enabled=False)` recovers the
original candidate-003 behavior without modifying that existing file.

Reset is inherited without override from `GymAlohaEnvironment`; constructor
uses `maximum_episode_steps=500`, default observation and render settings,
exactly as `scripts/smolvla_sim_gate.py`. Tests compare actual reset robot state,
all camera tensors and object/contact snapshots bitwise at each target seed.

## Success, stopping and interpretation

`seatable` means reward >=4, done and `is_success=true` within 500 total steps,
with finite commands inside the Action Contract and arm delta <=0.06 rad
(2e-6 float32 comparison allowance). Timeout alone is never success. State
limit violations and collisions are recorded separately; this diagnostic is
not a safety/teacher gate pass. Stop each rollout at task success, truncation,
teacher refusal or budget. Nonfinite state/action, out-of-bound command, all-IK
failure, source/image drift or failed calibration/control stops the campaign
as an instrument failure. Preserve partial evidence; do not reinterpret it as
five valid failures. No automatic retry or tuning after observing results.

Requested decision rule retained as a hypothesis to assess: >=1/5 seatable
locates a learning/interaction gap on the solved seeds; 0/5 suggests a Gate 4
position-control envelope ceiling and motivates a protocol review. Mixed
results are split per seed. **A failed finite multistart controller is not an
exhaustive reachability proof.** Therefore 0/5 will be reported as "no solution
found by this registered controller", with a protocol-ceiling interpretation
explicitly qualified. Failures before INSERT cannot establish a seating wall.
Even 5/5 does not establish that DAgger/data scale is the unique remaining cause.
The seven policy 0/5 records and M2 non-acceptance remain immutable.

Residual: pre-step Euclidean distance between actual socket body position and
the live peg-composed calibrated socket position, converted m -> mm; record
privileged-only minimum (same 1901 scope), whole-rollout minimum, final pre-step
value, steps, reward, handoff and failure phase. No residual threshold replaces
the environment's success signal.

## Runtime and delivery

WSL Bash -> `scripts/run_m2_container.sh vla-sim-xpu`, offline local Docker,
existing image only, default 6 GB memory/no extra swap and 2 CPU quota. Pure
simulation/IK; no policy weights. Expected bounded work: two 500-step controls
plus five 500-step targets (at most 3500 actions; wall time depends on IK).
Run focused pytest including reset parity and Ruff in that container before
the probe. Evidence is create-only under
`runs/m2-t2-seating-feasibility-gate4-2026-09-06/` with source and image identity,
calibration/control traces and one JSON per seed. Results go to the dated
report pair; append current evidence to the architecture. No commit/push.

User decisions after a qualified 0/5: whether to preregister tolerance changes,
a new demonstrated-seatable pose group, or a different control mode. Prefer
first establishing successful controls and localizing residual/contact geometry;
do not lower success tolerances merely to reclassify failures. No option is
selected or authorized by this registration.

Pre-execution formatting amendment: the calibration progress print was wrapped
to meet the repository's 100-character Ruff line limit. Instrument SHA-256
changed from `54e0508473d4143ebcb65b4f7ec6d8d74330c4146bbae10856b7b94a8ff329a3`
to `4257afe72d180884a362d79467399bc8bb15f35ac7dd56cafe1a617caa0ad044` before
any container execution. Controller, criteria and seed semantics are unchanged.
