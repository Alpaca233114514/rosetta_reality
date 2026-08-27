# Athena Plan047 runtime identity audit

Status: `passed_before_exact`.

Plan047 keeps the official MoveIt 2.5.9 LMA candidate generator and OMPL
RRTConnect planner. Its only registered change is deterministic selection among
already-valid IK candidates: maximize the minimum arm-joint limit margin, then
minimize the maximum joint displacement from the start, then retain the lower
fixed attempt index.

The new sidecar reports the registered selection identity and diagnostics. Two
fresh-sidecar position-priority requests produced identical goals, trajectories,
attempt counts, candidate counts, selected attempts, selected margins, and
selected start deltas. Two full-pose regression requests passed. Five MuJoCo /
MoveIt model-parity samples passed with maximum position error
`3.188872858294072e-16 m` and zero orientation error.

Replaying the Plan046 failure diagnostic preserved the frozen candidate
generator boundary: requests with no valid LMA candidate still failed, while
multi-candidate requests selected and reported the maximum-margin branch. The
near-limit multi-candidate request improved its goal margin from
`0.0475104891201368 rad` to `0.04802262733195661 rad`. This is diagnostic
evidence only; train-only exact remains the acceptance authority.

Tuning, development, collection, policy Gate seeds, validation/hidden inputs,
recovery labels, and CUDA training remain sealed.
