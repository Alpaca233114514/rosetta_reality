# M2 SmolVLA Athena Plan046 local exact audit

Status: **failed train-only exact; all later gates remain sealed**.

The official full-pose Cartesian backoff was exercised and advanced the run
from Plan045's step 225 to orient step 257. It produced two additional terminal
completions and 32 additional collision-checked waypoints. No clip, unexpected
collision, commanded physical-margin breach, or observed physical-margin
breach occurred.

The remaining failure is now tied to the active right wrist limit. Only
0.00128 rad remained above the unchanged tightened command margin. A smaller
0.025 effective Cartesian fraction is solvable, but its first-valid LMA
continuation consumes another 0.000615 rad of that reserve. Further fraction
shrinking would postpone rather than remove the boundary.

The next single-axis hypothesis is therefore joint-limit-aware selection among
deterministic official LMA candidates, not a looser pose tolerance, smaller
joint margin, additional seed set, or label path.

Tuning, development, collection, policy Gate seeds, validation/hidden episodes,
recovery labels and CUDA training were not opened.
