# M2 SmolVLA Athena Plan047 local exact audit

Status: **failed train-only exact; all later gates remain sealed**.

Plan047 exercised deterministic maximum-margin selection across the fixed
official LMA multistart candidates. Where multiple valid candidates existed,
the registered objective was applied. The run reached orient step 256 with no
clip, joint projection, unexpected collision, commanded physical-margin
breach, or observed physical-margin breach.

The failure boundary is not another candidate-ranking failure. The first
orient request and every registered full-pose Cartesian backoff returned zero
valid candidates. The prior approach position-priority completion retained
about 0.03 rad of bounded orientation relaxation while right wrist rotation had
only 0.00141 rad above the unchanged command margin. A discontinuous full-pose
orient request cannot re-enter the feasible manifold from that endpoint.

The next single-axis hypothesis is bounded orientation progress at the
already-reached approach position. It must retain the official full-pose LMA
and OMPL checks, frozen pose tolerances, frozen physical and command margins,
and all seed and label seals.

Tuning, development, collection, policy Gate seeds, validation/hidden episodes,
recovery labels and CUDA training were not opened.
