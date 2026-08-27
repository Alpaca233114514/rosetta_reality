# M2 SmolVLA Athena Plan044 runtime identity audit

Status: **passed before train-only exact**.

Plan044 uses the official MoveIt 2.5.9 LMA plugin in its documented
`position_only_ik` mode for two added arm groups. The runtime identity binds
those groups, zero rotational IK weight, OMPL RRTConnect, per-request seed
reset, the unchanged 22-link collision model, and a maximum `1e-5` rad terminal
goal normalization that is rechecked with `PlanningScene::isPathValid`.

The exact Plan043 failure request now succeeds at the full Cartesian fraction
on the first deterministic IK attempt. Its maximum position error is
`0.0008290217 m`, orientation relaxation is `0.0299898344 rad`, and minimum
path joint-limit margin is `0.0594898606 rad`, above the unchanged
`0.0454046226 rad` registered margin.

Two fresh sidecars using the same image and seed returned exactly equal goal
and trajectory vectors with equal attempt counts. Three earlier repeat attempts
that exposed process-random-stream and terminal-tolerance effects remain
preserved as negative evidence in the JSON companion.

No dataset row, validation or hidden episode, later-stage seed, recovery label,
model weight, or CUDA training path was opened by this audit.
