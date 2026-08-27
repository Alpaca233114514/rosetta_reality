# M2 SmolVLA Athena Plan045 runtime identity audit

Status: **passed before train-only exact**.

Plan045 keeps the official MoveIt 2.5.9 LMA plugin and its documented
`position_only_ik` mode. The single registered change replaces unstable
enumeration of every subgroup under `bimanual` with explicit lookup of the
original `left_arm` and `right_arm` full-pose groups. The runtime identity
reports that selection contract directly.

Two ordinary full-pose requests now reach deterministic LMA and OMPL and both
succeed; neither reproduces Plan044's pre-solver subgroup-count rejection. The
registered Plan043 failure target still succeeds at full Cartesian fraction on
the first position-only LMA attempt, with `0.0008290217 m` maximum position
error and `0.0594898606 rad` minimum path joint-limit margin. Two fresh
sidecars again return exactly equal goal and trajectory vectors.

Five MoveIt/MuJoCo FK parity samples pass with maximum position error
`3.19e-16 m` and zero measured orientation error. No dataset row, validation
or hidden episode, later-stage seed, recovery label, model weight, CUDA
training path, or sealed policy gate was opened by this audit.
