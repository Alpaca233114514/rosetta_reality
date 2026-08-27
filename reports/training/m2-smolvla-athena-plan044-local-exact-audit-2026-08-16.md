# M2 SmolVLA Athena Plan044 local exact audit

Status: **failed train-only exact**. No later seed or label boundary opened.

Plan044 stopped at approach step 97 with
`bimanual_subgroup_count_differs`, before either LMA or OMPL ran. The new SRDF
correctly added two position-only arm groups, but the old full-pose branch still
enumerated every subgroup under `bimanual` and required exactly two. MoveIt now
reported the two original groups plus the two new position-only groups.

This is an adapter group-selection defect, not a failure of the registered
position-priority hypothesis: that fallback was never exercised. There were no
commanded or observed margin breaches and no unexpected collision. The exact
report, execution log, and stopped container remain immutable negative evidence.
