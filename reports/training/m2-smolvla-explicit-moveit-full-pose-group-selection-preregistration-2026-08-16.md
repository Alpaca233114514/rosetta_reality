# Explicit MoveIt full-pose group selection preregistration

Plan044 never exercised the registered position-priority hypothesis. Adding its
two SRDF groups changed generic subgroup enumeration under `bimanual` from two
entries to four, so the old full-pose adapter stopped at a count assertion.

The sole Plan045 change is to select `left_arm` and `right_arm` explicitly for
full-pose LMA, matching the existing explicit selection of
`left_arm_position_priority` and `right_arm_position_priority` for
position-only LMA. Missing groups and order mismatches remain fail-closed.

No tolerance, Action Contract, collision rule, joint margin, OMPL setting,
controller, horizon, seed, stage boundary, or label rule changes.
