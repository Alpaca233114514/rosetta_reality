# M2 SmolVLA Gym joint-name adapter repair preregistration

Status: **Plan035 is locally implemented, preregistered and statically
verified; it is the third and final local exact attempt in this run bound**.

Plan034 passed the CSR-storage boundary, then raised before exact-report
creation because terminal feedforward used the Action Contract name
`left_waist` for MuJoCo lookup. The pinned Gym model name is
`vx300s_left/waist`; the evaluator already owns the complete ordered 12-joint
Gym mapping in `LEFT_JOINTS` and `RIGHT_JOINTS`.

Plan035 changes only that call argument. The action vector order, direct DOF,
CSR validation, force balance, correction bound, margins, planner, teacher,
thresholds, Action Contract, seeds and labels are unchanged. Ruff and all 70
focused tests passed in the network-disabled, read-only Linux boundary. The
official sidecar remains 22/22 and the existing five-sample FK parity report
remains exact within `3.188872858294072e-16` m / `0.0` rad.

Plan035 SHA-256 is
`e7b048ea99d1ea7fd0eb997ce63baf986ad5a575131d8ee84dc24646d9168197`.
Only train-only exact episode 2 / seed 10 may run. Irrespective of its result,
no fourth local exact is authorized in the current boundary.
