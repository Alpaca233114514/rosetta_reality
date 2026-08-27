# Active-set Cartesian trust-region preregistration

Status: **preregistered; not executed**.

Plan050 keeps the verified official MoveIt 2.5.9 LMA + OMPL runtime and changes
only the waypoint orchestration used after the orient full-pose path is
exhausted. The arm owning the current minimum Action Contract joint margin is
selected automatically. A symmetric six-axis Cartesian trust region first
seeks at least 1 mrad of joint-margin improvement while holding that arm's
orientation. When no such candidate exists, it advances orientation by the
largest feasible registered fraction. Every candidate is still validated to 1
mm / 3 mrad at its waypoint and must pass the unchanged tightened joint-margin,
collision and OMPL path gates.

The sealed Plan047 request produced 15 valid candidates from 90 symmetric
probes. A 1.5 mm coordinate step improved minimum joint margin from 0.04681 to
0.05378 rad, and the same trust region admitted a 12.5% orientation-progress
candidate at 0.04970 rad. No task-specific direction is registered.

Exact must pass before tuning is opened; later seeds, validation, hidden data,
collection, policy gates and recovery labels remain sealed. CUDA training is
not authorized.
