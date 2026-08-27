# M2 SmolVLA official pick_ik joint-limit preregistration

Plan047 proved that deterministic candidate ranking cannot help when official
full-pose LMA produces zero valid candidates. A bounded Cartesian stencil can
recover a few local orient waypoints, but that is a task-specific workaround.

Plan048 replaces only the full-pose kinematics plugin with the version-pinned
MoveIt-compatible `pick_ik/PickIkPlugin`. Its built-in joint-limit-avoidance
cost is enabled in deterministic local mode. The existing position-only LMA
groups remain unchanged.

Rosetta continues to reject any candidate that violates the frozen exact pose
tolerances, tightened joint path constraints, physical bounds, self-collision
check or OMPL path validation. No approximate solution, looser teacher gate,
smaller joint margin, new seed set or label path is authorized.

Before exact, the new image must pass dependency/plugin identity, the captured
Plan047 failure request, fresh-process repeat determinism, model parity, static
checks and related tests. Tuning and all later gates remain sealed.
