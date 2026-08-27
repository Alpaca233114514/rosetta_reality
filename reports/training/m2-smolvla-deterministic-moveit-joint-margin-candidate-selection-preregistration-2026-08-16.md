# Deterministic MoveIt joint-margin candidate selection preregistration

Plan046 showed that smaller Cartesian fractions continue the same local wrist
branch and consume the remaining tightened joint margin. Plan047 may change one
axis only: retain the official MoveIt 2.5.9 LMA candidate generator and frozen
256-attempt / 2-second deterministic budget, but rank every valid candidate
instead of accepting the first one.

The primary objective is maximum minimum arm joint-limit margin. Ties prefer
the smaller maximum start-to-goal joint delta, then the lower deterministic
attempt index. Bounds, registered joint path constraints, self-collision,
task-space tolerances and OMPL path validation remain hard filters.

Before exact, the new runtime must expose selection diagnostics, preserve
ordinary full-pose and position-priority behavior, repeat deterministically,
and retain MoveIt/MuJoCo model parity. Final pose gates, all margins, seeds,
labels and CUDA prohibition remain unchanged.
