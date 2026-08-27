# M2 SmolVLA official pick_ik global preregistration

Plan048 proved that the version-pinned plugin loads, but its local gradient
mode cannot leave the current wrist branch. Plan049 changes only the plugin's
documented mode to the global memetic optimizer.

The global solver is fixed to one thread, a fixed population and fixed
generation limits. Rosetta's independent exact-pose, tightened joint-margin,
bounds, collision and OMPL checks remain authoritative. The captured failure
request and a fresh-process repeat must pass before exact is authorized.

No Cartesian stencil, approximate pose gate, smaller joint margin, later seed
set or recovery-label path is opened.
