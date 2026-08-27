# M2 SmolVLA Athena Plan048 pick_ik local runtime audit

Status: **failed before exact; all evaluation gates remain sealed**.

The version-pinned `pick_ik/PickIkPlugin` loaded for both full-pose groups, and
the position-priority groups remained on LMA. The captured Plan047 first-orient
request still produced zero candidates in local gradient mode after the fixed
outer multistart budget.

This is a fail-closed runtime result: exact was not started, no custom Cartesian
stencil was enabled, and no pose tolerance, joint margin, seed boundary or
label boundary changed.

The next isolated plugin-axis test is the plugin's documented global memetic
mode with one thread and a fixed configuration, followed by a fresh-process
repeat check before any exact run.
