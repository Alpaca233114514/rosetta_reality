# Athena Plan049 official Pick IK global runtime audit

Status: **failed captured-request runtime gate; exact was not opened**.

The version-pinned `ros-humble-pick-ik` package built successfully and the
sidecar identity bound Pick IK 1.1.2, global mode, one memetic thread and every
registered optimizer parameter. Two fresh sidecars then replayed the immutable
Plan047 first-orient request. Both returned the identical
`bimanual_pick_ik_failed` response with zero valid IK candidates.

This rules out a corrupt or missing dependency as the cause. Reinstalling the
same package would reproduce the same geometry boundary. Local LMA probes have
already shown collision-free, exact-pose waypoint solutions when orientation
progress is combined with a small bounded Cartesian retreat, so the next axis
is a deterministic trust-region waypoint strategy around official MoveIt IK,
joint-limit validation and OMPL planning. It must not relax the final 1 mm / 3
mrad gates, the tightened joint margin, collision checks, or any seed/label
seal.

No tuning, development, collection, policy-gate, validation or hidden seed was
loaded. No recovery label was written and no CUDA training was started.
