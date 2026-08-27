# M2 SmolVLA Athena Plan056 local exact audit — 2026-08-16

Status: **failed train-only exact**. Immutable negative evidence.

Plan056 extended the already-evidenced right-finger/table task-contact scope to
lift. The run cleared the Plan055 collision boundary and preserved both grasp
contacts for 56 lift decisions. It failed at step 443 in `lift` with one Mink
QP inverse-kinematics failure: weighted IK error `0.0015606494501644484`
exceeded the unchanged `0.001` acceptance gate.

No teacher failure, clip, projection, margin breach, trust-region event, or
unexpected collision occurred. The failure boundary is now the local Mink QP
solver in lift, after contact feedback and lift contact scoping were repaired.
