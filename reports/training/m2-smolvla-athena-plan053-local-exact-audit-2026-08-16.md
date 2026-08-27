# M2 SmolVLA Athena Plan 053 local exact audit — 2026-08-16

Status: **failed train-only exact**. This is immutable negative evidence.

Constraint-anchored restoration removed the Plan 052 dead end: Plan 053 completed 18 orientation-progress replans and two constraint-anchored margin restorations with zero IK failures, commanded or observed margin breaches, adapter clips, and joint-limit projections. It nevertheless consumed all 750 registered steps in `orient`.

The sealed frontier probe shows that official MoveIt LMA + OMPL can solve a larger bounded orientation target while preserving the original 1 mm / 3 mm pose gates, joint margins, collision checks, seed boundary, and horizon. The next single numeric axis is therefore the per-command orientation target budget; no acceptance tolerance is relaxed.

All later seeds, hidden/validation data, recovery labels, CUDA training, and SSH remained sealed.
