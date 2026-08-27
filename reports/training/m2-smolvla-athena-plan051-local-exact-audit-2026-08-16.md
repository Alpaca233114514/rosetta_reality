# M2 SmolVLA Athena Plan 051 local exact audit — 2026-08-16

Status: **failed train-only exact by horizon exhaustion**. This is immutable negative evidence.

The feedback-aligned basis removed the Plan 050 endpoint failure: Plan 051 completed all 750 steps with zero IK failure, margin breach, adapter clip, projection, or unexpected collision. It exercised the new basis eight times. However, every activation chose margin restoration and none chose orientation progress, so the run remained in `orient` until the unchanged horizon expired.

This isolates a selection-policy starvation bug. Plan 052 may change only the lexicographic order: select the largest feasible orientation-progress fraction first, using minimum joint margin as the secondary candidate objective; use margin restoration only when no registered orientation-progress candidate exists. Hard MoveIt path constraints and all frozen thresholds remain unchanged.

Tuning, development, collection, policy Gate, validation/hidden, recovery-label, and CUDA boundaries remained sealed.
