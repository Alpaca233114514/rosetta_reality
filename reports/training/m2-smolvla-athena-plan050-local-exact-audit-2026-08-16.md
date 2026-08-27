# M2 SmolVLA Athena Plan 050 local exact audit — 2026-08-16

Status: **failed train-only exact**. This is immutable negative evidence.

Plan 050 exercised the preregistered active-set Cartesian trust region without changing the frozen teacher, Action Contract joint margins, pose gates, seed boundaries, or label boundary. It produced ten margin-restoration events and two orientation-progress events, advanced the exact episode from the prior Plan 047 failure at step 256 to step 712, and produced no commanded/observed margin breach, adapter clip, joint-limit projection, or unexpected collision.

The remaining failure is an official MoveIt LMA endpoint failure in `orient`. At step 712, `right_wrist_rotate` had only `0.00020236261367780073` rad remaining above the unchanged command margin. The direct request and registered fixed-world-axis trust-region sequence did not produce an executable continuation. The exact report is bound by SHA256 `7bb8771d7fc0c3e75394852c242b498eeafff876ad45fb1abdf9bf456859815b`.

Tuning seed 1900, development seeds 2000–2004, collection seeds, policy Gate seeds, validation/hidden episodes, recovery labels, and CUDA training remained sealed.
