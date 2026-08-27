# Athena Plan041 local exact audit

Plan041 again stopped at the third `orient` request before the contact-phase controller could run. Raising the upstream subgroup IK timeout from 0.50 to 2.00 seconds did not remove the failure; there were still no collision, clipping, controller, teacher, or joint-margin failures.

The MoveIt 2.5.9 implementation explains the instability: `RobotState::setFromIKSubgroups` uses the request start state first, then samples subsequent restart seeds from a lazily constructed unseeded RNG until a wall-clock timeout. More wall-clock time changes how many unknown seeds are tried but does not make the exact gate reproducible.

Plan042 may copy that official subgroup structure into the sidecar while replacing only its seed scheduler: start state first, then a fixed-seed, fixed-count sequence, unchanged official LMA calls, and the same full-state bounds/constraint/collision validation. The 2.0-second outer cap and every Plan041 safety/controller/seed/label gate remain frozen. The JSON companion is the exact authority.
