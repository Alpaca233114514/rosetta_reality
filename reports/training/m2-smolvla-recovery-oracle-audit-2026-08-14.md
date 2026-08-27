# M2 SmolVLA recovery-oracle audit — 2026-08-14

## Outcome

The local state-conditioned recovery boundary is implemented, but **recovery
label collection and a new furnace remain unauthorized**. The oracle reproduces
one proven source trajectory at its exact train-only simulator state, yet both
registered attempts fail on the independent tuning seed before the unseen
development seeds are opened.

No SSH connection, download, CUDA training, checkpoint reuse, validation/test
episode access, policy Gate 4 seed execution, or recovery-label write occurred.

## Immutable identities

- dataset: `lerobot/aloha_sim_insertion_human` at
  `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`;
- Action Contract: `configs/sim/aloha_insertion_smolvla.yaml`;
- oracle core SHA-256:
  `0b781f11832cb2bd00d7b704ddf2a2f90b6f435a3785cb0b70af750a9413bd1d`;
- evaluator SHA-256:
  `7b4e22a93a3995bb7a3eb4af5c113d3d6746ab1379be5f6a53a6f70fdca50b65`;
- recovery manifest contract SHA-256:
  `c93bfa9f5c5b841d18a9c94da4dbd9f9f3c53b5b8721e6ba5677a8d7ae4f840b`.

The future data contract is create-only and requires state-conditioned labels,
clean target actions, a passed oracle report, train-only source episodes, a
sealed hidden test, and mutually disjoint collection, oracle-evaluation, and
policy Gate 4 simulator seeds.

## Oracle protocol

`src/rosetta_reality/sim/recovery_oracle.py` receives only the current 14-D
robot state and already observed task reward. It searches a monotonic bounded
window of a successful train-only reference bank. A reward event unlocks the
post-contact phase. It has no environment-step or timestamp input and raises
instead of falling back to a same-index expert action outside the registered
neighborhood.

The Gym-ALOHA diagnostic retargets sparse reference anchors with local-branch
inverse kinematics and interpolates joint corrections. This is an attempted
cross-pose reference construction, not accepted recovery supervision.

## Evidence

All 11 registered source episode/seed pairs replayed to reward 4. The exact
control uses train episode 2 and simulator seed 10:

- source replay and oracle control both reached reward 4 and success in 294 actions;
- maximum state distance was `0.00564728770405054`;
- OOD failures and IK failures were both zero;
- hidden-test loaded and recovery-label written were both false.

Oracle `001` evaluated only dedicated tuning seed 1900. The nearest successful
source was episode 21/seed 28 at initial four-object-XY distance
`0.0533965453505516`. It returned reward 0 after 500 steps, with no OOD or IK
failure, and stalled at reference index 20. Its progression gate was `0.01` and
the observed distance was approximately `0.0155`.

Oracle `002` changed only that registered progression distance to `0.02` and
again passed exact control. On the same tuning seed it returned reward 0 after
500 steps, with no OOD or IK failure, and merely moved the stall to reference
index 25 at distance approximately `0.0373`.

The create-only evidence checksums are:

- `001` exact: `1a8117c198eb6b6ec8aa82ddf6736a1a3aa17796b94f4540ca1c07f1647ffea4`;
- `001` tuning: `9852c309a6afbc1097d71658e2110581468c281d373bf3df1dcb90a7fe48cfa6`;
- `002` exact: `350a5c98857e825afa891259b93f87c072aa7d13dc207e9a030875158fe24749`;
- `002` tuning: `47d05efbafec0d6cf37cfc17d4201889ffed86b84b1e46253dcdff34fcb060de`.

The earlier dense-IK prototype also exceeded a 15-minute diagnostic command
budget without producing a report. Sparse anchors repaired the performance
defect but not the cross-pose task failure.

## Decision

Robot-state proximity plus translated source trajectories is not a proven
state-conditioned recovery oracle. Increasing the progression threshold again
would fit the single tuning seed without evidence of a valid recovery target,
so unseen development seeds 2000--2004 remain unopened. Collection seeds
3000--3004 and policy Gate seeds 1000--1004 were not executed.

Therefore:

1. do not create a recovery dataset manifest or records;
2. do not label policy-deviated states with these actions;
3. do not start a recovery-data furnace or ask for SSH yet;
4. preserve both attempts as negative evidence;
5. the next oracle must condition on task/object geometry and expose an
   independently validated state-conditioned teacher before DAgger labels.
