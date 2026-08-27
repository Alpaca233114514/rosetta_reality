# M2 SmolVLA object-geometry teacher audit — 2026-08-14

## Outcome

The object-geometry-conditioned teacher boundary is implemented, but its
train-only exact gate still fails. **Recovery labels, tuning/development seeds,
and a new furnace remain unauthorized.** No SSH connection, download, CUDA
training, validation/hidden episode access, policy Gate seed execution, or
recovery-label write occurred.

The latest plan, `009`, reaches step 98 with finite, contract-projected actions
and no adapter clipping, then rejects the next approach target because fixing
`right_wrist_rotate` at its Action Contract limit leaves an 11.97 mm projected
task-space residual. The registered maximum is 3 mm; it was not relaxed.

## Immutable identities

- dataset revision: `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`;
- Action Contract: `configs/sim/aloha_insertion_smolvla.yaml`;
- current teacher SHA-256:
  `74b788fd316ef0f723d871db5c5cf550999998c37826843a268a31c4edddf5fd`;
- current evaluator SHA-256:
  `36bc6287912d0fa6405939515530e156db31e0ca0b58e10200dbd8b3793d4119`;
- frozen Gym-ALOHA adapter SHA-256:
  `e9c1005d0ae085e82e0c96e9d18527dce7d4749268a71756116cdffbb98d6e7d`;
- current plan `009` SHA-256:
  `33bdc1f4ec9d1315ccb742853e1a9f323e47b596e0f89bbcf045de182b6ef5fb`.

## Protocol boundary

`src/rosetta_reality/sim/geometry_teacher.py` receives current robot state,
object/end-effector poses, contacts, collision count, and already observed
reward. Its public decision API has no environment-step or timestamp input.
Runtime targets do not consume the source action timeline. Calibration uses
only rigid grasp and terminal object transforms from successful train episode 2
at simulator seed 10, which replays to reward 4 in 294 actions.

The simulator-specific evaluator performs bounded Cartesian/orientation
feedback, post-IK Action Contract projection, active-set re-solving with
projected joints fixed, independent projected-pose verification, and rejects
any additional environment-side clipping. All reports are create-only and
scoped by plan SHA and stage.

## Exact evidence

| Plan | Steps / phase | Result |
|---|---:|---|
| `003` | 25 / approach | unbounded grasp-orientation jump; IK error `0.0106843` |
| `004` | 78 / descend | accurate unconstrained IK crossed `right_forearm_roll`; projection was not yet verified |
| `005` | 16 / approach | redundant 0.12-rad joint-delta limiter raised projected error to `0.0156707` |
| `006` | 66 / approach | non-binding delta limiter exposed joint-limit projection error `0.00856694` |
| `007` | 66 / approach | active-set solve improved projection to `0.00438492`, still above 3 mm |
| `008` | 98 / approach | split translation/orientation path reached `0.000884032`, but a duplicate solver flag rejected it |
| `009` | 99 / approach | threshold-aligned evaluator accepted that step; next limited target failed at `0.0119651` |

The report SHA-256 values, exact plan identities, and machine-readable metrics
are recorded in the JSON companion. Every exact report records hidden-test,
validation, recovery-label, collection-seed, and policy-Gate access as false.

## Decision

The current failure is kinematic-path feasibility, not an excuse to weaken the
Action Contract or projected-pose threshold. Therefore:

1. do not run tuning seed 1900 or development seeds 2000--2004;
2. do not open collection seeds 3000--3004 or policy Gate seeds 1000--1004;
3. do not create recovery records or start a recovery-data furnace;
4. preserve plans `003`--`009` and their reports as immutable negative evidence;
5. next design a joint-limit-aware geometric path planner, with an independently
   registered exact gate before any new seed group is opened.
