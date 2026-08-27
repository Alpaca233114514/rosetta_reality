# M2 SmolVLA MoveIt runtime-boundary preregistration — 2026-08-15

## Decision and negative evidence

Athena plan `022` is preserved as immutable infrastructure-incomplete evidence.
Its train-only calibration reproduced reward `4` in 294 steps, but exact
stopped at approach step 98 with `start_state_out_of_bounds` before an OMPL
path existed. The Action Contract command and pinned Interbotix URDF limit were
both `3.14158`, while the Gym/MJCF observed state could reach mathematical pi,
a difference of only about `1.27e-5` rad. More importantly, the remote host lacked the official Interbotix
ament package and meshes; MoveIt loaded no collision geometry while the old
identity protocol still returned `ok`. The create-only exact report SHA-256 is
`a8615ca2935e0a4dcf4e968eb3b7884257f0824b6678ac94eddedc0f67583328`.
This result is not collision-checked planner acceptance and is not a conclusive
RRTConnect failure.

Plan `023` repairs only that official-runtime boundary. It does not replace or
modify the upstream planner:

1. the pinned Interbotix archive, xacro sources and every referenced mesh/URDF
   resource must pass SHA-256 checks before the sidecar starts;
2. the C++ sidecar fails at startup when MoveIt's model contains zero collision
   links or shapes, and its JSON identity reports the exact positive counts;
3. at Rosetta-to-MoveIt request ingress, MoveIt's own
   `RobotState::enforceBounds` may reconcile an arm state only when the nearest
   official bound is at most `0.00002` rad away;
4. every reconciled joint and delta is returned to Python and written into the
   exact report; any larger violation remains `start_state_out_of_bounds`;
5. a returned waypoint is bounded against the original observed state, not
   only the reconciled MoveIt state, so reconciliation cannot enlarge the
   registered per-command joint delta.

The planner remains the official ROS 2 Humble MoveIt `2.5.9` planning pipeline,
official `ompl_interface/OMPLPlanner`, OMPL `1.7.0`
`geometric::RRTConnect`, and official
`lma_kinematics_plugin/LMAKinematicsPlugin`. Rosetta remains a thin process,
frame, Action Contract and evidence adapter; it does not implement sampling,
graph search, collision checking or IK.

## Frozen scientific boundary

Teacher geometry, upstream Mink/DAQP settings, calibration episode, train-only
exact seed `10`, Action Contract, `0.001` maximum weighted error, `0.003`
maximum projected error, `0.23561944901923448`-rad command bound and zero
orientation relaxation are unchanged from plan `022`. The `0.00002`-rad value
is not a task-space tolerance and does not relax either pose gate or the Action
Contract; it is only a strict adapter from the observed simulator state to the
known pi versus `3.14158` model bound.

Tuning seed `1900`, development seeds `2000`--`2004`, collection seeds
`3000`--`3004`, policy-Gate seeds `1000`--`1004`, validation episodes, hidden
episodes and recovery-label writes remain sealed. A passed exact report would
authorize only a new review decision; it does not automatically open tuning.
No optimizer step or CUDA training is authorized.

## Frozen runtime identity and boundary smoke

| Item | Frozen identity |
|---|---|
| plan `023` SHA-256 | `26c6caa9b5397d66d378ba0f9404c07e6bef5be27d8c0dab43587b9a234e3954` |
| sidecar source SHA-256 | `1b2ac29b7fdd7ce9a576218f6255d1c02e0ec2ef3133bc5b75dc85597e134d00` |
| sidecar executable SHA-256 | `3d0d822f6e624dd155e4a36621983b9a6b562cd441f7fae3d1c4f7ecf15448fa` |
| Python client SHA-256 | `8cf399be162a6cb6d08350892c544cbd3ae57dace8a621031d7e89c42db1c26e` |
| evaluator SHA-256 | `5e055267f3f7f9e66fabce4417270120d85de6623fd1d64acf364b21b6ee7e6f` |
| composed URDF / SRDF | `544b5f299a5955b4c705cd42f952f4ab4fb898327f6206ea37117533190f63f4` / `65bd2a440fb3565fae4373d0218a95afd9564fcb2e4bfdcece06ac4e82c0321d` |
| mesh manifest SHA-256 | `63edd159854e2eaa99bbe640c76b6b65e0e0ac517081689796d31c1db579e04d` |
| URDF-source manifest SHA-256 | `c74d4712fe206303ef081a3f81c65c6aa2a8b1a0b29b3d52b174e9c6cbc8ccf1` |
| collision geometry | 22 links / 22 shapes |

The final create-only runtime smoke used an observed
`right_wrist_rotate = pi`. MoveIt reconciled exactly one joint to `3.14158`,
with maximum reconciliation `1.2653589793298892e-5` rad. Official LMA and
RRTConnect then returned a collision/bounds-valid path, while the first command
measured from the original pi observation was exactly
`0.23561944901923448` rad and did not exceed the frozen command bound. A
separate `3.14161` request had a `3.0000000000196536e-5`-rad violation and was
rejected as `start_state_out_of_bounds`. The final smoke evidence hashes are:

- identity JSONL: `cd565f721fae315c821f55ade50264ba0368f4889f8f5d1fddab7adf7efd625b`;
- reconciled path JSONL: `009e63dfc008715a3f918e28a9292a73951a3282bf853c0881dfd3c60803be85`;
- oversized rejection JSONL: `4e0a44512658380eabdba4ae4a224409da9f258ede5804f66462742f0510e8e5`.

Waypoint count, path length, IK branch and observed planning time are smoke
diagnostics, not acceptance identities.

## Static and protocol evidence

Create-only attempt `athena-plan023-static-validation-018` passed before exact
execution. Its workspace archive SHA-256 is
`1f249835e16ed59fdedab55defa7844cb4b1f4da18e8081f027f04c574833ac5`.
Ruff passed, all 29 focused tests passed, and all official mesh and URDF-source
files matched their manifests. Five MoveIt-to-Gym FK parity samples passed with
maximum position error `3.188872858294072e-16` m and maximum orientation error
`0.0` rad. The parity report SHA-256 is
`09d5872bae5cb8ff34d724c7d02c7c9c1c852a2c3dad4c54a97575ff4f7f54a0`;
the immutable execution-log SHA-256 is
`a2736b7655cc8ddb77c54d7c16e91e5e779f34b86c16b9afb568433e93eb8809`.

## Athena execution order

Athena must verify the watchdogs, official package identities, per-file
resource manifests, collision-geometry counts, sidecar binary and workspace
hashes before running Ruff and the focused protocol suite. A direct boundary
smoke must prove that the known pi representation is reconciled within the
registered tolerance and that a larger violation is rejected. Only then may
the create-only calibration plus train-only exact stage run in a durable
session. Any failure ends the furnace at a clean evidence boundary with later
stages still sealed.
