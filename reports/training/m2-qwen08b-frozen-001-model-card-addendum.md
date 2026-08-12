# Model Card Evaluation Addendum — m2-qwen08b-frozen-001

This addendum records evaluation evidence produced after the immutable export of
`m2-qwen08b-frozen-001-base-dc7cdfe2`. It does not modify or replace the five files covered by the
export manifest SHA-256 `9b8955e1902d191560732222ffd261d0595b7ce9aaf1d2cd9e197e0430b08369`.

## Model and intended use

- Base model: `Qwen/Qwen3.5-0.8B-Base`
- Base revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Adaptation: frozen backbone with trainable Rosetta projector, state encoder, fusion, and action head
- Dataset: `lerobot/aloha_sim_insertion_human@cc571a3c661df81b566dbfde3d5c1e85fcdf7884`
- Intended use: development-scale VLA research in the pinned ALOHA insertion simulation contract
- Prohibited interpretation: this artifact is not a general robot controller and has not been validated on
  physical hardware

## Offline evaluation

| Split | Samples | Action MAE | RMSE | Raw invalid action rate | Projection element rate | Final invalid rate |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 495 | 0.0230362 | 0.0396249 | 33.74% | 1.403% | 0.0% |
| Hidden test | 495 | 0.0291032 | 0.0578747 | 22.63% | 0.945% | 0.0% |

The validation MAE is 87.01% lower than the pre-training train-action-mean baseline. Final invalid rates
are zero only after Rosetta Action Contract projection. The raw invalid rates show that the policy still
depends materially on that projection.

## Simulation evaluation

Gate 3 was rerun after correcting the collision classifier to ignore only same-arm internal gripper-finger
contacts. The 20-step rollout had zero reset or accumulated unexpected collisions, zero raw/executed limit
violations, zero joint-limit violations, and no task success.

The strict Gate 4 protocol used five deterministic seeds and the full 300-step environment horizon. Its
predeclared acceptance required at least 20% task success and zero calibrated unexpected collisions.

| Metric | Result |
|---|---:|
| Successful episodes | 0 / 5 |
| Task success rate | 0.0% |
| Maximum reward | 0.0 |
| Evaluated control steps | 1,500 |
| Invalid action rate | 0.0% |
| Raw limit violation rate | 0.01905% |
| Executed limit violation rate | 0.0% |
| Joint-limit violations | 0 |
| Calibrated unexpected collisions | 0 |
| Mean policy inference latency | 1.4495 s / step |
| Mean simulation step latency | 0.1525 s / step |

Strict Gate 4 status is `failed`; both `safety_execution_status` and `task_capability_status` are `failed`.
The safety status fails because one trajectory produced four raw out-of-contract action elements before
projection. The task status fails because all five trajectories had zero reward and zero success.

## Collision metric correction

The historical metric counted eight normal contact points per frame between the two fingers of each gripper.
The corrected classifier excludes a robot-robot pair only when both geoms are gripper fingers and their arm
namespace before `/` is identical. Cross-arm gripper contacts and non-gripper robot/table/object contacts
remain unexpected. The corrected Gate 4 report records canonical unexpected-pair counts and observed none.

Historical collision counts must not be used as severity evidence. Task failure is established independently
by 0/5 success and maximum reward 0.

## Known limitations

- The artifact has only one full training run; exact artifact reload is verified, but from-scratch training
  reproducibility has not been independently demonstrated.
- Only five hidden-test episodes, one instruction, and one top camera were evaluated.
- Frozen attention-mean pooled features may omit spatial detail needed for insertion.
- Offline features use frame stride 5 while online control evaluates every simulator step.
- CPU inference is far slower than the 20 ms interval implied by the 50 Hz Action Contract.
- The original export Model Card predates hidden-test and Gate 3/4 evaluation; this addendum is a separate,
  post-export audit record and is not included in the immutable artifact manifest.
- Experiment provenance records a dirty workspace tree. The cache/artifact branch label differs from the
  final active branch label, although the 107-file workspace tree hash remains the binding code snapshot.
- The configured `checkpoint_every_epochs: 5` was not consumed by the runtime; the completed run saved one
  checkpoint per epoch.

## Evidence

- Validation report SHA-256: `56d39427ac3e7db45767fe2377f9a82ee49def83df7fd7561a66086719baf78d`
- Hidden-test report SHA-256: `4916a9a1170edb8c1e015658210fca4d04f2cf294f791bbe11e736218f1ece7f`
- Corrected Gate 3 SHA-256: `c90634d85def7038f55d636b5e0d904ae205b53aef7bb4fe7af65b698054647e`
- Strict Gate 4 SHA-256: `b5d2313ae4e0f9fb619d1fb21b4c61c9f574eebe3384366ac2fdedf448191733`

## Release decision

Research pipeline evidence is sufficient to retain this artifact as a frozen 0.8B reference checkpoint.
Task capability is not accepted. Do not publish it as a successful insertion policy, deploy it to a physical
robot, or use it as justification to begin 9B scale-up.
