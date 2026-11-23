# Base independent five-scene result — 2026-09-27

The authorized independent Exp1 base measurement has actually completed and
passed independent evidence verification. All five registered scenes were
measured; every episode ended on the original physical safety checks before
task success. This is a completed negative diagnostic result, not a formal
Gate4 pass or evidence of base task success.

Execution identity is `three-round-base-20260927-005`, with scientific plan
`configs/diagnostics/base-five-scene-independent-20260927-003.json` (SHA256
`39abec6dce8679c30bc5638abe17030af1c9d0d4018cd7e7892f4994867293e6`).
It used workspace `20260927T125840Z-66a5bbcdbdcd-70819c996cda` (composite
`70819c996cdabc16404a3cac5d3602a60106a4b55151a51398aa2cb4f4d79b6e`).
Base revision, weight SHA, exact historical canonical ALOHA interface, processor,
train-only normalization, seeded Gaussian noise, BF16, Action Contract,
slot0 execution and five original scene/noise seeds remained registered.
No trained endpoint was rerun and no optimizer update occurred.

## Actual episodes

| Seed | Executed steps | Maximum reward | Task success observed | Joint-limit counts | Unexpected-contact counts | Outcome |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1000 | 2 | 0 | false | 0 | 1 | physical_safety_stop |
| 1001 | 2 | 0 | false | 2 | 2 | physical_safety_stop |
| 1002 | 25 | 0 | false | 0 | 2 | physical_safety_stop |
| 1003 | 2 | 0 | false | 2 | 1 | physical_safety_stop |
| 1004 | 2 | 0 | false | 1 | 1 | physical_safety_stop |
| Total | 33 | — | 0/5 observed | 5 | 7 | all five measured |

Seeds1000/1003/1004 stop on left/right gripper-finger contact. Seed1001 stops
on left fingers contacting the right gripper bar, with two limit counts.
Seed1002 stops on left fingers contacting the table. The recorded limits are
counts, and the contact totals count contact pairs at observed steps; they are
not independent accidents. This trace does not record which physical joints
caused each limit count, so it does not assign joint names without evidence.

All recorded raw, processed and executed actions and robot states were finite.
Independent scalar replay verifies the original float32 contract projection
of raw slot0 equals every executed action, with zero raw clipped fields.
No command follows the first recorded safety stop in any episode. Those stops
end the current scene; all remaining independent seeds still execute.

The 0/5 is an observed-success fraction under safety censoring, not the outcome
of five complete 500-step formal episodes. It cannot answer what would have
happened after continuing unsafe trajectories. The unchanged historical base
Gate3 remains failed and formal Gate4 remains `not_measured`;
`original_gate4_protocol_equivalent=false`, `formal_gate_result=null` and
`m2_complete=false`. Comparison with the historical trained five complete
episodes is not a matched-runtime formal success-rate comparison.

## Independent verification

The base reviewer independently fetched every ordinary file in the closed job,
without model weights or symlink targets. **45/45** files (81,762 bytes) matched
fresh remote SHA256 observations. Archive SHA256 is
`4776115a83b734a1b0467e66816e54a90c7a65f0f0f4a5b658b6e2f5c8d753bd`.
The independent transfer receipt and original files are retained at
`.work/base-run-005-independent-received/`.

`scripts/verify_base_five_scene_diagnostic.py` independently re-read all five
JSONL traces, recalculated steps, reward maxima, success, raw clipping, physical
limits/contact counts and stop boundaries, checked the projected actions and
episode-summary equality, and verified the plan/contract/model/manifest,
parameter integrity and completion bindings. The result is
`reports/training/base-five-scene-independent-verification-20260927-005.json`,
SHA256 `b2da352b0b739ef2225713701f440dd6dc48891119435f592f65d3f3a3adf8f5`.
These are independent calculations on recorded evidence, not additional policy
runs or an independent remeasurement of simulator physics.

The final `completion.json` reports `completed`, binding summary SHA256
`762df71273aa08a9f950a284b0eeab5272844d023847d804ceef8804d15f3598`.
Completion receipt SHA256 is
`d108325d7ad7311acc7bf42377e2748ab6ebcc15f10bf89361b2be4de9b697b6`.
All five scene/noise seeds appear in the denominator, with no incomplete or
unmeasured seeds. Full policy parameter/buffer content hashes before and after
are equal: `fa35561681ccc85b2d1a98c072975c205f323d967e49aaf38f2ba99b0c8a9024`.
Actual optimizer updates are zero.

## Runtime, resources and lifecycle

Actual runtime is the registered AutoDL platform container on NVIDIA GeForce
RTX4090D, torch `2.8.0+cu128`, LeRobot `0.6.2`, gym-aloha `0.1.4`, Trackio
`0.28.0`, CUDA/BF16, with `nested_docker_used=false`. The worker runs with Hub
and datasets offline flags; the recorded `network_disabled` value is the
existing loader's offline-runtime declaration, not a separate network-isolation
measurement.

All `init`, `doctor`, `prepare`, `check` and `measure` stages returned 0.
The complete worker lasted about49.10seconds; model-loading through final
parameter-hash resource check recorded32.93seconds, and the batch recorded
11.95seconds after its initial parameter hash. These are different timing
boundaries. Final recorded RSS is2,334,281,728bytes, measured CUDA peak allocated
1,263,593,472bytes and peak reserved1,329,594,368bytes, within the registered
12GiB RSS and8/10GiB CUDA caps. CUDA peaks cover the registered post-load,
post-initial-hash measurement window; initialization transients are separate.

The outer work-only guard was armed and later exited after the worker. Both
cleanup receipts prove their owned process groups empty and report
`shutdown_requested=false`; no job requests platform shutdown. These receipts
do not prove current platform billing state. Exp1 completion does not authorize
power-off before the user's complete Exp1/2/3 round is finished.

Earlier003 and004 executions remain preserved:003 stopped on missing doctor
environment roots;004 passed doctor but miscounted cached-input weight symlink
bytes as generated output. Their separate failure reports are retained. The005
job places the create-only, fully verified input artifact outside the generated
job-output tree, preserving the256MiB evidence cap. No failed identity was
overwritten, no protected historical artifact was repaired in place, and no
Gate threshold was lowered.
