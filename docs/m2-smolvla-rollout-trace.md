# Canonical step-5000 rollout diagnostics

This is instrumentation, not a policy repair. Existing Gate 4 remains 0/5 and
M2 remains incomplete. No collection or GPU run is authorized by this document.
Historical training, processor, decoder and Gate entrypoints are unchanged.

## Ownership and execution

- `scripts/diagnose_canonical_rollout.py`: separate `collect --plan` and
  `verify --directory` commands. Verification imports no model or dataset.
- `src/rosetta_reality/eval/rollout_trace.py`: scoped policy/environment wrappers
  around the original `scripts.smolvla_sim_gate._rollout`. Original prediction
  tuples, actions and environment results are returned without modification.
- `src/rosetta_reality/eval/rollout_trace_verify.py`: independent standard-library
  replay of hashes, event order, state continuity and gripper support arithmetic.

Only one rollout may run in the process while the engine environment binding is
temporarily wrapped. Nested traced calls are rejected. The binding and lock are
restored on success and failure. The collector adds no inference, optimizer,
random sampling, action correction, simulation step or success threshold.

Use WSL Bash and an existing pinned Linux Docker image, with the repository
read-only and networking disabled, for verification and synthetic tests:

```bash
python scripts/diagnose_canonical_rollout.py verify --directory runs/NEW_TRACE
python -m pytest -q -p no:cacheprovider tests/test_rollout_trace.py
```

These are container commands, not authorization to run them in Windows Python
or on the WSL host. Tests use temporary outputs and synthetic policies/environments;
they do not instantiate a real SmolVLA model or MuJoCo environment.

## Output contract

Each directory is create-only and represents one episode. An existing directory
is rejected before policy configuration or environment construction.

| File | Content |
| --- | --- |
| `identity.json` | Registration, source/checkpoint/processor/config inventories, rollout options, action names, gripper indices and observation units |
| `trace.jsonl` | Ordered `prediction`, `step_started`, `step` events, each with schema version, episode index and environment seed |
| `summary.json` | Complete/incomplete status, failure type, prediction/started/completed/recorded counters, support statistics, original rollout metrics and timing |
| `manifest.json` | SHA-256 of the preceding three files and completion status |

Prediction events retain both internal gripper values for every chunk slot,
their dtype and support boundary, decoded and postprocessed first actions,
and observation tensor shapes/dtypes/hashes. Images are not serialized.
Step events reference their prediction and retain executed action, pre/post
robot state and observed gripper openings, pre/post physical snapshots, reward,
success and termination flags. Joint states are observations; they are not
assumed equal to the commanded actions. Snapshot body availability and missing
fields are explicit; missing measurements are never replaced with zeros.

Support is `abs(internal) > pi/2`, with the boundary rounded to the internal
tensor dtype. This avoids misclassifying an exactly encoded float32 endpoint.
For each side, `support.first_action` and `support.full_chunk` report sample
count, outside count/fraction, maximum excess and first prediction/slot.
`executed_support` includes only predictions with a persisted completed step.
Full chunks are predictions, never counts of executed actions. All internal
values are retained unchanged; sine-decoded legality does not establish support.

`step_started` is flushed before the environment call and `step` afterward.
An environment exception leaves execution outcome uncertain. A post-step logging
failure can leave completed greater than recorded; no next action is attempted.
The partial trace and incomplete summary are retained where storage remains
writable. If finalization itself cannot write, missing summary/manifest means
incomplete, as stated in the initial identity record. A truncated JSON line or
hash mismatch fails verification rather than silently dropping evidence.

Original rollout latencies now include observation overhead and are not directly
comparable with the historical uninstrumented run. Wall and serialization times
are separately reported; serialization time is not total instrumentation cost.
There is no expert trace and no first-expert-deviation claim.

## Future collection registration

There is deliberately no executable default plan. A new plan must contain:

- `schema_version: 1`, unique `run_id`, explicit `execution_authorized`,
  `purpose: diagnostic_only_unchanged_step5000`, `optimizer_steps: 0`;
- a live `deadline` no more than one hour ahead and the existing guarded worker's
  shutdown `watchdog` PID/start ticks; this CLI does not provision or launch a watchdog;
- `sources`, a repository-relative path-to-SHA map including `REQUIRED_SOURCES`
  from the entrypoint and the dependency inventory reviewed for the new worker;
- `training_plan`, `action_contract`, `artifact_config`, each with `path`/`sha256`;
- `checkpoint` with repository-relative `path` and the complete `files` inventory;
- fresh repository-relative `output` and `rollout` containing `seed`,
  `policy_noise_seed`, `maximum_steps`, `noise_mode`, `project_policy_output`.

The entrypoint requires the original step-5000 weight SHA, complete saved
processor inventory, training-plan SHA, artifact config and Action Contract
from immutable canonical attempt 004. It accepts one original seed 1000–1004,
the identical policy noise seed, 500 steps, seeded standard-normal noise and
the original action projection. Future registered collection should start with
seed 1000. Native saved processors are loaded through `load_gate_context` and
the original CUDA adapter supplies predictions. The live worker must retain its
existing resource admission, deadline, shutdown and return-evidence lifecycle.

## Validation boundary

Synthetic tests compare traced/untraced actions, noise arrays, state transitions,
rewards, termination and safety metrics exactly, excluding measured latency.
They cover horizon exhaustion, early success, timeout, prediction/environment/
write failures, unavailable physics, existing outputs, both grippers and
first/future slots, dtype boundaries and independently rejected evidence tampering.
Real CUDA parity, physics trajectories and any explanation or improvement of
the existing 0/5 remain unmeasured.

Local verification on 2026-09-16 used existing image
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`,
WSL Bash to Linux Docker, network disabled, repository read-only, two CPUs and
3 GiB memory/swap cap. Final diagnostic suite: **43 passed**, including float16,
bfloat16, float32 and float64. Related action/adapter/scaling/sampler/posttrain
regressions: **45 passed, one CUDA-only pixel test skipped**. Ruff passed for all
four new Python files. Initial lint line-length/lambda findings were corrected;
no scientific threshold or historical source pin was relaxed. No weights, real
dataset, model inference, optimizer update, simulation rollout or GPU was used.
