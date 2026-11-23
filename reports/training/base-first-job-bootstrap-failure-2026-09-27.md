# Base first-job bootstrap failure — 2026-09-27

Execution identity `three-round-base-20260927-003` exited at its first `doctor`
command with return code 1. The independent base measurement did not begin:
there are no episode traces or measurement outputs. This is a runtime bootstrap
failure, not a base-policy task failure or completed Exp1.

The actual doctor traceback reaches `scripts/autodl_doctor.py` root validation
and states that `ROSETTA_AUTODL_PLATFORM_ROOT` must be set by
`scripts/run_autodl.sh`. The immutable job registration omitted this environment
entry and the other complete doctor root declarations. The repair must preserve
the doctor assertions, provide the complete registered environment and existing
durable directories, and use a fresh execution identity. The measured-policy
source/plan remains `base-five-scene-independent-20260927-003`; the failed job
identity must not be overwritten or rerun.

The base reviewer independently retrieved the entire closed job and compared all
14 returned files to fresh remote per-file SHA256 observations: 14/14 matched.
Archive SHA256 is
`5cb06c25e549654279b71d89bff9be5aa7313563620d5c48eca4e1c5d6790a47`.
The local evidence is
`.work/base-run-003-independent-received/extracted/three-round-base-20260927-003/`.

The worker receipt lists only `doctor` with return code 1 and `error_type` of
`RuntimeError`. Both worker and guard cleanup receipts report
`owned_group_empty`, with `shutdown_requested=false`; the guard exits because
the worker exited. These receipts support the owned-job closeout and absence of
a shutdown request by this job. They do not establish platform billing state or
justify shutting down while the complete experiment round remains unfinished.

Preparation, artifact validation and five-scene measurement were registered as
later commands but never reached. Forward, episode and optimizer counts are zero
for this execution. The already passed synthetic regression suite and actual
base/cache identity checks remain preparation evidence. Exp1 remains unmeasured
pending the corrected fresh job and independent result verification.
