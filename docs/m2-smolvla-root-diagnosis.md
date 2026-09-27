# SmolVLA root diagnosis and full-trajectory probe preparation

Current local result: `reports/training/m2-smolvla-root-diagnosis-result-2026-09-22.md`
and its JSON companion. This is saved-array analysis and diagnostic infrastructure,
not another policy training run. Canonical Gate 4 remains 0/5; M2 is incomplete.

## Interfaces and evidence

- `scripts/analyze_root_neighborhoods.py --plan ... --output ...` reads sealed
  nonhidden numeric arrays and 2500/5000 predictions. Outputs per-episode/group/
  noise metrics, event statuses and train-only neighbor identities. Initial 2D
  geometry and current state are separate proxies; neither is full visual distance.
- `scripts/verify_root_neighborhoods.py --plan ... --input ... --output ...`
  independently reconstructs neighbors and arithmetic without production helpers.
- `scripts/prepare_root_diagnosis.py --output ...` inventories checkpoint bytes,
  training ingress and saved Gate evidence, and generates three seed-1002 drafts.
  Its temporary registered copy is used only for file-identity validation, never
  persisted or passed to model authorization. Saved plans remain draft.
- `scripts/prepare_trajectory_probe.py --plan ... --output ...` accepts only the
  pinned historical canonical plan and generates smoke/A/B drafts, schedules,
  train-derived threshold files and a saved-array evaluation protocol.

All output directories must be new. Historical plans, snapshots and failed
attempts remain immutable. Plan 001 of the neighborhood analysis predates final
formatting; the initial executed analysis uses plan 002. Resource-enforced replay
uses `configs/vla/root_neighborhoods_20260922_003.json`; all three metric/event/
neighbor output files are byte-identical to 002.

## Full-trajectory preparation boundary

Smoke uses episode 2 frames 0/499, batch 1, two updates. A uses episode 2,
batch 4, 500 updates; B uses episodes 2/49/4/23, batch 4, 2000 updates.
A/B have four complete 500-frame passes per episode with deterministic shuffles.
They are independent fresh-base fit probes, not a matched causal comparison.
Normalization remains inherited train40 statistics, not subset-only statistics.
Warmup is 16; decay is 500/2000. Smoke retains the A schedule but stops at two
updates. Quarter checkpoints are declared for A/B; no candidate is selected here.

`training.trajectory_probe.observe_effective_runtime` observes the native optimizer
factory and first update, reads an actual FeatureStack, named parameters, optimizer
groups, LR, checkpoint grid and sample ledger. It handles an Accelerate optimizer
wrapper by checking the underlying optimizer identity. The default pinned loss
uses a **global valid-element denominator**; weighted per-sample training is rejected.
The descriptor is derived from the actual batch mask and hash-pinned source;
it is not an independent measurement of an executed full-model loss tensor.

`training.trajectory_scoring.score_saved_predictions` (also re-exported from
`trajectory_probe`) consumes saved records only, checks identical four-noise
targets, masks, declared horizon/dimensions and full 500-frame coverage. Partial
analysis requires explicit nonempty frame identities. Every array value including
padding must be finite. It excludes exact tail padding, separates both
arms and grippers, and retains opening/closing/censored events and low-aperture
hold bias. Use the generated evaluation protocol's sealed thresholds. It does
not classify offline fit as task success. Opposite-direction events never match;
full-window misses and censored tails differ. A truncated tail cannot establish
sustained hold. Schema 2 explicitly reports `identity_verified=false`: numeric
cohort validity does not authenticate the model or processor. Invoke separately
for each checkpoint.

Native optimizer factory/update, Accelerate, FeatureStack and both observers now
pass a CPU integration test with synthetic tensors and a one-parameter policy.
Wrong loss denominators stop before an update; four exposures count as two unique
frames. Actual SmolVLA/processor composition is still unmeasured; future execution needs
a separately sealed integration preflight and resource window. Merely toggling
the draft's authorization flags is insufficient. Full source-closure/runtime
identity, independent model reload and matched source-state rollout are pending.
Missing source physics means training-scene rollout remains unmeasured.

## Reproducibility repairs

The verifier accumulates any-step success like the original Gate, cross-checks
trace robot states against independent observations, and keeps diagnostic physics
snapshots distinct from integration-state vectors. `trace_attribution` prevents
A/trace differences from being attributed when A/A is unmatched. Raw comparisons
remain available and never prove a unique cause.

Historical artifact metadata was reserialized by `canonical_fullframes_stages.render`
using `iris_runtime.save`, dropping the original final LF. The retained config SHA
is `978ec7b1...`; adding exactly one LF in memory gives the manifest's fixed native
SHA `0d2255fc...`. The new validator checks both full fixed digests. It neither
rewrites old evidence nor accepts arbitrary semantically equivalent JSON.

Seed 1002 is preregistered for the new A/A/full-trace comparison because the old
004/005 behavior differs there. This cannot reconstruct old 004's missing trace,
prove uninstrumented parity, or restore arbitrary wrapper/physics state.

## Local runtime

`scripts/run_root_diagnosis_local.sh` is invoked through WSL Bash. Its first
argument is the freshly verified Windows host path of the existing runs directory;
remaining arguments are the container command. It uses the existing fixed
Linux image, network disabled, two CPUs and a default 3 GiB container cap. Set
`ROSETTA_ROOT_MEMORY_MIB=1536` for the saved-array plan. The bounded command timeout
defaults to 180 seconds and sends TERM followed by KILL after five seconds.
Before loading arrays, the analysis CLI verifies actual cgroup memory/CPU limits
and the sealed runner's timeout declaration against its plan. It copies only
code/config/docs into an ephemeral container tree and mounts evidence beneath it,
avoiding Windows directory-junction escapes without relaxing path validators.
No model or raw dataset is mounted through a new download path.

The neighborhood plan's 1536 MiB field was an intended analysis budget; this
runner enforced 3 GiB rather than that tighter bound. No 1536 MiB enforcement
claim is made for the completed analysis. The user-authorized 16 GB host budget
was retained. Replay 003 corrects this: observed cgroup memory is 1610612736 bytes,
CPU quota 2 and command deadline 180 seconds. Historical 002 is unchanged.

Latest packages: `runs/root-evidence-20260922-002/`,
`runs/root-neighborhoods-20260922-003/`,
`runs/root-neighborhoods-independent-20260922-003/`,
`runs/root-readout-20260922-001/`, and
`runs/trajectory-probe-prepared-20260922-003/`.
Draft 003 validates the complete declared v2/VLA source inventory and structural
contract before writing each plan. Smoke declares two active updates separately
from its 500-update scheduler reference. No draft grants training authorization.
Validation and retained failures: `runs/root-diagnosis-followup-20260922-001/`
and the original `runs/root-diagnosis-20260922-001/qa/`.
Follow-up: `reports/training/m2-smolvla-root-diagnosis-followup-2026-09-22.{md,json}`.
