# M2 T2 teacher gate protocol preregistration (2026-08-29)

Freezes the staged teacher gate of
`m2-smolvla-t2-state-conditioned-teacher-gate-design-2026-08-28.md`
(sha256 `8aaea3672abd85659cd1b46e73b0fa5acb39acbe7701752eea22af4c045b146f`)
into binding thresholds and the executable runner
`scripts/run_teacher_gate.py`. JSON companion:
`m2-smolvla-t2-teacher-gate-protocol-preregistration-2026-08-29.json`.

Protocol id: `m2-t2-teacher-gate-001`. Every stage writes create-only
immutable evidence under `runs/…-003/teacher_gate/<candidate>/`; a failed
stage stops the gate; no stage involves a policy checkpoint.

## Teacher contract (frozen)

Observation type: the existing frozen `InsertionGeometry` — robot state
(14-D), object geometry, event/contact flags, observed reward. Its field set
is checksum-asserted at every G0 run: adding any time, index, episode, seed,
dataset or policy field fails the gate by construction. Output: one absolute
standard-space action under the unchanged Action Contract, or one explicit
`TeacherRefusal` (`out_of_support`, `corrupted_geometry`, `unsafe_state`,
`workspace_violation`). Deterministic given inputs; off-support states
refuse, never degrade.

## Stages and frozen thresholds

- **G0 contract conformance** — observation field set exact, byte-determinism
  under replay, non-degenerate explicit refusal on the registered
  off-support observation. The bundled candidate
  `gate-conformance-probe-001` can only ever run G0 and its pass validates
  the harness, not a teacher.
- **G1 exact calibration reproduction** — episode 2 / seed 10, budget 500
  steps, terminal reward 4.0 with task success, zero refusals, zero
  joint-limit violations, zero unexpected collisions. Necessary and
  insufficient by design.
- **G2 cross-pose generalization** — reserved tuning group seeds
  `1900–1904` (five registered poses), every pose solved within the 500-step
  budget; a single failure fails the gate. This is the criterion the
  recovery oracle failed 0/1.
- **G3 recovery from deviated states** — at least 3 registered perturbed
  states per G2 pose (from bounded perturbations of successful trajectories
  and from the completed first-deviation trace states), restore task
  progress within 150 actions with all safety criteria held, refusals below
  10%. Locked until G2 passes and the deviated-state inventory is
  registered.
- **G4 boundary honesty** — registered negative suite with zero silent
  defaults. Locked until G2 passes.
- **G5 independent audit** — evidence-chain identity and ordering recheck
  without the authoring session.

## Seed and scope boundaries (frozen)

Opened by this protocol: exact episode 2 / seed 10 and the tuning seeds
1900–1904. Sealed: development 2000–2004, collection 3000–3004, policy-gate
1000–1004, hidden test. A gate pass permits exactly one thing — proposing
the next preregistration (recovery-data contract and policy-side
single-axis comparison). It opens no collection seeds, writes no label
manifest, authorizes no furnace, and does not modify the six-identity Gate 4
failure record. Thresholds are immutable once frozen.

## Implementation status at registration

The contract module (`src/rosetta_reality/sim/teacher_gate.py`), the staged
runner (G0 executable; G1/G2 rollout loop implemented and fail-closed behind
a declared simulator-execution capability; G3/G4 locked by rule; G5 audit)
and the focused test suite exist at registration time. The local sim image
lacks the Mink dependency the geometry-teacher candidate needs, so the first
real candidate registration and its G0 run are the next step after this
preregistration; the dependency boundary must be resolved by a separately
registered image decision or an IK-free candidate design.

## Amendment 2026-08-29 (simulator image dependency resolved)

The implementation-status section above records the missing Mink dependency
as a blocker. It is resolved by the registered image decision of this date:
the local `vla-sim-xpu` image was rebuilt from the unchanged, already-pinned
`docker/Dockerfile.smolvla-sim-xpu` (commit `3306a5f` added
`mink==1.2.0`/`qpsolvers==4.13.0`/`daqp==0.8.7` on 2026-08-27, but the local
image predated it). Build-time version assertions passed and the imports were
verified in-container. New image identity:
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`
(previous `sha256:f4a71c40…` remains pinned by the frozen historical runners
and is untouched). Rollout-stage evidence records
`ROSETTA_CONTAINER_IMAGE_ID` from this amendment forward.
