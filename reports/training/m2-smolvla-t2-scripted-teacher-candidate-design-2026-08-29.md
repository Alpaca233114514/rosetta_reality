# M2 T2 scripted-policy teacher candidate design (2026-08-29)

Design note for the first real teacher candidate of the frozen staged gate
`m2-t2-teacher-gate-001`
(`reports/training/m2-smolvla-t2-teacher-gate-protocol-preregistration-2026-08-29.md`).
This is a design, not a preregistration: gate thresholds are already frozen in
the protocol; the candidate identity becomes immutable only at the
create-only `register` step. Work branch: `codex/t2-scripted-teacher-candidate`.

## Motivation

The vcdropout axis closed as a registered negative result
(`m2-smolvla-vcdropout-gate34-result-2026-08-29.md`): the offset-250 gradient
gate passed (image sensitivity 0.226, dominance 0.408), yet Gate 4 failed 0/5
with new joint-limit/collision regressions. The state-dominant shortcut is
therefore a symptom, not the root cause; the surviving registered root cause
is T2, no state-conditioned recovery supervision. The design doc of the
staged teacher gate additionally records that the historical teacher chains
(recovery oracle plans 001/002, geometry teacher plans 003–058) cannot pass
G2 cross-pose generalization as-is. A state-based controller that re-plans
from the current observation is the cheapest known candidate class for G2.

## Verified facts (this session)

- Gate runner `scripts/run_teacher_gate.py`: `register` (module + factory +
  constructor + optional `--simulator-execution --observer-module`), then
  `g0` → `g1` → `g2` → `g3` → `g4`, plus `g5`. G0 is pure contract
  conformance (no simulator). G1 = seed 10, 500-step budget, terminal reward
  4.0, zero refusals/violations. G2 = seeds 1900–1904, every pose solved.
  G3/G4 are implemented but locked until G2 passes. Evidence is create-only
  under `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/teacher_gate/<candidate>/`.
- Rollout contract: the declared observer module must expose
  `build_observation(environment, contract) -> InsertionGeometry`; the
  decision action is clipped through the unchanged Action Contract before
  `environment.step`; success reads `info["is_success"]`.
- Frozen observation `InsertionGeometry`
  (`src/rosetta_reality/sim/geometry_teacher.py`): `robot_state` (14-D),
  `left_eef`, `right_eef`, `socket`, `peg` (`GeometryPose`), `observed_reward`,
  `socket_grasp_contact`, `peg_grasp_contact`, `socket_on_table`,
  `peg_on_table`, `peg_socket_contact`, `pin_contact`,
  `unexpected_collision_count`. The field set is checksum-asserted at every
  G0 run; adding any time, index, episode, seed, dataset or policy field
  fails the gate by construction.
- The pinned sim image (`vla-sim-xpu`, image
  `sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`,
  per the 2026-08-29 protocol amendment) contains `gym-aloha==0.1.4` with
  `AlohaEnv`, tasks and pose samplers but **no scripted policy**, and
  `lerobot==0.6.2` also ships none. The scripted insertion controller must be
  vendored and adapted from upstream sources; it is not importable from the
  pinned dependencies.
- Local sim container rollouts are proven on this machine (Gate 3/4 `433`
  ran locally on Intel XPU with networking disabled). No AutoDL/SSH is
  needed for any teacher-gate stage or later label generation; AutoDL stays
  shut down.

## Proposed candidate

- Candidate module `src/rosetta_reality/sim/scripted_teacher.py` exposing
  `build_teacher()` returning an object with `identity`, `reset()`,
  `decide(InsertionGeometry) -> TeacherDecision` (the `TeacherCandidate`
  protocol in `src/rosetta_reality/sim/teacher_gate.py`).
- Suggested identity: `gym-aloha-scripted-insertion-adapter-001` (final id is
  frozen at registration).
- Input mapping (only frozen `InsertionGeometry` fields):
  `peg`/`socket` `GeometryPose` = box/target poses; `left_eef`/`right_eef` =
  end-effector positions; `robot_state` dims 6 and 13 = gripper openness;
  contact flags as stage guards; `unexpected_collision_count > 0` and
  corrupted/non-finite geometry map to explicit `TeacherRefusal`
  (`out_of_support`, `corrupted_geometry`, `unsafe_state`,
  `workspace_violation` — one class each, never silent degradation).
- Stage advancement must be **state-conditioned** (geometric thresholds on
  eef/object poses). If the upstream policy uses time-indexed stages, the
  adapter converts them to progress conditions; fidelity is then validated by
  G1/G2 evidence, never assumed.
- Observer module (new file, do not edit the frozen staged evaluator):
  extract exactly the frozen fields from `GymAlohaEnvironment`; reuse the
  geometry-extraction logic of `scripts/evaluate_aloha_geometry_teacher.py`
  as reference only.

## Open items for the implementation session

1. Locate and pin the authoritative scripted source (upstream ALOHA
   `aloha_sim` scripted env and/or the generator recorded on the
   `aloha_sim_insertion_scripted` dataset card). Record the exact upstream
   ref and verify its license before vendoring; do not reconstruct from
   memory.
2. Analyze the stage machine: time-indexed vs state-conditioned, and the
   exact per-stage inputs; design the state-conditioned conversion.
3. Confirm action semantics: absolute 14-D standard-space joint position
   targets at 50 Hz under the Action Contract, including gripper encoding.
4. Implement adapter + observer + focused tests (determinism under replay,
   refusal-region non-degeneracy, field-set conformance), pass Ruff and the
   container checks, then `register` → G0 → G1 → G2, each stage's
   create-only evidence preserved on failure with suffix increments.
5. On a G2 pass: G3 needs the registered deviated-state inventory (from the
   completed first-deviation traces), then G4, G5. A full gate pass permits
   exactly one thing — proposing the recovery-data contract preregistration.
   Collection seeds 3000–3004 stay sealed until that separate registration.

## Boundaries

Single axis = one new teacher candidate measured by the frozen gate. No
threshold may be widened; no collection, label manifest or furnace is
authorized by any stage result; the six-identity Gate 4 failure record stays
untouched; the hidden test stays sealed; AutoDL remains shut down.
