# Q0 shared-translation teacher screening candidate

Candidate/provider identity: `mink-coupled-translation-teacher-20260927-001`.
This is an experimental candidate, not an admitted data source. The current
user request authorizes remote execution preparation and a bounded run; the
older revision-002 authorization snapshot is historical. Execution still
requires a fresh stage/runtime/deadline registration and actual benchmark.

## Evidence and hypothesis

The controlling revision is
`reports/training/experiment-redesign-received-20260927-002/reports/training/three-experiment-design-2026-09-27-revision-002.md`.
It requires one Q0 route, unchanged teacher gates, and failure preservation.
The old teacher-003 closure's prose says 3/5 whereas its per-seed JSON says
2/5; neither admits a provider. The source files remain unchanged.

`reports/training/m2-smolvla-t2-seating-feasibility-probe-2026-09-03.md`
reports a single seed, 60-step seating probe with the right arm held at its
live pose. This does not test moving both arms together. The hypothesis is
that finite common relocation of both end-effector targets can reduce the
loaded IK residual while preserving their commanded relative pose. This is
not evidence that the task is solvable or that the controller will succeed.

## One structural axis

`src/rosetta_reality/sim/coupled_translation_teacher.py` subclasses the existing
geometric teacher. Only INSERT replaces each existing IK-seed solve with six
solves at common world-frame translations, in order:
`(0,0,0), (+.02,0,0), (-.02,0,0), (0,+.02,0), (0,-.02,0), (0,0,+.02)` metres.
Both target positions receive the same vector; orientations stay unchanged.
Selection is lexicographic: maximum position residual across arms, maximum
orientation residual across arms, then the fixed lattice index. Every
candidate is evaluated; there is no randomized search or retry expansion.
The original multistart branch selection/early acceptance then applies.

Targets remain anchored to the original stage target, never recursively
accumulated. Original stage transitions, hold branch, explicit refusals,
contact/height guards, arm-command bound (0.06 rad), Action Contract, physics,
reward, and all gate thresholds remain unchanged. Actual object-relative
pose is not guaranteed by preserving EEF target-relative pose. In particular,
the known APPROACH collision risk is not repaired in this single-axis run.

The candidate retains inherited current-source behavior, including the
existing zero-delta hold; it is not claimed byte-identical to historical 003.
All source identity must be sealed at staging. No dependencies are added.

## Bounded screening, not complete Q0

Register a fresh execution with module
`rosetta_reality.sim.coupled_translation_teacher`, factory `build_teacher`,
observer `rosetta_reality.sim.scripted_teacher_observer`, and declared
simulator execution. Run synthetic tests and environment checks first,
then G0, G1 (seed 10, at most 500 actions), G2 (1900–1904, at most 500 each)
only when the prior stage passes. Stop on stage failure; retain every seed
in the fixed G2 denominator. Record zero optimizer/model forward calls.
The supervisor must additionally reject any refusal, joint-limit violation,
or unexpected collision during G2; the legacy G2 acceptance alone checks
success and is insufficient to certify safety.

There are at most 3,000 environment actions in the screening. Each INSERT
action requires at most 6 lattice points times 9 original IK branches =
54 IK solves; other phases retain original cost. Worst-case 162,000 solves
is an arithmetic cap, not an estimated duration. A suggested work cap is
1,800 seconds, output cap 256 MiB, with at least 300 seconds for sealing,
retrieval verification and shutdown plus a later independent hard cutoff.
Admission requires measured target-solve and environment overhead to fit
the fixed action budget; if it cannot, stop rather than shorten gate
budgets, skip poses, or extend the deadline while running.

`scripts/benchmark_q0_candidate.py --output <new-report.json>` creates a
separate seed-10 reset and executes at most three zero-delta hold commands,
each equal to the current registered standard-space robot state. These
benchmark actions are not G1. Before each command it verifies finite shape
and rejects any required Action Contract clipping; after every command it
checks state/reward finiteness, physical joint limits, unexpected contacts,
actual clipping and premature termination. A safety failure stops immediately.
The reset is also checked for joint-limit violations and unexpected contacts.
This adds at most three environment actions beyond the 3,000 gate-action cap.

It still enumerates all 54 IK solves against the saved reset geometry without
modifying candidate settings or executing the IK outputs. It records package
versions, each solve duration/iteration count, the complete 54-solve loop,
cold reset/observation cost and maximum measured hold-step/render/observer
cost. Admission estimate is
`6*cold_reset + (54-solve loop + max_hold_step) * 3000 * 1.5 <= 1800 seconds`.
Cold reset is counted once for each of the six gate episodes, not per action.
An exceeded estimate exits nonzero. This remains an estimate: reset geometry
and three hold steps do not reproduce loaded contact solve/step latency.
A later live watchdog still owns the hard work cap. No estimate is improved
by omitting any of the registered 54 IK solves.

`scripts/run_teacher_gate.py` G3 and G4 currently raise unconditionally
(lines 438–451 at authoring); the required registered perturbation inventory
and negative suite are absent. G5 only compares present report code identities
(lines 453–474); it does not prove complete passing G0–G4. Therefore even G2
5/5 in this screening must stop as **Q0 incomplete**, never launch collection
or training. Complete Q0 still requires at least three preregistered
perturbations per G2 pose, each at most 150 actions, the frozen refusal/safety
thresholds, a registered G4 suite, and an independent complete-chain audit.

Synthetic tests in `tests/test_coupled_translation_teacher.py` cover shared
target geometry, deterministic tie breaking, residual-based selection,
all-failed refusal, command bound, solver restoration and unchanged early
phases. At authoring they have not been executed; they do not establish
real MuJoCo/Mink behavior, valid labels, teacher qualification or M2 success.
