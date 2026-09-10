# M2 SmolVLA vcdropout Gate 3/4 comparison result (2026-08-29)

Registered, hash-bound execution result of
`m2-smolvla-vcdropout-gate34-preregistration-2026-08-29` (suffix `433`).
JSON companion: `m2-smolvla-vcdropout-gate34-result-2026-08-29.json`.

Verdict: **Gate 3 passed; Gate 4 failed `0/5` — the visual-conditioning
state-dropout axis is closed at this development scale as a registered
negative result.** The sixth consecutive identity to fail the development
task evaluation, and the first whose training-time treatment measurably
changed the diagnosed gradient-level pathology.

## 1. Gate 3 — passed

Report `runs/…-003/gates/gate3-smolvla-sim-433.json`
(sha256 `a05c902da5903382…` full value in the JSON companion), rendered from
the registered simulation plan
`configs/vla/m2-smolvla450m-vcdropout-cuda-b64-001-sim-433.yaml`
(sha256 `14838fef385693b2…`). All eight criteria passed: 20-step rollout at
seed `20260809` with seeded standard-normal noise, finite actions, executed
and projected policy actions within the Action Contract, joint limits
respected, zero unexpected collisions, artifact reload verified. Runtime:
local WSL `vla-sim-xpu` container, Intel XPU, bf16, network disabled.

One create-only attempt preceded this pass and is preserved under
`runs/…/orchestration/vcd-gate34-attempt1-archive/`: the wrapper originally
crashed while the container rebuilt the saved processors because the Rosetta
processor steps were not registered; the fix imports
`rosetta_reality.vla.processor` for its registration side effect, and no gate
evidence existed before the archived attempt.

## 2. Gate 4 — failed

Report `runs/…-003/gates/gate4-smolvla-sim-433.json`. Five 500-step episodes,
environment seeds = policy noise seeds `1000..1004`:

| seed | success | max reward | unexpected collisions | joint-limit violations |
|---|---|---|---|---|
| 1000 | false | 0 | 7 | 48 |
| 1001 | false | 0 | 13 | 15 |
| 1002 | false | 0 | 0 | 20 |
| 1003 | false | 0 | 10 | 14 |
| 1004 | false | 0 | 4 | 34 |

Aggregate: `task_success_rate` `0.0` (< `0.2`), `joint_limit_violations`
`131`, `unexpected_collisions` `34`, mean rollout length `500.0`, mean
action smoothness L2 `0.33099`, invalid-action rate `0.0`, all
raw/projected/executed limit-violation rates `0.0`. Failed criteria:
`joint_limits_respected`, `maximum_unexpected_collisions`,
`minimum_task_success_rate`. Passed criteria: every action-integrity class —
the pipeline, adapter chain and projection boundary behaved exactly as
registered; the failure is behavioral, not mechanical.

## 3. Interpretation

- The treatment did what it was designed to do at the gradient level: the
  frozen offset-250 gate passed (`image_sensitivity` `0.226087` versus the
  Zen-baseline ≤~8% shift; `state_dominance_score` `0.408466`). The closed
  loop nevertheless produced zero success and zero reward — the same frame
  as all five historical controls — while adding 131 joint-limit violations
  and 34 unexpected collisions that the cleanest control (Zen-firstaction)
  did not have.
- Therefore the state-dominant shortcut was a real, gradient-level symptom
  but not the root cause of closed-loop failure: partially removing it
  without adding recovery or skill supervision does not produce competent
  behavior, and naive half-batch state dropout degrades closed-loop safety
  margins. Offline gradient health and closed-loop competence remain
  decoupled, extending the campaign's offline/online decoupling finding from
  MAE to modality-gradient metrics.
- Per the registered interpretation rules this closes the visual-
  conditioning state-dropout axis with immutable negative evidence. No
  threshold was widened, no seed was added, and the hidden test remains
  sealed.

## 4. Standing tally and next axis

Gate 4: Faust, Aster, Way, Zen-uniform (`411`), Zen-firstaction (`422`),
vcdropout (`433`) — all failed `0/5`; every other stage (training, selection,
export/reload, Gate 3) passed for all of them. M2 remains **not complete**.

The only untried primary axis is T2: a state-conditioned recovery teacher
that passes its own gate before any collection or furnace is registered
(design staged in
`reports/training/m2-smolvla-t2-state-conditioned-teacher-gate-design-2026-08-28.md`,
design only). This result strengthens its motivation: the policy needs
supervision that repairs deviated states, not merely a rebalanced modality
prior. The vcdropout deploy artifact and both Zen artifacts remain in the
local artifact root; the AutoDL instance is shut down and must not be
released.
