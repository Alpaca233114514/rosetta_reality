# M2 SmolVLA vcdropout Gate 3/4 comparison preregistration (2026-08-29)

Registered, hash-bound protocol for the separately registered Gate 3 / Gate 4
closed-loop comparison of the visual-conditioning state-dropout arm. The
frozen offset-250 gradient gate passed on the exported candidate artifact
2026-08-28, which is the only event that permits this registration. Nothing
in this document reopens training, thresholds or the gradient gate.

JSON companion: `m2-smolvla-vcdropout-gate34-preregistration-2026-08-29.json`.

## 1. Hypothesis and registered comparison

- Axis context: the module-gradient diagnostic proved a state-dominant
  shortcut (state-shuffle moves trainable-module gradients 1.5–3.4x while
  image shuffle moves them ≤~8%); the T4 treatment (training-only
  whole-sample normalized-state dropout, dedicated RNG) trained arm
  `m2-smolvla450m-vcdropout-cuda-b64-001`, and its frozen offset-250 gradient
  gate passed (`image_sensitivity` `0.226087`, `state_dominance_score`
  `0.408466`).
- Registered question: does the treated artifact behave measurably better in
  the closed loop than the immutable historical controls under the identical
  protocol? The historical frame: Faust, Aster, Way, Zen-uniform (`411`) and
  Zen-firstaction (`422`) all failed Gate 4 `0/5` with reward `0` on seeds
  1000–1004.
- Primary endpoint: Gate 4 `minimum_task_success_rate` (task success on at
  least `0.2` of the five registered seeds), exactly as enforced by the
  frozen engine. Secondary observations: per-seed success, maximum reward,
  rollout length, collision/joint-limit/invalid-action classes, smoothness
  and latency. Offline MAE is explicitly non-gating and non-comparative here.

## 2. Frozen protocol (engine-enforced)

| Field | Value |
|---|---|
| simulation engine | `scripts/smolvla_sim_gate.py` `5b76127a2e2d0e0049181a1d0ab12297474cbc8eb433c4fcdb6466c35c53c5ae` |
| gate wrapper | `scripts/smolvla_vcdropout_sim_gate.py` `dcf40b0936de765e95e7529a8128cf0a068f7743cf6e4457f90a71d778ee0e58` |
| protocol module | `scripts/smolvla_vcdropout_protocol.py` `4081f94eb9bcc959be0b71f59f5aff99886b42620ee1a48ab8fef83792dc4ebe` |
| Action Contract | `configs/sim/aloha_insertion_smolvla.yaml` `fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62` |
| candidate formal plan | `configs/vla/smolvla_450m_aloha_insertion_vcdropout_cuda_b64_001.yaml` `fd6784ae221eb96c9ac36531e9fa14d73182f243a3a1936c660ddb86a9509e39` |
| evaluated artifact | `m2-smolvla450m-vcdropout-cuda-b64-001-step0237-deploy-001`, manifest `abb8497a7424df2ad1c3e03eb364e581d9d0f05e12dfdf3a59d8bb4ed67f80ea` |
| entry permit | `runs/…/diagnostics/vcdropout-gradient-gate-974130aba90568e3.json` `10179ba2a8a57cbf39925a2fddc9b9c7ce6020ad88a084d5f76c31f6bd6809aa` |
| prior Gate 3 projection failure | `runs/m2-smolvla450m-aloha-insertion-001/gates/gate3-smolvla-sim-001.json` `5df3b887984d1c8fd47084c3315a71e3894eac18cc80b1eb08b2120583ed26ed` |
| prior Gate 4 task failure | `runs/…-003/gates/gate4-smolvla-sim-003.json` `86c35a4dbcb70761a5ec6787fd95e6dcbd11e26738aaab9e34500be02aff8a46` |
| report suffix | `433` (campaign series after Zen `411`/`422`) |
| Gate 3 | seed `20260809`, policy noise seed `20260809`, 20 steps, seeded standard-normal noise, `action_contract_clip` projection at the VLA output boundary, unprojected actions retained as non-gating diagnostics, maximum unexpected collisions `0` |
| Gate 4 | environment seeds = policy noise seeds = `1000..1004`, 500 steps per episode, `minimum_task_success_rate` `0.2`, maximum unexpected collisions `0`, requires the matching passed Gate 3 report |
| collision classifier | explicit task-contact allowlist (peg/right-finger, sockets/left-finger), same-arm internal gripper contacts non-gating, every other robot-scene contact unexpected |
| execution boundary | local WSL `vla-sim-xpu` container, image `sha256:f4a71c4020cd54d2a878f01628d591af9572f0784458f4c821008f8aea30393c`, memory limit `6g`, network disabled, `ROSETTA_TORCH_DEVICE=xpu`, bf16 autocast |

The wrapper renders one create-only simulation plan
(`configs/vla/m2-smolvla450m-vcdropout-cuda-b64-001-sim-433.yaml`) whose
identity fields bind the artifact manifest, the derived gate-facing selection
record, the inventory backup record, the Action Contract, the engine/wrapper
code hashes and both prior failure reports. The engine's own XPU-native
online policy class and runtime probe are used unchanged; no policy code is
cloned or altered.

## 3. Acceptance and interpretation rules

- Gate 3 criteria (all must hold): non-empty rollout, finite actions,
  executed and projected policy actions within the Action Contract, joint
  limits respected, zero unexpected collisions, artifact reload verified.
- Gate 4 criteria (all must hold): the Gate 3 criteria above aggregated over
  five episodes plus `minimum_task_success_rate >= 0.2`.
- Gate 4 runs only after a passed Gate 3 report whose artifact manifest sha,
  simulation plan sha and workspace code identity match exactly.
- Any failure is a registered negative result: it closes the visual-
  conditioning axis at this development scale with immutable evidence and
  returns the workflow to diagnosis. No threshold may be widened, no seed
  added, no protocol re-run under a different identity to produce a pass.
- A Gate 4 pass does not complete M2 by itself: it marks the first candidate
  that survived the registered task evaluation and authorizes the next
  registered stage (hidden-test boundary stays sealed; further claims require
  their own registration).

## 4. Boundaries

- Hidden-test episodes stay sealed; no checkpoint, threshold, seed or
  simulator parameter may change after this registration.
- No further training of this arm and no reuse of its optimizer state.
- The gradient gate, its six criteria and the preregistered treatment are
  immutable; this comparison consumes their authorization, it does not
  extend it.
- Episode reports are create-only under the durable run root; a crashed
  episode identity must never be regenerated under a different code identity.
