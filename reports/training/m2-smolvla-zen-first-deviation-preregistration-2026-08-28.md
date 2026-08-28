# M2 SmolVLA Zen first-deviation trace preregistration — 2026-08-28

## 1. Authority and scope

This preregisters the next registered diagnostic step after the Zen formal
furnace audit (`reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md`,
its section 10 option 3). It is a **non-gating local diagnostic**: no training,
no checkpoint reuse, no new furnace, no hidden-test access, and no closed-loop
acceptance claim. It follows the established Aster trace precedent
(`scripts/diagnose_smolvla_aster_trajectory.py`, evidence
`runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/aster-trajectory-520c8ec87c1618fc.json`)
with the Zen-native create-only adaptation layer.

## 2. Hypothesis and question

The Zen campaign established that both arms fail Gate 4 `0/5` with reward `0`
while the first-action arm commits zero safety violations, and that offline
first-action MAE does not transfer to closed-loop success. The open question
is **where the selected Zen first-action artifact first departs from the
expert trajectory** at the registered reset, and whether its early divergence
profile matches the Aster-era state-dominant failure mode.

Quantitative comparison baseline (immutable Aster trace, same reset):

- step-zero action MAE `0.0204168`; first post-step state MAE `0.0055942`;
- state-MAE crossings: `0.01` at step 1, `0.025` at step 4, `0.05` at step 24,
  `0.1` at step 28; maximum state MAE `0.222958`.

The trace is descriptive. Either result (earlier, equal, or later divergence)
is recorded as-is; nothing here authorizes recovery labels or a T2 collection
run by itself.

## 3. Frozen identities

| Field | Value |
|---|---|
| traced artifact | `m2-smolvla450m-zen-cuda-b64-firstaction-001-step0316-deploy-001` |
| artifact manifest sha256 | `d6b2a7ff922605daf04670dd8e57a582fc4f5f5dcb1efd78ff37aec3357d0653` (from `gates/gate4-smolvla-sim-422.json`) |
| selection decision | `runs/…/selection/m2-smolvla450m-zen-cuda-b64-firstaction-001-selection.json`, sha256 `540de9b07c4ddb2a06420f2902d5c51219ab73cdc6d2a4063f91d30176a8b882`, step 316, first-action MAE `0.022150604739519103` |
| trace script | `scripts/diagnose_smolvla_zen_trajectory.py`, sha256 `583734d5a2af48ca44827d02857556cd2e2db1c5b65b4bf49df496ba4c14b138` |
| runner | `scripts/run_zen_trajectory_trace.sh`, sha256 `7b02f0080d1bea1ca27294c44c7366c84bad908792cb3f89ea492dfdc8c9810e` |
| frozen gate engine | `scripts/smolvla_sim_gate.py`, sha256 `5b76127a2e2d0e0049181a1d0ab12297474cbc8eb433c4fcdb6466c35c53c5ae` |
| zen protocol module | `scripts/smolvla_zen_protocol.py`, sha256 `330175327f6de66fde5fc2cb823eb243543a770dfe4db0f6e527d671f1653bbc` |
| simulation adapter | `src/rosetta_reality/sim/gym_aloha.py`, sha256 `e9c1005d0ae085e82e0c96e9d18527dce7d4749268a71756116cdffbb98d6e7d` (identical at `6fe7535` and `d75d1c9`, i.e. the exact code the remote r42b gates validated) |
| action boundary | `src/rosetta_reality/vla/processor.py`, sha256 `6751d4dd901da27e0a299bd9426fa484540e85dc12f1f1a62694e063d07e2384` |
| action space | `src/rosetta_reality/vla/action_space.py`, sha256 `4321d7d76e39db8644500be4c02f6de89caadfd58832047fde493450df1cfbeb` |
| Action Contract | `configs/sim/aloha_insertion_smolvla.yaml`, sha256 `fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62` |
| projection precedent | `runs/m2-smolvla450m-aloha-insertion-001/gates/gate3-smolvla-sim-001.json`, sha256 `5df3b887984d1c8fd47084c3315a71e3894eac18cc80b1eb08b2120583ed26ed` |
| noise precedent | `runs/…/gates/gate4-smolvla-sim-422.json`, sha256 `21a851abdda6b9f642a6021a768905bf0878ff37d3d13ccd75145affabf98003` |
| dataset | `lerobot/aloha_sim_insertion_human`, revision `cc571a3c661df81b566dbfde3d5c1e85fcdf7884` (resolved and recorded at run time) |
| runtime | local WSL Docker `vla-sim-xpu`, Intel XPU, image `sha256:f4a71c4020cd54d2a878f01628d591af9572f0784458f4c821008f8aea30393c`, memory `6g`, networking disabled |

The gate-facing selection record and the trace simulation plan are rendered at
run time (create-only, drift-checked) under
`runs/<experiment_id>/selection/…-selection-trace-gate.json` and
`runs/<experiment_id>/plans/m2-smolvla450m-zen-firstaction-trace-sim-001.yaml`,
following the registered Zen gate wrapper precedent, because the derivation
requires the artifact's `model.safetensors` sha256 which exists only after the
deploy artifact is transferred locally.

## 4. Protocol

- train episode 2 / simulator seed 10 / policy noise seed 10 / maximum 320
  steps — identical to the registered Aster trace protocol;
- two independent `GymAlohaEnvironment` instances reset with the same seed;
  cross-environment reset state MAE must be `<= 1e-7`;
- expert side: time-indexed dataset actions through the Action Contract
  (reference only, never a recovery oracle);
- policy side: reloaded Zen first-action deploy artifact, receding-horizon
  first action, seeded standard-normal noise, Action Contract clip projection
  at the VLA output boundary;
- recorded: per-step state/action MAE, first crossings of
  `[0.005, 0.01, 0.025, 0.05, 0.1]`, first reward/done/violation events,
  object/EEF pose deltas, contact and joint-limit diagnostics;
- artifact loading goes through the frozen `smolvla_sim_gate._load_artifact`
  validation (manifest checksums, selection identity, contract equality,
  precedents, resource boundary) — fail closed on any mismatch.

## 5. Acceptance criteria (of the diagnostic run itself)

1. a complete `zen-trajectory-<hash>.json` under
   `runs/<experiment_id>/diagnostics/` with all recorded values finite;
2. deterministic independent resets verified;
3. hidden-test episodes never loaded;
4. orchestration status/log written under `runs/<experiment_id>/orchestration/`.

Policy task performance is explicitly **not** an acceptance criterion; Gate 4
already established reward `0` for this artifact.

## 6. Stop conditions

- any frozen-identity or artifact-checksum mismatch (fail closed, no silent
  fallback);
- non-finite action/state values;
- container OOM or XPU runtime failure (no automatic retry; report and stop);
- no re-run with changed seeds, steps, artifact or code under this
  registration; any change requires a new preregistration.

## 7. Execution prerequisites (currently pending)

1. the AutoDL instance must be powered on by the user (it was already shut
   down after the audit) and the two selected deploy artifacts transferred to
   the local artifact root with checksum verification;
2. Docker Desktop running with the pinned `vla-sim-xpu` image
   `sha256:f4a71c40…` present;
3. then `scripts/run_zen_trajectory_trace.sh` from WSL Bash.

## 8. Expected evidence

- `runs/<experiment_id>/diagnostics/zen-trajectory-<hash>.json`;
- `runs/<experiment_id>/orchestration/zen-trajectory-trace-001.{log,status}`;
- rendered plan and derived selection record (create-only, drift-checked);
- a completion report `m2-smolvla-zen-first-deviation-trace-2026-08-28.{md,json}`
  recording the measured divergence profile against the Aster baseline.

## 9. What not to conclude

- the trace does not gate M2, does not rehabilitate or rank either Zen arm,
  and does not measure closed-loop task success;
- time-indexed expert actions after divergence must not be relabeled as
  recovery targets;
- a cleaner or dirtier early-divergence profile alone authorizes nothing; the
  T2 recovery-distribution axis still requires its own teacher-gate design and
  preregistration.
