# Iris Gate state-contract adapter correction

Gate attempt `iris-002-gate34-20260913-002` completed doctor, source acceptance
and native data/model loading, then stopped before its first bridge forward:
`ValueError: Simulator observation has an invalid ALOHA state.` No closed-loop
action, optimizer update, or measured Gate outcome occurred.

The wrapper had used the policy config's upstream `observation.state` placeholder
(6) as the simulator state dimension, although the actual dataset contract is
14-dimensional. The unchanged original engine already documents this distinction
and exposes `_dataset_state_dimension` / `_validate_policy_contract_shape`.
The wrapper now calls those original helpers, retaining action/chunk shape checks.
It does not change the model config, state ordering or simulator observations.

Attempt 002 workspace: `20260913T152502Z-8f5631b0d857-0e55784fa493`, source archive
SHA256 `0e55784fa493b64540896ad8bba1de40f6e95bb11dcf6dabaae2fc2df04c03cf`.
All 21 evidence files (285,094 bytes) passed local recovery verification.
Archive SHA256 `293a75ab321fad9cc9df6552f6ba4ba59a9980efd0d2aff754c5fe5d4b8616be`;
manifest SHA256 `429e4578ccd8017d8ef0e3f5732fa3019202d4ca4e01715854294f0058e7a34e`.
The receipt was delivered and the live console confirmed shutdown. Preserve both
precondition failures and their untouched templates/workspaces.

The new regression test supplies a 6-dimensional config placeholder and a
14-dimensional dataset state contract, checks that 14 is used, and verifies that
an incorrect policy action dimension still fails. Relevant local checks:
**52 passed**, Ruff, formatting and shell syntax passed in the fixed offline
Docker image. The previous root-isolation tests remain included.

Continue the same user-authorized acceptance under fresh identity
`iris-002-gate34-20260913-003`, with the same two Iris 002 final checkpoints,
Gate suffixes 451/452, seeds, zero optimizer, 3600/4200-second bounds and original
engine/thresholds from `m2-smolvla-iris-gate34-plan-2026-09-13.md`. No failed job
is retried in place and no scientific selection or training parameter changes.

Template `iris-gate-template-003-2026-09-13.json`, SHA256
`08e30497289d87704d96d9f1427512e9768d8b6e473b6c83cea7ae082d5a16a3`.
Actual bridge and Gate outcomes remain pending; M2 remains incomplete.
