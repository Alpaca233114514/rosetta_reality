# GPU audit 003: independent stage run names

Run 002 passed the native preflight and real loss, all 155 gradient tensors,
and full action exact-parity checks for episode 49 at frames 0, 249 and 499.
Parameters remained unchanged. Its 20 evidence members were recovered and
SHA-verified. The smoke launcher then correctly refused a launch-manifest
collision because the new audit plan reused one run name across preflight
and smoke. The observer records zero delivered samples and zero updates.

Run `training-chain-gpu-audit-20260914-003` assigns distinct preflight and smoke
run names. Historical evidence and create-only protection remain intact.
The same registered resource, two-update, data, noise, numerical and Gate bounds
apply. No formal training or model selection is authorized by this amendment.
