# Iris Gate precondition failure and isolated-root correction

The first Gate campaign `iris-002-gate34-20260913-001` completed CUDA doctor and
artifact/backup acceptance, then stopped before model inference or closed-loop
actions with:

```text
ValueError: Train-only dataset view identity differs from the normalization report.
```

The native loader resolves the normalization report's dataset-view path against
`ROSETTA_RUN_ROOT`. The new wrapper had set that variable to its isolated Gate
output root before calling the loader. Both referenced files had the correct SHA,
but the computed data-view directory identity differed. This was a wrapper path
integration defect, not a failed safety rollout or scientific Gate outcome.
Gate 3/4 remain unmeasured for that attempt; optimizer updates were zero.

The failed workspace was `20260913T151530Z-471d427bcd83-2f0bc98c8480`, archive
SHA256 `2f0bc98c84809e66a99ecb2c98ac4f98594173a7c6611e4a4cf8098ef1fc6313`.
All 21 evidence files (280,257 bytes) were recovered and verified. Archive SHA:
`de57ceab6c8dbdc875f9c877031aac21c0bd9fd18514a0bd1b80aa666d76f224`;
manifest SHA:
`95b8147e213ac89cb01410177074d0048d171471b2c1b56315cc1ac9882a7723`.
The receipt was delivered and the live console subsequently confirmed shutdown.
The failed job/template and source weights remain untouched.

The correction temporarily uses the registered durable run/data root during
native loading and restores the Gate output root in `finally`, on success and
on exception. The original normalization SHA/path containment checks, simulator
engine and acceptance thresholds remain unchanged. Two targeted regression tests
cover root selection and restoration. The relevant local suite now has **51 passed**,
plus Ruff, formatting and shell syntax checks in the same fixed offline Docker.

Continue the user's authorized acceptance task under a fresh identity:
`iris-002-gate34-20260913-002`. It uses the same two original Iris 002 final
checkpoints, suffixes 451/452 in its own output root, zero optimizer, Gate 3 then
conditional Gate 4, the same 3600/4200-second budget and protected lifecycle from
`m2-smolvla-iris-gate34-plan-2026-09-13.md`. There is no retry inside the failed job
and no checkpoint/seed/threshold search. The parent monitors directly.

Template: `iris-gate-template-002-2026-09-13.json`, SHA256
`d0b17a982a4043e91e02441b49ee7cec70e23dd43e590f2d5202606b9f583954`.
Real bridge/Gate results remain pending; the offline negative result and M2 status
are unchanged.
