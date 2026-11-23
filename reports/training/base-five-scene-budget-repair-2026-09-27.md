# Base five-scene budget boundary repair — 2026-09-27

The active create-only plan is now
`configs/diagnostics/base-five-scene-independent-20260927-003.json`.
Plans 001 and 002 retain their original bytes and must not execute with the
changed source. The pinned base model, canonical interface, five seeds, noise,
500-step episode bound, action semantics, physical safety stops and original
formal Gate history are unchanged.

The previous diagnostic checked resources only before prediction. A final
prediction could cross the deadline or transient CUDA allocation cap, release
memory, and still issue its action or report episode completion. The repaired
`scripts/diagnose_smolvla_base_five_scene.py` checks immediately after prediction,
again immediately before an environment action, after persisting every step,
after reset, after terminal trace hashing, after environment close, and during
final parameter integrity capture and overall closeout. A late prediction never
issues an environment action. An already executed terminal step retains its
trace but cannot produce a completed episode after a budget violation.

CUDA checks include both current allocation/reservation and their recorded peak.
The measurement peak window is reset exactly once after policy construction and
the initial full parameter/buffer hash, before any simulator episode. This
separates initialization transients from measured inference peaks while retaining
current model residency in the limits. The 1800-second work clock still starts
before policy construction; initialization does not obtain extra wall time.
The external work-only guard must continue to cover preparation/load as well.

`make_budget_checker` accepts a clock and resource reader for deterministic
fault injection. `make_episode_runner` accepts these optional test dependencies
without changing production actions. The default reader uses Linux process RSS
and CUDA current/peak allocation/reservation. The registration still requires
12 GiB RSS, 8/10 GiB CUDA allocated/reserved and 256 MiB evidence limits.

`summary.json` records the measured episodes and may use
`measurements_complete_pending_closeout`; it is not final completion evidence.
The new create-only `completion.json` owns final status and binds the summary
SHA. It is written after the summary resource check, including anticipated
completion bytes. Completion requires its status `completed`, process exit 0,
all five measured episodes, and `parameter-integrity.json` proving unchanged
parameters/buffers. Budget or integrity failures report `blocked` and exit 1;
partial measurements and traces remain. No shutdown call was added.

The test entry remains:

```bash
python -m unittest discover -s tests -p test_base_five_scene_diagnostic.py -v
```

The suite now contains 18 tests, including injected inference deadline crossings,
transient allocated/reserved peaks with current occupancy returned to zero,
terminal-step and environment-close overruns, final-integrity failures,
post-summary closeout failure, and anticipated evidence bytes. These are fake
physics tests, not real model acceptance. Local read-only AST checks passed;
the registered Linux runtime must execute the full suite before actual Exp1.
Implementation or synthetic test completion does not complete the experiment.

New source/plan identities:

| File | SHA256 |
| --- | --- |
| `scripts/diagnose_smolvla_base_five_scene.py` | `f129f4db99622fca800cf159685e55c70df10ad1b36b37f728f97eada7ae7848` |
| `tests/test_base_five_scene_diagnostic.py` | `4bf301759868547d5c0407703672ae29cdca66fde1d0186a7db6470a0b0dc85d` |
| `configs/diagnostics/base-five-scene-independent-20260927-003.json` | `39abec6dce8679c30bc5638abe17030af1c9d0d4018cd7e7892f4994867293e6` |
