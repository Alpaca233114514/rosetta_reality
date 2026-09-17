# Repaired training-chain SSH verification — 2026-09-14

User-authorized engineering test of the current dirty feature branch, with emphasis
on actual training frames and adjacent evaluation/Gate correctness. Historical
plans, checkpoints and results remain immutable. No commit, push or formal furnace.

## Admission and bounds

- Original instance: `44db45aec7-cd880eb6`, SSH alias `rosetta-autodl`, port 20497.
  Chrome rejected GPU startup because the host had zero available cards. Start
  this same instance without a card for CPU/source/data verification only.
- Use the existing AutoDL platform container and existing environment; no nested
  Docker, package changes, model/data downloads, cloning or instance release.
- Stage the current reviewed source into a fresh content-addressed workspace via
  `scripts/stage_autodl_from_wsl.sh`. Record exact archive and per-file hashes
  before execution; never overwrite a historical workspace.
- No-card work: at most 1800 seconds, one CPU thread, 1.5 GiB process-tree RSS and
  at most 256 MiB added source/evidence. Stop on resource, identity or split drift.
- Production policy optimizer budget is zero during this CPU phase; tiny-policy
  regression optimizer calls must be distinguished from production model updates.
- GPU doctor, actual 450M forward/backward and independent model reload require
  a card and a separately sealed small smoke identity. No GPU success is inferred
  from CPU results, historical results or saved-array verification.

## Measurements

1. Verify current source/runtime/cache identities and run the repaired training,
   temporal loader, observer, loss, optimizer, selector, reload and Gate regressions.
2. Check actual nonhidden cached frames, action chunk tails/padding, raw versus
   processed values and historical consumed identities. Report exposures, unique
   input frames, target coverage, evaluation times and deployment use separately.
3. Extend checks to video timestamps/decoder agreement and real dataset integration
   where supported, without materializing hidden rows or changing data semantics.
4. Preserve every failed attempt and report unsupported runtime paths explicitly.
   A test pass is bounded engineering evidence, not zero-defect assurance, a new
   model-quality claim, a new Gate 3/4 result or M2 completion.
5. Recover and hash-check evidence, then shut down this session's instance after
   confirming no unrelated worker is active. Verify final state in Chrome.

CUDA/model/Gate execution remains unmeasured while the original card is unavailable.

The live no-card dialog advertises 0.5 CPU cores and 2 GB memory. The CPU phase
therefore admits only sequential bounded subprocesses and stops before 1.5 GiB
combined worker RSS; heavy model work is excluded. The UI price is 0.10 CNY/hour
with 0.01 CNY minimum; this 30-minute CPU phase is bounded at 0.05 CNY.
