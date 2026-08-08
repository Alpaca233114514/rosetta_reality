# M1 Acceptance — Dataset Pipeline

Status: complete for the bounded acceptance slice on 2026-08-09.

The closed slice is episode 0 of `lerobot/aloha_sim_insertion_human`. M1
establishes a revision-pinned, robot-agnostic data path and a CPU-only smoke
boundary; it does not claim model-weight integration, physical action
semantics, simulator control, or formal training.

## Acceptance evidence

All model and data checks below were run in WSL from the repository root:

- `python scripts/check_env.py` — Python 3.13.5, PyTorch 2.11.0+cpu, CUDA not
  available; the environment check completed successfully.
- `pytest -m "not data"` — 31 passed, 1 skipped, 4 deselected.
- `python scripts/train.py --dry-run` — CPU dry-run succeeded with prediction
  shape `(2, 8, 7)` and finite Smooth L1 loss `0.404727`.
- `python scripts/prepare_data.py inspect` — read-only inspection succeeded
  for resolved revision
  `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`; metadata, statistics, and 9
  checksum entries were present for the 500-frame episode cache.
- `pytest -m data` — 1 passed and 3 skipped. The passing case is
  the closed M1 insertion fixture. The skipped cases correspond to additional
  dataset configurations without a complete matching local cache; they are
  not counted as M1 acceptance evidence.
- `ruff check .` — all checks passed.

No model weights or new dataset cache were downloaded during acceptance. The
next milestone is M2, subject to its development-scale model, action contract,
and simulation gates in `AGENTS.md`.
