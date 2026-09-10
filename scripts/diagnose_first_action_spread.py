"""Read-only train-episode first-action spread on one verified revision.

A constant mean is a descriptive baseline, not a lower bound on task error.
These statistics alone cannot identify why a policy deviates at reset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from rosetta_reality.vla.vision_diagnostics import load_frame_zero_context  # noqa: E402


def main() -> int:
    context = load_frame_zero_context(REPOSITORY_ROOT, "train")
    actions, states = context["actions"], context["states"]
    print(f"episodes at frame 0: {len(actions)} (train only; hidden rows excluded at scan)")
    state_spread = float(np.abs(states[:, None, :] - states[None, :, :]).max())
    print(f"frame-0 state pairwise max diff: {state_spread:.6f}")
    acts = actions
    mean_action = acts.mean(axis=0)
    mad = float(np.abs(acts - mean_action).mean())
    print(f"first-action mean-abs deviation from episode mean (14 dims): {mad:.4f}")
    pairwise = float(np.abs(acts[:, None, :] - acts[None, :, :]).mean())
    print(f"mean abs pairwise difference between episodes' first actions: {pairwise:.4f}")
    for name, dim in (
        ("left_waist", 0),
        ("left_shoulder", 1),
        ("left_elbow", 2),
        ("grip_L", 6),
        ("right_waist", 7),
        ("right_elbow", 9),
        ("grip_R", 13),
    ):
        print(
            f"  {name:14s} std={acts[:, dim].std():.4f} "
            f"range=[{acts[:, dim].min():+.3f},{acts[:, dim].max():+.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
