"""First-step deviation measurement: teacher vs dataset expert (read-only).

Quantifies the phenomenon the user reports ("first step already deviates")
for the teacher line: resets the exact calibration episode (episode 2 /
seed 10), runs the registered candidate-003 teacher for the first N
decisions, and compares each commanded action against the dataset expert's
same-index action (the Gate 2 replay source). Light by design: one short
rollout and a two-row parquet read; writes no evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
for root in (str(SRC_ROOT),):
    if root not in sys.path:
        sys.path.insert(0, root)

from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.sim.geometric_teacher import build_teacher  # noqa: E402
from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment  # noqa: E402
from rosetta_reality.sim.scripted_teacher_observer import (  # noqa: E402
    build_observation,
)


def _expert_first_actions(episode: int, count: int) -> list[torch.Tensor]:
    import pyarrow.dataset as arrow_dataset

    cache_root = Path(
        "/workspace/data/lerobot--aloha_sim_insertion_human"
    )
    revisions = sorted(path for path in cache_root.iterdir() if path.is_dir())
    if len(revisions) != 1:
        raise RuntimeError(f"expected one pinned revision, found {len(revisions)}")
    dataset = arrow_dataset.dataset(revisions[0] / "data", format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "action"],
        filter=arrow_dataset.field("episode_index") == episode,
    )
    rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
    if not rows:
        raise RuntimeError(f"episode {episode} not found in the pinned cache")
    return [
        torch.as_tensor(row["action"], dtype=torch.float32) for row in rows[:count]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--steps", type=int, default=12)
    arguments = parser.parse_args()

    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    expert = _expert_first_actions(2, arguments.steps)
    teacher = build_teacher()
    environment = GymAlohaEnvironment(
        contract, maximum_episode_steps=arguments.steps
    )
    actions: list[torch.Tensor] = []
    states: list[Any] = []
    try:
        environment.reset(seed=arguments.seed)
        teacher.reset()
        for _step in range(arguments.steps):
            geometry = build_observation(environment, contract)
            decision = teacher.decide(geometry)
            if decision.refusal is not None:
                print(f"teacher refused at step {_step}: {decision.refusal.detail}")
                break
            actions.append(decision.action.clone())
            states.append(geometry.robot_state.clone())
            environment.step(contract.clip(decision.action)[0])
    finally:
        environment.close()

    print(f"{'step':>4} {'action MAE':>10} {'state MAE':>9}  expert-vs-teacher first 6 dims")
    for index in range(min(len(actions), len(expert))):
        action_error = float(
            torch.mean(torch.abs(actions[index] - expert[index]))
        )
        state_error = (
            float(torch.mean(torch.abs(states[index] - expert[index])))
            if index < len(states)
            else float("nan")
        )
        preview = " ".join(
            f"{float(actions[index][dimension]):+.3f}/{float(expert[index][dimension]):+.3f}"
            for dimension in range(6)
        )
        print(f"{index:>4} {action_error:>10.4f} {state_error:>9.4f}  {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
