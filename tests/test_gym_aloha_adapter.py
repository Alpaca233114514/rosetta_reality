from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeEnvironment:
    def __init__(self) -> None:
        self.state = np.zeros(14, dtype=np.float32)
        self.last_action: np.ndarray | None = None
        self.closed = False

    def reset(self, *, seed: int | None = None):
        self.state.fill(0)
        return {
            "agent_pos": self.state.copy(),
            "pixels": {"top": np.zeros((4, 5, 3), dtype=np.uint8)},
        }, {"seed": seed}

    def step(self, action: np.ndarray):
        self.last_action = action.copy()
        self.state = action.copy()
        return {"agent_pos": self.state.copy(), "pixels": {}}, 1.0, False, False, {}

    def close(self) -> None:
        self.closed = True


def test_gym_adapter_converts_observation_and_clips_action() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    fake = FakeEnvironment()
    environment = GymAlohaEnvironment(contract, environment=fake)

    observation = environment.reset(seed=7)
    assert observation["robot_state"].shape == (14,)
    assert observation["images"]["top"].shape == (3, 4, 5)
    assert observation["images"]["top"].dtype == torch.float32

    action = torch.zeros(14)
    action[0] = 99
    next_observation, reward, done, info = environment.step(action)

    assert fake.last_action is not None
    assert fake.last_action[0] == contract.dimensions[0].maximum
    assert next_observation["robot_state"][0] == contract.dimensions[0].maximum
    assert reward == 1.0
    assert not done
    assert info["clipped_fields"] == ["left_waist"]
    assert info["terminated"] is False
    assert info["truncated"] is False
    environment.close()
    assert fake.closed


def test_collision_metric_ignores_only_same_arm_internal_gripper_contacts(
    monkeypatch,
) -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    environment = GymAlohaEnvironment(contract, environment=FakeEnvironment())
    pairs = (
        (
            "vx300s_left/10_left_gripper_finger",
            "vx300s_left/10_right_gripper_finger",
        ),
        (
            "vx300s_right/10_left_gripper_finger",
            "vx300s_right/10_right_gripper_finger",
        ),
        (
            "vx300s_left/10_left_gripper_finger",
            "vx300s_right/10_left_gripper_finger",
        ),
        ("table", "vx300s_left/7_upper_arm"),
        ("peg", "vx300s_left/10_left_gripper_finger"),
        ("socket", "vx300s_right/7_upper_arm"),
    )
    monkeypatch.setattr(environment, "contact_pairs", lambda: pairs)

    assert environment.unexpected_collision_count() == 3
