"""State-conditioned recovery reference policy tests."""

import pytest
import torch

from rosetta_reality.sim.recovery_oracle import (
    OracleOutOfDistributionError,
    OracleReferenceTrajectory,
    StateConditionedTrajectoryOracle,
)


def _reference() -> OracleReferenceTrajectory:
    states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    return OracleReferenceTrajectory(
        states=states,
        actions=states + 100,
        source_episode=2,
        source_seed=10,
        first_progress_index=3,
        terminal_reward=4.0,
        terminal_success=True,
    )


def test_oracle_uses_current_state_not_a_time_index() -> None:
    oracle = StateConditionedTrajectoryOracle(
        _reference(),
        maximum_lookahead=3,
        maximum_state_distance=0.1,
        maximum_progress_state_distance=0.05,
    )

    decision = oracle.decide(torch.tensor([4.0, 5.0]), observed_reward=0.0)

    assert decision.reference_index == 2
    assert torch.equal(decision.action, torch.tensor([104.0, 105.0]))
    assert decision.state_distance == pytest.approx(0.0)
    assert oracle.cursor == 2


def test_oracle_progress_event_unlocks_post_contact_reference() -> None:
    oracle = StateConditionedTrajectoryOracle(
        _reference(),
        maximum_lookahead=5,
        maximum_state_distance=20.0,
        maximum_progress_state_distance=1.0,
        post_progress_skip=1,
    )

    locked = oracle.decide(torch.tensor([10.0, 11.0]), observed_reward=0.0)
    unlocked = oracle.decide(torch.tensor([10.0, 11.0]), observed_reward=1.0)

    assert locked.reference_index == 3
    assert locked.progress_unlocked is False
    assert unlocked.reference_index == 5
    assert unlocked.progress_unlocked is True
    assert unlocked.candidate_start == 4


def test_oracle_refuses_out_of_distribution_state() -> None:
    oracle = StateConditionedTrajectoryOracle(
        _reference(),
        maximum_lookahead=2,
        maximum_state_distance=0.5,
        maximum_progress_state_distance=0.1,
    )

    with pytest.raises(OracleOutOfDistributionError, match="outside"):
        oracle.decide(torch.tensor([-100.0, -100.0]), observed_reward=0.0)


def test_oracle_advances_only_when_current_state_reaches_next_reference_neighborhood() -> None:
    reference = OracleReferenceTrajectory(
        states=torch.tensor([[0.0], [0.1], [0.2], [0.3]]),
        actions=torch.tensor([[1.0], [1.1], [1.2], [1.3]]),
        source_episode=2,
        source_seed=10,
        first_progress_index=3,
        terminal_reward=4.0,
        terminal_success=True,
    )
    oracle = StateConditionedTrajectoryOracle(
        reference,
        maximum_lookahead=2,
        maximum_state_distance=0.5,
        maximum_progress_state_distance=0.06,
    )

    stalled = oracle.decide(torch.tensor([0.0]), observed_reward=0.0)
    advanced = oracle.decide(torch.tensor([0.06]), observed_reward=0.0)

    assert stalled.reference_index == 0
    assert advanced.reference_index == 1
    assert oracle.cursor == 1


def test_oracle_rejects_failed_or_nonfinite_reference() -> None:
    with pytest.raises(ValueError, match="failed source"):
        OracleReferenceTrajectory(
            states=torch.zeros(2, 2),
            actions=torch.zeros(2, 2),
            source_episode=0,
            source_seed=0,
            first_progress_index=1,
            terminal_reward=0.0,
            terminal_success=False,
        )
    with pytest.raises(ValueError, match="NaN or Inf"):
        OracleReferenceTrajectory(
            states=torch.tensor([[0.0, 0.0], [float("nan"), 0.0]]),
            actions=torch.zeros(2, 2),
            source_episode=0,
            source_seed=0,
            first_progress_index=1,
            terminal_reward=4.0,
            terminal_success=True,
        )
