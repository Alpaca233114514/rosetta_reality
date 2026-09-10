"""Protect diagnostic interpretation, total budget and Gate reset semantics."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_gate4_seating_feasibility as probe  # noqa: E402


@pytest.mark.parametrize("count", range(6))
def test_seed_specific_interpretation(count):
    rows = [{"seed": seed, "seatable": index < count, "error": None}
            for index, seed in enumerate(probe.SEEDS)]
    verdict = probe.classify(rows)
    if count == 0:
        assert "not_impossibility_proof" in verdict
    elif count == 5:
        assert verdict == "all_seatable_learning_interaction_gap"
    else:
        assert verdict == "mixed_seed_specific_learning_interaction_gap"


def test_partial_or_broken_probe_is_not_zero_of_five():
    with pytest.raises(ValueError):
        probe.classify([])
    rows = [{"seed": seed, "seatable": False, "error": "IK failure"}
            for seed in probe.SEEDS]
    assert probe.classify(rows) == "inconclusive_instrument_failure"


@pytest.mark.parametrize("reward,done,info,expected", [
    (4, True, {"is_success": True}, True),
    (3, True, {"is_success": True}, False),
    (4, False, {"is_success": True}, False),
    (0, True, {"truncated": True}, False),
    (4, True, {}, False),
])
def test_success_is_not_timeout_or_intermediate_reward(reward, done, info, expected):
    assert probe.task_success(reward, done, info) is expected


def test_evidence_never_overwritten(tmp_path):
    path = tmp_path / "result.json"
    probe.write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        probe.write_new(path, {"original": False})
    assert '"original": true' in path.read_text()


def test_total_budget_and_terminal_guard_run_before_simulation():
    env = object.__new__(probe.AuditedEnvironment)
    env.rows = [{}] * 500
    with pytest.raises(RuntimeError, match="500-step"):
        env.step(None)
    env.rows = []
    env.last_info = {"truncated": True}
    with pytest.raises(RuntimeError, match="after termination"):
        env.step(None)


def test_reset_is_inherited_unmodified_from_gate_adapter():
    assert probe.AuditedEnvironment.reset is probe.GymAlohaEnvironment.reset


def test_live_reset_bitwise_parity():
    """No dataset, policy, teacher tuning or simulation steps are involved."""
    import torch

    contract = probe.load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    for seed in probe.SEEDS:
        reference = probe.GymAlohaEnvironment(contract, maximum_episode_steps=500)
        diagnostic = probe.AuditedEnvironment(contract)
        try:
            first = reference.reset(seed=seed)
            second = diagnostic.reset(seed=seed)
            assert torch.equal(first["robot_state"], second["robot_state"])
            assert first["images"].keys() == second["images"].keys()
            for camera in first["images"]:
                assert torch.equal(first["images"][camera], second["images"][camera])
            assert reference.diagnostic_snapshot() == diagnostic.diagnostic_snapshot()
        finally:
            reference.close()
            diagnostic.close()
