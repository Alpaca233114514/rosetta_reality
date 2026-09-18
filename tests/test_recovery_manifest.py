"""Recovery dataset identity and isolation tests."""

import json

import pytest

from rosetta_reality.data.recovery_manifest import (
    REQUIRED_FIELDS,
    RecoveryDatasetManifest,
    load_recovery_manifest,
    save_recovery_manifest,
)


def _manifest(**changes: object) -> RecoveryDatasetManifest:
    payload: dict[str, object] = {
        "dataset_id": "aloha-recovery-001",
        "source_repo_id": "lerobot/aloha_sim_insertion_human",
        "source_revision": "a" * 40,
        "source_manifest_sha256": "b" * 64,
        "action_contract_sha256": "c" * 64,
        "oracle_implementation_sha256": "d" * 64,
        "oracle_evaluation_report_sha256": "e" * 64,
        "oracle_protocol": "state_and_reward_conditioned_monotonic_retrieval_v1",
        "authorized_train_episodes": (2, 4, 9),
        "source_episodes": (2,),
        "validation_episodes": (7,),
        "hidden_test_episodes": (1,),
        "collection_simulator_seeds": (3000, 3001),
        "oracle_evaluation_seeds": (2000, 2001),
        "policy_gate4_seeds": (1000, 1001),
        "sample_count": 20,
        "state_dimension": 14,
        "action_dimension": 14,
        "records_sha256": "f" * 64,
        "fields": {name: name for name in REQUIRED_FIELDS},
        "perturbation_contract": {
            "source": "policy_visited_state",
            "maximum_state_distance": 0.05,
        },
    }
    payload.update(changes)
    return RecoveryDatasetManifest(**payload)


def test_recovery_manifest_round_trip_is_create_only(tmp_path) -> None:
    manifest = _manifest()

    path = save_recovery_manifest(tmp_path / "recovery", manifest)

    assert load_recovery_manifest(path) == manifest
    assert save_recovery_manifest(path.parent, manifest) == path


def test_recovery_manifest_rejects_time_indexed_or_hidden_labels() -> None:
    with pytest.raises(ValueError, match="state-conditioned oracle"):
        _manifest(time_indexed_reference=True)
    with pytest.raises(ValueError, match="sealed hidden test"):
        _manifest(hidden_test_loaded=True)


def test_recovery_manifest_rejects_split_or_seed_leakage() -> None:
    with pytest.raises(ValueError, match="subset of train"):
        _manifest(source_episodes=(7,))
    with pytest.raises(ValueError, match="must be disjoint"):
        _manifest(collection_simulator_seeds=(1000, 3000))


def test_recovery_manifest_refuses_overwrite(tmp_path) -> None:
    root = tmp_path / "recovery"
    path = save_recovery_manifest(root, _manifest())
    payload = json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(FileExistsError, match="different recovery manifest"):
        save_recovery_manifest(root, _manifest(sample_count=21))

    assert json.loads(path.read_text(encoding="utf-8")) == payload
