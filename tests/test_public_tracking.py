import sqlite3
from pathlib import Path

import pytest

from rosetta_reality.tracking import sanitize_metrics, validate_public_payload
from scripts.sync_trackio import _run_snapshots, _scan_database, _snapshot_project


def test_public_payload_accepts_revision_identity_and_metrics() -> None:
    payload = {
        "model_id": "lerobot/smolvla_base",
        "model_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "formal_plan_sha256": "a" * 64,
        "normalization_source_split": "train",
        "workspace_dirty": True,
        "paper_url": "https://arxiv.org/abs/2506.01844",
        "loss": 0.5,
        "test_split_loaded": False,
    }

    validate_public_payload(payload)
    assert sanitize_metrics({"loss": 0.5, "message": "ignored"}, mode="train") == {
        "train/loss": 0.5
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"hf_token": "redacted"},
        {"value": "hf_" + "abcdefghijklmnopqrstuvwxyz"},
        {"value": "ghp_" + "a" * 36},
        {"value": "sk-proj-" + "a" * 24},
        {"value": "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12},
        {"url": "https://person:password@example.invalid/run"},
        {"path": "C:" + "\\Users\\person\\model"},
        {"path": "/" + "home/person/model"},
        {"path": "/" + "root/private/run.log"},
        {"path": "/" + "workspace/checkpoints/model"},
        {"message": "artifact at /" + "tmp/home/run.json"},
        {"url": "https://example.invalid/run?write_token=value"},
    ],
)
def test_public_payload_rejects_sensitive_fields(payload: dict) -> None:
    with pytest.raises(ValueError):
        validate_public_payload(payload)


def test_public_metrics_reject_nonfinite_values_and_sensitive_keys() -> None:
    with pytest.raises(ValueError, match="finite"):
        sanitize_metrics({"loss": float("nan")}, mode="train")
    with pytest.raises(ValueError, match="sensitive"):
        sanitize_metrics({"api_token": 1.0}, mode="train")


def test_trackio_scan_and_sync_use_one_immutable_database_snapshot(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    snapshot_root = tmp_path / "snapshot"
    source_root.mkdir()
    snapshot_root.mkdir()
    source = source_root / "approved.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE metrics (value TEXT)")
        connection.execute("INSERT INTO metrics VALUES (?)", ('{"loss": 0.5}',))

    snapshot = _snapshot_project(source_root, snapshot_root, "approved")
    scan = _scan_database(snapshot)
    with sqlite3.connect(source) as connection:
        connection.execute("INSERT INTO metrics VALUES (?)", ("hf_secret",))
    with sqlite3.connect(snapshot) as connection:
        rows = connection.execute("SELECT value FROM metrics").fetchall()

    assert scan == {"tables": 1, "rows": 1, "text_values": 1}
    assert rows == [('{"loss": 0.5}',)]


def test_trackio_snapshot_binds_formal_run_and_final_step(tmp_path: Path) -> None:
    database = tmp_path / "project.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE configs (run_id TEXT, run_name TEXT, config TEXT)"
        )
        connection.execute(
            "CREATE TABLE metrics (run_id TEXT, run_name TEXT, step INTEGER)"
        )
        connection.execute(
            "INSERT INTO configs VALUES (?, ?, ?)",
            (
                "run-1",
                "formal-001",
                '{"experiment_id":"experiment-001","phase":"formal",'
                '"formal_plan_sha256":"' + "a" * 64 + '"}',
            ),
        )
        connection.executemany(
            "INSERT INTO metrics VALUES (?, ?, ?)",
            [("run-1", "formal-001", 10), ("run-1", "formal-001", 20)],
        )

    assert _run_snapshots(database) == [
        {
            "run_id": "run-1",
            "run_name": "formal-001",
            "experiment_id": "experiment-001",
            "phase": "formal",
            "formal_plan_sha256": "a" * 64,
            "maximum_logged_step": 20,
        }
    ]
