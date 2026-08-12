"""Scan local Trackio data for sensitive content, then refresh a public static Space."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import stable_hash  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.tracking import validate_public_payload  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("VLA experiment config must be a mapping.")
    tracking = value.get("tracking")
    if not isinstance(tracking, dict):
        raise ValueError("VLA experiment tracking config must be a mapping.")
    if tracking.get("space_sdk") != "static" or tracking.get("visibility") != "public":
        raise ValueError("Only the approved public static Trackio Space may be synced.")
    return value


def _trackio_root() -> Path:
    raw = os.environ.get("TRACKIO_DIR")
    if not raw:
        raise ValueError("TRACKIO_DIR must identify the mounted durable Trackio directory.")
    root = Path(raw)
    if not root.is_absolute():
        raise ValueError("TRACKIO_DIR must be absolute.")
    return root


def _scan_database(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Trackio project database is missing: {path.name}.")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    scanned_rows = 0
    scanned_values = 0
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (table_name,) in table_rows:
            if not isinstance(table_name, str) or not table_name.replace("_", "").isalnum():
                raise ValueError("Trackio database contains an unsafe table identifier.")
            cursor = connection.execute(f'SELECT * FROM "{table_name}"')
            columns = [description[0] for description in cursor.description or []]
            for batch in iter(lambda: cursor.fetchmany(512), []):
                for row in batch:
                    scanned_rows += 1
                    for column, value in zip(columns, row, strict=True):
                        if value is None or isinstance(value, (int, float)):
                            continue
                        if isinstance(value, bytes):
                            try:
                                value = value.decode("utf-8")
                            except UnicodeDecodeError as error:
                                raise ValueError(
                                    "Trackio public sync refuses non-text binary database values."
                                ) from error
                        if not isinstance(value, str):
                            raise ValueError("Trackio database contains an unsupported value type.")
                        scanned_values += 1
                        try:
                            decoded = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            decoded = value
                        if column == "key" and isinstance(decoded, str):
                            validate_public_payload(
                                {decoded: 0},
                                context="trackio_database_key",
                            )
                        else:
                            validate_public_payload(decoded, context="trackio_database")
    finally:
        connection.close()
    return {"tables": len(table_rows), "rows": scanned_rows, "text_values": scanned_values}


def _reject_media(root: Path) -> None:
    media_root = root / "media"
    if media_root.is_dir() and any(path.is_file() for path in media_root.rglob("*")):
        raise ValueError("Trackio public sync refuses media files.")


def _snapshot_project(source_root: Path, snapshot_root: Path, project: str) -> Path:
    """Create one transactionally consistent SQLite snapshot for scan and sync."""

    source = source_root / f"{project}.db"
    if not source.is_file():
        raise FileNotFoundError(f"Trackio project database is missing: {source.name}.")
    destination = snapshot_root / source.name
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.force:
        parser.error("--force is required to refresh the exact approved static snapshot.")
    experiment = _load(args.config.resolve())
    tracking = experiment["tracking"]
    root = _trackio_root()
    project = str(tracking["project"])
    _reject_media(root)

    from huggingface_hub import HfApi

    identity = HfApi().whoami()
    if identity.get("name") != str(tracking["space_id"]).split("/", maxsplit=1)[0]:
        raise PermissionError("The active Hub identity does not own the approved Trackio Space.")
    with tempfile.TemporaryDirectory(prefix="rosetta-trackio-sync-") as temporary:
        snapshot_root = Path(temporary)
        snapshot_database = _snapshot_project(root, snapshot_root, project)
        scan = _scan_database(snapshot_database)
        previous_trackio_dir = os.environ.get("TRACKIO_DIR")
        os.environ["TRACKIO_DIR"] = str(snapshot_root)
        try:
            import trackio

            synced_space = trackio.sync(
                project=project,
                space_id=tracking["space_id"],
                sdk="static",
                force=True,
                run_in_background=False,
            )
        finally:
            if previous_trackio_dir is None:
                os.environ.pop("TRACKIO_DIR", None)
            else:
                os.environ["TRACKIO_DIR"] = previous_trackio_dir
    info = HfApi().space_info(repo_id=tracking["space_id"])
    if info.sdk != "static" or info.private:
        raise RuntimeError(
            "Synced Trackio Space visibility or SDK differs from the approved config."
        )
    report = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": experiment["experiment_id"],
        "project": project,
        "space_id": synced_space,
        "space_sdk": info.sdk,
        "visibility": "public",
        "public_payload_scan": scan,
        "contains_sensitive_data": False,
        "media_uploaded": False,
        "test_split_loaded": False,
    }
    digest = stable_hash(report)[:16]
    run_root = Path(os.environ.get("ROSETTA_RUN_ROOT", REPOSITORY_ROOT / "runs"))
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "tracking"
        / f"space-sync-{digest}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Trackio Space: https://huggingface.co/spaces/{synced_space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
