"""Conservatively audit a prepared LeRobot v3 cache without rewriting source data.

The first cleaning pass is deliberately conservative. It records clearly
corrupt rows and refuses to rewrite a cache when a row-level problem is found,
because changing Parquet rows without rewriting the corresponding video frames
can silently break observation/action alignment. A cache with no findings is
marked as validated-clean and receives an immutable JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from rosetta_reality.data.config import (
    DatasetConfig,
    load_dataset_config,
    resolve_dataset_cache_root,
)
from rosetta_reality.data.manifest import (
    DatasetManifest,
    find_dataset_manifests,
    load_dataset_manifest,
    validate_cache_checksums,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "data" / "aloha_sim_insertion.yaml"


def _scalar(value: Any, field: str) -> int | float:
    """Extract a scalar from Arrow's scalar or one-element-list representation."""

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Field {field!r} must be scalar, received {value!r}.")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Field {field!r} must be numeric, received {value!r}.")
    return value


def _vector(value: Any, field: str) -> torch.Tensor:
    """Convert a vector field while retaining a useful validation error."""

    if value is None:
        raise ValueError(f"Field {field!r} is null.")
    try:
        vector = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Field {field!r} is not a numeric vector.") from error
    if vector.ndim != 1:
        raise ValueError(f"Field {field!r} must be rank one, received {tuple(vector.shape)}.")
    return vector


def _relative(path: Path) -> str:
    """Render repository-local paths without machine-specific absolute paths."""

    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _matching_manifest(config: DatasetConfig, cache_root: Path) -> tuple[Path, DatasetManifest]:
    """Find the revision-pinned cache matching the current field selection."""

    expected_fields = asdict(config.fields)
    candidates: list[tuple[Path, DatasetManifest]] = []
    for path in find_dataset_manifests(cache_root, config.repo_id):
        manifest = load_dataset_manifest(path)
        if (
            manifest.episodes == config.episodes
            and manifest.cameras == config.cameras
            and manifest.fields == expected_fields
        ):
            candidates.append((path, manifest))
    if not candidates:
        raise FileNotFoundError(
            f"No prepared manifest matches {config.repo_id!r}, episodes "
            f"{list(config.episodes)!r}, cameras, and field mapping under {_relative(cache_root)}."
        )
    return candidates[-1]


def _load_task_map(root: Path) -> dict[int, str]:
    """Load LeRobot task labels when the optional task table is present."""

    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return {}
    import pyarrow.parquet as parquet

    rows = parquet.read_table(tasks_path).to_pylist()
    task_map: dict[int, str] = {}
    for row in rows:
        raw_index = row.get("task_index")
        raw_label = row.get("__index_level_0__", row.get("task"))
        if raw_index is None or raw_label is None:
            continue
        task_map[int(raw_index)] = str(raw_label)
    return task_map


def _scan_rows(config: DatasetConfig, root: Path) -> dict[str, Any]:
    """Scan selected rows for schema, numeric, key, and temporal anomalies."""

    import pyarrow.parquet as parquet

    required = (
        config.fields.state,
        config.fields.action,
        config.fields.timestamp,
        config.fields.episode_index,
        config.fields.frame_index,
        "index",
        "task_index",
    )
    optional = ("next.done",)
    selected = set(config.episodes)
    task_map = _load_task_map(root)
    rows_by_episode: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    total_rows = 0

    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No Parquet files found under {_relative(root / 'data')}.")

    for path in data_files:
        parquet_file = parquet.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        missing = sorted(set(required) - available)
        if missing:
            raise ValueError(f"{_relative(path)} is missing required columns: {missing}.")
        columns = list(
            dict.fromkeys((*required, *(name for name in optional if name in available)))
        )
        table = parquet.read_table(path, columns=columns)
        records = {name: table[name].to_pylist() for name in table.column_names}
        for row_index in range(table.num_rows):
            total_rows += 1
            raw_episode = records[config.fields.episode_index][row_index]
            try:
                episode = int(_scalar(raw_episode, config.fields.episode_index))
            except ValueError as error:
                issues.append(
                    {
                        "kind": "invalid_episode_index",
                        "file": _relative(path),
                        "row": row_index,
                        "detail": str(error),
                    }
                )
                continue
            if episode not in selected:
                continue

            raw_frame = records[config.fields.frame_index][row_index]
            raw_global_index = records["index"][row_index]
            raw_task_index = records["task_index"][row_index]
            row: dict[str, Any] = {
                "episode": episode,
                "file": _relative(path),
                "row": row_index,
                "done": records.get("next.done", [None] * table.num_rows)[row_index],
            }
            try:
                frame = int(_scalar(raw_frame, config.fields.frame_index))
                global_index = int(_scalar(raw_global_index, "index"))
                task_index = int(_scalar(raw_task_index, "task_index"))
                timestamp = float(
                    _scalar(records[config.fields.timestamp][row_index], config.fields.timestamp)
                )
                state = _vector(records[config.fields.state][row_index], config.fields.state)
                action = _vector(records[config.fields.action][row_index], config.fields.action)
            except ValueError as error:
                issues.append({"kind": "invalid_row", **row, "detail": str(error)})
                continue

            row.update(
                frame=frame,
                global_index=global_index,
                task_index=task_index,
                timestamp=timestamp,
            )
            expected_state_dim = config.expected_state_dim
            expected_action_dim = config.expected_action_dim
            if expected_state_dim is not None and state.numel() != expected_state_dim:
                issues.append(
                    {
                        "kind": "state_dimension",
                        **row,
                        "detail": f"expected {expected_state_dim}, received {state.numel()}",
                    }
                )
            if expected_action_dim is not None and action.numel() != expected_action_dim:
                issues.append(
                    {
                        "kind": "action_dimension",
                        **row,
                        "detail": f"expected {expected_action_dim}, received {action.numel()}",
                    }
                )
            if not bool(torch.isfinite(state).all()):
                issues.append({"kind": "state_nonfinite", **row})
            if not bool(torch.isfinite(action).all()):
                issues.append({"kind": "action_nonfinite", **row})
            if not math.isfinite(timestamp):
                issues.append({"kind": "timestamp_nonfinite", **row})
            if frame < 0 or global_index < 0:
                issues.append({"kind": "negative_index", **row})
            if task_map and task_index not in task_map:
                issues.append({"kind": "unknown_task_index", **row})
            rows_by_episode[episode].append(row)

    frame_key_counts = Counter(
        (row["episode"], row["frame"]) for rows in rows_by_episode.values() for row in rows
    )
    global_index_counts = Counter(
        row["global_index"] for rows in rows_by_episode.values() for row in rows
    )
    for key, count in frame_key_counts.items():
        if count > 1:
            issues.append({"kind": "duplicate_episode_frame", "key": list(key), "count": count})
    for key, count in global_index_counts.items():
        if count > 1:
            issues.append({"kind": "duplicate_global_index", "index": key, "count": count})

    info_path = root / "meta" / "info.json"
    fps: float | None = None
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("fps") is not None:
            fps = float(info["fps"])

    episode_summary: dict[str, Any] = {}
    for episode in config.episodes:
        rows = sorted(rows_by_episode.get(episode, []), key=lambda row: row["frame"])
        frames = [row["frame"] for row in rows]
        timestamps = [row["timestamp"] for row in rows]
        frame_gaps = [
            (previous, current)
            for previous, current in zip(frames, frames[1:])
            if current != previous + 1
        ]
        timestamp_steps = [
            current - previous for previous, current in zip(timestamps, timestamps[1:])
        ]
        nonincreasing = sum(step <= 0 for step in timestamp_steps)
        expected_step = 1.0 / fps if fps and fps > 0 else None
        bad_steps = (
            sum(abs(step - expected_step) > 1e-3 for step in timestamp_steps)
            if expected_step is not None
            else 0
        )
        if frame_gaps:
            issues.append(
                {
                    "kind": "frame_gaps",
                    "episode": episode,
                    "examples": [list(pair) for pair in frame_gaps[:5]],
                }
            )
        if nonincreasing:
            issues.append(
                {"kind": "timestamp_not_increasing", "episode": episode, "count": nonincreasing}
            )
        if bad_steps:
            issues.append(
                {
                    "kind": "timestamp_step_mismatch",
                    "episode": episode,
                    "count": bad_steps,
                    "expected_step": expected_step,
                }
            )
        done_frames = [row["frame"] for row in rows if row["done"] is True]
        if rows and rows[-1]["done"] is not None and done_frames != [rows[-1]["frame"]]:
            issues.append(
                {
                    "kind": "terminal_marker_mismatch",
                    "episode": episode,
                    "done_frames": done_frames,
                    "last_frame": rows[-1]["frame"],
                }
            )
        if (
            config.expected_frames is not None
            and len(config.episodes) == 1
            and len(rows) != config.expected_frames
        ):
            issues.append(
                {
                    "kind": "unexpected_frame_count",
                    "episode": episode,
                    "expected": config.expected_frames,
                    "received": len(rows),
                }
            )
        episode_summary[str(episode)] = {
            "rows": len(rows),
            "frame_range": [frames[0], frames[-1]] if frames else None,
            "frame_gaps": len(frame_gaps),
            "timestamp_range": [timestamps[0], timestamps[-1]] if timestamps else None,
            "timestamp_nonincreasing": nonincreasing,
            "timestamp_step_mismatch": bad_steps,
            "terminal_frames": done_frames,
        }

    return {
        "rows_scanned": total_rows,
        "selected_rows": sum(len(rows) for rows in rows_by_episode.values()),
        "task_labels": task_map,
        "episodes": episode_summary,
        "issues": issues,
        "fps": fps,
    }


def _sample_images(config: DatasetConfig, root: Path, manifest: DatasetManifest) -> dict[str, Any]:
    """Decode a small deterministic image sample through the real adapter."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from rosetta_reality.data.adapters import LeRobotV3Adapter

    adapter = LeRobotV3Adapter(
        repo_id=config.repo_id,
        revision=manifest.resolved_revision,
        root=root,
        episodes=config.episodes,
        cameras=config.cameras,
        fields=config.fields,
        embodiment=config.embodiment,
        license_name=config.license,
    )
    if not len(adapter):
        return {"frames": 0, "samples": []}
    indices = sorted({0, len(adapter) // 2, len(adapter) - 1})
    samples: list[dict[str, Any]] = []
    for index in indices:
        frame = adapter[index]
        camera_summary: dict[str, Any] = {}
        for camera_name, image in frame.images.items():
            camera_summary[camera_name] = {
                "shape": list(image.shape),
                "finite": bool(torch.isfinite(image).all()),
                "min": float(image.min()),
                "max": float(image.max()),
            }
        samples.append(
            {
                "dataset_index": index,
                "episode": frame.episode_id,
                "frame": frame.frame_index,
                "timestamp": frame.timestamp,
                "instruction": frame.instruction,
                "cameras": camera_summary,
            }
        )
    return {"frames": len(adapter), "samples": samples}


def _save_report(path: Path, report: dict[str, Any]) -> None:
    """Write a report once and refuse to replace a different prior result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != report:
            raise FileExistsError(f"Refusing to overwrite a different cleaning report at {path}.")
        return
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def clean(
    config: DatasetConfig,
    config_path: Path,
    report_path: Path | None = None,
) -> int:
    """Audit the selected cache and write an idempotent quality report."""

    cache_root = resolve_dataset_cache_root(config, REPOSITORY_ROOT)
    manifest_path, manifest = _matching_manifest(config, cache_root)
    root = manifest_path.parent
    checksum_files = validate_cache_checksums(root)
    row_report = _scan_rows(config, root)
    image_report = _sample_images(config, root, manifest)
    issues = [*row_report["issues"]]
    if row_report["selected_rows"] == 0:
        issues.append({"kind": "empty_selection", "episodes": list(config.episodes)})
    for sample in image_report["samples"]:
        for camera_name, camera in sample["cameras"].items():
            if not camera["finite"] or camera["min"] < 0 or camera["max"] > 1:
                issues.append(
                    {
                        "kind": "image_value_range",
                        "dataset_index": sample["dataset_index"],
                        "camera": camera_name,
                        "min": camera["min"],
                        "max": camera["max"],
                    }
                )

    status = "validated_clean" if not issues else "manual_review_required"
    report = {
        "schema_version": 1,
        "operation": "conservative_dataset_clean",
        "status": status,
        "source_format": manifest.source_format,
        "repo_id": manifest.repo_id,
        "requested_revision": manifest.requested_revision,
        "resolved_revision": manifest.resolved_revision,
        "config": _relative(config_path),
        "cache": _relative(root),
        "episodes": list(config.episodes),
        "cameras": config.cameras,
        "fields": asdict(config.fields),
        "checksums_verified": checksum_files,
        "rows_scanned": row_report["rows_scanned"],
        "selected_rows": row_report["selected_rows"],
        "rows_retained": row_report["selected_rows"] if not issues else 0,
        "rows_removed": 0,
        "fps": row_report["fps"],
        "episode_summary": row_report["episodes"],
        "task_labels": {
            str(task_index): label for task_index, label in row_report["task_labels"].items()
        },
        "image_sample": image_report,
        "issues": issues,
        "actions": [
            "Validated the revision-scoped cache without rewriting Parquet or video files.",
            "No rows were removed; row-level removal requires synchronized video rewriting.",
        ],
    }
    destination = report_path or (root / "cleaning_report.json")
    _save_report(destination, report)
    print(
        json.dumps(
            {
                "report": _relative(destination),
                "status": status,
                "issues": len(issues),
                "rows_removed": 0,
            },
            sort_keys=True,
        )
    )
    if issues:
        raise ValueError(
            f"Dataset cleaning found {len(issues)} issue(s); no source files were changed."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    config = load_dataset_config(args.config.resolve())
    return clean(
        config,
        args.config.resolve(),
        args.report.resolve() if args.report else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
