"""Read-only ALOHA qualification inventory. No models, downloads, or training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from rosetta_reality.diagnostics.cohort_qualification import file_sha, match, safe_path


def allowed_group(stats, allowed: set[int]) -> bool:
    """Do not read any raw row group whose episode membership is uncertain."""
    if stats is None or not stats.has_min_max or stats.null_count != 0:
        return False
    low, high = stats.min, stats.max
    if type(low) is not int or type(high) is not int or high < low or high - low > 1000:
        return False
    return all(episode in allowed for episode in range(low, high + 1))


def inventory(plan: dict, data_root: Path, *, read_rows: bool = False) -> dict:
    import pyarrow.parquet as pq  # Lazy: import of this script never loads ML/data.

    started = time.monotonic()
    permitted = set(plan["train_episodes"])
    if permitted & set(plan["hidden_episodes"] + plan["dev_episodes"]):
        raise ValueError("Train whitelist overlaps protected episodes")
    if set(plan["hidden_episodes"]) != {31, 6, 1, 24, 5}:
        raise ValueError("Original hidden exclusion must be preserved")
    files, episode_counts = [], {}
    for source in plan["sources"]:
        source_root = safe_path(data_root, source["root"])
        receipt_path = safe_path(source_root, "rosetta_download_manifest.json")
        if file_sha(receipt_path) != source["manifest_sha256"]:
            raise ValueError("Source receipt identity changed")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("dataset", receipt.get("repo")) != source["dataset"] or receipt.get("revision") != source["revision"]:
            raise ValueError("Receipt dataset or revision binding mismatch")
        for item in receipt["files"]:
            relative = item["path"]
            if not relative.startswith("data/") or not relative.endswith(".parquet"):
                continue
            if time.monotonic() - started > plan["wall_seconds"]:
                raise TimeoutError("Qualification inventory deadline reached")
            path = safe_path(source_root, relative)
            if file_sha(path) != item["sha256"]:
                raise ValueError("Parquet identity changed")
            table = pq.ParquetFile(path)
            names = table.schema_arrow.names
            if not {"episode_index", "frame_index", "timestamp", "observation.state", "action"} <= set(names):
                raise ValueError("Missing required ALOHA observation/action/time columns")
            episode_column = [i for i in range(table.metadata.num_columns)
                              if table.schema.column(i).path == "episode_index"]
            if len(episode_column) != 1:
                raise ValueError("Episode column is not unambiguous")
            groups = []
            for index in range(table.num_row_groups):
                metadata = table.metadata.row_group(index)
                stats = metadata.column(episode_column[0]).statistics
                admitted = allowed_group(stats, permitted)
                detail = {"row_group": index, "rows": metadata.num_rows,
                          "raw_rows_read": False, "admitted_by_episode_statistics": admitted,
                          "reason": None if admitted else "protected_or_unknown_episode_membership"}
                if read_rows and admitted:
                    for batch in table.iter_batches(
                        batch_size=128, row_groups=[index],
                        columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
                    ):
                        if time.monotonic() - started > plan["wall_seconds"]:
                            raise TimeoutError("Row audit deadline reached")
                        for row in batch.to_pylist():
                            if row["episode_index"] not in permitted:
                                raise ValueError("Episode statistics disagree with actual rows")
                            for name in ("observation.state", "action"):
                                vector = row[name]
                                if len(vector) != plan["expected_dimension"] or not all(
                                    type(v) in (int, float) and math.isfinite(v) for v in vector
                                ):
                                    raise ValueError("Nonfinite or incorrectly shaped state/action")
                            if not math.isfinite(row["timestamp"]):
                                raise ValueError("Nonfinite timestamp")
                            key = f"{source['dataset']}@{source['revision']}:{row['episode_index']}"
                            previous = episode_counts.get(key)
                            if previous is not None and (
                                row["frame_index"] != previous["last_frame"] + 1
                                or abs(row["timestamp"] - previous["last_timestamp"] - 1 / plan["expected_fps"]) > 1e-4
                            ):
                                raise ValueError("Frame/time continuity mismatch")
                            episode_counts[key] = {"frames_observed": (previous or {}).get("frames_observed", 0) + 1,
                                                   "last_frame": row["frame_index"], "last_timestamp": row["timestamp"]}
                    detail["raw_rows_read"] = True
                groups.append(detail)
            files.append({"dataset": source["dataset"], "revision": source["revision"],
                          "path": relative, "sha256": item["sha256"], "columns": names,
                          "row_groups": groups, "total_rows_from_metadata": table.metadata.num_rows})
    return {"schema": "rosetta.aloha_qualification_inventory.v1", "files": files,
            "episode_observations": episode_counts, "hidden_loaded": False,
            "row_group_exclusion_before_read": True, "rows_mode": read_rows,
            "video_alignment_verified": False, "semantic_contract_verified": False,
            "success_recovery_labels_verified": False, "training_ready": False,
            "elapsed_seconds": time.monotonic() - started,
            "limits": ["Shared protected/unknown row groups are skipped completely.",
                       "Finite rows and matching dimensions do not prove physical compatibility.",
                       "No video decoder or annotation generator is run.",
                       "Observed rows do not establish complete episodes or valid target slots."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    data = sub.add_parser("inventory")
    data.add_argument("--plan", type=Path, required=True)
    data.add_argument("--data-root", type=Path, required=True)
    data.add_argument("--read-train-rows", action="store_true")
    cohorts = sub.add_parser("cohort")
    cohorts.add_argument("--manifest", type=Path, required=True)
    cohorts.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.mode == "inventory":
            plan = json.loads(args.plan.read_text())
            if not 0 < plan["wall_seconds"] <= 180 or len(plan["sources"]) > 2:
                raise ValueError("Inventory exceeds registered wall/source bound")
            result = inventory(plan, args.data_root, read_rows=args.read_train_rows)
            result["plan_sha256"] = file_sha(args.plan)
        else:
            payload = json.loads(args.manifest.read_text())
            if not payload.get("protected_groups"):
                raise ValueError("D/dev/hidden/Gate/tuning group inventory is required")
            result = match(payload["candidates"], payload["pairs"], payload["protected_groups"],
                           args.evidence_root, payload["action_contract_sha256"])
            result["input_sha256"] = file_sha(args.manifest)
        with (args.output / "report.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception as exc:
        with (args.output / "failure.json").open("x") as stream:
            json.dump({"error_type": type(exc).__name__, "scope": "qualification_incomplete",
                       "training_ready": False}, stream, indent=2)
        raise


if __name__ == "__main__":
    main()
