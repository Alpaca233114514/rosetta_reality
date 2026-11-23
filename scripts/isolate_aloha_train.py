"""Explicitly authorized train-only export in a separate bounded process.

Parquet may decode protected rows internally. Only whitelisted rows cross the
export boundary. This is NOT the legacy no-protected-row-decoding contract.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from rosetta_reality.diagnostics.cohort_qualification import file_sha, receipt_revision, safe_path

AUTHORIZATION = "isolated_decode_train_export_only"
COLUMNS = ["episode_index", "frame_index", "timestamp", "observation.state", "action", "next.done"]


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_plan(plan):
    if plan.get("authorization") != AUTHORIZATION:
        raise PermissionError("Explicit isolated preprocessing authorization required")
    splits = []
    for name in ("train_episodes", "dev_episodes", "hidden_episodes"):
        values = plan[name]
        if not values or any(type(x) is not int or x < 0 for x in values) or len(set(values)) != len(values):
            raise ValueError("Invalid episode whitelist")
        splits.append(set(values))
    if splits[0] & (splits[1] | splits[2]) or splits[1] & splits[2]:
        raise ValueError("Protected split overlap")
    if splits[2] != {31, 6, 1, 24, 5} or splits[1] != {22, 13, 7, 33, 45} or splits[0] != set(range(50)) - splits[1] - splits[2]:
        raise ValueError("Original hidden/train split required")
    if not 0 < plan["wall_seconds"] <= 180 or not 0 < plan["max_output_bytes"] <= 32 * 1024**2:
        raise ValueError("Unbounded preprocessing plan")
    if not 0 < len(plan["sources"]) <= 2:
        raise ValueError("Invalid source count")
    if plan["expected_dimension"] != 14 or plan["expected_fps"] != 50 or plan["chunk_length"] != 50:
        raise ValueError("Not the registered ALOHA contract")
    for source in plan["sources"]:
        if type(source["expected_episode_frames"]) is not int or not 50 <= source["expected_episode_frames"] <= 1000:
            raise ValueError("Episode length must be explicitly bounded")
    return splits[0]


def export_worker(plan_path, data_root, output):
    plan = json.loads(plan_path.read_text())
    allowed = validate_plan(plan)  # Reject before opening any source or importing a reader.
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (plan["max_output_bytes"], plan["max_output_bytes"]))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    output.mkdir(parents=True, exist_ok=False)
    exports = []
    for number, source in enumerate(plan["sources"]):
        root = safe_path(data_root, source["root"])
        receipt_file = safe_path(root, "rosetta_download_manifest.json")
        if file_sha(receipt_file) != source["manifest_sha256"]:
            raise ValueError("Receipt drift")
        receipt = json.loads(receipt_file.read_text())
        if receipt.get("dataset", receipt.get("repo")) != source["dataset"] or receipt_revision(receipt) != source["revision"]:
            raise ValueError("Source identity mismatch")
        members = sorted((x for x in receipt["files"] if x["path"].startswith("data/") and x["path"].endswith(".parquet")), key=lambda x: x["path"])
        if not members or len({x["path"] for x in members}) != len(members):
            raise ValueError("Missing or duplicate source members")
        destination = output / f"train-{number:02d}.parquet"
        writer = None
        rows = 0
        try:
            for item in members:
                path = safe_path(root, item["path"])
                if path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
                    raise ValueError("Source member drift")
                dataset = ds.dataset(path, format="parquet")
                if not set(COLUMNS) <= set(dataset.schema.names):
                    raise ValueError("Missing source columns")
                scanner = dataset.scanner(columns=COLUMNS, filter=ds.field("episode_index").isin(sorted(allowed)),
                                          batch_size=128, batch_readahead=1, fragment_readahead=1, use_threads=False)
                for batch in scanner.to_batches():
                    if time.monotonic() - started > plan["wall_seconds"]:
                        raise TimeoutError("Isolation deadline")
                    # Inspect only episode IDs before any export; never stringify raw batches.
                    episodes = batch.column(batch.schema.get_field_index("episode_index")).to_pylist()
                    if any(type(ep) is not int or ep not in allowed for ep in episodes):
                        raise ValueError("Protected or unknown row crossed scanner boundary")
                    if not batch.num_rows:
                        continue
                    if writer is None:
                        writer = pq.ParquetWriter(destination, batch.schema, compression="zstd")
                    writer.write_batch(batch)
                    rows += batch.num_rows
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 1024**3:
                        raise MemoryError("Isolation RSS bound")
                    if sum(p.stat().st_size for p in output.glob("*.parquet")) > plan["max_output_bytes"]:
                        raise OSError("Isolation output bound")
        finally:
            if writer is not None:
                writer.close()
        if rows != len(allowed) * source["expected_episode_frames"]:
            raise ValueError("Unexpected train row count")
        exports.append({"path": destination.name, "sha256": file_sha(destination), "bytes": destination.stat().st_size,
                        "rows": rows, "dataset": source["dataset"], "revision": source["revision"],
                        "source_manifest_sha256": source["manifest_sha256"], "source_files": members,
                        "already_in_D": source["already_in_D"], "expected_episode_frames": source["expected_episode_frames"]})
    if sum(x["bytes"] for x in exports) > plan["max_output_bytes"]:
        raise OSError("Final isolation output bound")
    save(output / "manifest.json", {"schema": "rosetta.train_isolation.v1", "plan_sha256": file_sha(plan_path),
         "authorization": AUTHORIZATION, "exports": exports, "train_episodes": sorted(allowed),
         "protected_rows_may_be_decoded_inside_worker": True, "protected_rows_exported": 0,
         "hidden_not_decoded_claim": False, "video_decoded": False, "training_ready": False,
         "elapsed_seconds": time.monotonic() - started})


def verify_export(plan_path, export_root):
    """Independent process reads only derived train files, never source payloads."""
    plan = json.loads(plan_path.read_text())
    allowed = validate_plan(plan)
    import pyarrow.parquet as pq

    manifest = json.loads((export_root / "manifest.json").read_text())
    if manifest["plan_sha256"] != file_sha(plan_path) or manifest["train_episodes"] != sorted(allowed):
        raise ValueError("Isolation plan binding mismatch")
    if len(manifest["exports"]) != len(plan["sources"]):
        raise ValueError("Export source count mismatch")
    reports = []
    for item, source in zip(manifest["exports"], plan["sources"], strict=True):
        for key in ("dataset", "revision", "already_in_D", "expected_episode_frames"):
            if item[key] != source[key]:
                raise ValueError("Export source binding mismatch")
        if item["source_manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("Export receipt binding mismatch")
        path = safe_path(export_root, item["path"])
        if path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
            raise ValueError("Export file drift")
        episodes = {}
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=COLUMNS, use_threads=False):
            ids = batch.column(batch.schema.get_field_index("episode_index")).to_pylist()
            if any(type(ep) is not int or ep not in allowed for ep in ids):
                raise ValueError("Protected or unknown episode in export")
            for row in batch.to_pylist():
                ep = row["episode_index"]
                previous = episodes.setdefault(ep, {"frames": 0, "done_frames": [], "valid_target_slots": 0})
                frame = row["frame_index"]
                if type(frame) is not int or frame != previous["frames"]:
                    raise ValueError("Missing, duplicate or out-of-order frame")
                if type(row["timestamp"]) not in (int, float) or not math.isfinite(row["timestamp"]) or abs(row["timestamp"] - frame / plan["expected_fps"]) > 1e-4:
                    raise ValueError("Timestamp alignment mismatch")
                for name in ("observation.state", "action"):
                    vector = row[name]
                    if len(vector) != plan["expected_dimension"] or not all(type(v) in (int, float) and math.isfinite(v) for v in vector):
                        raise ValueError("Invalid state/action")
                if type(row["next.done"]) is not bool:
                    raise ValueError("Unknown terminal indicator")
                if row["next.done"]:
                    previous["done_frames"].append(frame)
                previous["frames"] += 1
        if set(episodes) != allowed or any(x["frames"] != source["expected_episode_frames"] for x in episodes.values()):
            raise ValueError("Incomplete train episode coverage")
        if sum(x["frames"] for x in episodes.values()) != item["rows"]:
            raise ValueError("Export row count mismatch")
        for value in episodes.values():
            length = value["frames"]
            value["valid_target_slots"] = sum(min(plan["chunk_length"], length - f) for f in range(length))
        reports.append({"dataset": source["dataset"], "revision": source["revision"], "already_in_D": source["already_in_D"],
                        "episodes": episodes, "frames": sum(x["frames"] for x in episodes.values()),
                        "valid_target_slots": sum(x["valid_target_slots"] for x in episodes.values()),
                        "payload_sha256": item["sha256"]})
    return {"schema": "rosetta.train_isolation.verification.v1", "status": "passed", "sources": reports,
            "manifest_sha256": file_sha(export_root / "manifest.json"), "plan_sha256": file_sha(plan_path),
            "protected_rows_observed_by_verifier": 0, "protected_rows_may_be_decoded_inside_worker": True,
            "finite_state_action": True, "frame_time_alignment_verified": True,
            "video_alignment_verified": False, "success_recovery_labels_verified": False,
            "done_is_not_success": True, "training_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "verify", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.mode == "export":
        export_worker(args.plan, args.data_root, args.output)
    elif args.mode == "verify":
        save(args.output / "verification.json", verify_export(args.plan, args.output))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        payload = args.output / "train-only"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="1")
        try:
            for mode in ("export", "verify"):
                command = [sys.executable, __file__, mode, "--plan", str(args.plan), "--output", str(payload)]
                if mode == "export":
                    command += ["--data-root", str(args.data_root)]
                # Raw decoder errors are deliberately not copied to audit logs.
                result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, timeout=plan["wall_seconds"], check=False)
                if result.returncode:
                    raise RuntimeError("Isolation phase failed: " + mode)
            save(args.output / "result.json", json.loads((payload / "verification.json").read_text()))
        except Exception as exc:
            save(args.output / "failure.json", {"error_type": type(exc).__name__, "training_ready": False})
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Error messages from decoders may include data. Emit type only.
        print(type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
