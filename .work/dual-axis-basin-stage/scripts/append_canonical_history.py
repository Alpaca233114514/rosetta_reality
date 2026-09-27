#!/usr/bin/env python3
"""Append seven verified Basin units without changing existing history bytes."""

import argparse
from pathlib import Path

from basin.io import child, dumps, read_bytes, sha, write_new
from basin.store import Store


NEW_IDS = (
    "canonical-gate-004", "canonical-gate-005", "canonical-original-reports",
    "canonical-saved-data", "canonical-step-2500", "canonical-step-5000",
    "canonical-training-001",
)


def snapshot(store, ids):
    result = {}
    for run_id in ids:
        manifest = store.verify(run_id)
        result[run_id] = {"manifest_sha256": sha(read_bytes(store.path(run_id) / "manifest.json")),
                          "files": manifest["files"]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    source = Store(args.source)
    destination = Store(args.destination)
    preflight = Path(args.preflight).resolve()
    receipt = Path(args.receipt).resolve()
    if source.root == destination.root or source.root.is_relative_to(destination.root) or destination.root.is_relative_to(source.root):
        raise ValueError("Source and destination histories overlap")
    if preflight.exists() or receipt.exists() or preflight == receipt:
        raise FileExistsError("Append evidence path already exists")
    if preflight.is_relative_to(source.root) or preflight.is_relative_to(destination.root) or receipt.is_relative_to(source.root) or receipt.is_relative_to(destination.root):
        raise ValueError("Append receipts must be outside history stores")
    if not source.root.is_dir() or not destination.root.is_dir():
        raise ValueError("Both histories must already exist")
    old_ids = sorted(path.name for path in destination.root.iterdir() if path.is_dir())
    source_ids = sorted(path.name for path in source.root.iterdir() if path.is_dir())
    if len(old_ids) != 8 or source_ids != sorted(NEW_IDS) or set(old_ids) & set(NEW_IDS):
        raise ValueError("Unexpected history IDs or destination collision")
    old = snapshot(destination, old_ids)
    incoming = snapshot(source, NEW_IDS)
    preflight_data = {"schema_version": 1, "status": "preflight_verified",
                      "destination_existing_count": len(old_ids), "source_new_count": len(incoming),
                      "old": old, "incoming": incoming,
                      "source_policy": "read_only", "append_policy": "Store.add_create_only"}
    write_new(preflight, dumps(preflight_data))

    added = []
    try:
        for run_id in NEW_IDS:
            record, events = source.get(run_id)
            artifacts = {name.removeprefix("artifacts/"): read_bytes(child(source.path(run_id), name))
                         for name in incoming[run_id]["files"] if name.startswith("artifacts/")}
            result = destination.add(record, events, artifacts)
            if result["status"] != "imported":
                raise ValueError("Create-only append did not create the new unit")
            added.append(run_id)
            if destination.verify(run_id)["files"] != incoming[run_id]["files"]:
                raise ValueError("Appended file digests differ from source")
        after_ids = sorted(path.name for path in destination.root.iterdir() if path.is_dir())
        after_old = snapshot(destination, old_ids)
        after_new = snapshot(destination, NEW_IDS)
        if after_ids != sorted(old_ids + list(NEW_IDS)) or after_old != old:
            raise ValueError("Destination history changed outside the seven appended units")
        if any(after_new[run_id]["files"] != incoming[run_id]["files"] for run_id in NEW_IDS):
            raise ValueError("Appended history content differs from source")
        if snapshot(source, NEW_IDS) != incoming:
            raise ValueError("Source history changed during append")
        status = "complete"
        error_type = None
    except Exception as exc:
        status = "failed"
        error_type = type(exc).__name__
        after_ids = sorted(path.name for path in destination.root.iterdir() if path.is_dir())
        after_old = snapshot(destination, old_ids)
        after_new = {run_id: snapshot(destination, [run_id])[run_id]
                     for run_id in added if destination.path(run_id).exists()}
    result = {"schema_version": 1, "status": status, "error_type": error_type,
              "preflight_sha256": sha(read_bytes(preflight)),
              "existing_ids": old_ids, "added_ids": added,
              "destination_ids_after": after_ids,
              "old_unchanged": after_old == old,
              "new_files_match_source": all(after_new.get(run_id, {}).get("files") == incoming[run_id]["files"]
                                            for run_id in added),
              "existing_manifest_sha256": {run_id: old[run_id]["manifest_sha256"] for run_id in old_ids},
              "appended_manifest_sha256": {run_id: after_new[run_id]["manifest_sha256"]
                                           for run_id in after_new}}
    write_new(receipt, dumps(result))
    print(dumps({"status": status, "existing": len(old_ids), "added": len(added),
                 "old_unchanged": result["old_unchanged"],
                 "new_files_match_source": result["new_files_match_source"],
                 "receipt_sha256": sha(read_bytes(receipt))}))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
