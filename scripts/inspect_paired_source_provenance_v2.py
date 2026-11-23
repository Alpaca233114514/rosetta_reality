"""Bounded metadata-only source capability audit inside registered Linux runtime.

Reads verified metadata schema/footer, never decodes source trajectory rows,
images, episode statistics, normalization values or protected observations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
        raise ValueError("invalid metadata path")
    target = root.joinpath(*p.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("metadata symlink escapes source")
    return target


def inspect(root, expected_receipt_sha):
    started = time.monotonic()
    receipt_path = root / "rosetta_download_manifest.json"
    if sha(receipt_path) != expected_receipt_sha:
        raise ValueError("source receipt identity mismatch")
    receipt = json.loads(receipt_path.read_text())
    dataset = receipt.get("dataset", receipt.get("repo"))
    revisions = {receipt["revision"]} if receipt.get("revision") else {
        item.get("revision") for item in receipt.get("source_files", [])
        if item.get("dataset") == dataset}
    if dataset != "lerobot/aloha_sim_insertion_human" or revisions != {"cc571a3c661df81b566dbfde3d5c1e85fcdf7884"}:
        raise ValueError("not registered original D")
    entries = {entry["path"]: entry for entry in receipt["files"]}

    def member(relative):
        entry = entries[relative]
        path = safe(root, relative)
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("metadata identity mismatch")
        if time.monotonic()-started > 45:
            raise TimeoutError("metadata work cap")
        return path

    info = json.loads(member("meta/info.json").read_text())
    import pyarrow.parquet as pq
    metadata = []
    for relative in sorted(entries):
        if relative.startswith("meta/episodes/") and relative.endswith(".parquet"):
            path = member(relative)
            file = pq.ParquetFile(path)
            metadata.append({"path": relative, "sha256": entries[relative]["sha256"],
                             "metadata_row_count": file.metadata.num_rows,
                             "columns": file.schema_arrow.names,
                             "schema": str(file.schema_arrow)})
    source_schema = []
    for relative in sorted(entries):
        if relative.startswith("data/") and relative.endswith(".parquet"):
            path = member(relative)
            file = pq.ParquetFile(path)
            source_schema.append({"path": relative, "sha256": entries[relative]["sha256"],
                                  "footer_row_count": file.metadata.num_rows,
                                  "columns": file.schema_arrow.names,
                                  "schema": str(file.schema_arrow)})
    columns = set(info["features"]) | {key for item in metadata+source_schema for key in item["columns"]}
    families = {
        "simulator_seed": [k for k in columns if "seed" in k.lower()],
        "scene_or_parent_identity": [k for k in columns if any(w in k.lower() for w in ("scene", "parent", "family"))],
        "object_pose": [k for k in columns if any(w in k.lower() for w in ("object", "peg", "socket", "env_state"))],
        "complete_integration_state": [k for k in columns if any(w in k.lower() for w in ("integration", "warmstart", "qvel", "qacc", "actuator", "mocap"))],
    }
    return {"schema": "rosetta.paired_source_provenance_capability.v2", "dataset": dataset,
            "revision": next(iter(revisions)), "receipt_sha256": expected_receipt_sha,
            "features": info["features"], "total_episodes": info["total_episodes"],
            "metadata": metadata, "trajectory_schemas": source_schema,
            "candidate_provenance_fields": {k: sorted(v) for k,v in families.items()},
            "raw_rows_decoded": 0, "images_decoded": 0, "protected_observations_exported": 0,
            "complete_physical_scene_inventory_created": False,
            "elapsed_seconds": time.monotonic()-started,
            "limitations": ["schema presence does not verify field values or original reset algorithm",
                            "robot14_state alone does not reconstruct object poses or native integration",
                            "image-derived geometry would require separate recorded uncertainty and camera identity audit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("create-only metadata audit output")
    result = inspect(args.root, args.receipt_sha256)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "metadata_inspected", "candidate_provenance_fields": result["candidate_provenance_fields"]}))


if __name__ == "__main__":
    main()
