"""Create-only verification of the unattended checkpoint result transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


def validate_manifest(value):
    if not isinstance(value, dict) or not isinstance(value.get("files"), dict):
        raise ValueError("Result manifest missing")
    pattern = re.compile(
        r"(?:(?:registration|permit|cpu-check|doctor-check|normalization-check|worker-exited)\.json|step-001280\.log|001280/(?:prefix-row-[0-9]{3}\.npz|prefix-observer\.json|arrays\.npz|result\.json|failure\.json|first-control\.json|first-control-failure\.npz))"
    )
    total = 0
    for name, item in value["files"].items():
        if (
            not pattern.fullmatch(name)
            or type(item.get("bytes")) is not int
            or item["bytes"] < 0
            or not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", ""))
        ):
            raise ValueError("Unexpected result path or identity")
        total += item["bytes"]
    if total != value.get("total_bytes") or not 0 < total <= 64 * 1024**2:
        raise ValueError("Result size bound differs")
    if not {"registration.json", "permit.json", "worker-exited.json"} <= set(value["files"]):
        raise ValueError("Closed worker identity missing")
    return value["files"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.receipt.exists():
        raise FileExistsError("Result or receipt already exists")
    if args.archive.stat().st_size > 65 * 1024**2:
        raise ValueError("Archive exceeds transfer bound")
    with tarfile.open(args.archive, "r:") as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members) or any(not m.isfile() for m in members):
            raise ValueError("Duplicate or non-regular archive member")
        marker = archive.getmember("handoff-manifest.json")
        if marker.size > 1024**2:
            raise ValueError("Manifest too large")
        raw = archive.extractfile(marker).read()
        files = validate_manifest(json.loads(raw))
        if {m.name for m in members} != set(files) | {"handoff-manifest.json"}:
            raise ValueError("Archive file set differs")
        args.output.mkdir(parents=True, exist_ok=False)
        for member in members:
            if member.name == "handoff-manifest.json":
                continue
            data = archive.extractfile(member).read()
            item = files[member.name]
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Result SHA mismatch")
            path = args.output / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        (args.output / "handoff-manifest.json").write_bytes(raw)
    worker = json.loads((args.output / "worker-exited.json").read_text())
    receipt = {
        "all_file_sha256_matched": True,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "file_count": len(files),
        "worker_result": worker,
    }
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(
        json.dumps(
            {
                "transfer_verified": True,
                "worker_error": worker.get("error"),
                "completed_steps": worker.get("completed_steps"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
