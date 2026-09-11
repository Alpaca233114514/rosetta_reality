"""Wait for closed checkpoint results and stream only their verified manifest files."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


def main():
    job = Path("runs/hestia-cuda-checkpoint-curve-20260911-002")
    deadline = time.time() + 600
    manifest_path = job / "handoff-manifest.json"
    while not manifest_path.exists():
        registration = job / "registration.json"
        if registration.exists():
            deadline = min(deadline, json.loads(registration.read_text())["shutdown_deadline_unix"])
        if time.time() >= deadline:
            raise TimeoutError("Result handoff was not ready before shutdown deadline")
        time.sleep(min(2, deadline - time.time()))
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if not 0 < manifest["total_bytes"] <= 64 * 1024**2:
        raise ValueError("Result transfer bound differs")
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        records = {
            "handoff-manifest.json": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            **manifest["files"],
        }
        for name, item in records.items():
            relative = PurePosixPath(name)
            source = job / name
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or source.is_symlink()
                or not source.resolve(strict=True).is_relative_to(job.resolve(strict=True))
            ):
                raise ValueError("Unsafe result path")
            data = source.read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Closed result changed before transfer")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    main()
