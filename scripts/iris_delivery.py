"""Bounded Iris result archive, safe extraction and exact receipt creation."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

from scripts.iris_runtime import save, sha

MAX_BYTES = 5 * 1024**3
EXCLUDED = {
    "watchdog.log",
    "active-child.json",
    "active-child.tmp",
    "handoff-manifest.json",
    "package.json",
    "transfer-receipt.json",
    "shutdown-request.json",
}


def safe_name(name):
    from pathlib import PurePosixPath

    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name
        or name == "."
    ):
        raise ValueError("Unsafe delivery path")
    return path


def package(job):
    job = Path(job)
    files = {}
    for path in sorted(job.rglob("*")):
        if path.is_symlink():
            raise ValueError("Delivery cannot include symlinks")
        if not path.is_file() or path.relative_to(job).as_posix() in EXCLUDED:
            continue
        name = path.relative_to(job).as_posix()
        safe_name(name)
        files[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    total = sum(entry["bytes"] for entry in files.values())
    if not files or len(files) > 10000 or total > MAX_BYTES:
        raise ValueError("Result transfer exceeds registered bounds")
    manifest = job / "handoff-manifest.json"
    save(manifest, {"schema_version": 1, "run": job.name, "files": files, "total_bytes": total})
    archive = job.parent / (job.name + ".tar")
    with archive.open("xb") as output, tarfile.open(fileobj=output, mode="w|") as tar:
        for name in [*files, "handoff-manifest.json"]:
            tar.add(job / name, arcname=name, recursive=False)
    manifest_sha = sha(manifest)
    save(
        job / "package.json",
        {
            "archive_sha256": sha(archive),
            "manifest_sha256": manifest_sha,
            "archive_bytes": archive.stat().st_size,
            "files": len(files),
            "total_bytes": total,
        },
    )
    return manifest_sha


def extract(archive, destination, expected_archive_sha, expected_manifest_sha):
    archive, destination = Path(archive), Path(destination)
    if (
        destination.exists()
        or archive.stat().st_size > MAX_BYTES + 32 * 1024**2
        or sha(archive) != expected_archive_sha
    ):
        raise ValueError("Delivery archive identity, size or destination differs")
    with tarfile.open(archive, "r:") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or len(names) > 10001:
            raise ValueError("Duplicate or excessive archive members")
        for member in members:
            safe_name(member.name)
            if not member.isfile() or member.size < 0:
                raise ValueError("Delivery only permits regular files")
        if "handoff-manifest.json" not in names:
            raise ValueError("Delivery manifest missing")
        manifest_member = tar.getmember("handoff-manifest.json")
        if manifest_member.size > 8 * 1024**2:
            raise ValueError("Manifest is too large")
        raw = tar.extractfile(manifest_member).read()
        import hashlib

        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha:
            raise ValueError("Delivery manifest SHA differs")
        manifest = json.loads(raw)
        files = manifest["files"]
        if set(names) != set(files) | {"handoff-manifest.json"}:
            raise ValueError("Delivery file set differs")
        total = 0
        for name, entry in files.items():
            safe_name(name)
            if type(entry.get("bytes")) is not int or entry["bytes"] != tar.getmember(name).size:
                raise ValueError("Delivery file size differs")
            total += entry["bytes"]
        if total != manifest["total_bytes"] or total > MAX_BYTES:
            raise ValueError("Delivery total byte count differs")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            expected = (
                expected_manifest_sha
                if member.name == "handoff-manifest.json"
                else files[member.name]["sha256"]
            )
            if sha(path) != expected:
                raise ValueError("Extracted file SHA differs: " + member.name)
    receipt = {
        "run": manifest["run"],
        "manifest_sha256": expected_manifest_sha,
        "archive_sha256": expected_archive_sha,
        "all_file_sha256_matched": True,
        "files": len(files),
        "total_bytes": total,
    }
    save(destination / "transfer-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--package-json", type=Path)
    args = parser.parse_args()
    if args.package_json is not None:
        package_report = json.loads(args.package_json.read_text())
        args.archive_sha256 = package_report["archive_sha256"]
        args.manifest_sha256 = package_report["manifest_sha256"]
        if package_report["archive_bytes"] != args.archive.stat().st_size:
            raise ValueError("Package archive size differs")
    if not args.archive_sha256 or not args.manifest_sha256:
        raise ValueError("Expected archive and manifest hashes required")
    print(
        json.dumps(
            extract(args.archive, args.destination, args.archive_sha256, args.manifest_sha256)
        )
    )


if __name__ == "__main__":
    main()
