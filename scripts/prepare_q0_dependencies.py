"""Install three reviewed binary wheels into a new run-only overlay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import urllib.request
import zipfile

WHEELS = (
    ("mink", "1.2.0", "https://files.pythonhosted.org/packages/f3/82/e4b336cabdd48f2d73ef0f5f547b4907b50bb8a9aec6403845c03e58c6bb/mink-1.2.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl", "62bf5ec5b990727e791d26a578658a72a80d20e21317b5824c72c2d00c29d938", 73617),
    ("qpsolvers", "4.13.0", "https://files.pythonhosted.org/packages/2f/d0/f234961cb3ffbe381c3d82cd9c40fc7e67200c9f6b62a0f60235e74134b3/qpsolvers-4.13.0-py3-none-any.whl", "7f23759a3d8835fa14b2c9eb4ba971188fde0c1460589f45897b00a75776eef7", 107113),
    ("daqp", "0.8.7", "https://files.pythonhosted.org/packages/3b/2b/ce4c2af256fd79b041e9c177285fc9014b3921a942a6a4cbe3246151b8a4/daqp-0.8.7-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl", "cd08316be8d1be489278085712bc09ca15e3b6d34eb7efc462aa43c0069a4b63", 870393),
)


def prepare(job: Path) -> None:
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Reviewed wheels require Linux Python 3.12")
    if not Path("/root/autodl-tmp").is_dir():
        raise RuntimeError("Registered AutoDL container required")
    target = job / "dependencies"
    if target.exists():
        raise FileExistsError("Dependency overlay must be new")
    wheel_dir = job / "reviewed-wheels"
    wheel_dir.mkdir()
    receipts = []
    for name, version, url, digest, size in WHEELS:
        destination = wheel_dir / url.rsplit("/", 1)[1]
        with urllib.request.urlopen(url, timeout=30) as response:
            if not response.url.startswith("https://files.pythonhosted.org/"):
                raise RuntimeError("Unexpected wheel origin")
            content = response.read(size + 1)
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError("Wheel byte identity mismatch")
        with destination.open("xb") as stream:
            stream.write(content)
        with zipfile.ZipFile(destination) as archive:
            for entry in archive.infolist():
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                    raise RuntimeError("Unsafe wheel path")
                if path.suffix == ".pth" or ".data/scripts/" in entry.filename:
                    raise RuntimeError("Wheel contains startup hooks or scripts")
        receipts.append({"name": name, "version": version, "sha256": digest,
                         "bytes": size, "origin": url, "source": f"https://pypi.org/pypi/{name}/{version}/json"})
    # Binary wheels only: no source builds, dependency resolver, upgrade or install hooks.
    result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                             "--no-deps", "--no-compile", "--disable-pip-version-check",
                             "--target", str(target), *map(str, sorted(wheel_dir.glob("*.whl")))],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError("Isolated reviewed-wheel installation failed")
    found = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions(path=[str(target)])}
    if found != {name: version for name, version, *_ in WHEELS}:
        raise RuntimeError("Unexpected installed overlay distributions")
    with (job / "dependency-receipt.json").open("x") as stream:
        json.dump({"wheels": receipts, "overlay": "dependencies", "base_environment_modified": False,
                   "system_torch_cuda_mujoco_modified": False}, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Record capabilities in the same container without constructing a policy.
    subprocess.run([sys.executable, "scripts/check_env.py"], check=True, timeout=30)
    print("Reviewed run-only dependency overlay installed and verified.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    prepare(parser.parse_args().job.resolve(strict=True))
