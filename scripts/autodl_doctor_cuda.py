"""Extend the immutable AutoDL doctor with the preregistered simulator package."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import autodl_doctor as historical_doctor  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=historical_doctor.DEFAULT_PROFILE)
    parser.add_argument("--config", type=Path, default=historical_doctor.DEFAULT_CONFIG)
    args = parser.parse_args()
    profile_path = args.profile.resolve()
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    expected = str(profile.get("packages", {}).get("gym-aloha", ""))
    installed = version("gym-aloha")
    if expected != "0.1.4" or installed != expected:
        raise RuntimeError("gym-aloha differs from the AutoDL runtime profile.")
    doctor_path = historical_doctor.doctor(profile_path, args.config.resolve())
    supplement = {
        "schema_version": 1,
        "status": "passed",
        "stage": "autodl_environment_doctor_simulator_supplement",
        "profile_sha256": file_sha256(profile_path),
        "doctor_report_sha256": file_sha256(doctor_path),
        "packages": {"gym-aloha": installed},
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "nested_docker_used": False,
        "hidden_test_loaded": False,
    }
    destination = doctor_path.with_name(f"{doctor_path.stem}-simulator.json")
    create_json(destination, supplement)
    print(json.dumps(supplement, indent=2, sort_keys=True))
    print(f"AutoDL simulator doctor supplement: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
