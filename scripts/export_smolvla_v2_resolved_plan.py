"""Export a v2 YAML/JSON plan as create-only JSON for Basin historical import."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from rosetta_reality.vla.training.plan import load_v2_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    if not plan_path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("The source plan must be inside the Rosetta repository.")
    plan = load_v2_plan(plan_path, REPOSITORY_ROOT)
    if plan.get("schema_version") != 2 or plan.get("role") != "vla":
        raise ValueError("Expected a SmolVLA version-2 plan.")
    payload = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
