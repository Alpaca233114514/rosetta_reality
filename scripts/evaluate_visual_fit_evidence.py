"""Score already-collected Hestia B/C evidence; no model execution entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compare", "reload"))
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Fit-strength output already exists")
    from rosetta_reality.vla import visual_fit

    if args.mode == "reload":
        result = visual_fit.compare_reload(args.first, args.second)
        passed = result["passed"]
    else:
        b, bm = visual_fit.read_bundle(args.first)
        c, cm = visual_fit.read_bundle(args.second)
        result = visual_fit.compare_arms(b, bm, c, cm)
        passed = result["offline_metric_criteria_passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": "evidence_scored", "criteria_passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
