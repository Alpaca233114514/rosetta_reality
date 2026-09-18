"""Seed 3 diagnostic CLI. Offline commands never load a policy or simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    validate = modes.add_parser("validate-plan")
    validate.add_argument("--plan", required=True, type=Path)
    validate.add_argument("--schema-only", action="store_true")
    for mode in ("collect", "collect-repro", "replay", "probe"):
        command = modes.add_parser(mode)
        command.add_argument("--plan", required=True, type=Path)
        if mode in ("replay", "probe"):
            command.add_argument("--input", required=True, type=Path)
    compare = modes.add_parser("compare-repro")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--repeat", required=True, type=Path)
    compare.add_argument("--traced", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    audit = modes.add_parser("audit-training")
    audit.add_argument("--plan", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    analyze = modes.add_parser("analyze")
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--probe", type=Path)
    analyze.add_argument("--training", type=Path)
    verify = modes.add_parser("verify")
    verify.add_argument("--directory", required=True, type=Path)
    verify.add_argument("--source", type=Path)
    args = parser.parse_args(argv)
    from rosetta_reality.eval.gate_diagnostic_io import load_json

    if args.mode == "validate-plan":
        from rosetta_reality.eval.gate_diagnostic_protocol import validate_plan

        result = validate_plan(load_json(args.plan), ROOT, check_files=not args.schema_only)
    elif args.mode in ("collect", "collect-repro", "replay", "probe"):
        from rosetta_reality.eval.gate_diagnostic_runtime import run_registered

        result = run_registered(
            load_json(args.plan), ROOT, args.mode, source=getattr(args, "input", None)
        )
    elif args.mode == "compare-repro":
        from rosetta_reality.eval.reproducibility import compare_campaign

        result = compare_campaign(args.baseline, args.repeat, args.traced, args.output)
    elif args.mode == "audit-training":
        from rosetta_reality.eval.gate_diagnostic_protocol import validate_plan
        from rosetta_reality.eval.gate_diagnostic_training import audit_training

        plan = load_json(args.plan)
        validate_plan(plan, ROOT, check_files=False)
        result = audit_training(plan, ROOT, args.output)
    elif args.mode == "analyze":
        from rosetta_reality.eval.gate_diagnostic_report import analyze

        result = analyze(args.input, args.output, probe=args.probe, training=args.training)
    else:
        from rosetta_reality.eval.gate_diagnostic_verify import verify_bundle

        result = verify_bundle(args.directory, source=args.source)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return (
        1
        if result["status"]
        in (
            "incomplete",
            "control_failed",
            "verified_incomplete",
            "diverged",
            "incomparable",
            "insufficient_evidence",
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
