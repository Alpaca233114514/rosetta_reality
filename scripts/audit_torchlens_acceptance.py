"""Measure the scope of acceptance, independently of the strict adoption gate."""

import argparse
import json
import sys
import time
from pathlib import Path

from rosetta_reality.diagnostics.torchlens_capture import (
    assess_acceptance,
    digest,
    resource_envelope,
    write_new,
)
from scripts.collect_torchlens_synthetic import run_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    resource_envelope()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "incomplete", "cases": {},
              "source_sha256": digest(Path(__file__).read_bytes()),
              "strict_criteria_changed": False, "live_integration_accepted": False}
    deadline = time.monotonic() + 175
    try:
        for case in ("normal", "python_rng", "numpy_rng"):
            root = args.output / case
            root.mkdir()
            arms = []
            for name, backend in (("control", "plain"), ("repeat", "plain"),
                                  ("observed", "torchlens"), ("reference", "reference")):
                command = [sys.executable, "scripts/collect_torchlens_synthetic.py", "--worker",
                           "--case", case, "--backend", backend, "--output", str(root / name),
                           "--run-id", "torchlens-acceptance-" + case]
                process = run_worker(command, root / f"{name}.log",
                                     max(0.01, deadline - time.monotonic()))
                if process.returncode:
                    raise RuntimeError(f"{case}/{name} failed; inspect retained log")
                arms.append(json.loads((root / name / "native.json").read_text()))
            report["cases"][case] = assess_acceptance(*arms)
        report["status"] = "complete"
    finally:
        report["files"] = {p.relative_to(args.output).as_posix(): digest(p.read_bytes())
                           for p in sorted(args.output.rglob("*")) if p.is_file()}
        write_new(args.output / "acceptance-audit.json", report)


if __name__ == "__main__":
    main()
