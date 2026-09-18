"""Bind existing normalization evidence and run native checks without model loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_existing(workspace, durable_runs, normalization):
    workspace = workspace.resolve(strict=True)
    durable_runs = durable_runs.resolve(strict=True)
    bindings = []
    for key in ("report", "dataset_view_manifest"):
        relative = Path(normalization[key])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "runs":
            raise ValueError("Normalization path must be inside workspace runs")
        source = (durable_runs / relative.relative_to("runs")).resolve(strict=True)
        if (
            not source.is_relative_to(durable_runs)
            or digest(source) != normalization[key + "_sha256"]
        ):
            raise ValueError("Existing normalization evidence identity differs")
        target = workspace / relative
        if key == "dataset_view_manifest":
            source, target = source.parent, target.parent
        if target.exists() or target.is_symlink():
            raise FileExistsError("Normalization binding already exists")
        if not target.parent.resolve().is_relative_to(workspace):
            raise ValueError("Normalization binding parent escaped")
        bindings.append((target, source, key == "dataset_view_manifest"))
    for target, source, directory in bindings:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--bind-existing", action="store_true")
    args = parser.parse_args()
    import evaluate_visual_native_small as native

    template = json.loads(args.template.read_text())
    historical, base, experiment = native._resolve_plan(Path(template["historical_plan"]))
    if args.bind_existing:
        bind_existing(ROOT, Path(os.environ["ROSETTA_RUN_ROOT"]), historical["normalization"])
    physical_sha = digest(ROOT / experiment["action_contract"]["derived"])
    report, manifest, view = native._validate_normalization(
        historical, experiment, base, physical_sha
    )
    result = {
        "status": "passed",
        "native_normalization_passed": True,
        "report_sha256": digest(report),
        "view_manifest_sha256": digest(manifest),
        "view_file_count": len(json.loads(manifest.read_text())["files"]),
        "model_weights_loaded": False,
        "optimizer_created": False,
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
