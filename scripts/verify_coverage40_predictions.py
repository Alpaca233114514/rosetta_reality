"""Verify the retrieved immutable coverage40 bundles without loading a model."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from rosetta_reality.vla.visual_coverage import compare_arms, file_hash, read_bundle


def compare_saved(expected, actual, path="root", counts=None):
    """Compare every recorded leaf; only historical resource peaks are excluded."""
    counts = {} if counts is None else counts
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key in {"peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"}:
                continue
            if key not in actual:
                raise ValueError(f"Missing reproduced field: {path}/{key}")
            compare_saved(value, actual[key], f"{path}/{key}", counts)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"List shape differs: {path}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare_saved(left, right, f"{path}/{index}", counts)
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Numeric reproduction differs: {path}")
        counts["numeric_leaves"] = counts.get("numeric_leaves", 0) + 1
    else:
        if expected != actual or type(expected) is not type(actual):
            raise ValueError(f"Recorded decision/identity differs: {path}")
        counts["other_leaves"] = counts.get("other_leaves", 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]:
        raise ValueError("Registered offline Linux container required")
    for name, sha in plan["sha256"].items():
        if file_hash(Path(name)) != sha:
            raise ValueError(f"Input/code identity changed: {name}")
    root = Path(plan["bundle_root"])
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(Path(plan["inventory"]).read_text())
    history = json.loads(Path(plan["history"]).read_text())
    for name, info in inventory["files"].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid local bundle path")
        if path.stat().st_size != info["bytes"] or file_hash(path) != info["sha256"]:
            raise ValueError(f"Transferred bytes differ: {name}")
    a, am = read_bundle(root / "A-first")
    b, bm = read_bundle(root / "B-first")
    for arm, metadata in (("A", am), ("B", bm)):
        if metadata["arm_identity"] != history["checkpoint_identities"][arm]:
            raise ValueError("Historical checkpoint identity differs")
        if metadata["common_identity"]["execution_contract_sha256"] != file_hash(
            root / "execution-contract.json"
        ):
            raise ValueError("Execution contract differs")
        if metadata["common_identity"]["physical_contract_sha256"] != file_hash(
            Path(plan["physical_contract"])
        ):
            raise ValueError("Physical contract differs")
        expected = json.loads((root / f"{arm}-reload-check.json").read_text())
        if expected != history["reload"][arm]["value"]:
            raise ValueError("Historical reload receipt differs")
    result = compare_arms(a, am, b, bm)
    comparisons = {}
    for arm in "AB":
        comparisons[arm] = compare_saved(history["metrics"][arm + "-first"], result[arm])
    comparisons["decisions"] = compare_saved(history["comparison"], result)
    with (output / "recomputed.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    report = {
        "status": "passed",
        "plan_sha256": file_hash(args.plan),
        "file_count": len(inventory["files"]),
        "total_bytes": sum(info["bytes"] for info in inventory["files"].values()),
        "historical_metrics_reproduced": comparisons,
        "float_tolerance": {"absolute": 1e-12, "relative": 1e-12},
        "historical_resource_peaks_not_remeasured": True,
        "array_shapes": {k: list(v.shape) for k, v in b.items()},
        "recomputed_sha256": file_hash(output / "recomputed.json"),
        "model_forwards": 0,
        "optimizer_steps": 0,
        "new_independent_model_reload": False,
        "hidden_test_loaded": False,
        "remote_files_modified": False,
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
