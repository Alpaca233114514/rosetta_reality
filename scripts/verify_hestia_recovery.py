"""Recompute recovered Hestia evidence without rerunning training or models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from rosetta_reality.vla import visual_coverage as legacy
from rosetta_reality.vla import visual_fit as fit
from rosetta_reality.vla import visual_fit_contract as checks
from rosetta_reality.vla.visual_fit_job import compare_historical_control
from scripts.verify_coverage40_predictions import compare_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]:
        raise ValueError("Registered offline Linux container required")
    for name, sha in plan["sha256"].items():
        if fit.file_hash(Path(name)) != sha:
            raise ValueError(f"Input/code identity changed: {name}")
    root = Path(plan["recovered_root"])
    inventory = json.loads(Path(plan["inventory"]).read_text())
    for name, info in inventory["files"].items():
        if fit.file_hash(root / name) != info["sha256"]:
            raise ValueError(f"Recovered file changed: {name}")

    def load(name):
        return json.loads((root / name).read_text())

    proof = load("C-integrity.json")
    result = load("result.json")
    for info in (
        result["comparison"],
        result["candidate_integrity"],
        proof["observation"],
        proof["schedule"],
        proof["resources"],
        proof["parameter_audit"],
    ):
        relative = Path(info["path"]).relative_to("runs/hestia-fit40-001")
        if fit.file_hash(root / relative) != info["sha256"]:
            raise ValueError("Broken historical evidence reference")
    checks.validate_plan(yaml.safe_load((root / "cpu/draft/main1280.yaml").read_text()), "C")
    checks.validate_observation(
        load("observation/result.json"),
        load("cpu/schedule.json"),
        load("main1280-resources.json"),
        "C",
    )
    b, bm = fit.read_bundle(root / "B-first")
    c, cm = fit.read_bundle(root / "C-first")
    if proof["model_sha256"] != cm["arm_identity"]["checkpoint_files"]["model.safetensors"]:
        raise ValueError("C manifest is not bound to the audited model")
    if proof["plan_sha256"] != fit.file_hash(root / "cpu/draft/main1280.yaml"):
        raise ValueError("C plan differs from its integrity proof")
    reloads = {}
    for arm in "BC":
        reloads[arm] = fit.compare_reload(root / f"{arm}-first", root / f"{arm}-reload")
        if reloads[arm] != load(f"{arm}-reload-check.json") or not reloads[arm]["passed"]:
            raise ValueError("Historical full-array reload does not reproduce")
    old, om = legacy.read_bundle(Path(plan["historical_b"]))
    historical = compare_historical_control(old, om, b, bm)
    if historical != load("historical-B-reproduction.json") or historical["status"] != "passed":
        raise ValueError("Historical B reproduction differs")
    comparison = fit.compare_arms(b, bm, c, cm)
    reproduced = compare_saved(load("comparison.json"), comparison)
    if (
        result["completed_optimizer_updates"] != 1280
        or result["completed_sample_exposures"] != 5120
    ):
        raise ValueError("Unexpected training extent")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.json").open("x") as stream:
        json.dump(comparison, stream, indent=2, allow_nan=False)
    report = {
        "status": "passed",
        "plan_sha256": fit.file_hash(args.plan),
        "historical_source_commit": load("registration.json")["source_commit"],
        "comparison_leaf_checks": reproduced,
        "comparison_sha256": fit.file_hash(output / "comparison.json"),
        "comparison_file_sha_exact": fit.file_hash(output / "comparison.json")
        == fit.file_hash(root / "comparison.json"),
        "recorded_reload_arrays_reproduced": reloads,
        "historical_B_reproduction": historical,
        "recorded_observation_and_lr_schedule_validated": True,
        "historical_updates": 1280,
        "historical_exposures": 5120,
        "current_checkpoint_weights_rehashed": False,
        "new_model_forwards": 0,
        "new_optimizer_steps": 0,
        "new_independent_model_reload": False,
        "hidden_test_loaded": False,
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
