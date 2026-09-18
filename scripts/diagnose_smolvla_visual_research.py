"""Plan-driven data-first visual research; no training or implicit GPU dispatch."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rosetta_reality.vla.visual_research import (  # noqa: E402
    analyze_predictions,
    require,
    seal_bundle,
    sha256,
    verify_bundle,
    write_json,
)


def load_plan(path, stage):
    plan = json.loads(Path(path).read_text())
    require(
        plan["schema_version"] == 1 and plan["status"] == "preregistered",
        "Unregistered research plan",
    )
    require(
        plan["optimizer_updates"] == 0 and plan["hidden_test_loaded"] is False,
        "Research scope differs",
    )
    require(stage in plan["authorized_stages"], "Stage is outside current authorization: " + stage)
    for key in ("sources", "evidence_sha256"):
        for name, digest in plan[key].items():
            p = (ROOT / name).resolve()
            require(
                p.is_relative_to(ROOT) and p.is_file() and sha256(p) == digest,
                "Identity drift: " + name,
            )
    plan["_sha256"] = sha256(path)
    return plan


def evidence(plan, output):
    from scripts.audit_smolvla_inventory import inventory, inventory_v2

    old, new = inventory(ROOT), inventory_v2(ROOT)
    missing = sorted(set(p for p in new if p.endswith(".py")) - set(old))
    write_json(
        output / "inventory.json",
        {
            "files": new,
            "file_count": len(new),
            "legacy_python_count": len(old),
            "newly_indexed_python": missing,
            "semantic_review_complete": False,
        },
    )
    report = {
        "status": "evidence_identity_verified",
        "verified_files": len(plan["evidence_sha256"]),
        "file_count": len(new),
        "newly_indexed_python_count": len(missing),
        "model_loaded": False,
        "optimizer_steps": 0,
        "old_report_claims_reused_only_at_original_scope": True,
    }
    write_json(output / "result.json", report)
    write_json(
        output / "runtime.json",
        {
            "system": platform.system(),
            "python": platform.python_version(),
            "profile": "configs/runtime/autodl_rtx4090.yaml",
            "actual_mode": "no_gpu_cpu_only",
            "nested_docker_used": False,
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        },
    )
    seal_bundle(output, {"stage": "evidence", "plan_sha256": plan["_sha256"]})
    return report


def analyze(plan, output, inputs):
    import numpy as np

    require(inputs is not None, "Analysis requires --input")
    manifest = verify_bundle(inputs)
    require(manifest["plan_sha256"] == plan["_sha256"], "Analysis plan differs from collection")
    with np.load(inputs / "arrays.npz", allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    require(
        not plan["analysis"].get("require_fixed_exposure_cohorts", False)
        or "input_seen_by_step2500" in arrays,
        "Registered fixed exposure cohorts absent from collection",
    )
    metadata = json.loads((inputs / "result.json").read_text())
    result = analyze_predictions(arrays, plan["train_episodes"], metadata["groups"])
    write_json(output / "result.json", result)
    seal_bundle(
        output,
        {
            "stage": "analyze",
            "input_manifest_sha256": sha256(inputs / "manifest.json"),
            "plan_sha256": plan["_sha256"],
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("evidence", "data", "pts", "pixels", "supervision", "collect", "analyze"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    plan = load_plan(args.plan, args.stage)
    require(sys.platform == "linux", "Research execution requires registered Linux container")
    require(
        all(os.environ.get(k) == "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")),
        "Offline environment required",
    )
    if args.stage == "collect":
        require(plan.get("gpu_authorized") is True, "Current authorization is no-card only")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.stage == "evidence":
            evidence(plan, args.output)
        elif args.stage in ("data", "pts", "pixels"):
            from scripts.visual_research_data import extract, pixels, video_pts

            {"data": extract, "pts": video_pts, "pixels": pixels}[args.stage](
                plan, args.output, ROOT
            )
        elif args.stage == "analyze":
            analyze(plan, args.output, args.input)
        elif args.stage == "supervision":
            from scripts.visual_research_data import supervision

            supervision(plan, args.output, args.input, ROOT)
        else:
            from scripts.visual_research_collect import collect

            collect(plan, args.output, ROOT)
    except BaseException as exc:
        write_json(
            args.output / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc).replace(str(ROOT), "."),
                "stage": args.stage,
                "plan_sha256": plan["_sha256"],
            },
        )
        raise
    print(json.dumps({"stage": args.stage, "output": args.output.as_posix(), "completed": True}))


if __name__ == "__main__":
    main()
