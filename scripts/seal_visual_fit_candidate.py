"""Bind C's full recovery checkpoint to its actual native observation and parameter audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def seal(
    plan_path: Path, observation_path: Path, schedule_path: Path, resources_path: Path, output: Path
):
    from run_smolvla_phase import _model_root
    from run_smolvla_v2 import _resolve_plan

    from rosetta_reality.vla import visual_fit as fit
    from rosetta_reality.vla import visual_fit_contract as checks

    def load(path):
        return json.loads(path.read_text())

    def reference(path):
        return {"path": path.resolve().relative_to(ROOT).as_posix(), "sha256": fit.file_hash(path)}

    audit_path = output.with_name(output.stem + "-parameters.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError("Candidate integrity evidence already exists")
    plan, _, experiment = _resolve_plan(plan_path.resolve())
    checks.validate_plan(plan, "C")
    checks.validate_experiment(experiment)
    checks.validate_observation(
        load(observation_path), load(schedule_path), load(resources_path), "C"
    )
    checkpoint = (
        Path(os.environ["ROSETTA_CHECKPOINT_ROOT"])
        / experiment["experiment_id"]
        / "smoke"
        / fit.RUN_NAMES["C"]
        / "checkpoints/001280"
    )
    checks.check_native_recovery(checkpoint, "C")
    model = checkpoint / "pretrained_model/model.safetensors"
    baseline = _model_root(experiment) / "model.safetensors"
    model_sha, base_sha = fit.file_hash(model), fit.file_hash(baseline)
    if model_sha in {base_sha, fit.CONTROL_MODEL_SHA256}:
        raise ValueError("C is an unchanged base or B checkpoint")
    import torch
    from safetensors import safe_open
    from smolvla_zen_validate import _zen_checkpoint_statistics

    counts = {group: {"total": 0, "changed": 0} for group in ("vlm", "expert", "projectors")}
    conversions = {}
    with (
        safe_open(baseline, framework="pt", device="cpu") as base,
        safe_open(model, framework="pt", device="cpu") as new,
    ):
        if set(base.keys()) != set(new.keys()):
            raise ValueError("Candidate tensor names differ from the pinned base")
        for key in base.keys():
            a, b = base.get_tensor(key), new.get_tensor(key)
            if a.shape != b.shape or not bool(torch.isfinite(b).all()):
                raise ValueError("Nonfinite or incompatible candidate tensor")
            if a.dtype != b.dtype:
                if a.dtype != torch.bfloat16 or b.dtype != torch.float32 or not torch.equal(a, b):
                    raise ValueError("Unexplained candidate dtype/value conversion")
                conversions[key] = {
                    "from": str(a.dtype),
                    "to": str(b.dtype),
                    "exact_values_equal": True,
                }
            group = "vlm" if ".vlm." in key else "expert" if ".lm_expert." in key else "projectors"
            counts[group]["total"] += 1
            counts[group]["changed"] += int(not torch.equal(a, b))
    if (
        counts["vlm"] != {"total": 345, "changed": 0}
        or counts["expert"]["total"] != 145
        or not counts["expert"]["changed"]
        or counts["projectors"]["total"] != 10
        or not counts["projectors"]["changed"]
    ):
        raise ValueError("Candidate update scope differs from frozen VLM/native expert")
    quarters = {}
    for step in (320, 640, 960, 1280):
        folder = checkpoint.parent / f"{step:06d}"
        checks.check_native_recovery(folder, "C", expected_step=step)
        metric = load(folder / "rosetta_checkpoint_metrics.json")
        if metric.get("step") != step or not metric.get("metrics"):
            raise ValueError("Quarter checkpoint metrics are missing")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in metric["metrics"].values()
        ):
            raise ValueError("Nonfinite quarter checkpoint metric")
        quarters[str(step)] = {
            p.relative_to(folder).as_posix(): fit.file_hash(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        }
    normal = checks.relative_file(ROOT, plan["normalization"]["report"])
    if fit.file_hash(normal) != checks.NORMALIZATION_SHA:
        raise ValueError("Candidate normalization identity changed")
    processor = _zen_checkpoint_statistics(checkpoint / "pretrained_model", load(normal))
    files = {
        p.relative_to(checkpoint).as_posix(): fit.file_hash(p)
        for p in sorted(checkpoint.rglob("*"))
        if p.is_file()
    }
    if (
        files["pretrained_model/model.safetensors"] != model_sha
        or fit.file_hash(baseline) != base_sha
    ):
        raise ValueError("Checkpoint/base changed during candidate audit")
    identity = {
        "run_name": fit.RUN_NAMES["C"],
        "checkpoint_files": {
            name.removeprefix("pretrained_model/"): sha
            for name, sha in files.items()
            if name.startswith("pretrained_model/")
        },
    }
    checks.validate_saved_recipe(
        load(checkpoint / "pretrained_model/train_config.json"), plan, "C", identity
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "passed",
        "step": 1280,
        "model_sha256": model_sha,
        "base_revision": fit.REVISIONS["base_revision"],
        "base_model_sha256": base_sha,
        "parameter_counts": counts,
        "processor_identity": processor,
        "exact_value_preserving_dtype_transitions": conversions,
        "recovery_state_complete": True,
    }
    with audit_path.open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    proof = {
        "status": "passed",
        "model_sha256": model_sha,
        "plan_sha256": fit.file_hash(plan_path),
        "initialization": "revision_pinned_base_model",
        "optimizer_resumed": False,
        "checkpoint_files": files,
        "quarter_checkpoints": quarters,
        "observation": reference(observation_path),
        "schedule": reference(schedule_path),
        "resources": reference(resources_path),
        "parameter_audit": reference(audit_path),
    }
    with output.open("x") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "observation", "schedule", "resources", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    seal(args.plan, args.observation, args.schedule, args.resources, args.output)
