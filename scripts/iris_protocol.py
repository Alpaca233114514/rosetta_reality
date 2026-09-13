"""Frozen Iris single-axis campaign and create-only plan preparation."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from scripts.iris_runtime import ROOT, sha

RUN = "iris-k-scene-20260913-002"
BASELINE = "reports/training/iris-preparation-20260913/historical-main1280.yaml"
BASELINE_SHA = "70fb068fc732b02c4bfd3cf04e70e5465e3094e8d32750bb670009e2f2f2507b"
STAGES = (
    "cpu",
    "doctor",
    "data",
    "benchmark",
    "forward-b1",
    "forward-b4",
    "calibrate",
    "seal-plans",
    "control-smoke",
    "control-smoke-first",
    "control-smoke-reload",
    "control-smoke-check",
    "treatment-smoke",
    "treatment-smoke-first",
    "treatment-smoke-reload",
    "treatment-smoke-check",
    "admit-main",
    "control-main",
    "treatment-main",
    "export",
    "control-first",
    "control-reload",
    "control-check",
    "treatment-first",
    "treatment-reload",
    "treatment-check",
    "analysis",
)
OWNERS = (
    "scripts/iris_protocol.py",
    "scripts/iris_runtime.py",
    "scripts/iris_calibration.py",
    "scripts/iris_stage.py",
    "scripts/run_iris_furnace.py",
    "scripts/iris_delivery.py",
    "scripts/analyze_iris.py",
    "scripts/run_smolvla_v2.py",
    "scripts/train_smolvla_v2.py",
    "scripts/launch_iris_worker.sh",
    "scripts/launch_iris_from_wsl.sh",
    "scripts/receive_iris_from_wsl.sh",
    "scripts/smolvla_forward_check.py",
    "scripts/evaluate_visual_native_small.py",
    "scripts/inspect_hestia_schedule.py",
    "scripts/run_visual_coverage_job.py",
    "scripts/diagnose_zen_noise_transfer.py",
    "src/rosetta_reality/vla/image_key_regularization.py",
    "configs/runtime/autodl_rtx4090.yaml",
    BASELINE,
)
UPSTREAM = {
    "policies/smolvla/modeling_smolvla.py": (
        "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d"
    ),
    "policies/smolvla/smolvlm_with_expert.py": (
        "996d3b0c713c0ed42b383aa2cf89b2e6f9868e337747c480a64adaecdc1073cf"
    ),
    "scripts/lerobot_train.py": "4d15d283ea54583f552b32088db0b6c195250905ca6daf06d4670383790e2059",
}


def estimate_main_seconds(records):
    import math

    if len(records) != 2:
        raise ValueError("Both fresh smoke timings required")
    timings = []
    for record in records:
        times = record.get("update_seconds", [])
        if (
            record.get("steps") != 2
            or len(times) != 2
            or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times)
        ):
            raise ValueError("Smoke update timings missing or invalid")
        timings.append(times[-1])
    return max(1200.0, sum(timings) * 1280 * 1.5) + 600


def baseline():
    import yaml

    if sha(ROOT / BASELINE) != BASELINE_SHA:
        raise ValueError("Historical recipe seal changed")
    return yaml.safe_load((ROOT / BASELINE).read_text())


def build_plan(original, name, *, coefficient=None, calibration=None, smoke=False, batch=4):
    plan = copy.deepcopy(original)
    run_name = RUN + "-" + name
    plan.update(
        plan_id=run_name,
        run_name=run_name,
        hypothesis="Image K scene variance regularization, lambda zero versus 0.01; fresh base",
        training_authorized=True,
        status="preregistered",
    )
    plan["optimizer_smoke"].update(
        run_name=run_name,
        steps=2 if smoke else 1280,
        save_freq=2 if smoke else 320,
        log_freq=1 if smoke else 16,
    )
    plan["preflight"].update(run_name=run_name, batch_size=batch)
    plan["features"] = [
        f for f in plan["features"] if f["name"] != "image_key_scene_regularization"
    ]
    if coefficient is not None:
        if calibration is None:
            raise ValueError("Training plans require sealed fresh-base calibration")
        plan["features"].insert(3, {"name": "image_key_scene_regularization"})
        plan["image_key_scene_contract"] = {
            "coefficient": coefficient,
            "layers": list(range(1, 16, 2)),
            "image_token_range": [0, 64],
            "calibration_path": calibration["path"],
            "calibration_sha256": calibration["sha256"],
        }
    plan["stop_conditions"] = [
        "Stop on any failed prerequisite, identity drift, nonfinite value or illegal action",
        "CUDA allocated <=8GiB, reserved <=10GiB, process tree RSS <=8GiB",
        "Shared work deadline 3600 seconds; protected shutdown at 4200 seconds",
        "Preserve all checkpoints; no resume, retry, hidden access or checkpoint search",
    ]
    return plan


def write_plan(job, name, **kwargs):
    import yaml

    from scripts.run_smolvla_v2 import _resolve_plan

    plan = build_plan(baseline(), name, **kwargs)
    files = set(plan["implementation_files"]) | set(OWNERS)
    files.update(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "src/rosetta_reality/vla/training").glob("*.py")
    )
    plan["implementation_files"] = {name: sha(ROOT / name) for name in sorted(files)}
    path = job / "plans" / (name + ".yaml")
    with path.open("x") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)
    resolved, _, _ = _resolve_plan(path)
    if resolved != plan:
        raise ValueError("Persisted native YAML plan round trip changed")
    return path


def bind_inputs():
    original = baseline()
    durable = Path(os.environ["ROSETTA_AUTODL_ROOT"]).resolve(strict=True)
    entries = [(v["path"], v["sha256"]) for v in original["prerequisites"].values()]
    norm = original["normalization"]
    entries.append((norm["report"], norm["report_sha256"]))
    for name, expected in entries:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe prerequisite path")
        source, dest = durable / relative, ROOT / relative
        if sha(source) != expected:
            raise ValueError("Durable prerequisite changed")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with dest.open("xb") as stream:
                stream.write(source.read_bytes())
        if sha(dest) != expected:
            raise ValueError("Workspace prerequisite differs")
    relative = Path(norm["dataset_view_manifest"])
    target = ROOT / relative.parent.parent
    source = durable / relative.parent.parent
    if not target.exists() and not target.is_symlink():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
    if (
        target.resolve() != source.resolve()
        or sha(ROOT / relative) != norm["dataset_view_manifest_sha256"]
    ):
        raise ValueError("Dataset view binding differs")


def verify_stage(job, stage, now):
    registration = json.loads((job / "registration.json").read_text())
    if (
        registration.get("id") != RUN
        or registration.get("execution_authorized") is not True
        or registration.get("shutdown_authorized") is not True
        or not registration["started"] <= now < registration["deadline"]
        or registration["deadline"] - registration["started"] != 3600
    ):
        raise ValueError("Missing or expired Iris registration")
    if stage not in STAGES:
        raise ValueError("Unregistered Iris stage")
    for prior in STAGES[: STAGES.index(stage)]:
        report = json.loads((job / (prior + ".done.json")).read_text())
        if report.get("stage") != prior or report.get("exit_code") != 0:
            raise ValueError("Iris prerequisite stage failed")
    for name, expected in registration["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Iris source drift: " + name)
    return registration
