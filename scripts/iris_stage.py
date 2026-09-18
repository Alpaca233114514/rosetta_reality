"""One fail-closed Iris stage, dispatched only by the registered supervisor."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.iris_protocol import RUN, UPSTREAM, baseline, verify_stage, write_plan  # noqa: E402
from scripts.iris_runtime import (  # noqa: E402
    budget,
    collect_native,
    load_native,
    save,
    sha,
    verify_checkpoint,
    verify_reload,
)


def load(path):
    return json.loads(Path(path).read_text())


def checkpoint_path(plan):
    active = plan["optimizer_smoke"]
    return (
        Path(os.environ["ROSETTA_CHECKPOINT_ROOT"])
        / plan["parent_experiment"]["experiment_id"]
        / "smoke"
        / active["run_name"]
        / "checkpoints"
        / f"{active['steps']:06d}"
        / "pretrained_model"
    )


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact inventory cannot contain symlinks")
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Artifact escaped its root")
            result[path.relative_to(root).as_posix()] = sha(path)
    if not result:
        raise ValueError("Empty artifact inventory")
    return result


def load_plan(job, name):
    from scripts.run_smolvla_v2 import _resolve_plan

    path = job / "plans" / (name + ".yaml")
    plan, _, _ = _resolve_plan(path)
    if name not in ("forward-b1", "forward-b4"):
        seals = load(job / "plans-seal.json")
        if sha(path) != seals[name]:
            raise ValueError("Training plan changed after sealing")
    return path, plan


def train(job, name, deadline):
    import lerobot.scripts.lerobot_train as trainer
    import torch

    import scripts.run_smolvla_v2 as launcher
    from rosetta_reality.vla.training.observed_launch import run_observed_launch
    from scripts.inspect_hestia_schedule import inspect_schedule

    path, plan = load_plan(job, name)
    steps = plan["optimizer_smoke"]["steps"]
    expected = inspect_schedule()["sample_identities"][: steps * 4]
    original_make, original_step, original_update = (
        trainer.make_policy,
        torch.optim.AdamW.step,
        trainer.update_policy,
    )
    selected = {}
    update = 0
    update_seconds = []
    log_path = job / (name + "-key-updates.jsonl")
    stream = log_path.open("x")

    def make_policy(*args, **kwargs):
        policy = original_make(*args, **kwargs)
        selected.update(
            {
                key: p
                for key, p in policy.named_parameters()
                if ".lm_expert.layers." in key
                and key.endswith(".self_attn.k_proj.weight")
                and int(key.split(".layers.")[1].split(".")[0]) % 2 == 1
            }
        )
        if len(selected) != 8 or any(not p.requires_grad for p in selected.values()):
            raise ValueError("Eight trainable K projections required")
        return policy

    def step(optimizer, *args, **kwargs):
        nonlocal update
        budget(deadline)
        if update >= steps:
            raise ValueError("Optimizer exceeded registered steps")
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("Nonfinite optimizer gradient")
        snapshots = {key: p.detach().clone() for key, p in selected.items()}
        gradient_norms = {}
        for key, p in selected.items():
            if p.grad is None:
                raise ValueError("K projection gradient missing")
            gradient_norms[key] = float(p.grad.detach().float().norm())
        result = original_step(optimizer, *args, **kwargs)
        update += 1
        row = {
            "step": update,
            "lr": [group["lr"] for group in optimizer.param_groups],
            "k_gradient_l2_after_clip": gradient_norms,
            "k_update_l2": {
                key: float((p.detach() - snapshots[key]).float().norm())
                for key, p in selected.items()
            },
        }
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        budget(deadline)
        return result

    def update_policy(*args, **kwargs):
        started_update = time.monotonic()
        result = original_update(*args, **kwargs)
        update_seconds.append(time.monotonic() - started_update)
        return result

    trainer.make_policy, torch.optim.AdamW.step, trainer.update_policy = (
        make_policy,
        step,
        update_policy,
    )
    sys.argv = ["run_smolvla_v2.py", "smoke", "--plan", str(path)]
    started = time.monotonic()
    try:
        run_observed_launch(
            trainer,
            launcher.main,
            expected_samples=expected,
            batch_size=4,
            output=job / (name + "-observation"),
        )
    finally:
        trainer.make_policy, torch.optim.AdamW.step, trainer.update_policy = (
            original_make,
            original_step,
            original_update,
        )
        stream.close()
    if update != steps or len(update_seconds) != steps:
        raise ValueError("Incomplete optimizer run")
    target = checkpoint_path(plan)
    checkpoints = [2] if steps == 2 else [320, 640, 960, 1280]
    all_files = {}
    for number in checkpoints:
        directory = target.parent.parent / f"{number:06d}"
        if not (directory / "training_state/optimizer_state.safetensors").is_file():
            raise ValueError("Recovery checkpoint optimizer state missing")
        metrics = load(directory / "rosetta_checkpoint_metrics.json")
        if metrics["step"] != number:
            raise ValueError("Checkpoint same-step metrics missing")
        all_files[str(number)] = inventory(directory)
    files = inventory(target)
    verify_checkpoint(target, files, plan)
    save(
        job / (name + "-checkpoint.json"),
        {
            "files": files,
            "checkpoints": all_files,
            "steps": steps,
            "seconds": time.monotonic() - started,
            "update_seconds": update_seconds,
            "plan_sha256": sha(path),
        },
    )


def evaluate(job, name, deadline):
    from scripts.diagnose_zen_noise_transfer import parameter_digests

    arm = name.split("-")[0]
    smoke = "-smoke-" in name
    plan_name = arm + ("-smoke" if smoke else "-main")
    path, plan = load_plan(job, plan_name)
    seal = load(job / (plan_name + "-checkpoint.json"))
    source = checkpoint_path(plan) if smoke else job / "exports" / arm
    context = load_native(
        path,
        checkpoint=source,
        checkpoint_files=seal["files"],
        split="train" if smoke else "non_hidden",
        deadline=deadline,
    )
    calibration = load(job / "calibration.json")
    current = parameter_digests(context.policy)
    if any(
        current.get(key) != value for key, value in calibration["frozen_parameter_sha256"].items()
    ):
        raise ValueError("Frozen VLM parameter changed")
    if smoke:
        context.episodes = context.episodes[:4]
        context.batches = context.batches[:4]
        context.targets = context.targets[:4]
        context.standards = context.standards[:4]
        context.image_sha256 = context.image_sha256[:4]
    collect_native(context, job / (name + ".npz"), deadline)


def run(job, stage, registration):
    deadline = registration["deadline"]
    parent = baseline()["parent_experiment"]["config"]
    if stage == "cpu":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_image_key_regularization.py",
                "tests/test_iris_runtime.py",
                "tests/test_iris_furnace.py",
                "tests/test_smolvla_training_features.py",
                "tests/test_smolvla_training_plan_schema.py",
            ],
            check=True,
        )
    elif stage == "doctor":
        from importlib.metadata import distribution, version

        installed = Path(distribution("lerobot").locate_file("lerobot"))
        for name, expected in UPSTREAM.items():
            if sha(installed / name) != expected:
                raise ValueError("Pinned upstream changed")
        if version("torch") != "2.8.0" and version("torch") != "2.8.0+cu128":
            raise ValueError("Registered CUDA torch differs")
        subprocess.run([sys.executable, "scripts/check_env.py"], check=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/autodl_doctor_cuda.py",
                "--profile",
                "configs/runtime/autodl_rtx4090.yaml",
                "--config",
                parent,
            ],
            check=True,
        )
    elif stage == "data":
        xml = job / "data.xml"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-m",
                "data",
                "tests/test_smolvla_visual_grounding_data.py",
                "--junitxml=" + str(xml),
            ],
            check=True,
        )
        suites = list(ET.parse(xml).getroot().iter("testsuite"))
        if (
            not suites
            or not sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(key, 0)) for s in suites for key in ("failures", "errors", "skipped"))
        ):
            raise ValueError("Required real-data tests failed or skipped")
    elif stage == "benchmark":
        subprocess.run(
            [sys.executable, "scripts/benchmark_smolvla.py", "--config", parent], check=True
        )
    elif stage.startswith("forward-"):
        path = write_plan(job, stage, batch=1 if stage.endswith("b1") else 4)
        subprocess.run(
            [sys.executable, "scripts/run_smolvla_v2.py", "preflight", "--plan", str(path)],
            check=True,
        )
    elif stage == "calibrate":
        from scripts.iris_calibration import calibrate

        calibrate(job / "plans/forward-b4.yaml", job / "calibration.json", deadline)
    elif stage == "seal-plans":
        calibration = {
            "path": (job / "calibration.json").relative_to(ROOT).as_posix(),
            "sha256": sha(job / "calibration.json"),
        }
        if load(job / "calibration.json").get("zero_coefficient_exact") is not True:
            raise ValueError("Calibration zero-control prerequisite failed")
        seals = {}
        for arm, coefficient in (("control", 0), ("treatment", 0.01)):
            for kind in ("smoke", "main"):
                name = arm + "-" + kind
                path = write_plan(
                    job,
                    name,
                    coefficient=coefficient,
                    calibration=calibration,
                    smoke=kind == "smoke",
                )
                seals[name] = sha(path)
        save(job / "plans-seal.json", seals)
    elif stage.endswith("-smoke") or stage.endswith("-main") and stage != "admit-main":
        train(job, stage, deadline)
    elif stage.endswith("-first") or stage.endswith("-reload"):
        evaluate(job, stage, deadline)
    elif stage.endswith("-check"):
        prefix = stage.removesuffix("-check")
        result = verify_reload(job / (prefix + "-first.npz"), job / (prefix + "-reload.npz"))
        save(job / (stage + ".json"), result)
    elif stage == "admit-main":
        records = [load(job / (arm + "-smoke-checkpoint.json")) for arm in ("control", "treatment")]
        from scripts.iris_protocol import estimate_main_seconds

        required = estimate_main_seconds(records)
        if deadline - time.time() < required:
            raise TimeoutError("Main training and reload do not fit the remaining budget")
        if shutil.disk_usage(Path(os.environ["ROSETTA_CHECKPOINT_ROOT"])).free < 17 * 1024**3:
            raise OSError("Full main checkpoint/export/transfer retention budget does not fit")
        save(
            job / "main-admission.json",
            {
                "status": "passed",
                "required_remaining_seconds": required,
                "remaining_seconds": deadline - time.time(),
            },
        )
    elif stage == "export":
        exports = job / "exports"
        exports.mkdir()
        for arm in ("control", "treatment"):
            _, plan = load_plan(job, arm + "-main")
            source = checkpoint_path(plan)
            files = load(job / (arm + "-main-checkpoint.json"))["files"]
            verify_checkpoint(source, files, plan)
            shutil.copytree(source, exports / arm)
            verify_checkpoint(exports / arm, files, plan)
        save(
            job / "selection-export.json",
            {
                "selection": "fixed_final_step_1280",
                "development_search": False,
                "artifact_status": "experimental_pending_validation",
                "m2_complete": False,
            },
        )
    elif stage == "analysis":
        from scripts.analyze_iris import analyze

        analyze(job)
    else:
        raise ValueError("Unknown stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    job = args.job.resolve()
    if job != ROOT / "runs" / RUN:
        raise ValueError("Job identity differs")
    registration = verify_stage(job, args.stage, time.time())
    from scripts.run_iris_furnace import alive

    watch = load(job / "watchdog.json")
    if not alive(watch["pid"], watch["ticks"]):
        raise ValueError("Iris watchdog is absent")
    run(job, args.stage, registration)


if __name__ == "__main__":
    main()
