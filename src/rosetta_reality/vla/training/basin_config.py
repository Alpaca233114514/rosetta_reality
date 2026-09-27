"""Create-only, file-only training configuration evidence for Basin."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "training_config.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _argument_map(arguments: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for argument in arguments:
        key, marker, value = argument.partition("=")
        if not key.startswith("--") or not marker or key in result:
            raise ValueError("Training arguments must be unique --key=value entries.")
        result[key] = value
    return result


def _expect_arguments(arguments: list[str], settings: dict[str, Any]) -> None:
    flags = _argument_map(arguments)
    expected = {
        "--seed": settings["seed"],
        "--batch_size": settings["batch_size"],
        "--steps": settings["max_steps"],
        "--accelerator.gradient_accumulation.steps": settings["gradient_accumulation_steps"],
        "--accelerator.mixed_precision": settings["precision"],
        "--dataset.repo_id": settings["dataset"]["identifier"],
        "--dataset.revision": settings["dataset"]["revision"],
        "--dataset.episodes": json.dumps(settings["dataset"]["episodes"], separators=(",", ":")),
        "--policy.optimizer_lr": settings["optimizer"]["learning_rate"],
        "--policy.optimizer_betas": (
            json.dumps(settings["optimizer"]["betas"], separators=(",", ":"))
            if settings["optimizer"]["betas"] is not None else None
        ),
        "--policy.optimizer_eps": settings["optimizer"]["eps"],
        "--policy.optimizer_weight_decay": settings["optimizer"]["weight_decay"],
        "--policy.optimizer_grad_clip_norm": settings["optimizer"]["gradient_clip_norm"],
        "--policy.scheduler_warmup_steps": settings["scheduler"]["warmup_steps"],
        "--policy.scheduler_decay_steps": settings["scheduler"]["decay_steps"],
        "--policy.scheduler_decay_lr": settings["scheduler"]["decay_learning_rate"],
    }
    for key, value in expected.items():
        if value is None:
            if key in flags:
                raise ValueError(f"Undeclared training argument: {key}.")
            continue
        if flags.get(key) != str(value):
            raise ValueError(f"Training argument differs from configuration: {key}.")


def write_training_config_bundle(
    destination: Path,
    *,
    plan: dict[str, Any],
    plan_path: Path,
    launch_path: Path,
    runtime_path: Path,
    phase: str,
    arguments: list[str],
) -> Path:
    """Record the declared effective CLI settings before any optimizer update."""
    if phase not in ("train", "smoke"):
        raise ValueError("Only optimizer phases have training config snapshots.")
    section = plan["training" if phase == "train" else "optimizer_smoke"]
    optimizer = plan["training"].get("optimizer") or {}
    scheduler = plan["training"].get("scheduler") or {}
    original = plan_path.read_bytes()
    launch_bytes, runtime_bytes = launch_path.read_bytes(), runtime_path.read_bytes()
    launch = json.loads(launch_bytes)
    runtime = json.loads(runtime_bytes)
    if (
        launch["formal_plan_sha256"] != _sha(original)
        or launch["runtime_experiment_sha256"] != _sha(runtime_bytes)
        or launch["mode"] != phase
        or launch["run_name"] != (plan["run_name"] if phase == "train" else section["run_name"])
    ):
        raise ValueError("Launch identity differs from the training configuration sources.")
    settings = {
        "seed": runtime["seed"],
        "batch_size": section["batch_size"],
        "gradient_accumulation_steps": 1,
        "max_steps": section.get("steps", 1),
        "optimizer": {"type": optimizer.get("type"), "learning_rate": optimizer.get("lr"),
                      "betas": optimizer.get("betas"), "eps": optimizer.get("eps"),
                      "weight_decay": optimizer.get("weight_decay"),
                      "gradient_clip_norm": optimizer.get("grad_clip_norm")},
        "scheduler": {"type": scheduler.get("type"), "warmup_steps": scheduler.get("num_warmup_steps"),
                      "decay_steps": scheduler.get("num_decay_steps"),
                      "peak_learning_rate": scheduler.get("peak_lr"),
                      "decay_learning_rate": scheduler.get("decay_lr")},
        "precision": plan["resources"]["mixed_precision"],
        "dataset": {"identifier": runtime["dataset"]["identifier"],
                    "revision": runtime["dataset"]["revision"],
                    "episodes": section["episodes"],
                    "view_manifest_sha256": launch["dataset_view_manifest_sha256"],
                    "normalization_report_sha256": launch["normalization_report_sha256"]},
    }
    _expect_arguments(arguments, settings)
    identity = {
        "plan_id": plan["plan_id"], "experiment_id": launch["experiment_id"],
        "code": launch["code_identity"],
        "plan_sha256": launch["formal_plan_sha256"],
        "experiment_config_sha256": launch["experiment_config_sha256"],
        "runtime_experiment_sha256": launch["runtime_experiment_sha256"],
        "action_contract_sha256": launch["action_contract_sha256"],
        "model_revision": runtime["model"]["revision"],
        "dataset_revision": runtime["dataset"]["revision"],
    }
    snapshot = {"schema": SCHEMA, "source_run": launch["run_name"], "phase": phase,
                "status": "launch_prepared", "parameters": {"training": settings, "identity": identity}}
    payloads = {"snapshot.json": _json_bytes(snapshot), "plan.json": _json_bytes(plan),
                "plan-original.raw": original, "launch.json": launch_bytes,
                "runtime.json": runtime_bytes}
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        _write_new(destination / name, data)
    manifest = {"schema": SCHEMA, "files": {name: _sha(data) for name, data in payloads.items()}}
    _write_new(destination / "manifest.json", _json_bytes(manifest))
    return destination
