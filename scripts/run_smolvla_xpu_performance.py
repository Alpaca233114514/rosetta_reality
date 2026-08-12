"""Audit and benchmark bounded SmolVLA XPU trainer optimizations."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_xpu_performance_003.yaml"
)
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_formal as formal_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path.name}.")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _repository_path(raw: str, *, require_file: bool = True) -> Path:
    relative = Path(raw)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Performance plan paths must be safe repository-relative paths.")
    path = (REPOSITORY_ROOT / relative).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("Performance plan path escaped the repository root.")
    if require_file and not path.is_file():
        raise FileNotFoundError(relative.as_posix())
    return path


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_performance_plan(
    plan_path: Path,
    *,
    require_runtime_evidence: bool = True,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    plan = _load_yaml(plan_path)
    parent = plan.get("parent_experiment", {})
    base_path = _repository_path(str(parent.get("config", "")))
    formal_value = plan.get("formal_plan", {})
    formal_path = _repository_path(str(formal_value.get("config", "")))
    if file_sha256(base_path) != parent.get("sha256"):
        raise ValueError("Performance plan parent experiment checksum is stale.")
    if file_sha256(formal_path) != formal_value.get("sha256"):
        raise ValueError("Performance plan formal-plan checksum is stale.")
    supersedes = plan.get("supersedes", {})
    superseded_plan = _repository_path(str(supersedes.get("plan", "")))
    if file_sha256(superseded_plan) != supersedes.get("sha256"):
        raise ValueError("Performance plan predecessor checksum is stale.")
    if supersedes.get("reason") == "per_element_fixed_forward_parity_failed":
        failed_parity = _repository_path(
            str(supersedes.get("failed_parity_report", "")),
            require_file=require_runtime_evidence,
        )
        if not _is_sha256(supersedes.get("failed_parity_report_sha256")):
            raise ValueError("Performance plan failed parity declaration is invalid.")
        if require_runtime_evidence and (
            file_sha256(failed_parity) != supersedes.get("failed_parity_report_sha256")
            or _load_json(failed_parity).get("status") != "failed"
        ):
            raise ValueError("Performance plan lost the failed parity evidence.")
    elif supersedes.get("reason") == "resource_and_compiler_cache_followup":
        evidence = supersedes.get("evidence", [])
        if not isinstance(evidence, list) or len(evidence) < 3:
            raise ValueError("The resource follow-up plan has incomplete evidence.")
        for item in evidence:
            if (
                not isinstance(item, dict)
                or not _is_sha256(item.get("sha256"))
                or not isinstance(item.get("status"), str)
                or not item["status"]
            ):
                raise ValueError("Performance predecessor evidence must be mappings.")
            evidence_path = _repository_path(
                str(item.get("path", "")),
                require_file=require_runtime_evidence,
            )
            if require_runtime_evidence:
                report = _load_json(evidence_path)
                if (
                    file_sha256(evidence_path) != item.get("sha256")
                    or report.get("status") != item.get("status")
                ):
                    raise ValueError("Performance predecessor evidence is stale.")
    else:
        raise ValueError("Performance plan has no approved predecessor reason.")
    formal_plan, formal_base_path, experiment = formal_runner._validate_plan(
        formal_path,
        require_runtime_evidence=require_runtime_evidence,
    )
    if formal_base_path != base_path:
        raise ValueError("Performance and formal plans do not share the same experiment.")

    protocol = plan.get("protocol", {})
    target = plan.get("target", {})
    parity = plan.get("parity", {})
    candidates = plan.get("candidates", {})
    order = protocol.get("candidate_order", [])
    test_episodes = {int(value) for value in experiment["dataset"]["test_episodes"]}
    protocol_episodes = [int(value) for value in protocol.get("episodes", [])]
    if (
        plan.get("schema_version") != 1
        or plan.get("role") != "vla"
        or plan.get("stage") != "m2_xpu_training_performance"
        or plan.get("status") != "preregistered"
        or parent.get("experiment_id") != experiment["experiment_id"]
        or protocol_episodes != formal_plan["training"]["episodes"]
        or set(protocol_episodes) & test_episodes
        or protocol.get("hidden_test_loaded") is not False
        or protocol.get("optimizer_created_only_after_baseline_and_parity") is not True
        or not isinstance(protocol.get("steps"), int)
        or protocol["steps"] <= int(protocol.get("discard_warmup_steps", 0))
        or protocol.get("save_checkpoint") is not False
        or protocol.get("num_workers") != 0
        or order != list(candidates)
        or not candidates
        or target.get("train_rows") != 20_000
        or target.get("minimum_dataset_passes") != 1.0
        or target.get("maximum_projected_wall_seconds") != 7_200
        or not math.isclose(
            float(target.get("minimum_steady_state_samples_per_second", 0.0)),
            20_000 / 7_200,
        )
        or parity.get("kind") != "masked_camera_encoder_skip"
        or parity.get("batch_size") != 1
        or parity.get("flow_time") != 0.5
        or parity.get("reference_empty_cameras") != 2
        or parity.get("candidate_empty_cameras") != 2
        or parity.get("expected_camera_slots") != 3
        or parity.get("expected_vision_encoder_calls_reference") != 3
        or parity.get("expected_vision_encoder_calls_candidate") != 1
    ):
        raise ValueError("Performance plan protocol, split, or target is invalid.")

    run_names: set[str] = set()
    for name, candidate in candidates.items():
        if (
            not isinstance(name, str)
            or not isinstance(candidate, dict)
            or candidate.get("batch_size") not in {2, 4, 8, 12, 16}
            or candidate.get("empty_cameras") not in {0, 2}
            or not isinstance(candidate.get("compile_model"), bool)
            or not isinstance(candidate.get("skip_fully_masked_camera_encoding"), bool)
            or not phase_runner.RUN_NAME_PATTERN.fullmatch(str(candidate.get("run_name", "")))
            or candidate.get("run_name") in run_names
        ):
            raise ValueError("A registered performance candidate is invalid.")
        run_names.add(str(candidate["run_name"]))
        if candidate["empty_cameras"] == 0 and candidate.get(
            "requires_empty_camera_parity"
        ) is not True:
            raise ValueError("Removing empty cameras requires the registered parity gate.")
        if candidate["empty_cameras"] == 0 and candidate.get(
            "skip_fully_masked_camera_encoding"
        ):
            raise ValueError("Empty-camera removal and masked-camera skipping cannot be combined.")
        if candidate["skip_fully_masked_camera_encoding"] and candidate.get(
            "requires_masked_camera_parity"
        ) is not True:
            raise ValueError("Masked-camera skipping requires the registered parity gate.")
        if candidate["compile_model"] and candidate.get("compile_mode") not in {
            "default",
            "reduce-overhead",
            "max-autotune",
        }:
            raise ValueError("A compiled candidate has no supported compile mode.")

    resources = plan.get("resources", {})
    memory_limit = resources.get("memory_limit")
    maximum_peak = target.get("maximum_peak_xpu_allocated_bytes")
    peak_guard = 4 * 1024**3 if memory_limit == "6g" else 7 * 1024**3
    if (
        resources.get("runtime") != "docker_linux_from_wsl"
        or resources.get("accelerator") != "xpu"
        or resources.get("mixed_precision") != experiment["resources"]["mixed_precision"]
        or memory_limit not in {experiment["resources"]["memory_limit"], "8g"}
        or resources.get("memory_swap_limit") != memory_limit
        or (
            memory_limit == "8g"
            and resources.get("authorization") != "user_explicit_2026-08-11"
        )
        or not isinstance(maximum_peak, int)
        or isinstance(maximum_peak, bool)
        or maximum_peak <= 0
        or maximum_peak > peak_guard
    ):
        raise ValueError("Performance resources differ from the registered XPU budget.")
    return plan, base_path, experiment, formal_path, formal_plan


def _decode_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Trackio row does not contain a JSON object.")
    return value


def _trackio_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [row.get(key) for row in rows]
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"Trackio metric is missing or non-finite: {key}.")
    return fmean(float(value) for value in values)


def _audit_baseline(plan: dict[str, Any]) -> dict[str, Any]:
    baseline = plan["baseline"]
    database = _repository_path(str(baseline["trackio_database"]))
    connection = _trackio_connection(database)
    try:
        config_row = connection.execute(
            "SELECT config FROM configs WHERE run_id = ?", (str(baseline["run_id"]),)
        ).fetchone()
        metric_rows = connection.execute(
            "SELECT step, timestamp, metrics FROM metrics WHERE run_id = ? ORDER BY step, id",
            (str(baseline["run_id"]),),
        ).fetchall()
    finally:
        connection.close()
    if config_row is None:
        raise ValueError("The registered Trackio baseline run is missing.")
    config = _decode_mapping(config_row[0])
    decoded_rows = [(row, _decode_mapping(row[2])) for row in metric_rows]
    training_rows = [
        (row, metrics)
        for row, metrics in decoded_rows
        if "train/step_s" in metrics
    ]
    metric_rows = [row for row, _metrics in training_rows]
    rows = [metrics for _row, metrics in training_rows]
    warmup = int(baseline["discard_warmup_steps"])
    steady = rows[warmup:]
    if (
        len(rows) != int(baseline["logged_steps"])
        or len(steady) == 0
        or config.get("experiment_id") != plan["parent_experiment"]["experiment_id"]
        or config.get("phase") != "formal"
        or config.get("batch_size") != baseline["batch_size"]
        or config.get("empty_cameras") != baseline["empty_cameras"]
        or bool(config.get("compile_model", False)) is not baseline["compile_model"]
        or bool(config.get("skip_fully_masked_camera_encoding", False))
        is not baseline["skip_fully_masked_camera_encoding"]
        or config.get("test_split_loaded") is not False
    ):
        raise ValueError("The registered Trackio baseline identity is invalid.")
    observed = {
        "mean_step_seconds": _mean_metric(steady, "train/step_s"),
        "mean_update_seconds": _mean_metric(steady, "train/update_s"),
        "mean_dataloading_seconds": _mean_metric(steady, "train/dataloading_s"),
        "mean_preprocessing_seconds": _mean_metric(steady, "train/preprocessing_s"),
        "mean_samples_per_second": _mean_metric(steady, "train/samples_per_s"),
    }
    for key, value in observed.items():
        if not math.isclose(value, float(baseline[key]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Registered baseline metric is stale: {key}.")
    first_timestamp = str(metric_rows[0][1])
    last_timestamp = str(metric_rows[-1][1])
    return {
        "status": "verified",
        "run_id": baseline["run_id"],
        "logged_steps": len(rows),
        "discarded_warmup_steps": warmup,
        "first_metric_timestamp": first_timestamp,
        "last_metric_timestamp": last_timestamp,
        **observed,
        "update_fraction_of_step": observed["mean_update_seconds"]
        / observed["mean_step_seconds"],
        "hidden_test_loaded": False,
    }


def _validate_prerequisites(
    plan: dict[str, Any],
    plan_path: Path,
    base_path: Path,
    experiment: dict[str, Any],
    formal_path: Path,
    formal_plan: dict[str, Any],
) -> tuple[Path, Path, Path, dict[str, Path]]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Performance work must run with networking disabled.")
    resources = plan["resources"]
    if (
        os.environ.get("ROSETTA_TORCH_DEVICE") != "xpu"
        or os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT") != resources["memory_swap_limit"]
    ):
        raise ValueError("The active XPU or memory budget differs from the performance plan.")
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    prerequisites = formal_runner._validate_prerequisites(
        formal_plan, experiment, base_path, contract_sha256
    )
    normalization_report, _view_manifest, dataset_root = formal_runner._validate_normalization(
        formal_plan, experiment, base_path, contract_sha256
    )
    registered_normalization = plan["normalization"]
    if (
        normalization_report != _repository_path(str(registered_normalization["report"]))
        or file_sha256(normalization_report) != registered_normalization["sha256"]
    ):
        raise ValueError("Performance normalization identity is invalid.")
    preflight_value = plan["preflight"]
    preflight_path = _repository_path(str(preflight_value["report"]))
    if file_sha256(preflight_path) != preflight_value["sha256"]:
        raise ValueError("Performance preflight checksum is stale.")
    formal_runner._validate_preflight(
        preflight_path,
        formal_plan,
        experiment,
        base_path,
        contract_sha256,
        file_sha256(normalization_report),
        file_sha256(formal_path),
    )
    if file_sha256(plan_path) == file_sha256(formal_path):
        raise ValueError("Performance and formal plans must remain separate artifacts.")
    return contract_path, normalization_report, dataset_root, prerequisites


def _runtime_experiment(
    experiment: dict[str, Any], plan: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    runtime = copy.deepcopy(experiment)
    runtime["resources"]["memory_limit"] = plan["resources"]["memory_limit"]
    runtime["resources"]["memory_swap_limit"] = plan["resources"]["memory_swap_limit"]
    runtime["model"]["policy"]["empty_cameras"] = int(candidate["empty_cameras"])
    runtime["model"]["policy"]["compile_model"] = bool(candidate["compile_model"])
    if candidate.get("compile_model"):
        runtime["model"]["policy"]["compile_mode"] = str(candidate["compile_mode"])
    protocol = plan["protocol"]
    runtime["phases"]["formal"] = {
        "episodes": list(protocol["episodes"]),
        "batch_size": int(candidate["batch_size"]),
        "steps": int(protocol["steps"]),
        "save_freq": int(protocol["save_freq"]),
        "save_checkpoint": False,
        "log_freq": int(protocol["log_freq"]),
        "num_workers": int(protocol["num_workers"]),
        "persistent_workers": bool(protocol["persistent_workers"]),
        "validation_gradients": False,
        "hidden_test_loaded": False,
    }
    return runtime


def _prepare_environment(
    *,
    base_path: Path,
    experiment: dict[str, Any],
    formal_path: Path,
    normalization_report: Path,
    performance_path: Path,
    run_name: str,
    skip_masked_camera_encoding: bool = False,
    compile_model: bool = False,
) -> dict[str, Any]:
    identity = workspace_code_identity(REPOSITORY_ROOT)
    os.environ.update(
        {
            "ROSETTA_VLA_PHASE": "performance_benchmark",
            "ROSETTA_VLA_EXPERIMENT_CONFIG": str(base_path),
            "ROSETTA_VLA_RUN_NAME": run_name,
            "ROSETTA_VLA_TRAIN_STATS_REPORT": str(normalization_report),
            "ROSETTA_VLA_FORMAL_PLAN_SHA256": file_sha256(formal_path),
            "ROSETTA_VLA_PERFORMANCE_PLAN_SHA256": file_sha256(performance_path),
            "ROSETTA_VLA_NORMALIZATION_SHA256": file_sha256(normalization_report),
            "ROSETTA_VLA_CODE_REVISION": str(identity["revision"]),
            "ROSETTA_VLA_WORKSPACE_TREE_SHA256": str(identity["workspace_tree_sha256"]),
            "ROSETTA_VLA_WORKSPACE_DIRTY": str(bool(identity["dirty"])).lower(),
            "ROSETTA_VLA_WORKSPACE_FILE_COUNT": str(identity["workspace_file_count"]),
            "ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING": (
                "1" if skip_masked_camera_encoding else "0"
            ),
        }
    )
    if compile_model:
        run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        cache_root = (
            run_root / "compiler_cache" / f"xpu-{file_sha256(performance_path)[:12]}"
        )
        triton_cache = cache_root / "triton"
        inductor_cache = cache_root / "inductor"
        triton_cache.mkdir(parents=True, exist_ok=True)
        (inductor_cache / "cache").mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    return identity


def _parity_report_path(experiment: dict[str, Any], plan_path: Path) -> Path:
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    return (
        run_root
        / str(experiment["experiment_id"])
        / "performance"
        / f"masked-camera-parity-{file_sha256(plan_path)[:12]}.json"
    )


def _run_parity(
    plan_path: Path,
    plan: dict[str, Any],
    base_path: Path,
    experiment: dict[str, Any],
    formal_path: Path,
    normalization_report: Path,
    dataset_root: Path,
    contract_path: Path,
) -> Path:
    import torch
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla import modeling_smolvla
    from smolvla_forward_check import _make_processors, _parse_config
    from torch.utils.data import DataLoader
    from train_smolvla_trackio import _install_masked_camera_encoder_skip

    destination = _parity_report_path(experiment, plan_path)
    if destination.exists():
        raise FileExistsError("The parity report already exists; it is create-only.")
    candidate = {
        "batch_size": int(plan["parity"]["batch_size"]),
        "empty_cameras": 2,
        "compile_model": False,
        "skip_fully_masked_camera_encoding": False,
    }
    runtime = _runtime_experiment(experiment, plan, candidate)
    model_root = phase_runner._model_root(experiment)
    output_dir = Path("/tmp/rosetta-smolvla-parity-unused")
    run_name = f"smolvla-parity-{file_sha256(plan_path)[:12]}"
    identity = _prepare_environment(
        base_path=base_path,
        experiment=experiment,
        formal_path=formal_path,
        normalization_report=normalization_report,
        performance_path=plan_path,
        run_name=run_name,
    )
    sys.argv = [
        "lerobot-train",
        *phase_runner._phase_arguments(
            runtime, "formal", run_name, model_root, dataset_root, output_dir
        ),
    ]
    cfg = _parse_config()
    cfg.validate()
    device = torch.device("xpu")
    torch.manual_seed(int(plan["parity"]["noise_seed"]))
    dataset = make_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    materialized = {
        int(value) for value in batch["episode_index"].detach().cpu().reshape(-1).tolist()
    }
    hidden_test = {int(value) for value in experiment["dataset"]["test_episodes"]}
    if materialized & hidden_test:
        raise ValueError("Parity materialized a sealed hidden-test episode.")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if (
        policy.config.add_image_special_tokens is not False
        or int(policy.config.prefix_length) != 0
    ):
        raise ValueError("The pinned model no longer satisfies the empty-camera parity premise.")
    preprocessor, _ = _make_processors(cfg, policy, dataset, device)
    for camera_key in dataset.meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(torch.get_default_dtype()) / 255
    batch = preprocessor(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    noise = torch.randn(
        actions.shape, generator=torch.Generator(device=device).manual_seed(
            int(plan["parity"]["noise_seed"])
        ), device=device, dtype=actions.dtype
    )
    flow_time = torch.full(
        (actions.shape[0],),
        float(plan["parity"]["flow_time"]),
        device=device,
        dtype=actions.dtype,
    )

    def fixed_forward() -> tuple[torch.Tensor, int]:
        images, image_masks = policy.prepare_images(batch)
        losses = policy.model.forward(
            images,
            image_masks,
            batch[f"{modeling_smolvla.OBS_LANGUAGE_TOKENS}"],
            batch[f"{modeling_smolvla.OBS_LANGUAGE_ATTENTION_MASK}"],
            state,
            actions,
            noise=noise,
            time=flow_time,
        )
        return losses.detach().float().cpu(), len(images)

    policy.train()
    started = time.perf_counter()
    vision_calls = {"reference": 0, "candidate": 0}
    phase = "reference"
    vision_model_class = modeling_smolvla.SmolVLMWithExpertModel
    original_embed_image = vision_model_class.embed_image

    def counted_embed_image(self: Any, image: torch.Tensor) -> torch.Tensor:
        vision_calls[phase] += 1
        return original_embed_image(self, image)

    vision_model_class.embed_image = counted_embed_image
    try:
        with torch.no_grad(), torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            reference, reference_images = fixed_forward()
            torch.xpu.synchronize()
            _install_masked_camera_encoder_skip()
            phase = "candidate"
            candidate_losses, candidate_images = fixed_forward()
            torch.xpu.synchronize()
    finally:
        vision_model_class.embed_image = original_embed_image
    elapsed = time.perf_counter() - started
    difference = (candidate_losses - reference).abs()
    reference_scalar = float(reference.mean().item())
    candidate_scalar = float(candidate_losses.mean().item())
    maximum_absolute = float(difference.max().item())
    mean_absolute = float(difference.mean().item())
    relative_scalar = abs(candidate_scalar - reference_scalar) / max(
        abs(reference_scalar), 1e-12
    )
    parity = plan["parity"]
    acceptance = {
        "camera_slot_count_unchanged": (
            reference_images == int(parity["expected_camera_slots"])
            and candidate_images == reference_images
        ),
        "vision_encoder_calls_reduced_from_three_to_one": (
            vision_calls["reference"]
            == int(parity["expected_vision_encoder_calls_reference"])
            and vision_calls["candidate"]
            == int(parity["expected_vision_encoder_calls_candidate"])
        ),
        "loss_tensor_shape_unchanged": reference.shape == candidate_losses.shape,
        "maximum_absolute_difference_within_limit": maximum_absolute
        <= float(parity["maximum_absolute_loss_tensor_difference"]),
        "mean_absolute_difference_within_limit": mean_absolute
        <= float(parity["maximum_mean_absolute_loss_tensor_difference"]),
        "relative_scalar_difference_within_limit": relative_scalar
        <= float(parity["maximum_relative_scalar_loss_difference"]),
        "optimizer_created": False,
        "gradients_enabled": False,
        "hidden_test_loaded": False,
    }
    status = (
        "passed"
        if all(
            (
                acceptance["camera_slot_count_unchanged"],
                acceptance["vision_encoder_calls_reduced_from_three_to_one"],
                acceptance["loss_tensor_shape_unchanged"],
                acceptance["maximum_absolute_difference_within_limit"],
                acceptance["mean_absolute_difference_within_limit"],
                acceptance["relative_scalar_difference_within_limit"],
                not acceptance["optimizer_created"],
                not acceptance["gradients_enabled"],
                not acceptance["hidden_test_loaded"],
            )
        )
        else "failed"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "stage": "smolvla_masked_camera_encoder_fixed_forward_parity",
        "experiment_id": experiment["experiment_id"],
        "performance_plan_sha256": file_sha256(plan_path),
        "formal_plan_sha256": file_sha256(formal_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": file_sha256(contract_path),
        "normalization_report_sha256": file_sha256(normalization_report),
        "model_revision": experiment["model"]["revision"],
        "dataset_revision": experiment["dataset"]["revision"],
        "batch_size": int(cfg.batch_size),
        "episodes_materialized": sorted(materialized),
        "noise_seed": int(parity["noise_seed"]),
        "flow_time": float(parity["flow_time"]),
        "reference_empty_cameras": int(parity["reference_empty_cameras"]),
        "candidate_empty_cameras": int(parity["candidate_empty_cameras"]),
        "reference_image_count": reference_images,
        "candidate_image_count": candidate_images,
        "reference_vision_encoder_calls": vision_calls["reference"],
        "candidate_vision_encoder_calls": vision_calls["candidate"],
        "loss_tensor_shape": list(reference.shape),
        "reference_scalar_loss": reference_scalar,
        "candidate_scalar_loss": candidate_scalar,
        "maximum_absolute_loss_tensor_difference": maximum_absolute,
        "mean_absolute_loss_tensor_difference": mean_absolute,
        "relative_scalar_loss_difference": relative_scalar,
        "acceptance": acceptance,
        "elapsed_seconds": elapsed,
        "accelerator_memory": {
            "allocated_bytes": int(torch.xpu.memory_allocated()),
            "reserved_bytes": int(torch.xpu.memory_reserved()),
            "maximum_allocated_bytes": int(torch.xpu.max_memory_allocated()),
        },
        "code_identity": identity,
        "network_disabled": True,
        "optimizer_created": False,
        "gradients_enabled": False,
        "hidden_test_loaded": False,
    }
    create_json(destination, report)
    if status != "passed":
        raise RuntimeError(f"Masked-camera parity failed; report: {destination.name}")
    return destination


def _validate_parity_report(
    path: Path,
    *,
    plan_path: Path,
    base_path: Path,
    experiment: dict[str, Any],
    formal_path: Path,
    normalization_report: Path,
    contract_path: Path,
) -> dict[str, Any]:
    report = _load_json(path.resolve())
    acceptance = report.get("acceptance", {})
    if (
        report.get("status") != "passed"
        or report.get("stage")
        != "smolvla_masked_camera_encoder_fixed_forward_parity"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("performance_plan_sha256") != file_sha256(plan_path)
        or report.get("formal_plan_sha256") != file_sha256(formal_path)
        or report.get("experiment_config_sha256") != file_sha256(base_path)
        or report.get("action_contract_sha256") != file_sha256(contract_path)
        or report.get("normalization_report_sha256") != file_sha256(normalization_report)
        or report.get("reference_empty_cameras") != 2
        or report.get("candidate_empty_cameras") != 2
        or report.get("reference_image_count") != 3
        or report.get("candidate_image_count") != 3
        or report.get("reference_vision_encoder_calls") != 3
        or report.get("candidate_vision_encoder_calls") != 1
        or report.get("optimizer_created") is not False
        or report.get("gradients_enabled") is not False
        or report.get("hidden_test_loaded") is not False
        or not isinstance(acceptance, dict)
        or acceptance.get("camera_slot_count_unchanged") is not True
        or acceptance.get("vision_encoder_calls_reduced_from_three_to_one") is not True
        or acceptance.get("loss_tensor_shape_unchanged") is not True
        or acceptance.get("maximum_absolute_difference_within_limit") is not True
        or acceptance.get("mean_absolute_difference_within_limit") is not True
        or acceptance.get("relative_scalar_difference_within_limit") is not True
    ):
        raise ValueError("The masked-camera parity report is invalid or stale.")
    return report


def _candidate_metrics(
    database: Path, run_name: str, expected_steps: int
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    connection = _trackio_connection(database)
    try:
        config_row = connection.execute(
            "SELECT run_id, config FROM configs WHERE run_name = ? ORDER BY id DESC LIMIT 1",
            (run_name,),
        ).fetchone()
        if config_row is None:
            raise ValueError("The candidate Trackio run is missing.")
        run_id = str(config_row[0])
        rows = connection.execute(
            "SELECT metrics FROM metrics WHERE run_id = ? ORDER BY step, id", (run_id,)
        ).fetchall()
    finally:
        connection.close()
    metrics = [
        value
        for row in rows
        if "train/step_s" in (value := _decode_mapping(row[0]))
    ]
    if len(metrics) != expected_steps:
        raise ValueError("The candidate Trackio run does not contain every registered step.")
    return run_id, _decode_mapping(config_row[1]), metrics


def _record_compile_cache_failure(
    *,
    candidate_name: str,
    plan_path: Path,
    plan: dict[str, Any],
    base_path: Path,
    experiment: dict[str, Any],
    formal_path: Path,
) -> Path:
    if candidate_name not in plan["candidates"]:
        raise ValueError("--candidate is not registered in the performance plan.")
    candidate = plan["candidates"][candidate_name]
    if candidate.get("compile_model") is not True:
        raise ValueError("Only a registered compile candidate may record this failure.")
    database = _repository_path(str(plan["baseline"]["trackio_database"]))
    connection = _trackio_connection(database)
    try:
        row = connection.execute(
            "SELECT run_id, config FROM configs WHERE run_name = ? ORDER BY id DESC LIMIT 1",
            (str(candidate["run_name"]),),
        ).fetchone()
        logged_training_rows = (
            []
            if row is None
            else connection.execute(
                "SELECT metrics FROM metrics WHERE run_id = ? ORDER BY step, id",
                (str(row[0]),),
            ).fetchall()
        )
    finally:
        connection.close()
    config = {} if row is None else _decode_mapping(row[1])
    training_steps = sum(
        "train/step_s" in _decode_mapping(metric_row[0])
        for metric_row in logged_training_rows
    )
    if training_steps != 0 or (
        row is not None
        and (
            config.get("performance_plan_sha256") != file_sha256(plan_path)
            or config.get("formal_plan_sha256") != file_sha256(formal_path)
            or config.get("compile_model") is not True
            or config.get("batch_size") != candidate["batch_size"]
            or config.get("test_split_loaded") is not False
        )
    ):
        raise ValueError("The compile failure Trackio identity is invalid.")
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "performance"
        / f"{candidate['run_name']}-failure.json"
    )
    report = {
        "schema_version": 1,
        "status": "failed_before_optimizer_step",
        "stage": "smolvla_xpu_training_performance_benchmark",
        "failure_code": "triton_default_cache_on_noexec_tmpfs",
        "error_signature": "spirv_utils shared library failed to map executable segment",
        "remediation": "dedicated_versioned_triton_and_inductor_cache_under_run_root",
        "experiment_id": experiment["experiment_id"],
        "candidate_name": candidate_name,
        "candidate": candidate,
        "run_name": candidate["run_name"],
        "trackio_run_id": None if row is None else str(row[0]),
        "trackio_config_persisted": row is not None,
        "logged_optimizer_steps": training_steps,
        "performance_plan_sha256": file_sha256(plan_path),
        "formal_plan_sha256": file_sha256(formal_path),
        "experiment_config_sha256": file_sha256(base_path),
        "code_revision": config.get("code_revision"),
        "workspace_tree_sha256": config.get("workspace_tree_sha256"),
        "evidence_source": "runner_terminal_exception_and_zero_persisted_training_steps",
        "network_disabled": True,
        "checkpoint_written": False,
        "hidden_test_loaded": False,
    }
    create_json(destination, report)
    return destination


def _project_candidate(
    *,
    plan: dict[str, Any],
    candidate: dict[str, Any],
    metrics: list[dict[str, Any]],
    total_wall_seconds: float,
) -> dict[str, Any]:
    warmup_count = int(plan["protocol"]["discard_warmup_steps"])
    steady = metrics[warmup_count:]
    if not steady:
        raise ValueError("The candidate has no post-warmup measurements.")
    step_seconds = _mean_metric(steady, "train/step_s")
    update_seconds = _mean_metric(steady, "train/update_s")
    dataloading_seconds = _mean_metric(steady, "train/dataloading_s")
    preprocessing_seconds = _mean_metric(steady, "train/preprocessing_s")
    samples_per_second = _mean_metric(steady, "train/samples_per_s")
    _mean_metric(steady, "train/loss")
    _mean_metric(steady, "train/grad_norm")
    all_step_seconds = [float(row["train/step_s"]) for row in metrics]
    startup_seconds = max(0.0, total_wall_seconds - sum(all_step_seconds))
    warmup_seconds = sum(all_step_seconds[:warmup_count])
    target = plan["target"]
    batch_size = int(candidate["batch_size"])
    required_steps = math.ceil(int(target["train_rows"]) / batch_size)
    projected = (
        startup_seconds
        + warmup_seconds
        + max(0, required_steps - warmup_count) * step_seconds
        + float(target["checkpoint_allowance_seconds"])
    )
    peak = max(int(row["train/xpu_max_allocated_bytes"]) for row in metrics)
    target_met = (
        projected <= float(target["maximum_projected_wall_seconds"])
        and samples_per_second
        >= float(target["minimum_steady_state_samples_per_second"])
        and peak <= int(target["maximum_peak_xpu_allocated_bytes"])
    )
    baseline_calls = int(target["train_rows"])
    return {
        "discarded_warmup_steps": warmup_count,
        "measured_steps": len(metrics),
        "mean_step_seconds": step_seconds,
        "mean_update_seconds": update_seconds,
        "mean_dataloading_seconds": dataloading_seconds,
        "mean_preprocessing_seconds": preprocessing_seconds,
        "mean_samples_per_second": samples_per_second,
        "update_fraction_of_step": update_seconds / step_seconds,
        "measured_total_wall_seconds": total_wall_seconds,
        "measured_startup_seconds": startup_seconds,
        "measured_warmup_step_seconds": warmup_seconds,
        "projected_optimizer_steps_for_one_pass": required_steps,
        "projected_policy_prefix_calls_for_one_pass": required_steps,
        "baseline_policy_prefix_calls_for_one_pass": baseline_calls,
        "policy_prefix_call_reduction_fraction": 1 - required_steps / baseline_calls,
        "projected_one_pass_wall_seconds": projected,
        "projected_one_pass_wall_minutes": projected / 60,
        "peak_xpu_allocated_bytes": peak,
        "target_met": target_met,
    }


def _run_candidate(
    *,
    candidate_name: str,
    parity_report: Path | None,
    plan_path: Path,
    plan: dict[str, Any],
    base_path: Path,
    experiment: dict[str, Any],
    formal_path: Path,
    normalization_report: Path,
    dataset_root: Path,
    contract_path: Path,
    baseline: dict[str, Any],
) -> Path:
    if candidate_name not in plan["candidates"]:
        raise ValueError("--candidate is not registered in the performance plan.")
    candidate = plan["candidates"][candidate_name]
    parity: dict[str, Any] | None = None
    if candidate.get("requires_empty_camera_parity") or candidate.get(
        "requires_masked_camera_parity"
    ):
        if parity_report is None:
            raise ValueError("This candidate requires --parity-report before optimizer creation.")
        parity = _validate_parity_report(
            parity_report,
            plan_path=plan_path,
            base_path=base_path,
            experiment=experiment,
            formal_path=formal_path,
            normalization_report=normalization_report,
            contract_path=contract_path,
        )
    runtime = _runtime_experiment(experiment, plan, candidate)
    model_root = phase_runner._model_root(experiment)
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    run_name = str(candidate["run_name"])
    output_dir = checkpoint_root / str(experiment["experiment_id"]) / "performance" / run_name
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    destination = (
        run_root / str(experiment["experiment_id"]) / "performance" / f"{run_name}.json"
    )
    if output_dir.exists() or destination.exists():
        raise FileExistsError("Candidate outputs are create-only; choose a new registered run.")
    identity = _prepare_environment(
        base_path=base_path,
        experiment=experiment,
        formal_path=formal_path,
        normalization_report=normalization_report,
        performance_path=plan_path,
        run_name=run_name,
        skip_masked_camera_encoding=bool(
            candidate["skip_fully_masked_camera_encoding"]
        ),
        compile_model=bool(candidate["compile_model"]),
    )
    sys.argv = [
        "lerobot-train",
        *phase_runner._phase_arguments(
            runtime, "formal", run_name, model_root, dataset_root, output_dir
        ),
    ]
    from train_smolvla_trackio import main as train_main

    started = time.perf_counter()
    train_main()
    total_wall_seconds = time.perf_counter() - started
    database = _repository_path(str(plan["baseline"]["trackio_database"]))
    run_id, trackio_config, metrics = _candidate_metrics(
        database, run_name, int(plan["protocol"]["steps"])
    )
    if (
        trackio_config.get("phase") != "performance_benchmark"
        or trackio_config.get("performance_plan_sha256") != file_sha256(plan_path)
        or trackio_config.get("formal_plan_sha256") != file_sha256(formal_path)
        or trackio_config.get("batch_size") != candidate["batch_size"]
        or trackio_config.get("empty_cameras") != candidate["empty_cameras"]
        or trackio_config.get("compile_model") is not candidate["compile_model"]
        or trackio_config.get("skip_fully_masked_camera_encoding")
        is not candidate["skip_fully_masked_camera_encoding"]
        or trackio_config.get("memory_limit") != plan["resources"]["memory_limit"]
        or trackio_config.get("memory_swap_limit")
        != plan["resources"]["memory_swap_limit"]
        or trackio_config.get("test_split_loaded") is not False
    ):
        raise ValueError("Candidate Trackio config differs from the registered plan.")
    projection = _project_candidate(
        plan=plan,
        candidate=candidate,
        metrics=metrics,
        total_wall_seconds=total_wall_seconds,
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "smolvla_xpu_training_performance_benchmark",
        "experiment_id": experiment["experiment_id"],
        "candidate_name": candidate_name,
        "run_name": run_name,
        "trackio_run_id": run_id,
        "performance_plan_sha256": file_sha256(plan_path),
        "formal_plan_sha256": file_sha256(formal_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": file_sha256(contract_path),
        "normalization_report_sha256": file_sha256(normalization_report),
        "model_revision": experiment["model"]["revision"],
        "dataset_revision": experiment["dataset"]["revision"],
        "candidate": candidate,
        "baseline": baseline,
        "parity_report_sha256": file_sha256(parity_report) if parity is not None else None,
        "metrics": projection,
        "target": plan["target"],
        "code_identity": identity,
        "network_disabled": True,
        "hidden_test_loaded": False,
        "checkpoint_written": False,
    }
    create_json(destination, report)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("audit", "parity", "benchmark", "record-compile-failure")
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--candidate")
    parser.add_argument("--parity-report", type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan, base_path, experiment, formal_path, formal_plan = _validate_performance_plan(
        plan_path
    )
    baseline = _audit_baseline(plan)
    if args.mode == "audit":
        print(json.dumps(baseline, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.mode == "record-compile-failure":
        if args.candidate is None:
            raise ValueError("record-compile-failure mode requires --candidate.")
        destination = _record_compile_cache_failure(
            candidate_name=args.candidate,
            plan_path=plan_path,
            plan=plan,
            base_path=base_path,
            experiment=experiment,
            formal_path=formal_path,
        )
        print(f"Performance report: {destination.name}")
        return 0
    contract_path, normalization_report, dataset_root, _ = _validate_prerequisites(
        plan, plan_path, base_path, experiment, formal_path, formal_plan
    )
    if args.mode == "parity":
        destination = _run_parity(
            plan_path,
            plan,
            base_path,
            experiment,
            formal_path,
            normalization_report,
            dataset_root,
            contract_path,
        )
    else:
        if args.candidate is None:
            raise ValueError("benchmark mode requires --candidate.")
        destination = _run_candidate(
            candidate_name=args.candidate,
            parity_report=args.parity_report,
            plan_path=plan_path,
            plan=plan,
            base_path=base_path,
            experiment=experiment,
            formal_path=formal_path,
            normalization_report=normalization_report,
            dataset_root=dataset_root,
            contract_path=contract_path,
            baseline=baseline,
        )
    print(f"Performance report: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
