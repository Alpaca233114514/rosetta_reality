"""Version-2 launch assembly tests (no model weights, no data, no training)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rosetta_reality.experiment import file_sha256
from rosetta_reality.features import create_json
from rosetta_reality.vla.training.launch import (
    build_training_arguments,
    compose_runtime_experiment,
    optimizer_arguments,
)

EXPERIMENT: dict[str, Any] = {
    "experiment_id": "m2-example",
    "seed": 17,
    "model": {
        "identifier": "lerobot/smolvla_base",
        "revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "policy": {
            "chunk_size": 50,
            "n_action_steps": 1,
            "empty_cameras": 2,
            "load_vlm_weights": False,
        },
        "adaptation": {
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            "train_state_proj": True,
        },
    },
    "dataset": {
        "identifier": "lerobot/aloha_sim_insertion_human",
        "revision": "cc571a3c661df81b566dbfde3d5c1e85fcdf7884",
        "rename_map": {"environment_state": "observation.environment_state"},
        "test_episodes": [31, 6, 1, 24, 5],
    },
    "tracking": {"project": "rosetta-reality-vla"},
}

OPTIMIZER = {
    "type": "adamw",
    "lr": 1.0e-4,
    "betas": [0.9, 0.95],
    "eps": 1.0e-8,
    "weight_decay": 1.0e-10,
    "grad_clip_norm": 10.0,
}
SCHEDULER = {
    "type": "cosine_decay_with_warmup",
    "num_warmup_steps": 2,
    "num_decay_steps": 8,
    "peak_lr": 1.0e-4,
    "decay_lr": 2.5e-6,
}


def _plan(*, features: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "run_name": "run-001",
        "resources": {"mixed_precision": "bf16", "memory_limit": "8g"},
        "tracking": {"project": "rosetta-reality-vla", "space_id": "unit/space"},
        "features": features
        if features is not None
        else [
            {"name": "trackio_logging"},
            {"name": "train_only_statistics"},
        ],
        "training": {
            "episodes": [49, 4],
            "batch_size": 8,
            "steps": 8,
            "save_freq": 4,
            "log_freq": 2,
            "optimizer": dict(OPTIMIZER),
            "scheduler": dict(SCHEDULER),
            "policy": {
                "empty_cameras": 2,
                "compile_model": False,
                "compile_mode": "default",
                "skip_fully_masked_camera_encoding": True,
            },
        },
        "optimizer_smoke": {
            "run_name": "run-001-smoke",
            "episodes": [49],
            "batch_size": 8,
            "steps": 2,
            "save_freq": 1,
            "save_checkpoint": True,
            "log_freq": 1,
            "num_workers": 0,
            "persistent_workers": False,
        },
        "validation": {
            "episodes": [22],
            "frame_offsets": [0],
            "total_samples": 1,
        },
    }


def test_optimizer_fragment_matches_the_frozen_faust_launcher() -> None:
    from scripts.run_smolvla_formal import _optimizer_arguments

    training = {
        "steps": 2500,
        "optimizer": {
            "type": "adamw",
            "lr": 1.0e-4,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-10,
            "grad_clip_norm": 10.0,
        },
        "scheduler": {
            "type": "cosine_decay_with_warmup",
            "num_warmup_steps": 125,
            "num_decay_steps": 2500,
            "peak_lr": 1.0e-4,
            "decay_lr": 2.5e-6,
        },
    }
    assert optimizer_arguments(training) == _optimizer_arguments(training)


def test_build_training_arguments_pins_the_upstream_cli_surface() -> None:
    arguments = build_training_arguments(
        _plan(),
        EXPERIMENT,
        mode="train",
        run_name="run-001",
        model_root=Path("/models/root"),
        dataset_root=Path("/data/root"),
        output_dir=Path("/out/run-001"),
        device="xpu",
    )
    assert arguments == [
        "--policy.path=/models/root",
        "--policy.pretrained_revision=c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "--policy.device=xpu",
        "--policy.push_to_hub=false",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=1",
        "--policy.empty_cameras=2",
        "--policy.load_vlm_weights=false",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=true",
        "--dataset.repo_id=lerobot/aloha_sim_insertion_human",
        "--dataset.root=/data/root",
        "--dataset.revision=cc571a3c661df81b566dbfde3d5c1e85fcdf7884",
        "--dataset.episodes=[49,4]",
        "--dataset.eval_split=0.0",
        '--rename_map={"environment_state":"observation.environment_state"}',
        "--output_dir=/out/run-001",
        "--job_name=run-001",
        "--seed=17",
        "--batch_size=8",
        "--steps=8",
        "--save_freq=4",
        "--save_checkpoint=true",
        "--save_checkpoint_to_hub=false",
        "--log_freq=2",
        "--eval_steps=0",
        "--env_eval_freq=0",
        "--num_workers=0",
        "--persistent_workers=false",
        "--dataloader_multiprocessing_context=null",
        "--accelerator.mixed_precision=bf16",
        "--accelerator.gradient_accumulation.steps=1",
        "--wandb.enable=true",
        "--wandb.disable_artifact=true",
        "--wandb.project=rosetta-reality-vla",
        "--policy.compile_model=false",
        "--policy.optimizer_lr=0.0001",
        "--policy.optimizer_betas=[0.9,0.95]",
        "--policy.optimizer_eps=1e-08",
        "--policy.optimizer_weight_decay=1e-10",
        "--policy.optimizer_grad_clip_norm=10.0",
        "--policy.scheduler_warmup_steps=2",
        "--policy.scheduler_decay_steps=8",
        "--policy.scheduler_decay_lr=2.5e-06",
    ]


def test_hidden_test_episodes_fail_closed() -> None:
    plan = _plan()
    plan["training"]["episodes"] = [49, 31]
    with pytest.raises(ValueError, match="hidden-test"):
        build_training_arguments(
            plan,
            EXPERIMENT,
            mode="train",
            run_name="run-001",
            model_root=Path("/models/root"),
            dataset_root=Path("/data/root"),
            output_dir=Path("/out/run-001"),
            device="xpu",
        )


def test_trackio_feature_controls_the_wandb_flags() -> None:
    plan = _plan(features=[{"name": "train_only_statistics"}])
    arguments = build_training_arguments(
        plan,
        EXPERIMENT,
        mode="train",
        run_name="run-001",
        model_root=Path("/models/root"),
        dataset_root=Path("/data/root"),
        output_dir=Path("/out/run-001"),
        device="xpu",
    )
    assert "--wandb.enable=false" in arguments
    assert not any(argument.startswith("--wandb.project=") for argument in arguments)


def test_smoke_mode_uses_the_registered_smoke_section() -> None:
    arguments = build_training_arguments(
        _plan(),
        EXPERIMENT,
        mode="smoke",
        run_name="run-001-smoke",
        model_root=Path("/models/root"),
        dataset_root=Path("/data/root"),
        output_dir=Path("/out/run-001-smoke"),
        device="cuda",
    )
    assert "--dataset.episodes=[49]" in arguments
    assert "--steps=2" in arguments
    assert "--save_freq=1" in arguments
    assert "--policy.device=cuda" in arguments
    assert "--policy.optimizer_lr=0.0001" in arguments


def test_compose_runtime_experiment_merges_plan_resources() -> None:
    plan = _plan()
    plan["resources"] = {
        "mixed_precision": "bf16",
        "memory_limit": "16g",
        "memory_swap_limit": "16g",
    }
    runtime = compose_runtime_experiment(EXPERIMENT, plan)
    assert runtime["resources"]["memory_limit"] == "16g"
    assert plan["resources"]["memory_limit"] == "16g"
    runtime["resources"]["memory_limit"] = "mutated"
    assert plan["resources"]["memory_limit"] == "16g"
    assert "resources" not in EXPERIMENT


def test_launch_manifest_binds_every_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.run_smolvla_v2 import _write_launch_manifest

    run_root = tmp_path / "run-root"
    run_root.mkdir()
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(run_root))
    files: dict[str, Path] = {}
    for name in (
        "plan.yaml",
        "base.yaml",
        "contract.yaml",
        "normalization.json",
        "view_manifest.json",
        "gate1.json",
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        files[name] = path
    runtime_experiment_path = tmp_path / "runtime-experiment.json"
    create_json(runtime_experiment_path, EXPERIMENT)
    plan = _plan()

    manifest_path = _write_launch_manifest(
        "train",
        "run-001",
        plan,
        files["plan.yaml"],
        files["base.yaml"],
        EXPERIMENT,
        files["contract.yaml"],
        files["normalization.json"],
        files["view_manifest.json"],
        runtime_experiment_path,
        {"gate1": files["gate1.json"]},
        {
            "revision": "unit",
            "workspace_tree_sha256": "0" * 64,
            "dirty": False,
            "workspace_file_count": 1,
        },
        None,
    )
    assert manifest_path == run_root / "m2-example" / "launch" / "run-001.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["stage"] == "smolvla_v2_formal_launch"
    assert report["mode"] == "train"
    assert report["features"] == ["trackio_logging", "train_only_statistics"]
    assert report["run_name"] == "run-001"
    assert report["hidden_test_loaded"] is False
    assert report["optimizer_contract"]["optimizer"]["lr"] == 1.0e-4
    assert set(report["prerequisites"]) == {"gate1"}
    assert report["runtime_experiment_sha256"] == file_sha256(runtime_experiment_path)
    json.dumps(report, allow_nan=False)
