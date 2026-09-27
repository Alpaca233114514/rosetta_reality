"""File-only Basin config capture; no weights, dataset loading or training."""

from __future__ import annotations

import copy
import ast
import hashlib
import json
from pathlib import Path

import pytest

from rosetta_reality.vla.training.basin_config import write_training_config_bundle
from rosetta_reality.vla.training.launch import build_training_arguments
from test_smolvla_training_launch import EXPERIMENT, _plan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_calls_capture_between_assembly_and_trainer() -> None:
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "run_smolvla_v2.py"
    module = ast.parse(launcher.read_text(encoding="utf-8"))
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    lines = {node.func.id: node.lineno for node in ast.walk(main)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id in {"build_training_arguments", "write_training_config_bundle", "train_main"}}
    assert lines["build_training_arguments"] < lines["write_training_config_bundle"] < lines["train_main"]


@pytest.mark.parametrize("phase,expected_name,expected_episodes,expected_steps", [
    ("train", "run-001", [49, 4], 8),
    ("smoke", "run-001-smoke", [49], 2),
])
def test_config_bundle_precedes_training_and_is_create_only(
    tmp_path: Path, phase: str, expected_name: str, expected_episodes: list[int], expected_steps: int
) -> None:
    plan = _plan()
    plan.update({"schema_version": 2, "role": "vla", "plan_id": "plan-001",
                 "parent_experiment": {"experiment_id": "m2-example", "sha256": "a" * 64}})
    experiment = copy.deepcopy(EXPERIMENT)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(experiment), encoding="utf-8")
    launch_path = tmp_path / "launch.json"
    launch = {"mode": phase, "run_name": expected_name, "experiment_id": "m2-example",
              "formal_plan_sha256": _sha(plan_path),
              "experiment_config_sha256": "a" * 64,
              "runtime_experiment_sha256": _sha(runtime_path),
              "action_contract_sha256": "b" * 64,
              "normalization_report_sha256": "c" * 64,
              "dataset_view_manifest_sha256": "d" * 64,
              "model_revision": experiment["model"]["revision"],
              "dataset_revision": experiment["dataset"]["revision"],
              "code_identity": {"revision": "test", "dirty": False}}
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    arguments = build_training_arguments(
        plan, experiment, mode=phase, run_name=expected_name,
        model_root=tmp_path / "model", dataset_root=tmp_path / "dataset",
        output_dir=tmp_path / "output", device="cpu",
    )
    bundle = tmp_path / "bundle"
    write_training_config_bundle(bundle, plan=plan, plan_path=plan_path,
                                 launch_path=launch_path, runtime_path=runtime_path,
                                 phase=phase, arguments=arguments)
    snapshot = json.loads((bundle / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["source_run"] == expected_name
    assert snapshot["parameters"]["training"]["dataset"]["episodes"] == expected_episodes
    assert snapshot["parameters"]["training"]["max_steps"] == expected_steps
    assert snapshot["parameters"]["training"]["optimizer"]["learning_rate"] == 1e-4
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["plan-original.raw"] == _sha(plan_path)
    with pytest.raises(FileExistsError):
        write_training_config_bundle(bundle, plan=plan, plan_path=plan_path,
                                     launch_path=launch_path, runtime_path=runtime_path,
                                     phase=phase, arguments=arguments)
    bad = ["--batch_size=999" if value.startswith("--batch_size=") else value for value in arguments]
    with pytest.raises(ValueError, match="--batch_size"):
        write_training_config_bundle(tmp_path / "rejected", plan=plan, plan_path=plan_path,
                                     launch_path=launch_path, runtime_path=runtime_path,
                                     phase=phase, arguments=bad)
    assert not (tmp_path / "rejected").exists()
