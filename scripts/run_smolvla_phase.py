"""Launch a gated SmolVLA phase only after immutable prerequisite evidence passes."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import file_sha256  # noqa: E402


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


def _absolute_root(environment: str) -> Path:
    raw = os.environ.get(environment)
    if not raw:
        raise ValueError(f"{environment} must be set by the Docker runner.")
    root = Path(raw)
    if not root.is_absolute():
        raise ValueError(f"{environment} must be absolute.")
    return root


def _validate_benchmark(
    path: Path,
    experiment: dict[str, Any],
    config_path: Path,
    contract_sha256: str,
) -> None:
    report = _load_json(path)
    if (
        report.get("status") != "complete"
        or report.get("stage") != "pre_training"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(config_path)
        or report.get("dataset_revision") != experiment["dataset"]["revision"]
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("hidden_test_loaded") is not False
        or report.get("normalization_source_split") != "train"
        or report.get("evaluated_split") != "validation"
    ):
        raise ValueError("Pre-training benchmark identity or hidden-test boundary is invalid.")


def _validate_gate(
    path: Path,
    *,
    expected_gate: str,
    experiment_id: str,
    contract_sha256: str,
    dataset_revision: str,
    allowed_replay_episodes: list[int],
) -> None:
    report = _load_json(path)
    if (
        report.get("status") != "passed"
        or report.get("gate") != expected_gate
        or report.get("experiment_id") != experiment_id
        or report.get("action_contract_sha256") != contract_sha256
    ):
        raise ValueError(f"{expected_gate} report is not bound to this SmolVLA contract.")
    if expected_gate == "m2_gate_2_dataset_action_replay" and (
        report.get("dataset_revision") != dataset_revision
        or report.get("episode") not in allowed_replay_episodes
        or report.get("acceptance_criteria", {}).get("timestamp_alignment") is not True
    ):
        raise ValueError("Gate 2 dataset identity or timestamp alignment is invalid.")


def _validate_tracking(path: Path, experiment: dict[str, Any]) -> None:
    report = _load_json(path)
    tracking = experiment["tracking"]
    if (
        report.get("status") != "complete"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("space_id") != tracking["space_id"]
        or report.get("space_sdk") != "static"
        or report.get("visibility") != "public"
        or report.get("contains_sensitive_data") is not False
        or report.get("media_uploaded") is not False
        or report.get("test_split_loaded") is not False
    ):
        raise ValueError("Trackio Space sync report is incomplete or unsafe.")


def _validate_preflight(
    path: Path,
    experiment: dict[str, Any],
    config_path: Path,
    contract_sha256: str,
) -> None:
    report = _load_json(path)
    loss = report.get("loss")
    if (
        report.get("status") != "passed"
        or report.get("stage") != "real_smolvla_no_optimizer_forward"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(config_path)
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("model_revision") != experiment["model"]["revision"]
        or report.get("vlm_dependency_revision")
        != experiment["model"]["vlm_dependency"]["revision"]
        or report.get("dataset_revision") != experiment["dataset"]["revision"]
        or report.get("episodes_loaded") != experiment["phases"]["smoke"]["episodes"]
        or report.get("hidden_test_loaded") is not False
        or report.get("network_disabled") is not True
        or report.get("optimizer_created") is not False
        or report.get("gradients_enabled") is not False
        or report.get("device") != os.environ.get("ROSETTA_TORCH_DEVICE")
        or report.get("mixed_precision") != experiment["resources"]["mixed_precision"]
        or not isinstance(loss, int | float)
        or isinstance(loss, bool)
    ):
        raise ValueError("The real SmolVLA no-optimizer preflight report is invalid.")


def _validate_smoke_acceptance(
    path: Path,
    experiment: dict[str, Any],
    config_path: Path,
    contract_sha256: str,
) -> None:
    report = _load_json(path)
    acceptance = report.get("acceptance", {})
    required = experiment["phases"]["smoke"]["acceptance"]
    if (
        report.get("status") != "passed"
        or report.get("stage") != "smolvla_tiny_smoke_acceptance"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(config_path)
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("tracking", {}).get("space_id") != experiment["tracking"]["space_id"]
        or report.get("tracking", {}).get("test_split_loaded") is not False
        or not isinstance(acceptance, dict)
        or any(acceptance.get(criterion) is not True for criterion in required)
        or acceptance.get("hidden_test_loaded") is not False
    ):
        raise ValueError("The SmolVLA tiny-smoke acceptance report is invalid.")


def _model_root(experiment: dict[str, Any]) -> Path:
    model = experiment["model"]
    root = _absolute_root("ROSETTA_MODELS_ROOT")
    path = root / str(model["identifier"]).replace("/", "--") / str(model["revision"])
    manifest = _load_json(path / "model_manifest.json")
    if (
        manifest.get("status") != "validated"
        or manifest.get("repo_id") != model["identifier"]
        or manifest.get("revision") != model["revision"]
    ):
        raise ValueError("Local SmolVLA snapshot manifest differs from the experiment.")
    dependency = model["vlm_dependency"]
    dependency_manifest = _load_json(path / str(dependency["manifest"]))
    if (
        dependency_manifest.get("status") != "validated"
        or dependency_manifest.get("repo_id") != dependency["identifier"]
        or dependency_manifest.get("revision") != dependency["revision"]
        or dependency_manifest.get("license") != dependency["license"]
    ):
        raise ValueError("Local VLM dependency manifest differs from the experiment.")
    namespace, name = str(dependency["identifier"]).split("/", maxsplit=1)
    reference = root / "hf_home" / "hub" / f"models--{namespace}--{name}" / "refs" / "main"
    if (
        not reference.is_file()
        or reference.read_text(encoding="utf-8").strip() != dependency["revision"]
    ):
        raise ValueError("The offline VLM dependency reference is not revision-pinned.")
    return path


def _dataset_root(experiment: dict[str, Any]) -> Path:
    dataset = experiment["dataset"]
    root = _absolute_root("ROSETTA_DATA_ROOT")
    path = root / str(dataset["identifier"]).replace("/", "--") / str(dataset["revision"])
    manifest = _load_json(path / "manifest.json")
    if (
        manifest.get("repo_id") != dataset["identifier"]
        or manifest.get("resolved_revision") != dataset["revision"]
    ):
        raise ValueError("Local dataset manifest differs from the experiment.")
    return path


def _phase_arguments(
    experiment: dict[str, Any],
    phase: str,
    run_name: str,
    model_root: Path,
    dataset_root: Path,
    output_dir: Path,
) -> list[str]:
    phase_config = experiment["phases"]["smoke" if phase == "preflight" else phase]
    if phase == "formal" and phase_config.get("stop_until_separately_preregistered") is True:
        raise RuntimeError("Formal training remains blocked until a separate resource review.")
    episodes = [int(value) for value in phase_config["episodes"]]
    test_episodes = {int(value) for value in experiment["dataset"]["test_episodes"]}
    if set(episodes) & test_episodes:
        raise ValueError("A training phase attempted to load hidden-test episodes.")
    policy = experiment["model"]["policy"]
    adaptation = experiment["model"]["adaptation"]
    device = os.environ.get("ROSETTA_TORCH_DEVICE")
    if not device:
        raise ValueError("ROSETTA_TORCH_DEVICE must be set by the Docker runner.")
    mixed_precision = str(experiment["resources"]["mixed_precision"])
    save_checkpoint = bool(phase_config.get("save_checkpoint", True))
    log_freq = int(phase_config.get("log_freq", 1))
    num_workers = int(phase_config.get("num_workers", 0))
    persistent_workers = bool(phase_config.get("persistent_workers", False))
    if log_freq <= 0:
        raise ValueError("SmolVLA phases require a positive log frequency.")
    if num_workers < 0 or (num_workers == 0 and persistent_workers):
        raise ValueError("Persistent workers require a positive worker count.")
    arguments = [
        f"--policy.path={model_root}",
        f"--policy.pretrained_revision={experiment['model']['revision']}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={policy['chunk_size']}",
        f"--policy.n_action_steps={policy['n_action_steps']}",
        f"--policy.empty_cameras={policy['empty_cameras']}",
        f"--policy.load_vlm_weights={str(policy['load_vlm_weights']).lower()}",
        f"--policy.freeze_vision_encoder={str(adaptation['freeze_vision_encoder']).lower()}",
        f"--policy.train_expert_only={str(adaptation['train_expert_only']).lower()}",
        f"--policy.train_state_proj={str(adaptation['train_state_proj']).lower()}",
        f"--dataset.repo_id={experiment['dataset']['identifier']}",
        f"--dataset.root={dataset_root}",
        f"--dataset.revision={experiment['dataset']['revision']}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        "--dataset.eval_split=0.0",
        f"--rename_map={json.dumps(experiment['dataset']['rename_map'], separators=(',', ':'))}",
        f"--output_dir={output_dir}",
        f"--job_name={run_name}",
        f"--seed={experiment['seed']}",
        f"--batch_size={phase_config['batch_size']}",
        f"--steps={phase_config['steps']}",
        f"--save_freq={phase_config['save_freq']}",
        f"--save_checkpoint={str(save_checkpoint).lower()}",
        "--save_checkpoint_to_hub=false",
        f"--log_freq={log_freq}",
        "--eval_steps=0",
        "--env_eval_freq=0",
        f"--num_workers={num_workers}",
        f"--persistent_workers={str(persistent_workers).lower()}",
        "--dataloader_multiprocessing_context=null",
        f"--accelerator.mixed_precision={mixed_precision}",
        "--accelerator.gradient_accumulation.steps=1",
        "--wandb.enable=true",
        "--wandb.disable_artifact=true",
        f"--wandb.project={experiment['tracking']['project']}",
    ]
    if "compile_model" in policy:
        arguments.append(
            f"--policy.compile_model={str(bool(policy['compile_model'])).lower()}"
        )
    if policy.get("compile_model"):
        compile_mode = str(policy.get("compile_mode", "default"))
        if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("Unsupported torch.compile mode for SmolVLA.")
        arguments.append(f"--policy.compile_mode={compile_mode}")
    if num_workers > 0:
        prefetch_factor = int(phase_config.get("prefetch_factor", 2))
        if prefetch_factor <= 0:
            raise ValueError("SmolVLA prefetch_factor must be positive.")
        arguments.append(f"--prefetch_factor={prefetch_factor}")
    return arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "smoke", "overfit", "formal"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--gate1-report", type=Path, required=True)
    parser.add_argument("--gate2-report", type=Path, required=True)
    parser.add_argument("--trackio-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--smoke-acceptance-report", type=Path)
    args = parser.parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("--run-name must be a lower-case path-safe identifier.")
    config_path = args.config.resolve()
    experiment = _load_yaml(config_path)
    resources = experiment["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT") != resources["memory_swap_limit"]
    ):
        raise ValueError("The active Docker memory limits differ from the preregistered budget.")
    contract_path = REPOSITORY_ROOT / experiment["action_contract"]["derived"]
    contract_sha256 = file_sha256(contract_path)
    _validate_benchmark(args.benchmark_report.resolve(), experiment, config_path, contract_sha256)
    _validate_gate(
        args.gate1_report.resolve(),
        expected_gate="m2_gate_1_scripted_action",
        experiment_id=experiment["experiment_id"],
        contract_sha256=contract_sha256,
        dataset_revision=experiment["dataset"]["revision"],
        allowed_replay_episodes=[
            *experiment["dataset"]["train_episodes"],
            *experiment["dataset"]["validation_episodes"],
        ],
    )
    _validate_gate(
        args.gate2_report.resolve(),
        expected_gate="m2_gate_2_dataset_action_replay",
        experiment_id=experiment["experiment_id"],
        contract_sha256=contract_sha256,
        dataset_revision=experiment["dataset"]["revision"],
        allowed_replay_episodes=[
            *experiment["dataset"]["train_episodes"],
            *experiment["dataset"]["validation_episodes"],
        ],
    )
    _validate_tracking(args.trackio_report.resolve(), experiment)
    if args.phase != "preflight":
        if args.preflight_report is None:
            raise ValueError("Optimizer phases require --preflight-report.")
        _validate_preflight(
            args.preflight_report.resolve(),
            experiment,
            config_path,
            contract_sha256,
        )
    if args.phase in {"overfit", "formal"}:
        if args.smoke_acceptance_report is None:
            raise ValueError("Overfit and formal phases require --smoke-acceptance-report.")
        _validate_smoke_acceptance(
            args.smoke_acceptance_report.resolve(),
            experiment,
            config_path,
            contract_sha256,
        )
    model_root = _model_root(experiment)
    dataset_root = _dataset_root(experiment)
    checkpoint_root = _absolute_root("ROSETTA_CHECKPOINT_ROOT")
    output_dir = checkpoint_root / str(experiment["experiment_id"]) / args.phase / args.run_name
    if output_dir.exists():
        raise FileExistsError("The requested phase output already exists; choose a new run name.")
    os.environ["ROSETTA_VLA_PHASE"] = args.phase
    os.environ["ROSETTA_VLA_EXPERIMENT_CONFIG"] = str(config_path)
    os.environ["ROSETTA_VLA_RUN_NAME"] = args.run_name
    sys.argv = [
        "lerobot-train",
        *_phase_arguments(
            experiment,
            args.phase,
            args.run_name,
            model_root,
            dataset_root,
            output_dir,
        ),
    ]
    if args.phase == "preflight":
        from smolvla_forward_check import main as preflight_main

        return preflight_main()
    from train_smolvla_trackio import main as train_main

    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
