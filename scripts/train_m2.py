"""Run the guarded overfit and resumable full-training stages for M2."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "m2_qwen08b_frozen_001.yaml"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.data.normalization import (  # noqa: E402
    DatasetStatistics,
    normalize,
)
from rosetta_reality.eval import action_metrics  # noqa: E402
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    load_experiment_config,
    stable_hash,
)
from rosetta_reality.features import (  # noqa: E402
    CachedFeatureDataset,
    create_json,
    load_feature_manifest,
    save_tensor_shard,
)
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.train.losses import (  # noqa: E402
    globally_normalized_scoped_first_action_loss,
    smooth_l1_action_loss,
    smooth_l1_action_loss_per_sample,
)
from rosetta_reality.train.m2 import (  # noqa: E402
    build_cached_policy,
    normalized_batch,
    predict_denormalized,
)


def _environment_root(name: str, default: str) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else REPOSITORY_ROOT / default


def _training_device(context: dict[str, Any]) -> torch.device:
    configured = str(context["experiment"]["resources"].get("training_device", "cpu"))
    device = torch.device(configured)
    if device.type == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("Experiment requires Intel XPU training, but XPU is unavailable.")
    if device.type not in {"cpu", "xpu"}:
        raise ValueError(f"Unsupported M2 training device: {device.type}.")
    return device


def _runtime_device_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "type": device.type,
        "torch_version": torch.__version__,
    }
    if device.type == "xpu":
        index = 0 if device.index is None else device.index
        report.update(
            {
                "index": index,
                "name": torch.xpu.get_device_name(index),
                "device_count": torch.xpu.device_count(),
            }
        )
    return report


def _training_state_with_noise(
    state: torch.Tensor,
    configured: dict[str, Any],
) -> torch.Tensor:
    """Apply the declared normalized-state jitter on training inputs only."""

    standard_deviation = float(configured.get("state_noise_std_normalized", 0.0))
    if not math.isfinite(standard_deviation) or not 0.0 <= standard_deviation < 1.0:
        raise ValueError("state_noise_std_normalized must be finite and in [0, 1).")
    if standard_deviation == 0.0:
        return state
    noisy = state + torch.randn_like(state) * standard_deviation
    if not bool(torch.isfinite(noisy).all()):
        raise FloatingPointError("Training state noise produced NaN or Inf.")
    return noisy


def _configured_action_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    configured: dict[str, Any],
) -> torch.Tensor:
    """Apply the experiment-bound action-loss protocol on every training path."""

    return smooth_l1_action_loss(
        predicted_actions,
        target_actions,
        first_action_weight=float(configured.get("first_action_loss_weight", 0.0)),
    )


def _model_action_loss(
    model: torch.nn.Module,
    observations: dict[str, torch.Tensor],
    state: torch.Tensor,
    target: torch.Tensor,
    configured: dict[str, Any],
    *,
    state_pairing: dict[str, Any] | None = None,
    paired_state: torch.Tensor | None = None,
    pairing_mask: torch.Tensor | None = None,
    early_phase_first_action: dict[str, Any] | None = None,
    frame_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the recorded branch and any one declared globally normalized branch."""

    prediction = model(
        observations,
        _training_state_with_noise(state, configured),
    )
    recorded_loss = _configured_action_loss(prediction, target, configured)
    if state_pairing is None:
        if paired_state is not None or pairing_mask is not None:
            raise ValueError("Unconfigured state-pairing tensors reached the training path.")
        combined_loss = recorded_loss
    else:
        if paired_state is None or pairing_mask is None:
            raise ValueError("Configured state pairing is missing from a training batch.")
        if (
            paired_state.shape != state.shape
            or pairing_mask.shape != (state.shape[0],)
            or pairing_mask.dtype != torch.bool
        ):
            raise ValueError("State-pairing batch shape differs from the recorded batch.")
        selected = pairing_mask.nonzero(as_tuple=False).flatten()
        paired_component = recorded_loss.new_zeros(())
        if selected.numel() > 0:
            paired_observations = {
                name: value.index_select(0, selected)
                for name, value in observations.items()
            }
            paired_prediction = model(
                paired_observations,
                _training_state_with_noise(
                    paired_state.index_select(0, selected), configured
                ),
            )
            paired_values = smooth_l1_action_loss_per_sample(
                paired_prediction,
                target.index_select(0, selected),
            )
            paired_component = (
                float(state_pairing["pairing_scale"])
                * paired_values.sum()
                / state.shape[0]
            )
        weight = float(state_pairing["weight"])
        combined_loss = (recorded_loss + weight * paired_component) / (1.0 + weight)

    if early_phase_first_action is None:
        return prediction, combined_loss
    if (
        state_pairing is not None
        or float(configured.get("first_action_loss_weight", 0.0)) != 0.0
        or float(configured.get("state_noise_std_normalized", 0.0)) != 0.0
    ):
        raise ValueError(
            "Early-phase first-action loss cannot run with another training objective."
        )
    if (
        not isinstance(frame_indices, torch.Tensor)
        or frame_indices.shape != (state.shape[0],)
        or frame_indices.dtype != torch.long
    ):
        raise ValueError("Early-phase first-action loss requires long frame indices per row.")
    maximum_frame = int(early_phase_first_action["maximum_frame_index_exclusive"])
    scope_mask = (frame_indices >= 0) & (frame_indices < maximum_frame)
    scoped_component = globally_normalized_scoped_first_action_loss(
        prediction,
        target,
        scope_mask,
        global_scale=float(early_phase_first_action["global_scale"]),
    )
    weight = float(early_phase_first_action["weight"])
    return prediction, (combined_loss + weight * scoped_component) / (1.0 + weight)


def _normalized_pairing_batch(
    batch: dict[str, torch.Tensor],
    statistics: DatasetStatistics,
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    paired = batch.get("paired_robot_state")
    mask = batch.get("pairing_mask")
    if paired is None and mask is None:
        return None, None
    if not isinstance(paired, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("State-pairing batch fields must both be tensors.")
    normalized = normalize(
        paired.to(device=device, dtype=torch.float32), statistics.state
    )
    return normalized, mask.to(device=device, dtype=torch.bool)


def _action_loss_protocol(
    configured: dict[str, Any],
    early_phase_first_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    weight = float(configured.get("first_action_loss_weight", 0.0))
    if early_phase_first_action is not None:
        scoped_weight = float(early_phase_first_action["weight"])
        return {
            "type": (
                "normalized_full_chunk_plus_globally_normalized_early_phase_"
                "first_action_smooth_l1"
            ),
            "first_action_weight": weight,
            "normalization_denominator": 1.0 + scoped_weight,
            "execution_alignment": "receding_horizon_first_action",
            "early_phase_first_action": dict(early_phase_first_action),
        }
    return {
        "type": (
            "full_chunk_smooth_l1"
            if weight == 0.0
            else "normalized_full_chunk_plus_first_action_smooth_l1"
        ),
        "first_action_weight": weight,
        "normalization_denominator": 1.0 + weight,
        "execution_alignment": "receding_horizon_first_action",
    }


def _to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def _optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _accelerator_rng_state(device: torch.device) -> dict[str, Any] | None:
    if device.type != "xpu":
        return None
    return {
        "type": "xpu",
        "states": [state.cpu() for state in torch.xpu.get_rng_state_all()],
    }


def _restore_accelerator_rng_state(value: Any, device: torch.device) -> None:
    if value is None and device.type == "cpu":
        return
    if (
        device.type != "xpu"
        or not isinstance(value, dict)
        or value.get("type") != "xpu"
        or not isinstance(value.get("states"), list)
    ):
        raise ValueError("Checkpoint accelerator RNG state differs from the training device.")
    torch.xpu.set_rng_state_all(value["states"])


def _checkpoint_due(
    epoch: int,
    *,
    every_epochs: int,
    improved: bool,
    terminal_epoch: bool,
) -> bool:
    """Keep periodic, best, and resumable terminal checkpoints."""

    if every_epochs <= 0:
        raise ValueError("checkpoint_every_epochs must be positive.")
    return improved or terminal_epoch or epoch % every_epochs == 0


def _feature_manifest(experiment_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    root = _environment_root("ROSETTA_FEATURE_ROOT", "feature_cache") / experiment_id
    candidates = sorted(root.glob("*/manifest.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one feature manifest for {experiment_id!r}, received {len(candidates)}."
        )
    return candidates[0]


def _statistics(manifest_path: Path, manifest: dict[str, Any]) -> DatasetStatistics:
    path = manifest_path.parent / manifest["normalization_path"]
    if file_sha256(path) != manifest["normalization_sha256"]:
        raise ValueError("Normalization checksum mismatch.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("source_split") != "train":
        raise ValueError("Training requires train-only normalization statistics.")
    return DatasetStatistics.from_dict(value["statistics"])


class _PairedStateDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose an immutable train-only simulator-state join alongside cached rows."""

    def __init__(
        self,
        base: CachedFeatureDataset,
        paired_robot_state: torch.Tensor,
        pairing_mask: torch.Tensor,
    ) -> None:
        self.base = base
        self.features = base.features
        self.robot_state = base.robot_state
        self.actions = base.actions
        self.episode_ids = base.episode_ids
        self.frame_indices = base.frame_indices
        self.paired_robot_state = paired_robot_state.to(torch.float32)
        self.pairing_mask = pairing_mask.to(torch.bool)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[index])
        item["paired_robot_state"] = self.paired_robot_state[index]
        item["pairing_mask"] = self.pairing_mask[index]
        return item


def _state_pairing_config(experiment: dict[str, Any]) -> dict[str, Any] | None:
    raw = experiment["training"].get("aligned_expert_replay_state_pairing")
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("enabled") is not True:
        raise ValueError("aligned_expert_replay_state_pairing must be an enabled mapping.")
    weight = float(raw.get("weight", -1.0))
    if weight != 1.0:
        raise ValueError("The v010 aligned state-pairing weight is fixed at 1.0.")
    relative = Path(str(raw.get("manifest", "")))
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[0] in {"runs", "checkpoints", "artifacts"}
    ):
        raise ValueError("State-pairing manifest must be relative to the experiment run root.")
    return raw


def _early_phase_first_action_protocol(
    experiment: dict[str, Any],
    train: CachedFeatureDataset,
) -> dict[str, Any] | None:
    """Bind a declared early-phase objective to exact immutable train keys."""

    raw = experiment["training"].get("early_phase_first_action_loss")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("early_phase_first_action_loss must be a mapping.")
    maximum_frame = int(raw["maximum_frame_index_exclusive"])
    expected_selected = int(raw["expected_selected_train_samples"])
    weight = float(raw["weight"])
    stride = int(experiment["dataset"]["frame_stride"])
    if weight != 1.0 or maximum_frame <= 0 or stride <= 0 or expected_selected <= 0:
        raise ValueError("Early-phase loss declaration is outside the fixed safe protocol.")
    expected_frames = list(range(0, maximum_frame, stride))
    train_episodes = [int(value) for value in experiment["dataset"]["split"]["train"]]
    if not expected_frames or len(set(train_episodes)) != len(train_episodes):
        raise ValueError("Early-phase loss has an empty frame scope or duplicate episodes.")
    selected = (train.frame_indices >= 0) & (train.frame_indices < maximum_frame)
    selected_count = int(selected.sum())
    if (
        selected_count != expected_selected
        or selected_count != len(train_episodes) * len(expected_frames)
    ):
        raise ValueError("Early-phase train-sample count differs from the declaration.")
    for episode in train_episodes:
        episode_frames = train.frame_indices[
            selected & train.episode_ids.eq(episode)
        ].tolist()
        if sorted(int(value) for value in episode_frames) != expected_frames:
            raise ValueError(
                "Early-phase frame keys do not exactly cover every train episode."
            )
    return {
        "schema_version": 1,
        "type": "globally_normalized_early_phase_first_action_smooth_l1_v1",
        "weight": weight,
        "minimum_frame_index_inclusive": 0,
        "maximum_frame_index_exclusive": maximum_frame,
        "selected_train_samples": selected_count,
        "total_train_samples": len(train),
        "global_scale": len(train) / selected_count,
        "selected_frames_per_episode": expected_frames,
        "selected_split": "train",
        "validation_uses_scoped_loss": False,
        "hidden_test_opened": False,
    }


def _load_state_pairing(
    *,
    experiment: dict[str, Any],
    config_path: Path,
    feature_manifest_path: Path,
    feature_manifest: dict[str, Any],
    train: CachedFeatureDataset,
    contract_path: Path,
) -> tuple[Dataset[dict[str, torch.Tensor]], dict[str, Any] | None]:
    configured = _state_pairing_config(experiment)
    if configured is None:
        return train, None
    relative = Path(str(configured["manifest"]))
    path = (
        _environment_root("ROSETTA_RUN_ROOT", "runs")
        / experiment["experiment_id"]
        / relative
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    json.dumps(manifest, allow_nan=False)
    train_episodes = [int(value) for value in experiment["dataset"]["split"]["train"]]
    scope = manifest.get("scope", {})
    protocol = manifest.get("protocol", {})
    expected_identity = {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "feature_cache_identity": feature_manifest["identity_hash"],
        "feature_manifest_sha256": file_sha256(feature_manifest_path),
        "dataset_revision": feature_manifest["identity"]["dataset"]["revision"],
        "dataset_manifest_sha256": feature_manifest["identity"]["dataset"][
            "manifest_sha256"
        ],
        "action_contract_sha256": file_sha256(contract_path),
        "scope": scope,
        "protocol": protocol,
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("identity") != expected_identity
        or manifest.get("identity_hash") != stable_hash(expected_identity)
        or scope.get("split") != "train"
        or scope.get("episodes") != train_episodes
        or scope.get("validation_split_opened") is not False
        or scope.get("test_split_opened") is not False
    ):
        raise ValueError("State-pairing manifest identity or train-only scope differs.")
    if (
        protocol.get("type") != "aligned_expert_replay_simulator_state_pairing_v1"
        or protocol.get("label_type") != "time_indexed_expert_reference"
        or protocol.get("state_conditioned") is not False
        or protocol.get("recovery_oracle") is not False
        or protocol.get("pre_action_state") is not True
        or int(protocol.get("candidate_seed_start", -1)) != 0
        or int(protocol.get("candidate_seed_count", -1)) != 256
        or int(protocol.get("top_k", -1)) != 5
        or float(protocol.get("maximum_pooled_4x4_mae", -1.0)) != 0.005
        or float(protocol.get("maximum_recorded_state_mae", -1.0)) != 0.025
    ):
        raise ValueError("State-pairing protocol differs from the pre-registered v010 rule.")
    tensor_record = manifest.get("tensor", {})
    tensor_relative = Path(str(tensor_record.get("path", "")))
    if (
        tensor_relative.is_absolute()
        or ".." in tensor_relative.parts
        or tensor_relative.name != "paired-states.pt"
    ):
        raise ValueError("State-pairing tensor path is unsafe or unexpected.")
    tensor_path = path.parent / tensor_relative
    if file_sha256(tensor_path) != tensor_record.get("sha256"):
        raise ValueError("State-pairing tensor checksum differs from the manifest.")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("identity_hash") != manifest.get("identity_hash")
        or payload.get("feature_cache_identity") != feature_manifest["identity_hash"]
    ):
        raise ValueError("State-pairing tensor identity differs from the manifest.")
    paired = payload.get("paired_robot_state")
    mask = payload.get("pairing_mask")
    episodes = payload.get("episode_ids")
    frames = payload.get("frame_indices")
    if (
        not isinstance(paired, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or not isinstance(episodes, torch.Tensor)
        or not isinstance(frames, torch.Tensor)
        or paired.shape != train.robot_state.shape
        or mask.shape != (len(train),)
        or mask.dtype != torch.bool
        or not torch.equal(episodes.to(torch.long), train.episode_ids)
        or not torch.equal(frames.to(torch.long), train.frame_indices)
        or not bool(torch.isfinite(paired).all())
    ):
        raise ValueError("State-pairing tensor shape, keys, or finite contract differs.")
    paired_count = int(mask.sum())
    declared_samples = manifest.get("samples", {})
    minimum = int(protocol.get("minimum_paired_samples", -1))
    if (
        int(declared_samples.get("total", -1)) != len(train)
        or int(declared_samples.get("paired", -1)) != paired_count
        or int(declared_samples.get("unpaired", -1)) != len(train) - paired_count
        or paired_count < minimum
        or minimum != 1980
        or not torch.equal(paired[~mask], train.robot_state[~mask])
    ):
        raise ValueError("State-pairing coverage or recorded-state fallback differs.")
    eligible_key_digest = stable_hash(
        [
            [int(episode), int(frame)]
            for episode, frame, selected in zip(episodes, frames, mask)
            if bool(selected)
        ]
    )
    if eligible_key_digest != protocol.get("eligible_key_sha256"):
        raise ValueError("State-pairing eligible-key digest differs.")
    episode_reports = manifest.get("episodes")
    if (
        not isinstance(episode_reports, list)
        or any(not isinstance(report, dict) for report in episode_reports)
        or [report.get("episode") for report in episode_reports]
        != train_episodes
    ):
        raise ValueError("State-pairing episode reports do not exactly cover train scope.")
    reported_pairs = 0
    chunk_length = int(protocol.get("action_chunk_length", -1))
    frame_stride = int(protocol.get("frame_stride", -1))
    if chunk_length != train.actions.shape[1] or frame_stride != int(
        experiment["dataset"]["frame_stride"]
    ):
        raise ValueError("State-pairing frame or chunk protocol differs from the cache.")
    for report in episode_reports:
        episode = int(report["episode"])
        episode_selection = episodes.eq(episode)
        source_frames = frames[episode_selection].to(torch.long)
        selected_frames = frames[episode_selection & mask].to(torch.long)
        declared_frames = report.get("eligible_frame_indices")
        paired_anchors = int(report.get("paired_anchor_count", -1))
        exclusive_stop = int(report.get("exclusive_valid_step_stop", -1))
        if (
            int(report.get("source_anchor_count", -1)) != source_frames.numel()
            or paired_anchors <= 0
            or paired_anchors != selected_frames.numel()
            or declared_frames != selected_frames.tolist()
            or not torch.equal(selected_frames, source_frames[:paired_anchors])
            or any(
                current - previous != frame_stride
                for previous, current in zip(
                    declared_frames[:-1], declared_frames[1:], strict=True
                )
            )
            or int(selected_frames[-1]) + chunk_length > exclusive_stop
        ):
            raise ValueError("State-pairing episode coverage is not the declared prefix rule.")
        reported_pairs += paired_anchors
    if reported_pairs != paired_count:
        raise ValueError("State-pairing per-episode counts differ from global coverage.")
    info = {
        "identity_hash": manifest["identity_hash"],
        "manifest_sha256": file_sha256(path),
        "tensor_sha256": tensor_record["sha256"],
        "paired_samples": paired_count,
        "total_samples": len(train),
        "pairing_scale": len(train) / paired_count,
        "weight": float(configured["weight"]),
        "eligible_key_sha256": eligible_key_digest,
    }
    return _PairedStateDataset(train, paired, mask), info


def _load_context(config_path: Path, feature_path: Path | None) -> dict[str, Any]:
    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    manifest_path = _feature_manifest(experiment["experiment_id"], feature_path)
    manifest = load_feature_manifest(manifest_path)
    identity = manifest.get("identity")
    if (
        not isinstance(identity, dict)
        or manifest.get("identity_hash") != stable_hash(identity)
        or identity.get("experiment_id") != experiment["experiment_id"]
        or identity.get("experiment_config_sha256") != file_sha256(config_path)
    ):
        raise ValueError("Feature cache and experiment configuration identities differ.")
    if (
        _state_pairing_config(experiment) is not None
        or experiment["training"].get("early_phase_first_action_loss") is not None
    ):
        derivation = manifest.get("derivation")
        if (
            not isinstance(derivation, dict)
            or derivation.get("type") != "verified_visible_identity_rebind_v1"
            or derivation.get("materialized_splits") != ["train", "validation"]
            or derivation.get("withheld_splits") != ["test"]
            or derivation.get("hidden_test_loaded") is not False
            or manifest.get("hidden_test_loaded") is not False
            or manifest.get("hidden_test_materialized") is not False
            or manifest.get("materialized_splits") != ["train", "validation"]
            or manifest.get("withheld_splits") != ["test"]
            or manifest.get("samples", {}).get("test") != 0
            or manifest.get("shards", {}).get("test") != []
        ):
            raise ValueError(
                "The configured train-only auxiliary objective requires the exact "
                "train/validation-only feature cache with hidden test withheld."
            )
    train = CachedFeatureDataset(manifest_path, "train")
    validation = CachedFeatureDataset(manifest_path, "validation")
    statistics = _statistics(manifest_path, manifest)
    contract_path = REPOSITORY_ROOT / experiment["action_contract"]
    contract = load_action_contract(contract_path)
    for split_name, dataset in (("train", train), ("validation", validation)):
        _, violations = contract.clip(dataset.actions)
        if bool(violations.any()):
            raise ValueError(
                f"{split_name} targets were not transformed into the Action Contract."
            )
    early_phase_first_action = _early_phase_first_action_protocol(experiment, train)
    training_dataset, state_pairing = _load_state_pairing(
        experiment=experiment,
        config_path=config_path,
        feature_manifest_path=manifest_path,
        feature_manifest=manifest,
        train=train,
        contract_path=contract_path,
    )
    return {
        "experiment": experiment,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "train": training_dataset,
        "validation": validation,
        "statistics": statistics,
        "contract": contract,
        "contract_path": contract_path,
        "config_path": config_path,
        "state_pairing": state_pairing,
        "early_phase_first_action": early_phase_first_action,
        "action_loss_protocol": _action_loss_protocol(
            experiment["training"], early_phase_first_action
        ),
    }


def _matching_report(paths: list[Path], predicate: Any, description: str) -> Path:
    matches = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") in ("passed", "complete") and predicate(value):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"Required passed {description} report was not found.")
    return sorted(matches)[-1]


def _prerequisites(
    context: dict[str, Any],
    *,
    require_smoke: bool,
    require_overfit: bool,
) -> dict[str, Path]:
    experiment = context["experiment"]
    identity = context["manifest"]["identity_hash"]
    root = _environment_root("ROSETTA_RUN_ROOT", "runs") / experiment["experiment_id"]
    benchmark_path = root / "benchmark" / f"pre-training-{identity[:16]}.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError("Pre-training benchmark must complete before any optimization.")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (
        benchmark.get("stage") != "pre_training"
        or benchmark.get("feature_cache_identity") != identity
        or benchmark.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Pre-training benchmark identity or hidden-test boundary is invalid.")

    contract_hash = file_sha256(context["contract_path"])
    gate_root = root / "gates"
    gate1 = _matching_report(
        list(gate_root.glob("gate1-*.json")),
        lambda value: value.get("action_contract_sha256") == contract_hash,
        "Gate 1",
    )
    dataset_identity = context["manifest"]["identity"]["dataset"]
    gate2 = _matching_report(
        list(gate_root.glob("gate2-*.json")),
        lambda value: (
            value.get("action_contract_sha256") == contract_hash
            and value.get("dataset_revision") == dataset_identity["revision"]
            and value.get("dataset_manifest_sha256") == dataset_identity["manifest_sha256"]
        ),
        "Gate 2",
    )
    reports = {"benchmark": benchmark_path, "gate1": gate1, "gate2": gate2}
    if require_smoke:
        smoke_path = root / "smoke" / f"optimizer-{identity[:16]}.json"
        if not smoke_path.is_file():
            raise FileNotFoundError("One-step optimizer smoke must pass before overfit.")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if (
            smoke.get("status") != "passed"
            or smoke.get("feature_cache_identity") != identity
            or smoke.get("action_loss_protocol") != context["action_loss_protocol"]
            or smoke.get("state_pairing") != context["state_pairing"]
        ):
            raise ValueError("Optimizer smoke report is not a matching passed gate.")
        reports["optimizer_smoke"] = smoke_path
    if require_overfit:
        overfit_path = root / "overfit" / f"overfit-{identity[:16]}.json"
        if not overfit_path.is_file():
            raise FileNotFoundError("Small-data overfit gate must pass before full training.")
        overfit = json.loads(overfit_path.read_text(encoding="utf-8"))
        if (
            overfit.get("status") != "passed"
            or overfit.get("feature_cache_identity") != identity
            or overfit.get("action_loss_protocol") != context["action_loss_protocol"]
            or overfit.get("state_pairing") != context["state_pairing"]
        ):
            raise ValueError("Small-data overfit report is not a matching passed gate.")
        reports["overfit"] = overfit_path
    return reports


def _new_model(context: dict[str, Any]) -> torch.nn.Module:
    train = context["train"]
    return build_cached_policy(
        context["experiment"],
        feature_dim=train.features.shape[-1],
        state_dim=train.robot_state.shape[-1],
        action_dim=train.actions.shape[-1],
        chunk_size=train.actions.shape[-2],
        statistics=context["statistics"],
    )


def optimizer_smoke(context: dict[str, Any]) -> Path:
    """Run exactly one guarded optimizer step after the immutable benchmark."""

    prerequisites = _prerequisites(
        context,
        require_smoke=False,
        require_overfit=False,
    )
    configured = context["experiment"]["training"]
    seed = int(configured["seed"])
    torch.manual_seed(seed)
    device = _training_device(context)
    if device.type == "xpu":
        torch.xpu.manual_seed_all(seed)
    model = _new_model(context).to(device)
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("Cached frozen backbone unexpectedly has trainable parameters.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configured["learning_rate"]),
        weight_decay=float(configured["weight_decay"]),
        foreach=False if device.type == "xpu" else None,
    )
    raw_batch = context["train"][0]
    batch = {
        key: value.unsqueeze(0)
        for key, value in raw_batch.items()
        if key
        in (
            "features",
            "robot_state",
            "actions",
            "frame_index",
            "paired_robot_state",
            "pairing_mask",
        )
    }
    observations, state, target = normalized_batch(
        batch,
        context["statistics"],
        device=device,
    )
    paired_state, pairing_mask = _normalized_pairing_batch(
        batch, context["statistics"], device=device
    )
    frame_indices = batch["frame_index"].to(device=device, dtype=torch.long)
    if context["state_pairing"] is not None and not bool(pairing_mask.item()):
        raise RuntimeError("Optimizer smoke must exercise an eligible paired-state row.")
    optimizer.zero_grad(set_to_none=True)
    prediction, loss = _model_action_loss(
        model,
        observations,
        state,
        target,
        configured,
        state_pairing=context["state_pairing"],
        paired_state=paired_state,
        pairing_mask=pairing_mask,
        early_phase_first_action=context["early_phase_first_action"],
        frame_indices=frame_indices,
    )
    if prediction.shape != target.shape or prediction.shape != (
        1,
        context["contract"].chunk_length,
        context["contract"].dimension,
    ):
        raise RuntimeError("Optimizer smoke prediction violates the M2 action shape contract.")
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Optimizer smoke loss is non-finite.")
    loss.backward()
    gradient_l2: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"Optimizer smoke gradient is non-finite: {name}.")
        component = name.split(".", maxsplit=1)[0]
        gradient_l2[component] = gradient_l2.get(component, 0.0) + float(
            parameter.grad.square().sum().cpu()
        )
    required_components = {"state_encoder", "fusion", "action_head"}
    if any(gradient_l2.get(component, 0.0) <= 0 for component in required_components):
        raise RuntimeError("Optimizer smoke did not reach every downstream trainable component.")
    clip_grad_norm_(
        model.parameters(),
        float(configured["gradient_clip_norm"]),
        foreach=False if device.type == "xpu" else None,
    )
    optimizer.step()
    with torch.no_grad():
        _, post_step_loss = _model_action_loss(
            model,
            observations,
            state,
            target,
            configured,
            state_pairing=context["state_pairing"],
            paired_state=paired_state,
            pairing_mask=pairing_mask,
            early_phase_first_action=context["early_phase_first_action"],
            frame_indices=frame_indices,
        )
    if not bool(torch.isfinite(post_step_loss)):
        raise FloatingPointError("Optimizer smoke post-step loss is non-finite.")

    report = {
        "schema_version": 1,
        "gate": "m2_one_step_optimizer_smoke",
        "status": "passed",
        "experiment_id": context["experiment"]["experiment_id"],
        "experiment_config_sha256": file_sha256(context["config_path"]),
        "feature_cache_identity": context["manifest"]["identity_hash"],
        "normalization_source_split": "train",
        "state_noise_std_normalized": float(
            configured.get("state_noise_std_normalized", 0.0)
        ),
        "action_loss_protocol": context["action_loss_protocol"],
        "state_pairing": context["state_pairing"],
        "sample_episode": int(raw_batch["episode_id"]),
        "sample_frame": int(raw_batch["frame_index"]),
        "steps": int(configured["optimizer_smoke_steps"]),
        "prediction_shape": list(prediction.shape),
        "initial_normalized_smooth_l1": float(loss.detach().cpu()),
        "post_step_normalized_smooth_l1": float(post_step_loss.cpu()),
        "gradient_l2_squared": gradient_l2,
        "runtime_device": _runtime_device_report(device),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "prerequisite_reports": {
            name: file_sha256(path) for name, path in prerequisites.items()
        },
    }
    destination = (
        _environment_root("ROSETTA_RUN_ROOT", "runs")
        / context["experiment"]["experiment_id"]
        / "smoke"
        / f"optimizer-{context['manifest']['identity_hash'][:16]}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return destination


def overfit(context: dict[str, Any]) -> Path:
    """Prove the downstream path can memorize a fixed tiny train-only slice."""

    prerequisites = _prerequisites(
        context,
        require_smoke=True,
        require_overfit=False,
    )
    configured = context["experiment"]["training"]
    sample_count = int(configured["overfit_samples"])
    if sample_count > len(context["train"]):
        raise ValueError("overfit_samples exceeds the training cache size.")
    seed = int(configured["seed"])
    torch.manual_seed(seed)
    device = _training_device(context)
    if device.type == "xpu":
        torch.xpu.manual_seed_all(seed)
    model = _new_model(context).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configured["learning_rate"]),
        weight_decay=float(configured["weight_decay"]),
        foreach=False if device.type == "xpu" else None,
    )
    batch = {
        "features": context["train"].features[:sample_count],
        "robot_state": context["train"].robot_state[:sample_count],
        "actions": context["train"].actions[:sample_count],
        "frame_index": context["train"].frame_indices[:sample_count],
    }
    if context["state_pairing"] is not None:
        batch.update(
            paired_robot_state=context["train"].paired_robot_state[:sample_count],
            pairing_mask=context["train"].pairing_mask[:sample_count],
        )
    observations, state, target = normalized_batch(
        batch,
        context["statistics"],
        device=device,
    )
    paired_state, pairing_mask = _normalized_pairing_batch(
        batch, context["statistics"], device=device
    )
    frame_indices = batch["frame_index"].to(device=device, dtype=torch.long)
    if context["state_pairing"] is not None and not bool(pairing_mask.any()):
        raise RuntimeError("Small-data overfit must include an eligible paired-state row.")
    model.train()
    with torch.no_grad():
        _, initial_value = _model_action_loss(
            model,
            observations,
            state,
            target,
            configured,
            state_pairing=context["state_pairing"],
            paired_state=paired_state,
            pairing_mask=pairing_mask,
            early_phase_first_action=context["early_phase_first_action"],
            frame_indices=frame_indices,
        )
        initial_loss = float(initial_value.cpu())
    losses: list[float] = []
    for _ in range(int(configured["overfit_steps"])):
        optimizer.zero_grad(set_to_none=True)
        prediction, loss = _model_action_loss(
            model,
            observations,
            state,
            target,
            configured,
            state_pairing=context["state_pairing"],
            paired_state=paired_state,
            pairing_mask=pairing_mask,
            early_phase_first_action=context["early_phase_first_action"],
            frame_indices=frame_indices,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Small-data overfit loss became non-finite.")
        loss.backward()
        clip_grad_norm_(
            model.parameters(),
            float(configured["gradient_clip_norm"]),
            foreach=False if device.type == "xpu" else None,
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        _, final_value = _model_action_loss(
            model,
            observations,
            state,
            target,
            configured,
            state_pairing=context["state_pairing"],
            paired_state=paired_state,
            pairing_mask=pairing_mask,
            early_phase_first_action=context["early_phase_first_action"],
            frame_indices=frame_indices,
        )
        final_loss = float(final_value.cpu())
    ratio = final_loss / initial_loss
    maximum_ratio = float(configured["overfit_maximum_loss_ratio"])
    passed = ratio <= maximum_ratio
    report = {
        "schema_version": 1,
        "gate": "m2_small_data_overfit",
        "status": "passed" if passed else "failed",
        "experiment_id": context["experiment"]["experiment_id"],
        "experiment_config_sha256": file_sha256(context["config_path"]),
        "feature_cache_identity": context["manifest"]["identity_hash"],
        "normalization_source_split": "train",
        "state_noise_std_normalized": float(
            configured.get("state_noise_std_normalized", 0.0)
        ),
        "action_loss_protocol": context["action_loss_protocol"],
        "state_pairing": context["state_pairing"],
        "samples": sample_count,
        "steps": int(configured["overfit_steps"]),
        "seed": seed,
        "initial_normalized_smooth_l1": initial_loss,
        "final_normalized_smooth_l1": final_loss,
        "last_noisy_normalized_smooth_l1": losses[-1],
        "loss_ratio": ratio,
        "maximum_loss_ratio": maximum_ratio,
        "runtime_device": _runtime_device_report(device),
        "prerequisite_reports": {
            name: file_sha256(path) for name, path in prerequisites.items()
        },
    }
    destination = (
        _environment_root("ROSETTA_RUN_ROOT", "runs")
        / context["experiment"]["experiment_id"]
        / "overfit"
        / f"overfit-{context['manifest']['identity_hash'][:16]}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Small-data overfit acceptance threshold was not reached.")
    return destination


def _evaluation(
    model: torch.nn.Module,
    dataset: CachedFeatureDataset,
    context: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=int(context["experiment"]["training"]["batch_size"]),
        shuffle=False,
    )
    predicted, target, raw_predicted = predict_denormalized(
        model,
        loader,
        context["statistics"],
        context["contract"],
        device=device,
    )
    return action_metrics(
        predicted,
        target,
        context["contract"],
        context["statistics"].action,
        raw_predicted=raw_predicted,
    )


def _checkpoint_payload(
    *,
    context: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_id: str,
    epoch: int,
    best_epoch: int,
    best_value: float,
    stale_epochs: int,
    train_loss: float,
    validation_metrics: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": context["experiment"]["experiment_id"],
        "experiment_config_sha256": file_sha256(context["config_path"]),
        "feature_cache_identity": context["manifest"]["identity_hash"],
        "run_id": run_id,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "stale_epochs": stale_epochs,
        "train_loss": train_loss,
        "validation_metrics": validation_metrics,
        "runtime_device": _runtime_device_report(device),
        "action_loss_protocol": context["action_loss_protocol"],
        "state_pairing": context["state_pairing"],
        "model_state": _to_cpu_tree(model.state_dict()),
        "optimizer_state": _to_cpu_tree(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state(),
        "accelerator_rng_state": _accelerator_rng_state(device),
        "model_contract": {
            "feature_dim": context["train"].features.shape[-1],
            "state_dim": context["train"].robot_state.shape[-1],
            "action_dim": context["train"].actions.shape[-1],
            "chunk_size": context["train"].actions.shape[-2],
            "prediction_parameterization": context["experiment"]["action_expert"].get(
                "prediction_parameterization", "absolute"
            ),
        },
    }


def train(
    context: dict[str, Any],
    *,
    run_id: str,
    resume: Path | None,
    stop_after_epoch: int | None,
) -> Path | None:
    """Train from scratch or explicitly resume a matching create-only checkpoint chain."""

    prerequisites = _prerequisites(
        context,
        require_smoke=True,
        require_overfit=True,
    )
    configured = context["experiment"]["training"]
    maximum_epochs = int(configured["maximum_epochs"])
    target_epoch = maximum_epochs if stop_after_epoch is None else stop_after_epoch
    if not 1 <= target_epoch <= maximum_epochs:
        raise ValueError("stop_after_epoch must be within the configured training horizon.")
    checkpoint_root = (
        _environment_root("ROSETTA_CHECKPOINT_ROOT", "checkpoints")
        / context["experiment"]["experiment_id"]
        / run_id
    )
    metric_root = (
        _environment_root("ROSETTA_RUN_ROOT", "runs")
        / context["experiment"]["experiment_id"]
        / "training"
        / run_id
    )
    seed = int(configured["seed"])
    torch.manual_seed(seed)
    device = _training_device(context)
    if device.type == "xpu":
        torch.xpu.manual_seed_all(seed)
    model = _new_model(context).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configured["learning_rate"]),
        weight_decay=float(configured["weight_decay"]),
        foreach=False if device.type == "xpu" else None,
    )
    start_epoch = 1
    best_epoch = 0
    best_value = float("inf")
    stale_epochs = 0
    if resume is None:
        checkpoint_root.mkdir(parents=True, exist_ok=False)
        metric_root.mkdir(parents=True, exist_ok=False)
    else:
        resume = resume.resolve()
        value = torch.load(resume, map_location="cpu", weights_only=False)
        if (
            value.get("run_id") != run_id
            or value.get("feature_cache_identity") != context["manifest"]["identity_hash"]
            or value.get("experiment_config_sha256") != file_sha256(context["config_path"])
            or value.get("action_loss_protocol") != context["action_loss_protocol"]
            or value.get("state_pairing") != context["state_pairing"]
            or resume.parent != checkpoint_root.resolve()
            or value.get("runtime_device", {}).get("type", "cpu") != device.type
        ):
            raise ValueError("Resume checkpoint identity, run, or location mismatch.")
        model.load_state_dict(value["model_state"], strict=True)
        optimizer.load_state_dict(value["optimizer_state"])
        _optimizer_state_to_device(optimizer, device)
        torch.set_rng_state(value["torch_rng_state"])
        _restore_accelerator_rng_state(value.get("accelerator_rng_state"), device)
        start_epoch = int(value["epoch"]) + 1
        best_epoch = int(value["best_epoch"])
        best_value = float(value["best_value"])
        stale_epochs = int(value["stale_epochs"])
    if start_epoch > target_epoch:
        raise ValueError("Resume checkpoint is already beyond the requested target epoch.")

    primary = context["experiment"]["evaluation"]["primary_metric"]
    completed_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, target_epoch + 1):
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            context["train"],
            batch_size=int(configured["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch in loader:
            observations, state, target = normalized_batch(
                batch,
                context["statistics"],
                device=device,
            )
            paired_state, pairing_mask = _normalized_pairing_batch(
                batch, context["statistics"], device=device
            )
            frame_indices = batch["frame_index"].to(
                device=device, dtype=torch.long
            )
            optimizer.zero_grad(set_to_none=True)
            prediction, loss = _model_action_loss(
                model,
                observations,
                state,
                target,
                configured,
                state_pairing=context["state_pairing"],
                paired_state=paired_state,
                pairing_mask=pairing_mask,
                early_phase_first_action=context["early_phase_first_action"],
                frame_indices=frame_indices,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Training loss became non-finite at epoch {epoch}.")
            loss.backward()
            clip_grad_norm_(
                model.parameters(),
                float(configured["gradient_clip_norm"]),
                foreach=False if device.type == "xpu" else None,
            )
            optimizer.step()
            count = target.shape[0]
            total_loss += float(loss.detach().cpu()) * count
            sample_count += count
        train_loss = total_loss / sample_count
        validation_metrics = _evaluation(model, context["validation"], context, device)
        value = float(validation_metrics[primary])
        improved = value < best_value
        if improved:
            best_value = value
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        epoch_report = {
            "schema_version": 1,
            "epoch": epoch,
            "train_normalized_smooth_l1": train_loss,
            "action_loss_protocol": context["action_loss_protocol"],
            "state_pairing": context["state_pairing"],
            "validation": validation_metrics,
            "best_epoch": best_epoch,
            "best_value": best_value,
            "stale_epochs": stale_epochs,
        }
        create_json(metric_root / f"epoch-{epoch:03d}.json", epoch_report)
        terminal_epoch = (
            epoch == target_epoch
            or stale_epochs >= int(configured["early_stopping_patience"])
        )
        if _checkpoint_due(
            epoch,
            every_epochs=int(configured["checkpoint_every_epochs"]),
            improved=improved,
            terminal_epoch=terminal_epoch,
        ):
            checkpoint_path = checkpoint_root / f"epoch-{epoch:03d}.pt"
            save_tensor_shard(
                checkpoint_path,
                _checkpoint_payload(
                    context=context,
                    model=model,
                    optimizer=optimizer,
                    run_id=run_id,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_value=best_value,
                    stale_epochs=stale_epochs,
                    train_loss=train_loss,
                    validation_metrics=validation_metrics,
                    device=device,
                ),
            )
        completed_epoch = epoch
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"validation_{primary}={value:.6f} best_epoch={best_epoch:03d}",
            flush=True,
        )
        if stale_epochs >= int(configured["early_stopping_patience"]):
            stopped_early = True
            break

    if completed_epoch < maximum_epochs and not stopped_early:
        print(f"Intentional partial stop after epoch {completed_epoch}; resume is required.")
        return None

    benchmark = json.loads(prerequisites["benchmark"].read_text(encoding="utf-8"))
    baseline_name = context["experiment"]["evaluation"]["compare_against"]
    baseline_value = float(benchmark["metrics"][baseline_name][primary])
    invalid_tolerance = float(
        context["experiment"]["evaluation"]["invalid_action_tolerance"]
    )
    best_checkpoint = torch.load(
        checkpoint_root / f"epoch-{best_epoch:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    best_validation_metrics = best_checkpoint["validation_metrics"]
    action_contract_accepted = (
        float(best_validation_metrics["raw_invalid_action_rate"]) <= invalid_tolerance
    )
    beat_baseline = best_value < baseline_value
    accepted = beat_baseline and action_contract_accepted
    manifest = {
        "schema_version": 1,
        "status": "complete" if accepted else "acceptance_failed",
        "experiment_id": context["experiment"]["experiment_id"],
        "run_id": run_id,
        "feature_cache_identity": context["manifest"]["identity_hash"],
        "experiment_config_sha256": file_sha256(context["config_path"]),
        "completed_epoch": completed_epoch,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_checkpoint": f"epoch-{best_epoch:03d}.pt",
        "best_validation_value": best_value,
        "primary_metric": primary,
        "baseline": baseline_name,
        "baseline_value": baseline_value,
        "beat_declared_baseline": beat_baseline,
        "action_contract_accepted": action_contract_accepted,
        "invalid_action_tolerance": invalid_tolerance,
        "state_noise_std_normalized": float(
            configured.get("state_noise_std_normalized", 0.0)
        ),
        "action_loss_protocol": context["action_loss_protocol"],
        "state_pairing": context["state_pairing"],
        "runtime_device": _runtime_device_report(device),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "prerequisites": {
            name: {"name": path.name, "sha256": file_sha256(path)}
            for name, path in prerequisites.items()
        },
        "checkpoints": {
            path.name: file_sha256(path) for path in sorted(checkpoint_root.glob("epoch-*.pt"))
        },
    }
    destination = metric_root / "training_manifest.json"
    create_json(destination, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not accepted:
        raise RuntimeError(
            "Full training failed the declared baseline or Action Contract acceptance."
        )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke_parser.add_argument("--feature-manifest", type=Path)
    overfit_parser = subparsers.add_parser("overfit")
    overfit_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    overfit_parser.add_argument("--feature-manifest", type=Path)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_parser.add_argument("--feature-manifest", type=Path)
    train_parser.add_argument("--run-id", required=True)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    context = _load_context(args.config.resolve(), args.feature_manifest)
    if args.command == "smoke":
        optimizer_smoke(context)
        return 0
    if args.command == "overfit":
        overfit(context)
        return 0
    train(
        context,
        run_id=args.run_id,
        resume=args.resume,
        stop_after_epoch=args.stop_after_epoch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
