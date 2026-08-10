"""Diagnose the offline-to-closed-loop gap for the frozen M2 policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "m2_qwen08b_frozen_001.yaml"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.data import ordered_feature_names, resolve_prepared_cache  # noqa: E402
from rosetta_reality.data.adapters.lerobot_v3 import LeRobotV3Adapter  # noqa: E402
from rosetta_reality.data.config import load_dataset_config  # noqa: E402
from rosetta_reality.data.normalization import (  # noqa: E402
    DatasetStatistics,
    denormalize,
    normalize,
)
from rosetta_reality.eval.diagnostics import (  # noqa: E402
    action_error_summary,
    cross_episode_shuffle_indices,
    nearest_cross_episode_indices,
    paired_cosine_distances,
    paired_l2_distances,
    pairwise_cosine_summary,
    pairwise_l2_summary,
    pearson_correlation,
    phase_labels,
    scalar_summary,
)
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
from rosetta_reality.models.backbones.qwen35 import Qwen35Backbone  # noqa: E402
from rosetta_reality.sim import (  # noqa: E402
    ActionContract,
    GymAlohaEnvironment,
    load_action_contract,
)
from rosetta_reality.train.m2 import build_cached_policy  # noqa: E402


def _run_root() -> Path:
    value = os.environ.get("ROSETTA_RUN_ROOT")
    return Path(value) if value else REPOSITORY_ROOT / "runs"


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _validation_scope(
    config_path: Path,
    dataset_path: Path,
    episodes: tuple[int, ...],
) -> dict[str, Any]:
    """Require exact validation coverage before any dataset or image decoding."""

    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    configured_dataset = (REPOSITORY_ROOT / experiment["dataset"]["config"]).resolve()
    if dataset_path.resolve() != configured_dataset:
        raise ValueError("Validation diagnostic dataset differs from the experiment config.")
    validation = tuple(int(value) for value in experiment["dataset"]["split"]["validation"])
    hidden_test = {int(value) for value in experiment["dataset"]["split"]["test"]}
    if (
        len(validation) != len(set(validation))
        or set(validation) & hidden_test
        or episodes != validation
    ):
        raise ValueError(
            "Validation diagnostics require the exact ordered validation split and no test."
        )
    return {
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "split": "validation",
        "episodes": list(validation),
        "test_split_opened": False,
    }


def _load_context(
    config_path: Path,
    feature_manifest_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    feature_manifest = load_feature_manifest(feature_manifest_path)
    artifact_manifest_path = artifact_root / "manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("status") != "verified":
        raise ValueError("Diagnostics require a verified exported artifact.")
    for name, expected in artifact_manifest["files"].items():
        if file_sha256(artifact_root / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}.")
    artifact_config = json.loads((artifact_root / "config.json").read_text(encoding="utf-8"))
    if artifact_config["feature_cache_identity"] != feature_manifest["identity_hash"]:
        raise ValueError("Artifact and feature-cache identities differ.")
    if artifact_config["experiment_id"] != experiment["experiment_id"]:
        raise ValueError("Artifact and experiment identifiers differ.")
    normalization_payload = json.loads(
        (artifact_root / "normalization.json").read_text(encoding="utf-8")
    )
    if normalization_payload.get("source_split") != "train":
        raise ValueError("Diagnostics require train-only normalization statistics.")
    statistics = DatasetStatistics.from_dict(normalization_payload["statistics"])
    model_payload = torch.load(artifact_root / "model.pt", map_location="cpu", weights_only=True)
    model_contract = model_payload["model_contract"]
    model = build_cached_policy(
        experiment,
        feature_dim=int(model_contract["feature_dim"]),
        state_dim=int(model_contract["state_dim"]),
        action_dim=int(model_contract["action_dim"]),
        chunk_size=int(model_contract["chunk_size"]),
        statistics=statistics,
    )
    model.load_state_dict(model_payload["model_state"], strict=True)
    model.eval()
    contract_path = REPOSITORY_ROOT / experiment["action_contract"]
    return {
        "experiment": experiment,
        "feature_manifest": feature_manifest,
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "feature_manifest_sha256": file_sha256(feature_manifest_path),
        "statistics": statistics,
        "model": model,
        "contract": load_action_contract(contract_path),
        "contract_sha256": file_sha256(contract_path),
    }


@torch.inference_mode()
def _predict(
    model: Any,
    dataset: CachedFeatureDataset,
    statistics: DatasetStatistics,
    contract: ActionContract,
    *,
    batch_size: int,
    feature_replacement: torch.Tensor | None = None,
    state_replacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_chunks: list[torch.Tensor] = []
    projected_chunks: list[torch.Tensor] = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        features = dataset.features[start:stop]
        states = dataset.robot_state[start:stop]
        if feature_replacement is not None:
            features = _replacement_batch(
                feature_replacement,
                original=features,
                start=start,
                stop=stop,
                total=len(dataset),
                name="feature",
            )
        if state_replacement is not None:
            states = _replacement_batch(
                state_replacement,
                original=states,
                start=start,
                stop=stop,
                total=len(dataset),
                name="state",
            )
        normalized_prediction = model(
            {"features": features.to(torch.float32)},
            normalize(states.to(torch.float32), statistics.state),
        )
        if not bool(torch.isfinite(normalized_prediction).all()):
            raise FloatingPointError("Diagnostic policy prediction contains NaN or Inf.")
        raw = denormalize(normalized_prediction, statistics.action).cpu()
        projected, _ = contract.clip(raw)
        raw_chunks.append(raw)
        projected_chunks.append(projected)
    return torch.cat(raw_chunks), torch.cat(projected_chunks)


def _replacement_batch(
    replacement: torch.Tensor,
    *,
    original: torch.Tensor,
    start: int,
    stop: int,
    total: int,
    name: str,
) -> torch.Tensor:
    """Select a constant or sample-aligned diagnostic replacement batch."""

    if replacement.ndim != 2 or replacement.shape[1:] != original.shape[1:]:
        raise ValueError(
            f"Diagnostic {name} replacement does not match the input feature shape."
        )
    if replacement.shape[0] == 1:
        return replacement.expand(stop - start, -1)
    if replacement.shape[0] == total:
        return replacement[start:stop]
    raise ValueError(
        f"Diagnostic {name} replacement must contain one or {total} samples."
    )


def _limit_violation_rate(raw: torch.Tensor, contract: ActionContract) -> float:
    _, mask = contract.clip(raw)
    return float(mask.to(torch.float32).mean())


def _phase_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    frames: torch.Tensor,
    boundaries: tuple[int, ...],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(phase_labels(frames, boundaries)):
        grouped[label].append(index)
    result: dict[str, dict[str, float | int]] = {}
    for label, indices in sorted(grouped.items()):
        selection = torch.tensor(indices, dtype=torch.long)
        result[label] = {
            "samples": len(indices),
            **action_error_summary(predicted[selection], target[selection]),
        }
    return result


def _dimension_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    absolute = (predicted - target).abs()
    return {
        name: {
            "chunk_mae": float(absolute[:, :, index].mean()),
            "first_action_mae": float(absolute[:, 0, index].mean()),
        }
        for index, name in enumerate(names)
    }


def _reset_metrics(
    dataset: CachedFeatureDataset,
    predicted: torch.Tensor,
) -> dict[str, Any]:
    selected = dataset.frame_indices == 0
    target = dataset.actions[selected, 0]
    prediction = predicted[selected, 0]
    features = dataset.features[selected]
    states = dataset.robot_state[selected]
    return {
        "samples": int(selected.sum()),
        "feature_pairwise_cosine_distance": pairwise_cosine_summary(features),
        "feature_pairwise_l2_distance": pairwise_l2_summary(features),
        "state_pairwise_l2_distance": pairwise_l2_summary(states),
        "target_first_action_pairwise_l2_distance": pairwise_l2_summary(target),
        "predicted_first_action_pairwise_l2_distance": pairwise_l2_summary(prediction),
        "first_action_mae": float((prediction - target).abs().mean()),
    }


def cached_policy_diagnostic(
    *,
    config_path: Path,
    feature_manifest_path: Path,
    artifact_root: Path,
    splits: tuple[str, ...],
    phase_boundaries: tuple[int, ...],
) -> Path:
    """Measure phase errors and vision/state ablations on immutable cached features."""

    context = _load_context(config_path, feature_manifest_path, artifact_root)
    experiment = context["experiment"]
    statistics = context["statistics"]
    contract = context["contract"]
    model = context["model"]
    train = CachedFeatureDataset(feature_manifest_path, "train")
    train_feature_mean = train.features.mean(dim=0, keepdim=True)
    train_state_mean = train.robot_state.mean(dim=0, keepdim=True)
    batch_size = int(experiment["training"]["batch_size"])

    split_reports: dict[str, Any] = {}
    for split in splits:
        dataset = train if split == "train" else CachedFeatureDataset(feature_manifest_path, split)
        raw, predicted = _predict(
            model,
            dataset,
            statistics,
            contract,
            batch_size=batch_size,
        )
        _, feature_ablated = _predict(
            model,
            dataset,
            statistics,
            contract,
            batch_size=batch_size,
            feature_replacement=train_feature_mean,
        )
        _, state_ablated = _predict(
            model,
            dataset,
            statistics,
            contract,
            batch_size=batch_size,
            state_replacement=train_state_mean,
        )
        actual = action_error_summary(predicted, dataset.actions)
        feature_ablation = action_error_summary(feature_ablated, dataset.actions)
        state_ablation = action_error_summary(state_ablated, dataset.actions)
        split_reports[split] = {
            "samples": len(dataset),
            "episodes": sorted(set(int(value) for value in dataset.episode_ids.tolist())),
            "actual": actual,
            "mean_feature_ablation": {
                **feature_ablation,
                "chunk_mae_delta_vs_actual": feature_ablation["chunk_mae"]
                - actual["chunk_mae"],
                "first_action_mae_delta_vs_actual": feature_ablation["first_action_mae"]
                - actual["first_action_mae"],
            },
            "mean_state_ablation": {
                **state_ablation,
                "chunk_mae_delta_vs_actual": state_ablation["chunk_mae"]
                - actual["chunk_mae"],
                "first_action_mae_delta_vs_actual": state_ablation["first_action_mae"]
                - actual["first_action_mae"],
            },
            "raw_limit_violation_rate": _limit_violation_rate(raw, contract),
            "phase_metrics": _phase_metrics(
                predicted,
                dataset.actions,
                dataset.frame_indices,
                phase_boundaries,
            ),
            "dimension_metrics": _dimension_metrics(
                predicted,
                dataset.actions,
                contract.dimension_names,
            ),
            "reset_metrics": _reset_metrics(dataset, predicted),
        }

    report = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "m2_cached_policy_offline_to_closed_loop_gap",
        "experiment_id": experiment["experiment_id"],
        "artifact_id": context["artifact_manifest"]["artifact_id"],
        "artifact_manifest_sha256": context["artifact_manifest_sha256"],
        "feature_cache_identity": context["feature_manifest"]["identity_hash"],
        "feature_manifest_sha256": context["feature_manifest_sha256"],
        "action_contract_sha256": context["contract_sha256"],
        "phase_boundaries": list(phase_boundaries),
        "ablation_definition": {
            "mean_feature": "replace each pooled feature with the train-split mean feature",
            "mean_state": "replace each robot state with the train-split mean state",
            "positive_mae_delta": "the ablated input carried predictive information",
        },
        "splits": split_reports,
    }
    digest = stable_hash(report)[:12]
    destination = (
        _run_root()
        / experiment["experiment_id"]
        / "diagnostics"
        / f"cached-policy-{digest}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination}")
    return destination


def _action_response_summary(
    reference: torch.Tensor,
    changed: torch.Tensor,
) -> dict[str, Any]:
    difference = changed.to(torch.float32) - reference.to(torch.float32)
    return {
        "chunk_mean_absolute_change": float(difference.abs().mean()),
        "first_action_mean_absolute_change": float(difference[:, 0].abs().mean()),
        "first_action_l2_change": scalar_summary(
            paired_l2_distances(changed[:, 0], reference[:, 0])
        ),
        "chunk_l2_change": scalar_summary(
            paired_l2_distances(
                changed.flatten(start_dim=1),
                reference.flatten(start_dim=1),
            )
        ),
    }


def _condition_report(
    *,
    raw: torch.Tensor,
    predicted: torch.Tensor,
    target: torch.Tensor,
    full_prediction: torch.Tensor,
    full_error: dict[str, float],
    contract: ActionContract,
) -> dict[str, Any]:
    error = action_error_summary(predicted, target)
    return {
        **error,
        "chunk_mae_delta_vs_full": error["chunk_mae"] - full_error["chunk_mae"],
        "first_action_mae_delta_vs_full": (
            error["first_action_mae"] - full_error["first_action_mae"]
        ),
        "raw_limit_violation_rate": _limit_violation_rate(raw, contract),
        "prediction_response_vs_full": _action_response_summary(
            full_prediction,
            predicted,
        ),
    }


def _reset_visual_necessity(
    dataset: CachedFeatureDataset,
    statistics: DatasetStatistics,
) -> dict[str, Any]:
    selected = dataset.frame_indices.eq(0)
    count = int(selected.sum())
    if count < 2:
        return {
            "samples": count,
            "pairs": 0,
            "interpretation_boundary": "At least two reset samples are required.",
        }
    features = dataset.features[selected]
    states = normalize(dataset.robot_state[selected], statistics.state)
    targets = dataset.actions[selected, 0]
    pair_indices = torch.triu_indices(count, count, offset=1)
    left, right = pair_indices[0], pair_indices[1]
    state_l2 = paired_l2_distances(states[left], states[right])
    feature_l2 = paired_l2_distances(features[left], features[right])
    feature_cosine = paired_cosine_distances(features[left], features[right])
    target_l2 = paired_l2_distances(targets[left], targets[right])
    return {
        "samples": count,
        "pairs": int(left.numel()),
        "normalized_state_l2": scalar_summary(state_l2),
        "near_identical_state_pair_rate_at_1e-6": float(
            state_l2.le(1e-6).to(torch.float32).mean()
        ),
        "feature_l2_per_sqrt_dimension": scalar_summary(
            feature_l2 / float(features.shape[1] ** 0.5)
        ),
        "feature_cosine_distance": scalar_summary(feature_cosine),
        "target_first_action_l2": scalar_summary(target_l2),
        "correlations": {
            "feature_cosine_vs_target_first_action_l2": pearson_correlation(
                feature_cosine,
                target_l2,
            ),
            "normalized_state_l2_vs_target_first_action_l2": pearson_correlation(
                state_l2,
                target_l2,
            ),
        },
        "interpretation_boundary": (
            "Non-zero target distance at near-identical reset state establishes a need for "
            "information outside robot state, but does not by itself prove that the cached "
            "visual representation makes that information learnable."
        ),
    }


def _nearest_state_dataset_report(
    *,
    query: CachedFeatureDataset,
    reference: CachedFeatureDataset,
    statistics: DatasetStatistics,
    query_split: str,
    reference_split: str,
) -> dict[str, Any]:
    query_states = normalize(query.robot_state, statistics.state)
    reference_states = normalize(reference.robot_state, statistics.state)
    matched, state_l2 = nearest_cross_episode_indices(
        query_states,
        query.episode_ids,
        reference_states,
        reference.episode_ids,
    )
    reference_features = reference.features[matched]
    feature_cosine = paired_cosine_distances(query.features, reference_features)
    feature_l2 = paired_l2_distances(query.features, reference_features)
    target_first_l2 = paired_l2_distances(
        query.actions[:, 0],
        reference.actions[matched, 0],
    )
    target_chunk_mae = (
        query.actions - reference.actions[matched]
    ).abs().mean(dim=(1, 2))
    frame_distance = (
        query.frame_indices - reference.frame_indices[matched]
    ).abs().to(torch.float32)
    cutoff = torch.quantile(state_l2, 0.1)
    near_indices = torch.nonzero(state_l2.le(cutoff), as_tuple=False).flatten().tolist()
    near_indices.sort(key=lambda index: float(target_first_l2[index]), reverse=True)
    examples = []
    for index in near_indices[:10]:
        reference_index = int(matched[index])
        examples.append(
            {
                "query_episode": int(query.episode_ids[index]),
                "query_frame": int(query.frame_indices[index]),
                "reference_episode": int(reference.episode_ids[reference_index]),
                "reference_frame": int(reference.frame_indices[reference_index]),
                "normalized_state_l2": float(state_l2[index]),
                "feature_cosine_distance": float(feature_cosine[index]),
                "target_first_action_l2": float(target_first_l2[index]),
                "target_chunk_mae": float(target_chunk_mae[index]),
            }
        )
    return {
        "query_split": query_split,
        "reference_split": reference_split,
        "queries": len(query),
        "different_episode_enforced": True,
        "normalized_state_l2": scalar_summary(state_l2),
        "feature_l2_per_sqrt_dimension": scalar_summary(
            feature_l2 / float(query.features.shape[1] ** 0.5)
        ),
        "feature_cosine_distance": scalar_summary(feature_cosine),
        "target_first_action_l2": scalar_summary(target_first_l2),
        "target_chunk_mae": scalar_summary(target_chunk_mae),
        "absolute_frame_distance": scalar_summary(frame_distance),
        "same_frame_rate": float(frame_distance.eq(0).to(torch.float32).mean()),
        "correlations": {
            "feature_cosine_vs_target_first_action_l2": pearson_correlation(
                feature_cosine,
                target_first_l2,
            ),
            "normalized_state_l2_vs_target_first_action_l2": pearson_correlation(
                state_l2,
                target_first_l2,
            ),
        },
        "lowest_state_distance_decile_cutoff": float(cutoff),
        "largest_action_differences_within_lowest_state_distance_decile": examples,
    }


def modality_audit(
    *,
    config_path: Path,
    feature_manifest_path: Path,
    artifact_root: Path,
    splits: tuple[str, ...],
    seed: int,
) -> Path:
    """Audit model modality reliance and dataset visual necessity without training."""

    if not splits or any(split not in {"train", "validation"} for split in splits):
        raise ValueError("Modality audit is restricted to train and validation splits.")
    if len(set(splits)) != len(splits):
        raise ValueError("Modality audit splits must be unique.")
    context = _load_context(config_path, feature_manifest_path, artifact_root)
    experiment = context["experiment"]
    statistics = context["statistics"]
    contract = context["contract"]
    model = context["model"]
    train = CachedFeatureDataset(feature_manifest_path, "train")
    train_feature_mean = train.features.mean(dim=0, keepdim=True)
    train_state_mean = train.robot_state.mean(dim=0, keepdim=True)
    batch_size = int(experiment["training"]["batch_size"])

    split_reports: dict[str, Any] = {}
    for split in splits:
        dataset = train if split == "train" else CachedFeatureDataset(
            feature_manifest_path,
            split,
        )
        within_frame_indices = cross_episode_shuffle_indices(
            dataset.episode_ids,
            frame_indices=dataset.frame_indices,
            seed=seed,
        )
        global_indices = cross_episode_shuffle_indices(
            dataset.episode_ids,
            seed=seed,
        )
        full_raw, full_prediction = _predict(
            model,
            dataset,
            statistics,
            contract,
            batch_size=batch_size,
        )
        constant_feature_raw, constant_feature_prediction = _predict(
            model,
            dataset,
            statistics,
            contract,
            batch_size=batch_size,
            feature_replacement=train_feature_mean,
        )
        within_frame_raw, within_frame_prediction = _predict(
            model,
            dataset,
            statistics,
            contract,
            batch_size=batch_size,
            feature_replacement=dataset.features[within_frame_indices],
        )
        global_raw, global_prediction = _predict(
            model,
            dataset,
            statistics,
            contract,
            batch_size=batch_size,
            feature_replacement=dataset.features[global_indices],
        )
        constant_state_raw, constant_state_prediction = _predict(
            model,
            dataset,
            statistics,
            contract,
            batch_size=batch_size,
            state_replacement=train_state_mean,
        )
        full_error = action_error_summary(full_prediction, dataset.actions)
        split_report: dict[str, Any] = {
            "samples": len(dataset),
            "episodes": sorted(set(int(value) for value in dataset.episode_ids.tolist())),
            "model_reliance": {
                "full": {
                    **full_error,
                    "raw_limit_violation_rate": _limit_violation_rate(
                        full_raw,
                        contract,
                    ),
                },
                "constant_train_mean_feature": _condition_report(
                    raw=constant_feature_raw,
                    predicted=constant_feature_prediction,
                    target=dataset.actions,
                    full_prediction=full_prediction,
                    full_error=full_error,
                    contract=contract,
                ),
                "within_frame_cross_episode_feature_shuffle": _condition_report(
                    raw=within_frame_raw,
                    predicted=within_frame_prediction,
                    target=dataset.actions,
                    full_prediction=full_prediction,
                    full_error=full_error,
                    contract=contract,
                ),
                "global_cross_episode_feature_shuffle": _condition_report(
                    raw=global_raw,
                    predicted=global_prediction,
                    target=dataset.actions,
                    full_prediction=full_prediction,
                    full_error=full_error,
                    contract=contract,
                ),
                "constant_train_mean_state": _condition_report(
                    raw=constant_state_raw,
                    predicted=constant_state_prediction,
                    target=dataset.actions,
                    full_prediction=full_prediction,
                    full_error=full_error,
                    contract=contract,
                ),
            },
            "shuffle_invariants": {
                "within_frame": {
                    "is_permutation": sorted(within_frame_indices.tolist())
                    == list(range(len(dataset))),
                    "different_episode_rate": float(
                        dataset.episode_ids
                        .ne(dataset.episode_ids[within_frame_indices])
                        .to(torch.float32)
                        .mean()
                    ),
                    "same_frame_rate": float(
                        dataset.frame_indices
                        .eq(dataset.frame_indices[within_frame_indices])
                        .to(torch.float32)
                        .mean()
                    ),
                },
                "global": {
                    "is_permutation": sorted(global_indices.tolist())
                    == list(range(len(dataset))),
                    "different_episode_rate": float(
                        dataset.episode_ids
                        .ne(dataset.episode_ids[global_indices])
                        .to(torch.float32)
                        .mean()
                    ),
                    "same_frame_rate": float(
                        dataset.frame_indices
                        .eq(dataset.frame_indices[global_indices])
                        .to(torch.float32)
                        .mean()
                    ),
                },
            },
            "dataset_visual_necessity": {
                "reset_pairs": _reset_visual_necessity(dataset, statistics),
                "nearest_state_within_split": _nearest_state_dataset_report(
                    query=dataset,
                    reference=dataset,
                    statistics=statistics,
                    query_split=split,
                    reference_split=split,
                ),
            },
        }
        if split != "train":
            split_report["dataset_visual_necessity"]["nearest_state_to_train"] = (
                _nearest_state_dataset_report(
                    query=dataset,
                    reference=train,
                    statistics=statistics,
                    query_split=split,
                    reference_split="train",
                )
            )
        split_reports[split] = split_report

    report = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "m2_visual_necessity_and_modality_reliance_audit",
        "optimizer_steps": 0,
        "test_split_opened": False,
        "seed": int(seed),
        "experiment_id": experiment["experiment_id"],
        "artifact_id": context["artifact_manifest"]["artifact_id"],
        "artifact_manifest_sha256": context["artifact_manifest_sha256"],
        "feature_cache_identity": context["feature_manifest"]["identity_hash"],
        "feature_manifest_sha256": context["feature_manifest_sha256"],
        "action_contract_sha256": context["contract_sha256"],
        "definitions": {
            "model_reliance": (
                "Measures whether changing one cached modality changes this trained policy and "
                "its action error; it does not establish dataset causality by itself."
            ),
            "within_frame_shuffle": (
                "Permutes cached visual features across different episodes at the exact same "
                "frame index while leaving robot state and targets fixed."
            ),
            "global_shuffle": (
                "Permutes cached visual features across different episodes without preserving "
                "frame index; this is a stronger but phase-confounded stress test."
            ),
            "constant_feature": (
                "Replaces every cached feature with the train-split mean feature."
            ),
            "constant_state": (
                "Replaces every robot state with the train-split mean state."
            ),
            "dataset_visual_necessity": (
                "Measures whether cross-episode samples with equal or nearby robot state require "
                "different expert actions and whether their cached visual features differ."
            ),
            "positive_mae_delta": (
                "The removed or mismatched modality carried predictive information."
            ),
        },
        "splits": split_reports,
    }
    digest = stable_hash(report)[:12]
    destination = (
        _run_root()
        / experiment["experiment_id"]
        / "diagnostics"
        / f"modality-audit-{digest}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination}")
    return destination


def _dataset_rows(
    root: Path,
    episode: int,
    *,
    action_field: str,
    state_field: str,
    timestamp_field: str,
    episode_field: str,
    frame_field: str,
) -> list[dict[str, Any]]:
    import pyarrow.dataset as arrow_dataset

    dataset = arrow_dataset.dataset(root / "data", format="parquet")
    table = dataset.to_table(
        columns=[action_field, state_field, timestamp_field, frame_field],
        filter=arrow_dataset.field(episode_field) == episode,
    )
    return sorted(table.to_pylist(), key=lambda row: int(row[frame_field]))


def _episode_initial_images(
    *,
    root: Path,
    revision: str,
    dataset_config: Any,
    episodes: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    adapter = LeRobotV3Adapter(
        repo_id=dataset_config.repo_id,
        revision=revision,
        root=root,
        episodes=episodes,
        cameras=dataset_config.cameras,
        fields=dataset_config.fields,
        embodiment=dataset_config.embodiment,
        license_name=dataset_config.license,
    )
    first_indices: dict[int, int] = {}
    for index in range(len(adapter)):
        reference = adapter.frame_reference(index)
        episode = int(reference.episode_id)
        if reference.frame_index == 0 and episode not in first_indices:
            first_indices[episode] = index
        if len(first_indices) == len(episodes):
            break
    missing = sorted(set(episodes) - set(first_indices))
    if missing:
        raise ValueError(f"Dataset has no frame-zero image for episodes: {missing}.")
    camera_name = next(iter(dataset_config.cameras))
    return {
        episode: adapter[first_indices[episode]].images[camera_name]
        for episode in episodes
    }


def export_initial_images(
    *,
    validation_config_path: Path,
    dataset_path: Path,
    episodes: tuple[int, ...],
) -> Path:
    """Decode selected dataset frame-zero images once for the isolated sim stage."""

    if not episodes:
        raise ValueError("At least one episode is required for initial-image export.")
    validation_scope = _validation_scope(
        validation_config_path,
        dataset_path,
        episodes,
    )
    dataset_config = load_dataset_config(dataset_path)
    unknown = sorted(set(episodes) - set(dataset_config.episodes))
    if unknown:
        raise ValueError(f"Episodes are outside the pinned dataset selection: {unknown}.")
    root, manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    dataset_manifest_sha256 = file_sha256(root / "manifest.json")
    identity = {
        "schema_version": 2,
        "dataset_repo_id": dataset_config.repo_id,
        "dataset_revision": manifest.resolved_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "episodes": list(episodes),
        "camera": next(iter(dataset_config.cameras)),
        "decoder": "LeRobotDataset video_backend=pyav",
        "validation_scope": validation_scope,
    }
    identity_hash = stable_hash(identity)
    images = _episode_initial_images(
        root=root,
        revision=manifest.resolved_revision,
        dataset_config=dataset_config,
        episodes=episodes,
    )
    destination = (
        _run_root()
        / "m2-qwen08b-frozen-001"
        / "diagnostics"
        / f"initial-images-{identity_hash[:12]}"
    )
    if destination.exists() and not destination.is_dir():
        raise FileExistsError("Initial-image artifact destination is not a directory.")
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for episode, source_image in images.items():
        image = source_image.to(torch.float32).cpu()
        filename = f"episode-{episode:06d}.pt"
        shard_path = destination / filename
        shard_payload = {
            "schema_version": 1,
            "identity_hash": identity_hash,
            "episode": episode,
            "image": image,
        }
        if shard_path.exists():
            existing = torch.load(shard_path, map_location="cpu", weights_only=True)
            if (
                not isinstance(existing, dict)
                or existing.get("schema_version") != 1
                or existing.get("identity_hash") != identity_hash
                or existing.get("episode") != episode
                or not isinstance(existing.get("image"), torch.Tensor)
                or not torch.equal(existing["image"], image)
            ):
                raise ValueError("Existing initial-image shard differs from a fresh decode.")
        else:
            save_tensor_shard(shard_path, shard_payload)
        files[str(episode)] = {
            "path": filename,
            "sha256": file_sha256(shard_path),
            "shape": list(image.shape),
        }
    manifest_payload = {
        **identity,
        "identity_hash": identity_hash,
        "files": files,
    }
    manifest_path = create_json(destination / "manifest.json", manifest_payload)
    receipt = {
        **identity,
        "identity_hash": identity_hash,
        "path": str(destination),
        "manifest_sha256": file_sha256(manifest_path),
        "files": files,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return destination


def _load_initial_images(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_revision: str,
    episodes: tuple[int, ...],
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if not path.is_dir():
        raise ValueError("Initial-image artifact must use the validation-only sharded format.")
    manifest_path = path / "manifest.json"
    payload = _read_json_object(manifest_path)
    if payload.get("schema_version") != 2:
        raise ValueError("Initial-image artifact schema is unsupported.")
    if payload.get("dataset_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Initial-image artifact dataset manifest differs.")
    if payload.get("dataset_revision") != expected_revision:
        raise ValueError("Initial-image artifact dataset revision differs.")
    episode_values = payload.get("episodes", [])
    if (
        not isinstance(episode_values, list)
        or any(type(value) is not int for value in episode_values)
        or len(episode_values) != len(set(episode_values))
    ):
        raise ValueError("Initial-image artifact episodes must be unique integers.")
    available_episodes = set(episode_values)
    if len(episodes) != len(set(episodes)) or set(episodes) != available_episodes:
        raise ValueError("Initial-image artifact must exactly cover requested episodes.")
    validation_scope = payload.get("validation_scope")
    if (
        not isinstance(validation_scope, dict)
        or validation_scope.get("split") != "validation"
        or validation_scope.get("test_split_opened") is not False
        or validation_scope.get("episodes") != episode_values
    ):
        raise ValueError("Initial-image artifact lacks an exact validation-only scope.")
    file_records = payload.get("files")
    expected_image_keys = {str(episode) for episode in episode_values}
    if not isinstance(file_records, dict) or set(file_records) != expected_image_keys:
        raise ValueError("Initial-image shard records differ from declared episodes.")
    identity_fields = (
        "schema_version",
        "dataset_repo_id",
        "dataset_revision",
        "dataset_manifest_sha256",
        "episodes",
        "camera",
        "decoder",
        "validation_scope",
    )
    declared_identity = {name: payload.get(name) for name in identity_fields}
    if payload.get("identity_hash") != stable_hash(declared_identity):
        raise ValueError("Initial-image artifact identity hash is invalid.")
    expected_filenames = {"manifest.json"} | {
        f"episode-{episode:06d}.pt" for episode in episode_values
    }
    if {item.name for item in path.iterdir()} != expected_filenames:
        raise ValueError("Initial-image artifact contains undeclared files.")
    images: dict[int, torch.Tensor] = {}
    for episode in episodes:
        record = file_records[str(episode)]
        expected_filename = f"episode-{episode:06d}.pt"
        if not isinstance(record, dict) or record.get("path") != expected_filename:
            raise ValueError(f"Initial-image shard path is invalid for episode {episode}.")
        shard_path = path / expected_filename
        if shard_path.is_symlink() or file_sha256(shard_path) != record.get("sha256"):
            raise ValueError(f"Initial-image shard checksum differs for episode {episode}.")
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(shard, dict)
            or shard.get("schema_version") != 1
            or shard.get("identity_hash") != payload.get("identity_hash")
            or shard.get("episode") != episode
        ):
            raise ValueError(f"Initial-image shard identity differs for episode {episode}.")
        image = shard.get("image")
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise ValueError(f"Initial-image artifact has no valid image for episode {episode}.")
        if not bool(torch.isfinite(image).all()):
            raise ValueError(f"Initial image for episode {episode} contains NaN or Inf.")
        images[episode] = image.to(torch.float32)
    identity = {
        "identity_hash": payload.get("identity_hash"),
        "manifest_sha256": file_sha256(manifest_path),
        "decoder": payload.get("decoder"),
        "dataset_revision": payload.get("dataset_revision"),
        "dataset_manifest_sha256": payload.get("dataset_manifest_sha256"),
        "episodes": sorted(available_episodes),
        "validation_scope": validation_scope,
    }
    return images, identity


def _validated_model_manifest(model_root: Path, expected_identifier: str) -> dict[str, Any]:
    manifest_path = model_root / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "validated"
        or manifest.get("repo_id") != expected_identifier
    ):
        raise ValueError("Pooling probe requires the validated Base model identity.")
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("Validated Base model manifest has no file records.")
    for relative, record in records.items():
        path = model_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Base model checksum mismatch during pooling probe: {relative}.")
    return manifest


def _pooling_dispersion(features: torch.Tensor) -> dict[str, Any]:
    l2 = pairwise_l2_summary(features)
    return {
        "feature_dimension": int(features.shape[1]),
        "finite": bool(torch.isfinite(features).all()),
        "pairwise_l2": l2,
        "mean_pairwise_l2_per_sqrt_dimension": (
            float(l2["mean"]) / float(features.shape[1] ** 0.5)
        ),
        "pairwise_cosine": pairwise_cosine_summary(features),
        "mean_per_dimension_standard_deviation": float(
            features.to(torch.float32).std(dim=0, unbiased=False).mean()
        ),
    }


@torch.inference_mode()
def pooling_probe(
    *,
    config_path: Path,
    initial_images_path: Path,
    episodes: tuple[int, ...],
) -> Path:
    """Compare candidate frozen representations on identity-bound reset images."""

    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    model_environment = str(experiment["backbone"]["local_root_environment"])
    model_value = os.environ.get(model_environment)
    if not model_value:
        raise ValueError(f"{model_environment} is required for pooling diagnostics.")
    model_root = Path(model_value)
    model_manifest = _validated_model_manifest(
        model_root,
        str(experiment["backbone"]["identifier"]),
    )
    dataset_config = load_dataset_config(REPOSITORY_ROOT / experiment["dataset"]["config"])
    dataset_root, dataset_manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=False,
    )
    images, image_identity = _load_initial_images(
        initial_images_path,
        expected_manifest_sha256=file_sha256(dataset_root / "manifest.json"),
        expected_revision=dataset_manifest.resolved_revision,
        episodes=episodes,
    )
    expected_scope = _validation_scope(
        config_path,
        REPOSITORY_ROOT / experiment["dataset"]["config"],
        episodes,
    )
    if image_identity.get("validation_scope") != expected_scope:
        raise ValueError("Pooling probe image artifact differs from the validation config.")
    configured = experiment["backbone"]
    dtype = getattr(torch, str(configured["dtype"]), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported pooling-probe dtype: {configured['dtype']!r}.")
    model_width = int(model_manifest["model_contract"]["hidden_size"])
    backbone = Qwen35Backbone(
        str(model_root),
        hidden_size=model_width,
        device="cpu",
        dtype=dtype,
        local_files_only=True,
        freeze=True,
        pooling="attention_masked_mean",
        prompt_template=str(configured["processor"]["prompt"]),
        prompt_mode=str(configured["processor"].get("prompt_mode", "auto")),
        model_kwargs={"low_cpu_mem_usage": True},
        processor_kwargs={
            "min_pixels": int(configured["processor"]["min_pixels"]),
            "max_pixels": int(configured["processor"]["max_pixels"]),
        },
    )
    model, _ = backbone.load()
    camera_name = next(iter(dataset_config.cameras))
    modes = ("attention_masked_mean", "image_token_mean", "image_spatial_2x2")
    features: dict[str, list[torch.Tensor]] = {mode: [] for mode in modes}
    sample_metadata: list[dict[str, Any]] = []
    for episode in episodes:
        encoded = backbone.prepare_inputs(
            {camera_name: images[episode]},
            dataset_config.expected_instruction or "",
        )
        encoded = {
            key: value.to("cpu") if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        outputs = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        final_hidden = outputs.hidden_states[-1]
        image_mask = backbone._image_token_mask(encoded, final_hidden, model)
        sample_metadata.append(
            {
                "episode": episode,
                "sequence_tokens": int(final_hidden.shape[1]),
                "image_tokens": int(image_mask.sum()),
                "image_grid_thw": encoded["image_grid_thw"].tolist(),
            }
        )
        for mode in modes:
            backbone.pooling = mode
            pooled = backbone._pool_final_hidden(final_hidden, encoded, model)
            if pooled.shape != (1, backbone.hidden_size):
                raise RuntimeError(f"Unexpected {mode} feature shape: {tuple(pooled.shape)}.")
            features[mode].append(pooled[0].to(torch.float32).cpu())

    report = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "qwen35_frozen_pooling_probe",
        "experiment_id": experiment["experiment_id"],
        "model": {
            "identifier": model_manifest["repo_id"],
            "revision": model_manifest["revision"],
            "manifest_sha256": file_sha256(model_root / "model_manifest.json"),
        },
        "dataset": {
            "revision": dataset_manifest.resolved_revision,
            "manifest_sha256": file_sha256(dataset_root / "manifest.json"),
            "initial_images": image_identity,
            "episodes": list(episodes),
        },
        "samples": sample_metadata,
        "representations": {
            mode: _pooling_dispersion(torch.stack(values))
            for mode, values in features.items()
        },
        "interpretation_limit": (
            "Dispersion measures whether representations distinguish reset images; it does not "
            "by itself prove that a downstream policy can use the differences for task success."
        ),
    }
    digest = stable_hash(report)[:12]
    destination = (
        _run_root()
        / experiment["experiment_id"]
        / "diagnostics"
        / f"pooling-probe-{digest}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination}")
    return destination


def _alignment_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("Image alignment requires matching [channel, height, width] tensors.")
    reference = reference.to(torch.float32)
    candidate = candidate.to(torch.float32)
    difference = reference - candidate
    pooled_reference = torch.nn.functional.avg_pool2d(reference.unsqueeze(0), kernel_size=4)
    pooled_candidate = torch.nn.functional.avg_pool2d(candidate.unsqueeze(0), kernel_size=4)
    return {
        "pixel_mae": float(difference.abs().mean()),
        "pixel_rmse": float(difference.square().mean().sqrt()),
        "pooled_4x4_mae": float((pooled_reference - pooled_candidate).abs().mean()),
    }


def _object_state(environment: GymAlohaEnvironment) -> list[float]:
    unwrapped = getattr(environment.raw_environment, "unwrapped", environment.raw_environment)
    control_environment = getattr(unwrapped, "_env", None)
    physics = getattr(control_environment, "physics", None)
    if physics is None:
        raise RuntimeError("Gym-ALOHA physics is unavailable for object-pose diagnostics.")
    return [float(value) for value in physics.data.qpos.copy()[16:]]


def _search_initial_seed(
    environment: GymAlohaEnvironment,
    reference_image: torch.Tensor,
    *,
    seed_start: int,
    seed_count: int,
    top_k: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + seed_count):
        observation = environment.reset(seed=seed)
        images = observation.get("images", {})
        if len(images) != 1:
            raise ValueError("Seed alignment requires exactly one configured simulator camera.")
        candidate_image = next(iter(images.values()))
        candidates.append(
            {
                "seed": seed,
                **_alignment_metrics(reference_image, candidate_image),
                "object_state": _object_state(environment),
            }
        )
    return sorted(candidates, key=lambda row: row["pooled_4x4_mae"])[:top_k]


def _replay_expert_actions(
    environment: GymAlohaEnvironment,
    contract: ActionContract,
    rows: list[dict[str, Any]],
    *,
    action_field: str,
    state_field: str,
    seed: int,
    maximum_steps: int,
    settle_steps: int,
    action_mode: str,
) -> dict[str, Any]:
    if action_mode not in {"contract_clipped", "source_raw"}:
        raise ValueError(f"Unsupported expert replay action mode: {action_mode!r}.")

    def diagnostic_step(
        action: torch.Tensor,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if action_mode == "contract_clipped":
            return environment.step(action)
        raw_observation, reward, raw_terminated, raw_truncated, raw_info = (
            environment.raw_environment.step(action.detach().to(torch.float32).cpu().numpy())
        )
        info = dict(raw_info)
        info.update(terminated=bool(raw_terminated), truncated=bool(raw_truncated))
        return (
            dict(GymAlohaEnvironment._observation(raw_observation)),
            float(reward),
            bool(raw_terminated or raw_truncated),
            info,
        )

    observation = environment.reset(seed=seed)
    initial_state = observation["robot_state"].to(torch.float32)
    dataset_initial_state = torch.as_tensor(rows[0][state_field], dtype=torch.float32)
    rewards: list[float] = []
    tracking_errors: list[float] = []
    reward_first_steps: dict[str, int] = {}
    clipped_elements = 0
    executed_outside_contract_elements = 0
    unexpected_collisions = 0
    unexpected_pairs: Counter[str] = Counter()
    terminated = False
    truncated = False
    last_action: torch.Tensor | None = None
    for step, row in enumerate(rows[:maximum_steps]):
        raw_action = torch.as_tensor(row[action_field], dtype=torch.float32)
        contract.validate_tensor(raw_action, allow_chunk=False)
        action, mask = contract.clip(raw_action)
        executed_action = action if action_mode == "contract_clipped" else raw_action
        last_action = executed_action
        clipped_elements += int(mask.sum())
        executed_outside_contract_elements += int(mask.sum()) if action_mode == "source_raw" else 0
        observation, reward, done, info = diagnostic_step(executed_action)
        terminated = terminated or bool(info.get("terminated", False))
        truncated = truncated or bool(info.get("truncated", False))
        rewards.append(float(reward))
        tracking_errors.append(
            float((observation["robot_state"] - executed_action).abs().mean())
        )
        for stage in range(1, 5):
            if reward >= stage and str(stage) not in reward_first_steps:
                reward_first_steps[str(stage)] = step
        for first, second in environment.contact_pairs():
            if environment.is_unexpected_collision_pair(first, second):
                unexpected_collisions += 1
                unexpected_pairs[" <-> ".join(sorted((first, second)))] += 1
        if done or bool(info.get("is_success", False)):
            break
    expert_action_steps = len(rewards)
    settle_steps_executed = 0
    if not terminated and not truncated and last_action is not None:
        for offset in range(settle_steps):
            observation, reward, done, info = diagnostic_step(last_action)
            rewards.append(float(reward))
            settle_steps_executed += 1
            terminated = terminated or bool(info.get("terminated", False))
            truncated = truncated or bool(info.get("truncated", False))
            for stage in range(1, 5):
                if reward >= stage and str(stage) not in reward_first_steps:
                    reward_first_steps[str(stage)] = maximum_steps + offset
            for first, second in environment.contact_pairs():
                if environment.is_unexpected_collision_pair(first, second):
                    unexpected_collisions += 1
                    unexpected_pairs[" <-> ".join(sorted((first, second)))] += 1
            if done or bool(info.get("is_success", False)):
                break
    return {
        "seed": seed,
        "action_mode": action_mode,
        "contract_projection_applied": action_mode == "contract_clipped",
        "steps": len(rewards),
        "expert_action_steps": expert_action_steps,
        "settle_steps_requested": settle_steps,
        "settle_steps_executed": settle_steps_executed,
        "terminated": terminated,
        "truncated": truncated,
        "success": max(rewards, default=0.0) >= 4,
        "maximum_reward": max(rewards, default=0.0),
        "reward_histogram": {
            str(value): count for value, count in sorted(Counter(rewards).items())
        },
        "reward_first_steps": reward_first_steps,
        "initial_state_mae": float((initial_state - dataset_initial_state).abs().mean()),
        "mean_target_tracking_mae": sum(tracking_errors) / max(1, len(tracking_errors)),
        "maximum_target_tracking_mae": max(tracking_errors, default=0.0),
        "source_clipped_elements": clipped_elements,
        "executed_outside_contract_elements": executed_outside_contract_elements,
        "unexpected_collisions": unexpected_collisions,
        "unexpected_collision_pairs": dict(sorted(unexpected_pairs.items())),
    }


def expert_replay_diagnostic(
    *,
    validation_config_path: Path,
    dataset_path: Path,
    contract_path: Path,
    episodes: tuple[int, ...],
    maximum_steps: int,
    candidate_seed_start: int,
    candidate_seed_count: int,
    top_k: int,
    initial_images_path: Path,
    settle_steps: int,
    action_mode: str,
    seed_map: dict[int, int] | None,
) -> Path:
    """Replay expert actions after a bounded image-based initial-seed search."""

    if not episodes:
        raise ValueError("At least one dataset episode is required.")
    validation_scope = _validation_scope(
        validation_config_path,
        dataset_path,
        episodes,
    )
    if maximum_steps <= 0 or candidate_seed_count <= 0 or top_k <= 0 or settle_steps < 0:
        raise ValueError(
            "Replay steps, candidate count, and top-k must be positive; settle steps may be zero."
        )
    if top_k > candidate_seed_count:
        raise ValueError("Seed-search top-k cannot exceed the candidate count.")
    if seed_map is not None and set(seed_map) != set(episodes):
        raise ValueError("Explicit seed map must cover exactly the requested episodes.")
    dataset_config = load_dataset_config(dataset_path)
    unknown = sorted(set(episodes) - set(dataset_config.episodes))
    if unknown:
        raise ValueError(f"Episodes are outside the pinned dataset selection: {unknown}.")
    root, manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    cleaning_path = root / "cleaning_report.json"
    cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
    if cleaning.get("status") != "validated_clean":
        raise ValueError("Expert replay requires a validated-clean dataset cache.")
    contract = load_action_contract(contract_path)
    contract.validate_order(ordered_feature_names(root, dataset_config.fields.action))
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if float(info["fps"]) != contract.frequency_hz:
        raise ValueError("Dataset and simulator control frequencies differ.")
    dataset_manifest_sha256 = file_sha256(root / "manifest.json")
    initial_images, initial_image_identity = _load_initial_images(
        initial_images_path,
        expected_manifest_sha256=dataset_manifest_sha256,
        expected_revision=manifest.resolved_revision,
        episodes=episodes,
    )
    if initial_image_identity.get("validation_scope") != validation_scope:
        raise ValueError("Expert replay image artifact differs from the validation config.")
    episode_reports: list[dict[str, Any]] = []
    for episode in episodes:
        print(f"Expert replay diagnostic episode={episode}", flush=True)
        rows = _dataset_rows(
            root,
            episode,
            action_field=dataset_config.fields.action,
            state_field=dataset_config.fields.state,
            timestamp_field=dataset_config.fields.timestamp,
            episode_field=dataset_config.fields.episode_index,
            frame_field=dataset_config.fields.frame_index,
        )
        if len(rows) < maximum_steps:
            raise ValueError(
                f"Episode {episode} has {len(rows)} rows, fewer than requested {maximum_steps}."
            )
        environment = GymAlohaEnvironment(
            contract,
            maximum_episode_steps=maximum_steps + settle_steps,
        )
        try:
            if seed_map is None:
                candidates = _search_initial_seed(
                    environment,
                    initial_images[episode],
                    seed_start=candidate_seed_start,
                    seed_count=candidate_seed_count,
                    top_k=top_k,
                )
            else:
                candidates = _search_initial_seed(
                    environment,
                    initial_images[episode],
                    seed_start=seed_map[episode],
                    seed_count=1,
                    top_k=1,
                )
            best = candidates[0]
            replay = _replay_expert_actions(
                environment,
                contract,
                rows,
                action_field=dataset_config.fields.action,
                state_field=dataset_config.fields.state,
                seed=int(best["seed"]),
                maximum_steps=maximum_steps,
                settle_steps=settle_steps,
                action_mode=action_mode,
            )
        finally:
            environment.close()
        episode_reports.append(
            {
                "episode": episode,
                "dataset_frames": len(rows),
                "seed_search_top_candidates": candidates,
                "selected_seed": int(best["seed"]),
                "selected_alignment": {
                    name: best[name]
                    for name in ("pixel_mae", "pixel_rmse", "pooled_4x4_mae")
                },
                "replay": replay,
            }
        )
        print(
            f"episode={episode} selected_seed={best['seed']} "
            f"max_reward={replay['maximum_reward']} success={replay['success']}",
            flush=True,
        )
    report = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "m2_expert_replay_with_image_aligned_initial_seed",
        "validation_scope": validation_scope,
        "dataset": {
            "repo_id": dataset_config.repo_id,
            "revision": manifest.resolved_revision,
            "manifest_sha256": dataset_manifest_sha256,
            "cleaning_report_sha256": file_sha256(cleaning_path),
            "info_sha256": file_sha256(info_path),
            "object_pose_field_available": False,
        },
        "action_contract_sha256": file_sha256(contract_path),
        "frequency_hz": contract.frequency_hz,
        "maximum_steps": maximum_steps,
        "settle_steps": settle_steps,
        "action_mode": action_mode,
        "initial_image_artifact": initial_image_identity,
        "seed_search": {
            "candidate_seed_start": candidate_seed_start,
            "candidate_seed_count": candidate_seed_count,
            "selection_mode": (
                "bounded_search" if seed_map is None else "explicit_seed_map_with_image_check"
            ),
            "explicit_seed_map": (
                None
                if seed_map is None
                else {str(key): value for key, value in sorted(seed_map.items())}
            ),
            "ranking_metric": "4x4 average-pooled full-frame pixel MAE",
            "top_k": top_k,
            "limitation": (
                "The source dataset does not expose the simulator object pose. The selected seed "
                "is a bounded image-nearest heuristic and does not prove exact pose identity."
            ),
        },
        "episodes": episode_reports,
        "aggregate": {
            "successful_episodes": sum(item["replay"]["success"] for item in episode_reports),
            "episode_count": len(episode_reports),
            "maximum_reward": max(
                item["replay"]["maximum_reward"] for item in episode_reports
            ),
        },
    }
    json.dumps(report, allow_nan=False)
    digest = stable_hash(report)[:12]
    destination = (
        _run_root()
        / "m2-qwen08b-frozen-001"
        / "diagnostics"
        / f"expert-replay-{digest}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"Report: {destination}")
    return destination


def _positive_boundaries(values: list[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or result[0] <= 0 or any(
        current <= previous for previous, current in zip(result, result[1:])
    ):
        raise argparse.ArgumentTypeError("Phase boundaries must be strictly increasing.")
    return result


def _seed_map(values: list[str] | None) -> dict[int, int] | None:
    if values is None:
        return None
    result: dict[int, int] = {}
    for value in values:
        episode_text, separator, seed_text = value.partition(":")
        if not separator:
            raise argparse.ArgumentTypeError("Seed-map entries must use EPISODE:SEED syntax.")
        try:
            episode = int(episode_text)
            seed = int(seed_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError("Seed-map entries must contain integers.") from error
        if episode in result:
            raise argparse.ArgumentTypeError(f"Duplicate seed-map episode: {episode}.")
        result[episode] = seed
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cached = subparsers.add_parser("cached-policy")
    cached.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cached.add_argument("--feature-manifest", type=Path, required=True)
    cached.add_argument("--artifact", type=Path, required=True)
    cached.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
    )
    cached.add_argument(
        "--phase-boundaries",
        nargs="+",
        type=int,
        default=(100, 200, 300, 400, 500),
    )
    modality = subparsers.add_parser("modality-audit")
    modality.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modality.add_argument("--feature-manifest", type=Path, required=True)
    modality.add_argument("--artifact", type=Path, required=True)
    modality.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    modality.add_argument("--seed", type=int, default=20260809)
    images = subparsers.add_parser("export-initial-images")
    images.add_argument("--validation-config", type=Path, required=True)
    images.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "data" / "aloha_sim_insertion_m2.yaml",
    )
    images.add_argument("--episodes", nargs="+", type=int, required=True)
    replay = subparsers.add_parser("expert-replay")
    replay.add_argument("--validation-config", type=Path, required=True)
    replay.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "data" / "aloha_sim_insertion_m2.yaml",
    )
    replay.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml",
    )
    replay.add_argument("--episodes", nargs="+", type=int, required=True)
    replay.add_argument("--maximum-steps", type=int, default=500)
    replay.add_argument("--candidate-seed-start", type=int, default=0)
    replay.add_argument("--candidate-seed-count", type=int, default=256)
    replay.add_argument("--top-k", type=int, default=5)
    replay.add_argument("--initial-images", type=Path, required=True)
    replay.add_argument("--settle-steps", type=int, default=0)
    replay.add_argument(
        "--action-mode",
        choices=("contract_clipped", "source_raw"),
        default="contract_clipped",
    )
    replay.add_argument("--seed-map", nargs="+")
    pooling = subparsers.add_parser("pooling-probe")
    pooling.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pooling.add_argument("--initial-images", type=Path, required=True)
    pooling.add_argument("--episodes", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if args.command == "cached-policy":
        boundaries = _positive_boundaries(args.phase_boundaries)
        cached_policy_diagnostic(
            config_path=args.config.resolve(),
            feature_manifest_path=args.feature_manifest.resolve(),
            artifact_root=args.artifact.resolve(),
            splits=tuple(args.splits),
            phase_boundaries=boundaries,
        )
        return 0
    if args.command == "modality-audit":
        modality_audit(
            config_path=args.config.resolve(),
            feature_manifest_path=args.feature_manifest.resolve(),
            artifact_root=args.artifact.resolve(),
            splits=tuple(args.splits),
            seed=args.seed,
        )
        return 0
    if args.command == "export-initial-images":
        export_initial_images(
            validation_config_path=args.validation_config.resolve(),
            dataset_path=args.dataset.resolve(),
            episodes=tuple(args.episodes),
        )
        return 0
    if args.command == "expert-replay":
        expert_replay_diagnostic(
            validation_config_path=args.validation_config.resolve(),
            dataset_path=args.dataset.resolve(),
            contract_path=args.contract.resolve(),
            episodes=tuple(args.episodes),
            maximum_steps=args.maximum_steps,
            candidate_seed_start=args.candidate_seed_start,
            candidate_seed_count=args.candidate_seed_count,
            top_k=args.top_k,
            initial_images_path=args.initial_images.resolve(),
            settle_steps=args.settle_steps,
            action_mode=args.action_mode,
            seed_map=_seed_map(args.seed_map),
        )
        return 0
    if args.command == "pooling-probe":
        pooling_probe(
            config_path=args.config.resolve(),
            initial_images_path=args.initial_images.resolve(),
            episodes=tuple(args.episodes),
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
