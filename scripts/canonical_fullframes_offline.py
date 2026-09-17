"""Run the sealed canonical train/development offline endpoint evaluation.

The command is intended for the registered AutoDL CUDA container.  It loads
only the 40 train and five development episodes from the sealed dataset view,
uses the step-5000 zero-copy endpoint artifact, and writes create-only metrics.
No optimizer is constructed and no simulator Gate is run here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SOURCE = ROOT / "src"
for _path in (SOURCE, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from canonical_fullframes_posttrain import validate_plan  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402

DEFAULT_PLAN = (
    ROOT
    / "reports/training/m2-smolvla-canonical-fullframes-posttrain-gate-plan-004-2026-09-15.json"
)
DEFAULT_EXPERIMENT = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
DEFAULT_ARTIFACT_ID = "canonical-fullframes-20260914-001-step5000-zero-copy-001"
TRAIN_EPISODES = [
    49,
    4,
    23,
    43,
    21,
    37,
    18,
    34,
    0,
    47,
    38,
    29,
    3,
    26,
    14,
    17,
    44,
    30,
    15,
    42,
    10,
    35,
    25,
    32,
    19,
    36,
    41,
    28,
    8,
    27,
    16,
    11,
    2,
    20,
    9,
    39,
    46,
    48,
    12,
    40,
]
DEV_EPISODES = [22, 13, 7, 33, 45]
FRAME_OFFSETS = [0, 125, 250, 375]
HIDDEN_EPISODES = {31, 6, 1, 24, 5}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    json.dumps(value, allow_nan=False)
    return value


def write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _build_policy(
    artifact: Path,
    config: dict[str, Any],
    normalization: dict[str, Any],
    dataset_meta: Any,
    contract: Any,
):
    from scripts.canonical_fullframes_runtime import load_context

    context = load_context(artifact)
    return context.policy, context.pre, context.post


def _indices(dataset: LeRobotDataset, episodes: list[int]) -> list[tuple[int, int, int]]:
    episode_ids = [int(v) for v in dataset.meta.episodes["episode_index"]]
    starts = [int(v) for v in dataset.meta.episodes["dataset_from_index"]]
    lengths = [int(v) for v in dataset.meta.episodes["length"]]
    metadata = {
        episode: (start, length)
        for episode, start, length in zip(episode_ids, starts, lengths, strict=True)
    }
    result: list[tuple[int, int, int]] = []
    for episode in episodes:
        require(episode not in HIDDEN_EPISODES, "A sealed hidden episode was requested")
        require(episode in metadata, f"Episode is missing from dataset metadata: {episode}")
        start, length = metadata[episode]
        for offset in FRAME_OFFSETS:
            require(offset < length, "Registered frame offset is outside an episode")
            absolute = start + offset
            require(
                absolute in dataset.absolute_to_relative_idx,
                "Registered frame is missing from dataset view",
            )
            result.append((episode, offset, dataset.absolute_to_relative_idx[absolute]))
    return result


def _baseline_from_train_view(dataset_root: Path, contract: Any) -> tuple[list[float], int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to compute the train-only action baseline"
        ) from error
    lower = [float(v) for v in contract.lower_bounds]
    upper = [float(v) for v in contract.upper_bounds]
    sums = [0.0] * len(lower)
    count = 0
    data_root = dataset_root / "data"
    for path in sorted(data_root.rglob("*.parquet")):
        from scripts.canonical_fullframes_contract import train_action_table

        table = train_action_table(parquet, path, TRAIN_EPISODES)
        episodes = table["episode_index"].to_pylist()
        actions = table["action"].to_pylist()
        for episode, row in zip(episodes, actions, strict=True):
            require(int(episode) in TRAIN_EPISODES, "Filtered baseline leaked nontraining row")
            values = (
                row
                if isinstance(row, list) and row and isinstance(row[0], (int, float))
                else row[0]
            )
            if not isinstance(values, list) or len(values) != len(lower):
                raise ValueError("Train action parquet shape differs from the Action Contract")
            for index, value in enumerate(values):
                clipped = min(upper[index], max(lower[index], float(value)))
                sums[index] += clipped
            count += 1
    require(count == 20000, f"Train-only baseline row count drift: {count}")
    return [value / count for value in sums], count


def _sync() -> None:
    torch.cuda.synchronize()


def _evaluate_cohort(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: LeRobotDataset,
    indices: list[tuple[int, int, int]],
    contract: Any,
    *,
    noise_mode: str,
    noise_seed: int | None,
    baseline: list[float],
) -> dict[str, Any]:
    if noise_mode == "seeded_standard_normal":
        require(noise_seed is not None, "Seeded noise requires a seed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
    else:
        require(noise_seed is None, "Zero noise cannot carry a random seed")
        generator = None
    policy.eval()
    lower = contract.lower_bounds.to(torch.float64)
    upper = contract.upper_bounds.to(torch.float64)
    total_abs = total_sq = first_abs = baseline_abs = baseline_sq = 0.0
    elements = first_elements = 0
    invalid = violations = target_violations = 0
    smooth_sum = smooth_elements = 0
    flow_losses: list[float] = []
    latencies: list[float] = []
    internal: list[torch.Tensor] = []
    sample_records: list[dict[str, Any]] = []

    for expected_episode, offset, relative_index in indices:
        from scripts.iris_runtime import budget

        budget(float(os.environ["ROSETTA_POSTTRAIN_DEADLINE"]))
        sample = dataset[relative_index]
        require(int(sample["frame_index"]) == offset, "Offline actual frame drift")
        batch = default_collate([sample])
        episode_value = batch.get("episode_index")
        raw_action = batch.get("action")
        action_is_pad = batch.get("action_is_pad")
        require(
            isinstance(episode_value, torch.Tensor)
            and int(episode_value.item()) == expected_episode,
            "Offline episode identity drift",
        )
        require(
            isinstance(raw_action, torch.Tensor) and list(raw_action.shape) == [1, 50, 14],
            "Offline action chunk shape drift",
        )
        require(
            isinstance(action_is_pad, torch.Tensor) and not bool(action_is_pad.any().item()),
            "Offline padding mask is not empty",
        )
        target = raw_action.detach().cpu().to(torch.float64)
        target = torch.maximum(torch.minimum(target, upper.view(1, 1, -1)), lower.view(1, 1, -1))
        for key in list(batch):
            value = batch[key]
            if (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and key.startswith("observation.images")
            ):
                from rosetta_reality.vla.image_scaling import canonical_rgb_uint8

                batch[key] = canonical_rgb_uint8(value)
        batch = preprocessor(batch)
        normalized_action = batch.get("action")
        require(
            isinstance(normalized_action, torch.Tensor), "Offline processor removed target action"
        )
        noise_shape = (1, 50, int(policy.config.max_action_dim))
        if noise_mode == "zeros":
            noise = torch.zeros(noise_shape, device="cuda", dtype=normalized_action.dtype)
        else:
            noise = torch.randn(
                noise_shape, generator=generator, device="cpu", dtype=torch.float32
            ).to(device="cuda", dtype=normalized_action.dtype)
        flow_time = torch.full((1,), 0.5, device="cuda", dtype=normalized_action.dtype)
        policy.reset()
        _sync()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            from scripts.iris_runtime import observations

            predicted_model = policy.predict_action_chunk(observations(batch), noise=noise)

        _sync()
        latencies.append(time.perf_counter() - started)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch, noise=noise, time=flow_time)
        predicted = postprocessor(predicted_model)
        require(
            isinstance(predicted, torch.Tensor) and list(predicted.shape) == [1, 50, 14],
            "Offline output chunk shape drift",
        )
        finite = torch.isfinite(predicted)
        invalid += int((~finite).sum().item())
        if not bool(finite.all()):
            raise FloatingPointError("Offline output is non-finite")
        predicted_cpu = predicted.detach().cpu().to(torch.float64)
        raw_adapter = None
        for step in getattr(postprocessor, "steps", []):
            candidate = getattr(step, "last_unclipped_action", None)
            if isinstance(candidate, torch.Tensor):
                raw_adapter = candidate.detach().cpu().to(torch.float64)
            candidate_internal = getattr(step, "last_model_action", None)
            if isinstance(candidate_internal, torch.Tensor):
                from scripts.canonical_fullframes_contract import internal_grippers

                internal.append(
                    internal_grippers(candidate_internal, contract.dimensions)
                    .detach()
                    .cpu()
                    .to(torch.float64)
                    .reshape(-1)
                )
        if raw_adapter is None:
            raw_adapter = predicted_cpu
        violations += int(
            ((predicted_cpu < lower.view(1, 1, -1)) | (predicted_cpu > upper.view(1, 1, -1)))
            .sum()
            .item()
        )
        target_violations += int(
            (
                (raw_action.detach().cpu().to(torch.float64) < lower.view(1, 1, -1))
                | (raw_action.detach().cpu().to(torch.float64) > upper.view(1, 1, -1))
            )
            .sum()
            .item()
        )
        error = predicted_cpu - target
        baseline_tensor = (
            torch.tensor(baseline, dtype=torch.float64).view(1, 1, -1).expand_as(target)
        )
        baseline_error = baseline_tensor - target
        total_abs += float(error.abs().sum())
        total_sq += float(error.square().sum())
        first_abs += float(error[:, 0].abs().sum())
        baseline_abs += float(baseline_error.abs().sum())
        baseline_sq += float(baseline_error.square().sum())
        elements += error.numel()
        first_elements += error[:, 0].numel()
        differences = predicted_cpu[:, 1:] - predicted_cpu[:, :-1]
        smooth_sum += float(differences.abs().sum())
        smooth_elements += differences.numel()
        loss_value = float(loss.detach().cpu().item())
        require(math.isfinite(loss_value), "Offline fixed-flow loss is non-finite")
        flow_losses.append(loss_value)
        sample_records.append(
            {
                "episode": expected_episode,
                "frame_offset": offset,
                "chunk_length": 50,
                "action_dimension": 14,
            }
        )

    require(len(sample_records) == len(indices), "Offline sample record count drift")
    return {
        "noise_mode": noise_mode,
        "noise_seed": noise_seed,
        "sample_count": len(indices),
        "action_mae": total_abs / elements,
        "action_rmse": math.sqrt(total_sq / elements),
        "first_action_mae": first_abs / first_elements,
        "fixed_flow_loss": sum(flow_losses) / len(flow_losses),
        "baseline_action_mae": baseline_abs / elements,
        "baseline_action_rmse": math.sqrt(baseline_sq / elements),
        "invalid_action_rate": invalid / elements,
        "action_contract_limit_violation_rate": violations / elements,
        "target_contract_projection_rate": target_violations / (len(indices) * 50 * 14),
        "action_smoothness_mean_abs_delta": smooth_sum / max(1, smooth_elements),
        "inference_latency_mean_seconds": sum(latencies) / len(latencies),
        "inference_latency_p95_seconds": sorted(latencies)[
            max(0, math.ceil(len(latencies) * 0.95) - 1)
        ],
        "internal_gripper_support": {
            "sampled": int(sum(value.numel() for value in internal)),
            "minimum": float(torch.cat(internal).min()) if internal else None,
            "maximum": float(torch.cat(internal).max()) if internal else None,
        },
        "sample_identities": sample_records,
    }


def run(args: argparse.Namespace) -> int:
    require(
        os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("HF_DATASETS_OFFLINE") == "1",
        "Offline evaluation requires networking-disabled environment",
    )
    require(torch.cuda.is_available(), "Canonical offline evaluation requires CUDA")
    plan = validate_plan(args.plan.resolve())
    artifact = Path(args.artifact).resolve()
    config = load_json(artifact / "config.json")
    normalization = load_json(artifact / "normalization.json")
    manifest = load_json(artifact / "manifest.json")
    from scripts.canonical_fullframes_contract import verify_files

    verify_files(artifact / "pretrained_model", plan["endpoint"]["files"])
    require(
        manifest.get("status") == "verified" and manifest.get("selected_checkpoint_step") == 5000,
        "Endpoint artifact identity is invalid",
    )
    require(
        manifest.get("reload", {}).get("exact_tensor_equality") is True,
        "Endpoint reload precondition is missing",
    )
    require(
        manifest.get("hidden_test_loaded") is False and config.get("hidden_test_loaded") is False,
        "Artifact hidden-test boundary is invalid",
    )
    contract = load_action_contract(ROOT / plan["gate_protocol_authority"]["action_contract"])
    dataset_root = Path(args.dataset_root).resolve()
    dataset_meta = LeRobotDatasetMetadata(
        config["dataset_id"], root=dataset_root, revision=config["dataset_revision"]
    )
    policy_cfg = SmolVLAConfig.from_pretrained(artifact / "pretrained_model", local_files_only=True)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, dataset_meta)
    episodes = [*TRAIN_EPISODES, *DEV_EPISODES]
    require(
        not set(episodes) & HIDDEN_EPISODES, "Offline episode set intersects sealed hidden episodes"
    )
    dataset = LeRobotDataset(
        config["dataset_id"],
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        revision=config["dataset_revision"],
        download_videos=False,
        return_uint8=True,
    )
    train_indices = _indices(dataset, TRAIN_EPISODES)
    dev_indices = _indices(dataset, DEV_EPISODES)
    require(len(train_indices) == 160 and len(dev_indices) == 20, "Offline cohort size drift")
    baseline, baseline_rows = _baseline_from_train_view(dataset_root, contract)
    policy, preprocessor, postprocessor = _build_policy(
        artifact, config, normalization, dataset_meta, contract
    )
    from scripts.diagnose_zen_noise_transfer import parameter_digests

    before = parameter_digests(policy)
    results: dict[str, Any] = {}
    for cohort, indices in (("train", train_indices), ("development", dev_indices)):
        results[f"{cohort}:zeros"] = _evaluate_cohort(
            policy,
            preprocessor,
            postprocessor,
            dataset,
            indices,
            contract,
            noise_mode="zeros",
            noise_seed=None,
            baseline=baseline,
        )
        for seed in [1000, 1001, 1002, 1003, 1004]:
            results[f"{cohort}:seeded_standard_normal:{seed}"] = _evaluate_cohort(
                policy,
                preprocessor,
                postprocessor,
                dataset,
                indices,
                contract,
                noise_mode="seeded_standard_normal",
                noise_seed=seed,
                baseline=baseline,
            )
    require(parameter_digests(policy) == before, "Offline evaluation changed model parameters")
    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "canonical_fullframes_offline_endpoint_evaluation",
        "plan_id": plan["plan_id"],
        "plan_sha256": file_sha256(args.plan.resolve()),
        "artifact_id": manifest["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact / "manifest.json"),
        "dataset_id": config["dataset_id"],
        "dataset_revision": config["dataset_revision"],
        "train_episodes": TRAIN_EPISODES,
        "development_episodes": DEV_EPISODES,
        "frame_offsets": FRAME_OFFSETS,
        "hidden_test_loaded": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "parameters_unchanged": True,
        "baseline": {
            "source": "train-only raw action parquet, projected to Action Contract",
            "train_rows": baseline_rows,
            "action_mean": baseline,
        },
        "results": results,
        "integrity": {
            "sample_count_train": len(train_indices),
            "sample_count_development": len(dev_indices),
            "full_chunk_length": 50,
            "action_dimension": 14,
            "primary_flow_time": 0.5,
            "diagnostic_noise_seeds": [1000, 1001, 1002, 1003, 1004],
        },
        "runtime": {
            "device": "cuda",
            "network_disabled": True,
            "mixed_precision": "bf16",
            "nested_docker_used": False,
        },
    }
    output = Path(args.output).resolve()
    write_json_create_only(output, report)
    max_report_bytes = int(
        plan["posttrain_runtime"]["storage_budget_basis"]["offline_reports_max_bytes"]
    )
    require(
        output.stat().st_size <= max_report_bytes, "Offline report exceeds the sealed storage cap"
    )
    print(
        json.dumps(
            {"status": report["status"], "output": str(output), "cohorts": sorted(results)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
