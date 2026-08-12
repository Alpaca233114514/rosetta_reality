"""Create no-weights evidence for a registered SmolVLA fixed-frame set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as arrow_dataset
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_fixed_overfit_002.yaml"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import file_sha256, stable_hash  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla import load_smolvla_experiment  # noqa: E402
from rosetta_reality.vla.fixed_samples import load_fixed_frame_protocol  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path.name}.")
    return value


def _absolute_root(environment: str, fallback: Path) -> Path:
    raw = os.environ.get(environment)
    root = Path(raw) if raw else fallback
    if raw and not root.is_absolute():
        raise ValueError(f"{environment} must be absolute.")
    return root.resolve()


def _dataset_root(experiment: dict[str, Any]) -> Path:
    root = _absolute_root("ROSETTA_DATA_ROOT", REPOSITORY_ROOT / "data/lerobot_m2")
    repo_id = str(experiment["dataset"]["identifier"]).replace("/", "--")
    return root / repo_id / str(experiment["dataset"]["revision"])


def _chunk_summary(chunk: np.ndarray) -> dict[str, Any]:
    return {
        "minimum": chunk.min(axis=0).tolist(),
        "maximum": chunk.max(axis=0).tolist(),
        "mean": chunk.mean(axis=0).tolist(),
        "left_gripper": {
            "minimum": float(chunk[:, 6].min()),
            "maximum": float(chunk[:, 6].max()),
            "mean": float(chunk[:, 6].mean()),
        },
        "right_gripper": {
            "minimum": float(chunk[:, 13].min()),
            "maximum": float(chunk[:, 13].max()),
            "mean": float(chunk[:, 13].mean()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Fixed-sample diagnosis requires networking disabled.")

    config_path = args.config.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    protocol = load_fixed_frame_protocol(experiment, "overfit")
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract = load_action_contract(contract_path)
    dataset_config = _load_yaml(REPOSITORY_ROOT / str(experiment["dataset"]["config"]))
    fields = dataset_config["fields"]
    dataset_root = _dataset_root(experiment)
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("repo_id") != experiment["dataset"]["identifier"]
        or manifest.get("resolved_revision") != experiment["dataset"]["revision"]
    ):
        raise ValueError("Prepared dataset identity differs from the experiment.")
    table = arrow_dataset.dataset(dataset_root / "data", format="parquet").to_table(
        columns=[fields["episode_index"], fields["frame_index"], fields["action"]],
        filter=arrow_dataset.field(fields["episode_index"]) == protocol.episode,
    )
    rows = sorted(table.to_pylist(), key=lambda row: int(row[fields["frame_index"]]))
    frame_indices = [int(row[fields["frame_index"]]) for row in rows]
    if not rows or frame_indices != list(range(len(rows))):
        raise ValueError("The registered fixed episode is not contiguous from frame zero.")
    actions = np.asarray([row[fields["action"]] for row in rows], dtype=np.float64)
    if actions.shape != (len(rows), contract.dimension) or not np.isfinite(actions).all():
        raise ValueError("Fixed-episode actions differ from the Action Contract.")
    lower = contract.lower_bounds.numpy().astype(np.float64, copy=False)
    upper = contract.upper_bounds.numpy().astype(np.float64, copy=False)
    tolerance = contract.source_overshoot_tolerances.numpy().astype(np.float64, copy=False)
    overshoot = np.maximum(np.maximum(lower - actions, actions - upper), 0.0)
    if np.any(overshoot > tolerance + 1e-12):
        raise ValueError("Fixed-episode source action exceeds registered tolerance.")
    projected = np.maximum(np.minimum(actions, upper), lower)

    chunk_length = int(contract.chunk_length)
    anchors: list[dict[str, Any]] = []
    chunks: list[np.ndarray] = []
    for frame_index in protocol.frame_indices:
        stop = frame_index + chunk_length
        if stop > len(projected):
            raise ValueError("A fixed anchor cannot provide the registered action chunk.")
        chunk = projected[frame_index:stop]
        chunks.append(chunk)
        anchors.append(
            {
                "episode_index": protocol.episode,
                "frame_index": frame_index,
                "chunk_frame_range_inclusive": [frame_index, stop - 1],
                "source_action": actions[frame_index].tolist(),
                "projected_action": projected[frame_index].tolist(),
                "chunk": _chunk_summary(chunk),
            }
        )
    aggregate = np.concatenate(chunks, axis=0)
    left = aggregate[:, 6]
    right = aggregate[:, 13]
    coverage = {
        "left_gripper_closed_or_low_present": bool(np.any(left <= 0.25)),
        "left_gripper_open_present": bool(np.any(left >= 0.50)),
        "right_gripper_closed_present": bool(np.any(right <= 0.10)),
        "right_gripper_open_present": bool(np.any(right >= 0.50)),
        "right_gripper_transition_chunks": sum(
            int(chunk[:, 13].min() <= 0.10 and chunk[:, 13].max() >= 0.50)
            for chunk in chunks
        ),
    }
    if not all(
        coverage[key]
        for key in (
            "left_gripper_closed_or_low_present",
            "left_gripper_open_present",
            "right_gripper_closed_present",
            "right_gripper_open_present",
        )
    ) or int(coverage["right_gripper_transition_chunks"]) < 1:
        raise ValueError("The fixed anchors do not cover registered gripper phases.")

    protocol_payload = protocol.as_dict()
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_fixed_sample_no_weights_diagnostic",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "diagnostic_script_sha256": file_sha256(Path(__file__)),
        "action_contract_sha256": file_sha256(contract_path),
        "dataset_revision": experiment["dataset"]["revision"],
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "fixed_sample_protocol": protocol_payload,
        "fixed_sample_protocol_sha256": stable_hash(protocol_payload),
        "fixed_sample_count": len(protocol.frame_indices),
        "action_chunk_length": chunk_length,
        "anchors": anchors,
        "aggregate_projected_target": _chunk_summary(aggregate),
        "coverage": coverage,
        "source_projection": {
            "projected_element_count": int((overshoot > 0.0).sum()),
            "projected_element_rate": float((overshoot > 0.0).mean()),
        },
        "model_weights_loaded": False,
        "optimizer_created": False,
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
        "episodes_loaded": [protocol.episode],
        "network_disabled": True,
    }
    json.dumps(report, allow_nan=False)
    run_root = _absolute_root("ROSETTA_RUN_ROOT", REPOSITORY_ROOT / "runs")
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "diagnostics"
        / f"fixed-samples-{stable_hash(report)[:16]}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
