"""Verify the explicit SmolVLA ALOHA adapter without loading model weights or data rows."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import file_sha256, stable_hash  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.processor import (  # noqa: E402
    BOUNDED_SINE_ACTION_ADAPTER,
    model_action_to_standard,
    standard_aloha_action_to_model,
    standard_aloha_state_to_pi,
)

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_001.yaml"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SmolVLA action-space diagnostic input must be an object.")
    return value


def _statistics(report: dict[str, Any], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    raw = report.get("effective_stats", {}).get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"Normalization report is missing {key} statistics.")
    mean = torch.tensor(raw.get("mean"), dtype=torch.float64)
    standard_deviation = torch.tensor(raw.get("std"), dtype=torch.float64)
    if (
        mean.ndim != 1
        or standard_deviation.shape != mean.shape
        or not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(standard_deviation).all())
        or bool((standard_deviation <= 0).any())
    ):
        raise ValueError(f"Normalization report has invalid {key} statistics.")
    return mean, standard_deviation


def diagnose(config_path: Path, normalization_path: Path) -> Path:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("SmolVLA action-space diagnostics require networking disabled.")
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    normalization = _load_json(normalization_path)
    if (
        normalization.get("experiment_id") != experiment["experiment_id"]
        or normalization.get("action_space") != action_space.as_dict()
        or normalization.get("source_split") != "train"
        or normalization.get("validation_episodes_loaded") is not False
        or normalization.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Normalization report differs from the repair experiment.")

    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract = load_action_contract(contract_path)
    action_mean, action_std = _statistics(normalization, "action")
    state_mean, state_std = _statistics(normalization, "observation.state")
    if action_mean.numel() != contract.dimension or state_mean.numel() != contract.dimension:
        raise ValueError("Repair normalization dimension differs from the Action Contract.")

    standard_actions = torch.stack(
        [
            contract.lower_bounds.to(torch.float64),
            (
                contract.lower_bounds.to(torch.float64)
                + contract.upper_bounds.to(torch.float64)
            )
            / 2,
            contract.upper_bounds.to(torch.float64),
        ]
    )[:, None, :]
    standard_states = standard_actions.clone()
    internal_actions = standard_aloha_action_to_model(
        standard_actions, action_space.representation_adapter
    )
    decoded_actions = model_action_to_standard(
        internal_actions, action_space.representation_adapter
    )
    internal_states = standard_aloha_state_to_pi(standard_states)
    normalized_actions = (internal_actions - action_mean.view(1, 1, -1)) / action_std.view(
        1, 1, -1
    )
    normalized_states = (internal_states - state_mean.view(1, 1, -1)) / state_std.view(
        1, 1, -1
    )

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy_shell = object.__new__(SmolVLAPolicy)
    upstream_actions = SmolVLAPolicy._pi_aloha_encode_actions_inv(
        policy_shell, standard_actions.clone()
    )
    upstream_states = SmolVLAPolicy._pi_aloha_decode_state(
        policy_shell, standard_states[:, 0, :].clone()
    )[:, None, :]
    action_indices = (
        [index for index in range(contract.dimension) if index not in (6, 13)]
        if action_space.representation_adapter == BOUNDED_SINE_ACTION_ADAPTER
        else list(range(contract.dimension))
    )
    parity_error = torch.maximum(
        (upstream_actions[..., action_indices] - internal_actions[..., action_indices])
        .abs()
        .max(),
        (upstream_states - internal_states).abs().max(),
    )
    if float(parity_error) > 1e-12:
        raise ValueError("Rosetta pi-Aloha adapter differs from the pinned upstream formulas.")
    round_trip_error = (decoded_actions - standard_actions).abs()
    maximum_error = float(round_trip_error.max())
    passed = maximum_error <= 1e-9
    if not passed:
        raise ValueError("Pinned upstream pi-ALOHA action adapter failed round-trip.")
    adapter_source = Path(inspect.getsourcefile(SmolVLAPolicy) or "")
    if not adapter_source.is_file():
        raise FileNotFoundError("Pinned upstream SmolVLA adapter source is unavailable.")
    bounded_probe: dict[str, Any] | None = None
    if action_space.representation_adapter == BOUNDED_SINE_ACTION_ADAPTER:
        arbitrary = torch.zeros(33, 1, contract.dimension, dtype=torch.float64)
        values = torch.linspace(-4 * torch.pi, 4 * torch.pi, 33, dtype=torch.float64)
        arbitrary[..., 6] = values[:, None]
        arbitrary[..., 13] = values.flip(0)[:, None]
        decoded_arbitrary = model_action_to_standard(
            arbitrary, action_space.representation_adapter
        )
        grippers = decoded_arbitrary[..., [6, 13]]
        bounded_probe = {
            "passed": bool(((grippers >= 0) & (grippers <= 1)).all()),
            "minimum": float(grippers.min()),
            "maximum": float(grippers.max()),
            "probe_internal_minimum": float(values.min()),
            "probe_internal_maximum": float(values.max()),
        }
        if bounded_probe["passed"] is not True:
            raise ValueError("Bounded gripper decoder emitted an illegal action.")

    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_action_space_no_weights_diagnostic",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "experiment_inheritance": experiment.get("experiment_inheritance"),
        "action_contract_sha256": file_sha256(contract_path),
        "normalization_report_sha256": file_sha256(normalization_path),
        "action_space": action_space.as_dict(),
        "upstream_repository": experiment["upstream"]["repository"],
        "upstream_revision": experiment["upstream"]["revision"],
        "upstream_adapter_source_sha256": file_sha256(adapter_source),
        "representative_action_points": [
            "contract_lower",
            "contract_midpoint",
            "contract_upper",
        ],
        "upstream_formula_parity_maximum_absolute_error": float(parity_error),
        "upstream_formula_parity_scope": (
            "state_and_arm_action_dimensions"
            if action_space.representation_adapter == BOUNDED_SINE_ACTION_ADAPTER
            else "state_and_all_action_dimensions"
        ),
        "bounded_gripper_output": bounded_probe,
        "round_trip": {
            "passed": passed,
            "maximum_absolute_standard_space_error": maximum_error,
            "per_dimension_maximum_absolute_error": round_trip_error.amax(dim=(0, 1)).tolist(),
        },
        "internal_action_range": {
            "minimum": internal_actions.amin(dim=(0, 1)).tolist(),
            "maximum": internal_actions.amax(dim=(0, 1)).tolist(),
        },
        "normalized_internal_action_range": {
            "minimum": normalized_actions.amin(dim=(0, 1)).tolist(),
            "maximum": normalized_actions.amax(dim=(0, 1)).tolist(),
        },
        "internal_state_range": {
            "minimum": internal_states.amin(dim=(0, 1)).tolist(),
            "maximum": internal_states.amax(dim=(0, 1)).tolist(),
        },
        "normalized_internal_state_range": {
            "minimum": normalized_states.amin(dim=(0, 1)).tolist(),
            "maximum": normalized_states.amax(dim=(0, 1)).tolist(),
        },
        "policy_level_adapter_enabled": action_space.adapt_to_pi_aloha,
        "model_weights_loaded": False,
        "dataset_rows_loaded": False,
        "optimizer_created": False,
        "gradients_enabled": False,
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
        "network_disabled": True,
    }
    json.dumps(report, allow_nan=False)
    run_root = Path(os.environ.get("ROSETTA_RUN_ROOT", REPOSITORY_ROOT / "runs"))
    if not run_root.is_absolute():
        run_root = (REPOSITORY_ROOT / run_root).resolve()
    digest = stable_hash(
        {
            "config": file_sha256(config_path),
            "normalization": file_sha256(normalization_path),
            "script": file_sha256(Path(__file__)),
        }
    )[:16]
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "diagnostics"
        / f"action-space-{digest}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.name}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--normalization-report", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.config.resolve(), args.normalization_report.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
