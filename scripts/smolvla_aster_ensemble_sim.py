"""Run Aster with a registered Gaussian action ensemble at inference time."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import smolvla_action_repair_sim_gate as action_repair  # noqa: E402
import smolvla_sim_gate as simulator  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402


def _mean_predictions(
    predictions: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average finite same-shape samples in standard action space."""

    if not predictions:
        raise ValueError("Gaussian action ensembling requires at least one sample.")
    reference_shapes = tuple(tensor.shape for tensor in predictions[0])
    if len(reference_shapes) != 2:
        raise ValueError("Each action sample must expose raw and processed tensors.")
    for prediction in predictions:
        if (
            len(prediction) != 2
            or tuple(tensor.shape for tensor in prediction) != reference_shapes
            or any(not bool(torch.isfinite(tensor).all()) for tensor in prediction)
        ):
            raise ValueError("Gaussian action samples must be finite and shape-aligned.")
    raw = torch.stack([prediction[0] for prediction in predictions]).mean(dim=0)
    processed = torch.stack([prediction[1] for prediction in predictions]).mean(dim=0)
    return raw, processed


class _EnsembleActionRepairOnlineSmolVLA(
    action_repair._ActionRepairOnlineSmolVLA
):
    def __init__(
        self,
        artifact: Path,
        config: dict[str, Any],
        normalization: dict[str, Any],
        contract: Any,
    ) -> None:
        super().__init__(artifact, config, normalization, contract)
        plan = simulator._load_yaml(action_repair._plan_path())
        inference = plan.get("inference")
        if not isinstance(inference, dict):
            raise ValueError("Aster ensemble plan has no inference contract.")
        self._ensemble_samples = int(
            inference.get("gaussian_samples_per_observation", -1)
        )
        if (
            inference.get("noise") != "seeded_standard_normal"
            or self._ensemble_samples != 4
            or inference.get("sample_aggregation")
            != "standard_action_arithmetic_mean"
        ):
            raise ValueError("Aster Gaussian ensemble contract is invalid.")

    def predict(
        self, observation: dict[str, Any], instruction: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predict_one = super().predict
        return _mean_predictions(
            [
                predict_one(observation, instruction)
                for _ in range(self._ensemble_samples)
            ]
        )


def main() -> int:
    plan_path = action_repair._plan_path()
    run_root = Path(os.environ["ROSETTA_RUN_ROOT"]).resolve()
    cache_root = run_root / "compiler_cache" / f"aster-ensemble-{file_sha256(plan_path)[:12]}"
    triton_cache = cache_root / "triton"
    inductor_cache = cache_root / "inductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)

    simulator._OnlineSmolVLA = _EnsembleActionRepairOnlineSmolVLA
    original_create_json = simulator.create_json

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        if payload.get("gate") in {
            "m2_gate_3_small_policy_rollout",
            "m2_gate_4_development_task_evaluation",
        } or payload.get("diagnostic") == "smolvla_non_gate_chunk_execution_strategy":
            payload["bounded_gripper_decoder"] = True
            payload["gaussian_samples_per_observation"] = 4
            payload["sample_aggregation"] = "standard_action_arithmetic_mean"
            payload["ensemble_sim_script_sha256"] = file_sha256(Path(__file__))
        original_create_json(path, payload)

    simulator.create_json = create_json
    return simulator.main()


if __name__ == "__main__":
    raise SystemExit(main())
