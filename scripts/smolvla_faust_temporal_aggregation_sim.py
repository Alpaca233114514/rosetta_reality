"""Run Faust with receding-horizon temporal aggregation across predicted chunks."""

from __future__ import annotations

import math
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


def _aggregate_current_action(
    predictions: list[tuple[int, torch.Tensor, torch.Tensor]],
    current_step: int,
    decay: float,
    weighting: str = "newer_predictions",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate all finite chunks covering ``current_step`` with registered weights."""

    if not predictions or not math.isfinite(decay) or decay <= 0.0:
        raise ValueError("Temporal aggregation requires predictions and positive decay.")
    selected: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for origin, raw, processed in predictions:
        offset = current_step - origin
        if (
            origin < 0
            or offset < 0
            or raw.ndim != 2
            or processed.ndim != 2
            or raw.shape != processed.shape
            or offset >= raw.shape[0]
            or not bool(torch.isfinite(raw).all())
            or not bool(torch.isfinite(processed).all())
        ):
            continue
        selected.append((origin, raw[offset], processed[offset]))
    if not selected:
        raise ValueError("No finite prediction covers the current control step.")
    if weighting == "newer_predictions":
        exponents = [current_step - origin for origin, _, _ in selected]
    elif weighting == "older_predictions_original_act_order":
        # ACT stores rows by query time (oldest to newest) and applies
        # exp(-k * arange(n)); preserve that exact ordering here.
        exponents = list(range(len(selected)))
    else:
        raise ValueError("Unsupported temporal-aggregation weighting order.")
    weights = torch.tensor(
        [math.exp(-decay * exponent) for exponent in exponents],
        dtype=torch.float64,
    )
    weights /= weights.sum()

    def aggregate(index: int) -> torch.Tensor:
        values = torch.stack([entry[index].to(torch.float64) for entry in selected])
        return (values * weights[:, None]).sum(dim=0).to(selected[-1][index].dtype)

    return aggregate(1), aggregate(2)


class _TemporalAggregationActionRepairOnlineSmolVLA(
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
            raise ValueError("Faust temporal-aggregation plan has no inference contract.")
        self._weighting = str(inference.get("temporal_aggregation", ""))
        self._decay = float(inference.get("temporal_aggregation_decay", float("nan")))
        supported = {
            ("exponential_newer_prediction_weighting", 0.05): "newer_predictions",
            (
                "exponential_older_prediction_weighting_original_act_order",
                0.01,
            ): "older_predictions_original_act_order",
        }
        self._weighting_order = supported.get((self._weighting, self._decay), "")
        if not self._weighting_order or inference.get(
            "aggregation_space"
        ) != "standard_action_space":
            raise ValueError("Faust temporal-aggregation contract is invalid.")
        self._history: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self._control_step = 0

    def configure_noise(self, mode: str, seed: int | None) -> None:
        super().configure_noise(mode, seed)
        self._history.clear()
        self._control_step = 0

    def predict(
        self, observation: dict[str, Any], instruction: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw, processed = super().predict(observation, instruction)
        self._history.append((self._control_step, raw, processed))
        earliest = self._control_step - raw.shape[0] + 1
        self._history = [entry for entry in self._history if entry[0] >= earliest]
        aggregated = _aggregate_current_action(
            self._history,
            self._control_step,
            self._decay,
            self._weighting_order,
        )
        self._control_step += 1
        return aggregated[0].unsqueeze(0), aggregated[1].unsqueeze(0)


def main() -> int:
    plan_path = action_repair._plan_path()
    plan = simulator._load_yaml(plan_path)
    run_root = Path(os.environ["ROSETTA_RUN_ROOT"]).resolve()
    cache_root = (
        run_root
        / "compiler_cache"
        / f"faust-temporal-aggregation-{file_sha256(plan_path)[:12]}"
    )
    triton_cache = cache_root / "triton"
    inductor_cache = cache_root / "inductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)

    simulator._OnlineSmolVLA = _TemporalAggregationActionRepairOnlineSmolVLA
    original_create_json = simulator.create_json

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        if payload.get("diagnostic") == "smolvla_non_gate_chunk_execution_strategy":
            payload["bounded_gripper_decoder"] = True
            payload["temporal_aggregation"] = plan["inference"][
                "temporal_aggregation"
            ]
            payload["temporal_aggregation_decay"] = plan["inference"][
                "temporal_aggregation_decay"
            ]
            payload["aggregation_space"] = "standard_action_space"
            payload["temporal_aggregation_sim_script_sha256"] = file_sha256(
                Path(__file__)
            )
        original_create_json(path, payload)

    simulator.create_json = create_json
    return simulator.main()


if __name__ == "__main__":
    raise SystemExit(main())
