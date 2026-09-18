"""Physical action semantics shared by datasets, policies, and simulators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ValueError(f"{context} is missing {key!r}.") from error


@dataclass(frozen=True, slots=True)
class ActionDimension:
    """One ordered physical or normalized action field."""

    name: str
    unit: str
    minimum: float
    maximum: float
    encoding: str | None = None
    source_overshoot_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("Action dimension name and unit must be non-empty.")
        if not self.minimum < self.maximum:
            raise ValueError(f"Action dimension {self.name!r} has invalid limits.")
        if self.source_overshoot_tolerance < 0:
            raise ValueError(
                f"Action dimension {self.name!r} has a negative source overshoot tolerance."
            )


@dataclass(frozen=True, slots=True)
class ActionContract:
    """Complete ordered Rosetta action contract for one embodiment."""

    name: str
    schema_version: int
    embodiment: str
    environment_id: str
    action_type: str
    semantics: str
    control_mode: str
    space: str
    reference_frame: str
    frequency_hz: float
    timestamp_alignment: str
    chunk_length: int
    chunk_execution: str
    simulator_expansion: str
    dimensions: tuple[ActionDimension, ...]
    chunk_execution_steps: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported action contract version: {self.schema_version}.")
        if not self.dimensions:
            raise ValueError("Action contract must contain at least one dimension.")
        if self.frequency_hz <= 0 or self.chunk_length <= 0:
            raise ValueError("Action frequency and chunk length must be positive.")
        if not 1 <= self.chunk_execution_steps <= self.chunk_length:
            raise ValueError("Action chunk execution steps must be within the chunk length.")
        names = self.dimension_names
        if len(set(names)) != len(names):
            raise ValueError("Action dimension names must be unique.")

    @property
    def dimension(self) -> int:
        """Number of ordered logical action fields."""

        return len(self.dimensions)

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Ordered physical names used for compatibility checks."""

        return tuple(dimension.name for dimension in self.dimensions)

    @property
    def lower_bounds(self) -> Tensor:
        """Return logical lower bounds in contract order."""

        return torch.tensor([dimension.minimum for dimension in self.dimensions])

    @property
    def upper_bounds(self) -> Tensor:
        """Return logical upper bounds in contract order."""

        return torch.tensor([dimension.maximum for dimension in self.dimensions])

    @property
    def source_overshoot_tolerances(self) -> Tensor:
        """Maximum source-data excess that the adapter may safely saturate."""

        return torch.tensor(
            [dimension.source_overshoot_tolerance for dimension in self.dimensions]
        )

    def validate_order(self, names: tuple[str, ...] | list[str]) -> None:
        """Reject equal-width data whose physical field ordering differs."""

        received = tuple(names)
        if received != self.dimension_names:
            raise ValueError(
                "Action ordering is incompatible with the Rosetta Action Contract: "
                f"expected {self.dimension_names!r}, received {received!r}."
            )

    def validate_tensor(self, action: Tensor, *, allow_chunk: bool = True) -> None:
        """Validate shape and numeric finiteness without clipping."""

        valid_rank = action.ndim >= 1 if allow_chunk else action.ndim == 1
        if not valid_rank or action.shape[-1] != self.dimension:
            expected = f"[..., {self.dimension}]" if allow_chunk else f"[{self.dimension}]"
            raise ValueError(
                f"Action must have shape {expected}, received {tuple(action.shape)}."
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("Action contains NaN or Inf.")

    def clip(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """Clip to physical limits and return a per-element clipping mask."""

        self.validate_tensor(action)
        lower = self.lower_bounds.to(device=action.device, dtype=action.dtype)
        upper = self.upper_bounds.to(device=action.device, dtype=action.dtype)
        clipped = torch.maximum(torch.minimum(action, upper), lower)
        return clipped, clipped.ne(action)


def load_action_contract(path: Path) -> ActionContract:
    """Load a checked-in YAML contract without importing a simulator."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Action contract must contain a YAML mapping.")
    action = _required(raw, "action", "Action contract")
    if not isinstance(action, dict):
        raise ValueError("Action contract field 'action' must be a mapping.")
    raw_dimensions = _required(action, "dimensions", "action")
    if not isinstance(raw_dimensions, list):
        raise ValueError("action.dimensions must be a list.")
    dimensions = tuple(
        ActionDimension(
            name=str(_required(value, "name", "action dimension")),
            unit=str(_required(value, "unit", "action dimension")),
            minimum=float(_required(value, "minimum", "action dimension")),
            maximum=float(_required(value, "maximum", "action dimension")),
            encoding=(None if value.get("encoding") is None else str(value["encoding"])),
            source_overshoot_tolerance=float(value.get("source_overshoot_tolerance", 0.0)),
        )
        for value in raw_dimensions
        if isinstance(value, dict)
    )
    declared_dimension = int(_required(action, "dimension", "action"))
    if declared_dimension != len(dimensions):
        raise ValueError(
            f"action.dimension declares {declared_dimension}, but {len(dimensions)} fields exist."
        )
    return ActionContract(
        name=str(_required(raw, "name", "Action contract")),
        schema_version=int(_required(raw, "schema_version", "Action contract")),
        embodiment=str(_required(raw, "embodiment", "Action contract")),
        environment_id=str(_required(raw, "environment_id", "Action contract")),
        action_type=str(_required(action, "type", "action")),
        semantics=str(_required(action, "semantics", "action")),
        control_mode=str(_required(action, "control_mode", "action")),
        space=str(_required(action, "space", "action")),
        reference_frame=str(_required(action, "reference_frame", "action")),
        frequency_hz=float(_required(action, "frequency_hz", "action")),
        timestamp_alignment=str(_required(action, "timestamp_alignment", "action")),
        chunk_length=int(_required(action, "chunk_length", "action")),
        chunk_execution=str(_required(action, "chunk_execution", "action")),
        simulator_expansion=str(_required(action, "simulator_expansion", "action")),
        dimensions=dimensions,
        chunk_execution_steps=int(action.get("chunk_execution_steps", 1)),
    )
