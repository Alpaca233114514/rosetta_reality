"""Versioned fixed-frame sampling contracts for bounded SmolVLA overfit runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FixedFrameProtocol:
    """The immutable sample identity used by smoke and overfit phases."""

    schema_version: int
    episode: int
    frame_indices: tuple[int, ...]
    phases: tuple[str, ...]
    sampler: str
    hidden_test_loaded: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe representation."""

        value = asdict(self)
        value["frame_indices"] = list(self.frame_indices)
        value["phases"] = list(self.phases)
        return value


def load_fixed_frame_protocol(
    experiment: Mapping[str, Any], phase: str
) -> FixedFrameProtocol:
    """Load and strictly validate the registered fixed-frame protocol."""

    repair = experiment.get("repair_protocol")
    if not isinstance(repair, Mapping):
        raise ValueError("The experiment has no repair protocol.")
    raw = repair.get("fixed_sample_overfit")
    if not isinstance(raw, Mapping):
        raise ValueError("The repair experiment has no fixed-sample protocol.")
    raw_frames = raw.get("frame_indices")
    raw_phases = raw.get("phases")
    if (
        not isinstance(raw_frames, list)
        or not raw_frames
        or any(type(value) is not int or value < 0 for value in raw_frames)
        or len(raw_frames) != len(set(raw_frames))
        or raw_frames != sorted(raw_frames)
        or not isinstance(raw_phases, list)
        or not raw_phases
        or any(not isinstance(value, str) or not value for value in raw_phases)
        or len(raw_phases) != len(set(raw_phases))
    ):
        raise ValueError("The fixed-sample frame or phase identity is invalid.")
    protocol = FixedFrameProtocol(
        schema_version=int(raw.get("schema_version", 0)),
        episode=int(raw.get("episode", -1)),
        frame_indices=tuple(raw_frames),
        phases=tuple(raw_phases),
        sampler=str(raw.get("sampler", "")),
        hidden_test_loaded=raw.get("hidden_test_loaded"),
    )
    if (
        protocol.schema_version != 1
        or protocol.episode < 0
        or protocol.sampler != "deterministic_epoch_permutation"
        or protocol.hidden_test_loaded is not False
        or phase not in protocol.phases
    ):
        raise ValueError("The active phase is not covered by the fixed-sample protocol.")
    dataset = experiment.get("dataset")
    phases = experiment.get("phases")
    if not isinstance(dataset, Mapping) or not isinstance(phases, Mapping):
        raise ValueError("The experiment dataset or phase mapping is missing.")
    train_episodes = dataset.get("train_episodes")
    hidden_episodes = dataset.get("test_episodes")
    registered_phase = "overfit" if phase == "overfit_resume" else phase
    phase_config = phases.get(registered_phase)
    if (
        not isinstance(train_episodes, list)
        or protocol.episode not in train_episodes
        or not isinstance(hidden_episodes, list)
        or protocol.episode in hidden_episodes
        or not isinstance(phase_config, Mapping)
        or phase_config.get("episodes") != [protocol.episode]
    ):
        raise ValueError("The fixed sample is outside the registered training phase.")
    return protocol


def resolve_fixed_dataset_indices(
    protocol: FixedFrameProtocol,
    dataset_from_indices: Sequence[int],
    dataset_to_indices: Sequence[int],
    episode_indices_to_use: Sequence[int] | None,
    absolute_to_relative_idx: Mapping[int, int] | None,
) -> list[int]:
    """Map registered episode/frame identities to the active dataset view."""

    if episode_indices_to_use is None or list(episode_indices_to_use) != [protocol.episode]:
        raise ValueError("The active dataset does not contain exactly the fixed episode.")
    if len(dataset_from_indices) != len(dataset_to_indices):
        raise ValueError("Dataset episode boundaries are inconsistent.")
    if protocol.episode >= len(dataset_from_indices):
        raise ValueError("The fixed episode is outside the dataset metadata.")
    start = int(dataset_from_indices[protocol.episode])
    stop = int(dataset_to_indices[protocol.episode])
    if stop <= start:
        raise ValueError("The fixed episode has no frames.")
    resolved: list[int] = []
    for frame_index in protocol.frame_indices:
        absolute = start + frame_index
        if absolute >= stop:
            raise ValueError("A fixed frame is outside the registered episode.")
        if absolute_to_relative_idx is None:
            relative = absolute
        else:
            if absolute not in absolute_to_relative_idx:
                raise ValueError("A fixed frame is absent from the active dataset view.")
            relative = int(absolute_to_relative_idx[absolute])
        resolved.append(relative)
    if len(resolved) != len(set(resolved)):
        raise ValueError("Fixed frame identities resolved to duplicate dataset indices.")
    return resolved
