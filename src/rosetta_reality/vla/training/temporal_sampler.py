"""Opt-in sealed, explicit sample order for a new bounded temporal comparison."""

from __future__ import annotations

import json
from pathlib import Path

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.training.plan import repository_relative_path


def load_schedule(context, declaration):
    root = Path(__file__).resolve().parents[4]
    relative = repository_relative_path(declaration.get("path"), context="Temporal schedule")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Temporal schedule escaped the repository or is absent")
    if file_sha256(path) != declaration.get("sha256"):
        raise ValueError("Temporal schedule checksum changed")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("status") != "preregistered"
        or value.get("training_authorized") is not True
    ):
        raise ValueError("Temporal schedule must be preregistered and authorized, not a draft")
    if context.phase != "smoke" or context.plan.get("scope") != "bounded_temporal_sampling":
        raise ValueError("Temporal sampler requires a separately registered bounded smoke scope")
    samples = value.get("sample_identities")
    phase = context.plan["optimizer_smoke"]
    if not isinstance(samples, list) or len(samples) != phase["batch_size"] * phase["steps"]:
        raise ValueError("Temporal schedule length differs from the active update budget")
    allowed = set(context.experiment["dataset"]["train_episodes"])
    forbidden = set(context.experiment["dataset"]["validation_episodes"]) | set(
        context.experiment["dataset"]["test_episodes"]
    )
    for pair in samples:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(x) is not int or x < 0 for x in pair)
        ):
            raise ValueError("Temporal sample must be a nonnegative integer episode/frame pair")
        if pair[0] not in allowed or pair[0] in forbidden:
            raise ValueError("Temporal schedule crosses the training split")
    if set(ep for ep, _ in samples) != set(phase["episodes"]):
        raise ValueError("Temporal schedule does not cover the active training episodes")
    if value.get("seed") != context.experiment["seed"]:
        raise ValueError("Temporal schedule seed differs")
    return samples


def sampler_class(original, samples, seed):
    class SealedTemporalSampler(original):
        def __init__(
            self,
            dataset_from_indices,
            dataset_to_indices,
            episode_indices_to_use=None,
            drop_n_first_frames=0,
            drop_n_last_frames=0,
            shuffle=False,
            seed=0,
            absolute_to_relative_idx=None,
        ):
            if (
                drop_n_first_frames
                or drop_n_last_frames
                or len(dataset_from_indices) != len(dataset_to_indices)
            ):
                raise ValueError(
                    "Temporal sampler does not permit implicit dropping or invalid boundaries"
                )
            if seed != registered_seed or set(episode_indices_to_use or []) != {
                ep for ep, _ in samples
            }:
                raise ValueError("Native sampler seed or split differs from the sealed schedule")
            indices = []
            for ep, frame in samples:
                if ep >= len(dataset_from_indices):
                    raise ValueError("Temporal episode outside native metadata")
                start, stop = int(dataset_from_indices[ep]), int(dataset_to_indices[ep])
                absolute = start + frame
                if start < 0 or not start <= absolute < stop:
                    raise ValueError("Temporal frame outside its episode")
                if absolute_to_relative_idx is not None:
                    if absolute not in absolute_to_relative_idx:
                        raise ValueError("Temporal sample absent from the active dataset view")
                    absolute = absolute_to_relative_idx[absolute]
                indices.append(absolute)
            self._fixed_indices = tuple(indices)
            self._num_frames = len(indices)
            # Native __iter__ is retained, but the schedule already owns shuffle.
            self.shuffle, self.seed = False, seed
            self._epoch, self._start_index, self._absolute_to_relative = 0, 0, None

        def _frame_index(self, position):
            return self._fixed_indices[position]

    registered_seed = seed
    return SealedTemporalSampler
