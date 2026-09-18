"""Run a future authorized SmolVLA repair smoke with projected train targets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import train_smolvla_trackio as historical_trainer  # noqa: E402

from rosetta_reality.experiment import file_sha256, stable_hash  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.checkpoint_memory import (  # noqa: E402
    install_checkpoint_memory_trim,
)
from rosetta_reality.vla.fixed_samples import (  # noqa: E402
    load_fixed_frame_protocol,
    resolve_fixed_dataset_indices,
)
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402


def _repair_context() -> tuple[dict[str, Any], Any, Path, Path]:
    raw = os.environ.get("ROSETTA_VLA_EXPERIMENT_CONFIG")
    if not raw:
        raise ValueError("ROSETTA_VLA_EXPERIMENT_CONFIG must be set by the repair launcher.")
    config_path = Path(raw)
    if not config_path.is_absolute() or not config_path.is_file():
        raise ValueError("The repair experiment config must be an absolute existing file.")
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    if not contract_path.is_file():
        raise FileNotFoundError("The repair Action Contract is missing.")
    return experiment, action_space, config_path, contract_path


def _validate_projected_statistics(
    experiment: dict[str, Any],
    action_space: Any,
    config_path: Path,
    contract_path: Path,
) -> None:
    raw = os.environ.get("ROSETTA_VLA_TRAIN_STATS_REPORT")
    if not raw:
        raise ValueError("Repair training requires a projected train-only statistics report.")
    path = Path(raw)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(config_path)
        or report.get("action_contract_sha256") != file_sha256(contract_path)
        or report.get("action_space") != action_space.as_dict()
        or report.get("target_projection", {}).get("mode") != "action_contract_clip"
        or report.get("target_projection", {}).get("stage") != "before_normalization"
        or report.get("validation_episodes_loaded") is not False
        or report.get("hidden_test_loaded") is not False
        or os.environ.get("ROSETTA_VLA_NORMALIZATION_SHA256") != file_sha256(path)
    ):
        raise ValueError("Repair train-only statistics do not match the action-space contract.")


def _install_projection(
    lerobot_train: Any,
    experiment: dict[str, Any],
    action_space: Any,
    contract_path: Path,
) -> None:
    if action_space.target_projection != "action_contract_clip":
        raise ValueError("The repair trainer requires Action Contract target projection.")
    contract = load_action_contract(contract_path)
    original = lerobot_train.make_pre_post_processors

    def make_pre_post_processors(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        preprocessor, postprocessor = original(*args, **kwargs)
        ensure_smolvla_action_boundary(
            preprocessor,
            postprocessor,
            contract,
            action_space,
            action_contract_sha256=file_sha256(contract_path),
            upstream_revision=str(experiment["upstream"]["revision"]),
        )
        return preprocessor, postprocessor

    lerobot_train.make_pre_post_processors = make_pre_post_processors


def _validate_fixed_sample_evidence(
    experiment: dict[str, Any], config_path: Path, phase: str
) -> None:
    raw = os.environ.get("ROSETTA_VLA_FIXED_SAMPLE_REPORT")
    if not raw:
        raise ValueError("Repair optimizer work requires fixed-sample evidence.")
    path = Path(raw)
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = load_fixed_frame_protocol(experiment, phase)
    protocol_payload = protocol.as_dict()
    if (
        not isinstance(report, dict)
        or report.get("status") != "passed"
        or report.get("stage") != "smolvla_fixed_sample_no_weights_diagnostic"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(config_path)
        or report.get("dataset_revision") != experiment["dataset"]["revision"]
        or report.get("fixed_sample_protocol") != protocol_payload
        or report.get("fixed_sample_protocol_sha256") != stable_hash(protocol_payload)
        or report.get("fixed_sample_count") != len(protocol.frame_indices)
        or report.get("model_weights_loaded") is not False
        or report.get("optimizer_created") is not False
        or report.get("validation_episodes_loaded") is not False
        or report.get("hidden_test_loaded") is not False
        or os.environ.get("ROSETTA_VLA_FIXED_SAMPLE_SHA256") != file_sha256(path)
    ):
        raise ValueError("Fixed-sample evidence does not match the active repair run.")


def _install_fixed_frame_sampler(
    lerobot_train: Any, experiment: dict[str, Any], phase: str
) -> None:
    protocol = load_fixed_frame_protocol(experiment, phase)
    original_sampler = lerobot_train.EpisodeAwareSampler

    class RegisteredFixedFrameSampler(original_sampler):
        """Drop-in sampler restricted to the preregistered frame identities."""

        def __init__(
            self,
            dataset_from_indices: list[int],
            dataset_to_indices: list[int],
            episode_indices_to_use: list[int] | None = None,
            drop_n_first_frames: int = 0,
            drop_n_last_frames: int = 0,
            shuffle: bool = False,
            seed: int = 0,
            absolute_to_relative_idx: dict[int, int] | None = None,
        ) -> None:
            if drop_n_first_frames != 0 or drop_n_last_frames != 0:
                raise ValueError("Fixed-frame repair does not allow implicit frame dropping.")
            fixed_indices = resolve_fixed_dataset_indices(
                protocol,
                dataset_from_indices,
                dataset_to_indices,
                episode_indices_to_use,
                absolute_to_relative_idx,
            )
            self._fixed_indices = tuple(fixed_indices)
            self._num_frames = len(self._fixed_indices)
            self.shuffle = shuffle
            self.seed = seed
            self._epoch = 0
            self._start_index = 0
            self._absolute_to_relative = None

        @property
        def indices(self) -> list[int]:
            return list(self._fixed_indices)

        def _frame_index(self, position: int) -> int:
            if position < 0 or position >= self._num_frames:
                raise IndexError("Fixed-frame sampler position is out of range.")
            return self._fixed_indices[position]

        def _epoch_generator(self, epoch: int):
            import torch

            epoch_seed = int(
                np.random.SeedSequence([self.seed, epoch]).generate_state(
                    1, dtype=np.uint64
                )[0]
            )
            return torch.Generator().manual_seed(epoch_seed)

    RegisteredFixedFrameSampler.__name__ = "RegisteredFixedFrameSampler"
    lerobot_train.EpisodeAwareSampler = RegisteredFixedFrameSampler


def main() -> None:
    if os.environ.get("ROSETTA_VLA_ACTION_REPAIR_OPTIMIZER_AUTHORIZED") != "1":
        raise RuntimeError(
            "Repair optimizer work is locked until the registered diagnostics are accepted."
        )

    import lerobot.scripts.lerobot_train as lerobot_train

    import rosetta_reality.tracking.trackio_lerobot as trackio_bridge

    experiment, action_space, config_path, contract_path = _repair_context()
    protocol = experiment.get("repair_protocol", {})
    phase = os.environ.get("ROSETTA_VLA_PHASE")
    authorized_phases = protocol.get("authorized_phases", [])
    resume_is_registered_overfit = (
        phase == "overfit_resume"
        and "overfit" in authorized_phases
        and "explicit_resume_completes"
        in experiment.get("phases", {}).get("overfit", {}).get("acceptance", [])
    )
    if (
        experiment.get("status") != "preregistered_action_repair_smoke_and_overfit"
        or protocol.get("optimizer_authorized") is not True
        or (phase not in authorized_phases and not resume_is_registered_overfit)
        or protocol.get("hidden_test_loaded") is not False
    ):
        raise PermissionError("The loaded repair experiment does not authorize this phase.")
    _validate_projected_statistics(
        experiment,
        action_space,
        config_path,
        contract_path,
    )
    _validate_fixed_sample_evidence(experiment, config_path, str(phase))
    if os.environ.get("ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING") == "1":
        historical_trainer._install_masked_camera_encoder_skip()
    historical_trainer._install_train_only_statistics(lerobot_train)
    _install_projection(lerobot_train, experiment, action_space, contract_path)
    _install_fixed_frame_sampler(lerobot_train, experiment, str(phase))
    install_checkpoint_memory_trim(lerobot_train)
    # The historical Trackio bridge intentionally reads a plain formal config.
    # Keep that hash-bound implementation immutable and inject this already
    # checksum-validated overlay only for the fresh repair process.
    trackio_bridge._experiment_config = lambda: experiment
    lerobot_train.WandBLogger = trackio_bridge.TrackioLogger
    try:
        lerobot_train.main()
    finally:
        trackio_bridge.finish_trackio()


if __name__ == "__main__":
    main()
