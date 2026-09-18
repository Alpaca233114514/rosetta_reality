"""Registry of plan-declared training features for the pinned LeRobot trainer.

Every feature is an explicit local extension of the pinned upstream trainer:

- installations happen in the exact order the plan declares them, and a failed
  installation rolls back the already-installed features in reverse order;
- double installation fails closed instead of silently stacking wrappers —
  the failure mode that invalidated ``aster-b8-002`` cannot recur here;
- every feature restores the untouched upstream surface for tests and
  diagnostics;
- heavyweight imports (torch, lerobot) happen inside ``install`` so importing
  this package never loads models or mutates the environment.

Feature implementations reuse the frozen, unit-tested modules
``vla/horizon_loss.py``, ``vla/state_robustness.py``, ``vla/processor.py`` and
``vla/fixed_samples.py`` wherever they exist.  New experiments may add a new,
separately named implementation module without mutating those frozen modules.
The historical ``scripts/train_smolvla_*`` stack remains untouched as
provenance for the completed Faust, Aster and Way runs.
"""

from __future__ import annotations

import ctypes
import gc
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.accelerator_memory import empty_accelerator_cache
from rosetta_reality.vla.training.context import TrainingContext
from rosetta_reality.vla.training.masked_camera import (
    install_masked_camera_encoder_skip,
    restore_masked_camera_encoder_skip,
)


def _lerobot_train_module() -> Any:
    import lerobot.scripts.lerobot_train as lerobot_train

    return lerobot_train


def _load_modeling_module() -> Any:
    import lerobot.policies.smolvla.modeling_smolvla as modeling_smolvla

    return modeling_smolvla


def _marker(name: str) -> str:
    return f"_rosetta_v2_feature_{name}_installed"


class TrainingFeature(ABC):
    """Base contract for one installable, restorable trainer extension."""

    name: str

    @abstractmethod
    def install(self, context: TrainingContext) -> None:
        """Wrap the pinned upstream surface declared by this feature."""

    @abstractmethod
    def restore(self, context: TrainingContext) -> None:
        """Remove the wrapper installed by :meth:`install`."""


def _convert_statistics(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import torch

    converted: dict[str, dict[str, Any]] = {}
    for feature, raw_statistics in value.items():
        if not isinstance(raw_statistics, dict):
            raise ValueError("Train-only feature statistics must be mappings.")
        converted[feature] = {}
        for statistic, raw_value in raw_statistics.items():
            if not isinstance(raw_value, list) or not raw_value:
                raise ValueError("Train-only statistics must be non-empty lists.")
            dtype = torch.int64 if statistic == "count" else torch.float64
            converted[feature][statistic] = torch.tensor(raw_value, dtype=dtype)
    return converted


class TrainOnlyStatisticsFeature(TrainingFeature):
    """Inject train-only normalization statistics into the dataset builder.

    Migrated from the frozen ``train_smolvla_trackio`` trainer; the report path
    now comes from the plan's normalization section instead of an environment
    variable.
    """

    name = "train_only_statistics"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("train_only_statistics declares no parameters.")

    def _load_report(self, context: TrainingContext) -> dict[str, Any]:
        path = context.normalization_report
        if not path.is_file():
            raise FileNotFoundError(f"Train-only statistics report is missing: {path.name}.")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(report, dict)
            or report.get("status") != "complete"
            or report.get("stage") != "smolvla_train_only_normalization"
            or report.get("source_split") != "train"
            or report.get("validation_episodes_loaded") is not False
            or report.get("hidden_test_loaded") is not False
            or not isinstance(report.get("effective_stats"), dict)
            or not isinstance(report.get("train_episodes"), list)
        ):
            raise ValueError("SmolVLA train-only normalization report is invalid.")
        return report

    def install(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False):
            raise RuntimeError("train_only_statistics is already installed.")
        report = self._load_report(context)
        allowed_episodes = {int(value) for value in report["train_episodes"]}
        _convert_statistics(report["effective_stats"])
        original = lerobot_train.make_train_eval_datasets

        def make_train_eval_datasets(cfg: Any) -> tuple[Any, Any]:
            requested = {int(value) for value in (cfg.dataset.episodes or [])}
            if not requested or not requested.issubset(allowed_episodes):
                raise ValueError(
                    "Training episodes are outside the train-only normalization scope."
                )
            dataset, eval_dataset = original(cfg)
            dataset.meta.stats.update(_convert_statistics(report["effective_stats"]))
            if eval_dataset is not None:
                eval_dataset.meta.stats.update(
                    _convert_statistics(report["effective_stats"])
                )
            return dataset, eval_dataset

        make_train_eval_datasets._rosetta_v2_original = original  # type: ignore[attr-defined]
        lerobot_train.make_train_eval_datasets = make_train_eval_datasets
        setattr(lerobot_train, _marker(self.name), True)

    def restore(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False) is not True:
            raise RuntimeError("No train-only statistics wrapper is installed.")
        current = lerobot_train.make_train_eval_datasets
        lerobot_train.make_train_eval_datasets = getattr(
            current, "_rosetta_v2_original", None
        ) or current
        setattr(lerobot_train, _marker(self.name), False)


class MaskedCameraSkipFeature(TrainingFeature):
    """Skip encoding fully masked placeholder cameras (parity-checked path)."""

    name = "masked_camera_skip"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("masked_camera_skip declares no parameters.")

    def install(self, context: TrainingContext) -> None:
        install_masked_camera_encoder_skip(_load_modeling_module())

    def restore(self, context: TrainingContext) -> None:
        restore_masked_camera_encoder_skip(_load_modeling_module())


class ActionBoundaryProjectionFeature(TrainingFeature):
    """Install the registered Action Contract projection processor boundary.

    Migrated from the frozen ``train_smolvla_action_repair`` trainer.
    """

    name = "action_boundary_projection"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("action_boundary_projection declares no parameters.")

    def install(self, context: TrainingContext) -> None:
        from rosetta_reality.sim import load_action_contract
        from rosetta_reality.vla.processor import ensure_smolvla_action_boundary

        action_space = context.action_space
        if action_space.target_projection != "action_contract_clip":
            raise ValueError(
                "The action-boundary feature requires Action Contract target projection."
            )
        contract = load_action_contract(context.contract_path)
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False):
            raise RuntimeError("action_boundary_projection is already installed.")
        original = lerobot_train.make_pre_post_processors

        def make_pre_post_processors(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            preprocessor, postprocessor = original(*args, **kwargs)
            ensure_smolvla_action_boundary(
                preprocessor,
                postprocessor,
                contract,
                action_space,
                action_contract_sha256=file_sha256(context.contract_path),
                upstream_revision=str(context.experiment["upstream"]["revision"]),
            )
            return preprocessor, postprocessor

        make_pre_post_processors._rosetta_v2_original = original  # type: ignore[attr-defined]
        lerobot_train.make_pre_post_processors = make_pre_post_processors
        setattr(lerobot_train, _marker(self.name), True)

    def restore(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False) is not True:
            raise RuntimeError("No action-boundary projection wrapper is installed.")
        current = lerobot_train.make_pre_post_processors
        lerobot_train.make_pre_post_processors = getattr(
            current, "_rosetta_v2_original", None
        ) or current
        setattr(lerobot_train, _marker(self.name), False)


class FixedFrameSamplerFeature(TrainingFeature):
    """Restrict the dataloader to preregistered fixed-frame identities."""

    name = "fixed_frame_sampler"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        phase = parameters.get("phase")
        if phase not in {"smoke", "overfit", "overfit_resume"}:
            raise ValueError("fixed_frame_sampler requires a supported phase parameter.")
        self._phase = str(phase)

    def install(self, context: TrainingContext) -> None:
        import numpy as np

        from rosetta_reality.vla.fixed_samples import (
            load_fixed_frame_protocol,
            resolve_fixed_dataset_indices,
        )

        protocol = load_fixed_frame_protocol(context.experiment, self._phase)
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False):
            raise RuntimeError("fixed_frame_sampler is already installed.")
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
                    raise ValueError(
                        "Fixed-frame repair does not allow implicit frame dropping."
                    )
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
        RegisteredFixedFrameSampler._rosetta_v2_original = original_sampler  # type: ignore[attr-defined]
        lerobot_train.EpisodeAwareSampler = RegisteredFixedFrameSampler
        setattr(lerobot_train, _marker(self.name), True)

    def restore(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False) is not True:
            raise RuntimeError("No fixed-frame sampler is installed.")
        current = lerobot_train.EpisodeAwareSampler
        lerobot_train.EpisodeAwareSampler = getattr(
            current, "_rosetta_v2_original", None
        ) or current
        setattr(lerobot_train, _marker(self.name), False)


class HorizonWeightProfileFeature(TrainingFeature):
    """Install the plan-bound temporal weighting on the pinned flow loss."""

    name = "horizon_weight_profile"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("horizon_weight_profile declares no parameters.")

    def install(self, context: TrainingContext) -> None:
        from rosetta_reality.vla.horizon_loss import (
            install_horizon_weight_profile,
            profile_from_plan,
        )

        profile = profile_from_plan(
            context.plan, int(context.experiment["model"]["policy"]["chunk_size"])
        )
        install_horizon_weight_profile(_load_modeling_module(), profile)

    def restore(self, context: TrainingContext) -> None:
        from rosetta_reality.vla.horizon_loss import restore_horizon_weight_profile

        restore_horizon_weight_profile(_load_modeling_module())


class StateRobustnessJitterFeature(TrainingFeature):
    """Install train-only normalized state jitter on optimizer forwards."""

    name = "state_robustness_jitter"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("state_robustness_jitter declares no parameters.")

    def install(self, context: TrainingContext) -> None:
        from rosetta_reality.vla.state_robustness import (
            install_state_robustness_profile,
            profile_from_plan,
        )

        profile = profile_from_plan(context.plan)
        install_state_robustness_profile(_load_modeling_module(), profile)

    def restore(self, context: TrainingContext) -> None:
        from rosetta_reality.vla.state_robustness import (
            restore_state_robustness_profile,
        )

        restore_state_robustness_profile(_load_modeling_module())


class StateConditioningDropoutFeature(TrainingFeature):
    """Install the registered training-only whole-state dropout treatment."""

    name = "state_conditioning_dropout"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("state_conditioning_dropout declares no parameters.")

    def install(self, context: TrainingContext) -> None:
        from rosetta_reality.vla.visual_conditioning import (
            install_visual_conditioning_profile,
            profile_from_plan,
        )

        profile = profile_from_plan(context.plan)
        install_visual_conditioning_profile(_load_modeling_module(), profile)

    def restore(self, context: TrainingContext) -> None:
        from rosetta_reality.vla.visual_conditioning import (
            restore_visual_conditioning_profile,
        )

        restore_visual_conditioning_profile(_load_modeling_module())


class VisionFrontEndUnfreezeFeature(TrainingFeature):
    """Make exactly the declared visual front-end trainable in the trainer.

    The pinned upstream flags cannot express "visual front-end trainable,
    language model frozen" (``train_expert_only=true`` freezes the whole VLM,
    switching it off opens the text stack).  This feature wraps
    ``make_policy``: once the pinned constructor has applied the frozen
    baseline (``freeze_vision_encoder=true`` / ``train_expert_only=true`` /
    ``train_state_proj=true`` from the parent experiment), the declared
    ``vision_model`` + ``connector`` parameters flip to trainable and the
    resolved scope is written to a create-only diagnostics report before the
    trainer builds the optimizer from ``policy.parameters()``.  Validation,
    export and deployment never install this feature, and the saved policy
    config keeps the frozen-baseline flags.
    """

    name = "vision_front_end_unfreeze"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("vision_front_end_unfreeze declares no parameters.")

    def install(self, context: TrainingContext) -> None:
        from rosetta_reality.vla.vision_front_end import (
            apply_front_end_scope,
            scope_report_text,
        )

        adaptation = context.experiment["model"]["adaptation"]
        if (
            adaptation.get("freeze_vision_encoder") is not True
            or adaptation.get("train_expert_only") is not True
            or adaptation.get("train_state_proj") is not True
        ):
            raise ValueError(
                "vision_front_end_unfreeze requires the frozen-baseline parent "
                "adaptation (freeze_vision_encoder/train_expert_only/train_state_proj "
                "all true); an explicit adaptation change needs a new parent "
                "experiment registration."
            )
        contract = context.plan.get("vision_front_end_contract")
        if not isinstance(contract, dict) or contract.get("profile") != "visual_front_end_unfreeze":
            raise ValueError("The vision front-end treatment contract is missing.")
        if (
            contract.get("activation_checkpointing")
            != "per_module_nonreentrant_vision_front_end"
        ):
            raise ValueError(
                "The vision front-end contract must declare the registered "
                "per-module non-reentrant activation checkpointing."
            )

        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False):
            raise RuntimeError("vision_front_end_unfreeze is already installed.")
        original = lerobot_train.make_policy
        feature_marker = self.name
        scope_state: dict[str, Any] = {}

        def make_policy(*args: Any, **kwargs: Any) -> Any:
            policy = original(*args, **kwargs)
            scope = apply_front_end_scope(policy.model)
            from rosetta_reality.vla.vision_front_end import (
                install_front_end_checkpointing,
            )

            checkpointing = install_front_end_checkpointing(policy.model)
            destination = _diagnostics_destination(
                context, f"vision-front-end-scope-{context.run_name}.json"
            )
            if destination.exists():
                raise FileExistsError(
                    f"Vision front-end scope report is create-only: {destination.name}."
                )
            destination.write_text(
                scope_report_text(
                    scope,
                    activation_checkpointing=checkpointing,
                    feature=feature_marker,
                    phase=context.phase,
                    run_name=context.run_name,
                    experiment_id=str(context.experiment["experiment_id"]),
                    plan_sha256=file_sha256(context.plan_path),
                ),
                encoding="utf-8",
            )
            scope_state["scope"] = scope
            return policy

        make_policy._rosetta_v2_original = original  # type: ignore[attr-defined]
        make_policy._rosetta_v2_scope_state = scope_state  # type: ignore[attr-defined]
        lerobot_train.make_policy = make_policy
        setattr(lerobot_train, _marker(self.name), True)

    def restore(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False) is not True:
            raise RuntimeError("No vision front-end unfreeze wrapper is installed.")
        current = lerobot_train.make_policy
        lerobot_train.make_policy = getattr(
            current, "_rosetta_v2_original", None
        ) or current
        setattr(lerobot_train, _marker(self.name), False)


def release_checkpoint_headroom(device: str | None = None) -> None:
    """Return unreachable host and unused CUDA/XPU allocations before serialization."""

    gc.collect()
    import torch

    empty_accelerator_cache(torch, device)
    libc = ctypes.CDLL(None)
    malloc_trim = getattr(libc, "malloc_trim", None)
    if malloc_trim is not None:
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)


def _diagnostics_destination(context: TrainingContext, filename: str) -> Path:
    """Resolve the create-only durable diagnostics path for one run."""

    import os

    run_root = os.environ.get("ROSETTA_RUN_ROOT")
    if not run_root:
        raise RuntimeError("Training diagnostics require ROSETTA_RUN_ROOT.")

    directory = (
        Path(run_root) / str(context.experiment["experiment_id"]) / "diagnostics"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _meter_value(metric: Any) -> float:
    """Return this-step value from a pinned LeRobot ``AverageMeter`` or float.

    The pinned trainer routes every ``train_metrics`` assignment through
    ``MetricsTracker.__setattr__`` into ``AverageMeter.update``, so the step's
    own reading lives in ``.val``; plain floats stay supported for callers
    that assign scalars directly.
    """

    value = getattr(metric, "val", metric)
    return float(value)


class GradientClipDiagnosticsFeature(TrainingFeature):
    """Record pre/post-clip global and per-module gradient norms per step.

    Audit finding O2.  ``update_policy`` is wrapped only to capture the policy
    object so per-parameter names resolve through ``named_parameters()``; the
    measurement itself wraps ``Accelerator.clip_grad_norm_``.  The only
    instrumented clip path is the registered positive ``grad_clip_norm``; the
    schema refuses this feature when the plan clips at zero or below because
    that upstream fallback path is not measured.  Non-finite gradients fail
    closed instead of being logged as numbers.
    """

    name = "gradient_clip_diagnostics"
    _CLIP_MARKER = "_rosetta_v2_clip_diagnostics_installed"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("gradient_clip_diagnostics declares no parameters.")
        self._original_clip: Any = None
        self._original_update: Any = None
        self._state: dict[str, Any] = {"update": 0, "policy": None, "names": None}

    @staticmethod
    def _norms(parameters: Any, names: dict[int, str] | None) -> dict[str, float]:
        import math

        import torch

        squared: dict[str, float] = {}
        for index, parameter in enumerate(parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(
                    "Non-finite gradient observed by clip diagnostics."
                )
            group = (
                names.get(id(parameter), "unresolved").split(".", 1)[0]
                if names is not None
                else "unresolved"
            )
            squared[group] = squared.get(group, 0.0) + float(
                gradient.to(torch.float64).square().sum()
            )
        module_norms = {
            group: math.sqrt(value) for group, value in squared.items()
        }
        module_norms["global"] = math.sqrt(sum(squared.values()))
        return module_norms

    def install(self, context: TrainingContext) -> None:
        import accelerate

        lerobot_train = _lerobot_train_module()
        if getattr(accelerate.Accelerator, self._CLIP_MARKER, False):
            raise RuntimeError("gradient_clip_diagnostics is already installed.")
        optimizer = context.plan.get("training", {}).get("optimizer", {})
        clip_norm = optimizer.get("grad_clip_norm")
        if (
            not isinstance(clip_norm, int | float)
            or isinstance(clip_norm, bool)
            or float(clip_norm) <= 0.0
        ):
            raise ValueError(
                "gradient_clip_diagnostics requires a positive registered "
                "grad_clip_norm; the zero-clip fallback path is not instrumented."
            )
        destination = _diagnostics_destination(
            context, f"gradient-clip-{context.run_name}.jsonl"
        )
        if destination.exists():
            raise FileExistsError(f"Clip diagnostics are create-only: {destination.name}.")
        self._original_clip = accelerate.Accelerator.clip_grad_norm_
        self._original_update = lerobot_train.update_policy
        state = self._state
        state["destination"] = destination

        @wraps(self._original_update)
        def update_policy(*args: Any, **kwargs: Any) -> Any:
            policy = args[1] if len(args) > 1 else kwargs.get("policy")
            if policy is not None and state["policy"] is not policy:
                state["policy"] = policy
                state["names"] = {
                    id(parameter): name
                    for name, parameter in policy.named_parameters()
                }
            return self._original_update(*args, **kwargs)

        def clip_grad_norm_(
            self_accelerator: Any, parameters: Any, max_norm: float, **kwargs: Any
        ) -> Any:
            import json

            # The trainer passes a fresh `policy.parameters()` generator per
            # update; materialize once so the diagnostics passes and the
            # original clip observe the same live tensors instead of an
            # exhausted iterator.
            materialized = list(parameters)
            pre_clip = self._norms(materialized, state["names"])
            total_norm = self._original_clip(
                self_accelerator, materialized, max_norm, **kwargs
            )
            post_clip = self._norms(materialized, state["names"])
            state["update"] = int(state["update"]) + 1
            with open(state["destination"], "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "update": state["update"],
                            "registered_grad_clip_norm": float(max_norm),
                            "pre_clip_l2": pre_clip,
                            "post_clip_l2": post_clip,
                        },
                        allow_nan=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            return total_norm

        setattr(clip_grad_norm_, self._CLIP_MARKER, True)
        update_policy._rosetta_v2_original = self._original_update  # type: ignore[attr-defined]
        lerobot_train.update_policy = update_policy
        accelerate.Accelerator.clip_grad_norm_ = clip_grad_norm_
        setattr(accelerate.Accelerator, self._CLIP_MARKER, True)

    def restore(self, context: TrainingContext) -> None:
        import accelerate

        lerobot_train = _lerobot_train_module()
        if getattr(accelerate.Accelerator, self._CLIP_MARKER, False) is not True:
            raise RuntimeError("No clip-diagnostics wrapper is installed.")
        if self._original_clip is not None:
            accelerate.Accelerator.clip_grad_norm_ = self._original_clip
        current_update = lerobot_train.update_policy
        lerobot_train.update_policy = getattr(
            current_update, "_rosetta_v2_original", None
        ) or current_update
        setattr(accelerate.Accelerator, self._CLIP_MARKER, False)


class CheckpointMetricSnapshotFeature(TrainingFeature):
    """Write the exact same-step metric row into every checkpoint directory.

    Audit finding T9.  ``update_policy`` is wrapped to capture the step's final
    loss/grad-norm/LR/step-time values, and ``save_checkpoint`` is wrapped to
    serialize them next to the weights.  A checkpoint whose step has no exact
    captured row fails closed instead of writing an approximate snapshot.
    """

    name = "checkpoint_metric_snapshot"
    _SNAPSHOT_NAME = "rosetta_checkpoint_metrics.json"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("checkpoint_metric_snapshot declares no parameters.")
        self._captured: dict[int, dict[str, float]] = {}

    @staticmethod
    def _memory_snapshot(device: str) -> dict[str, Any]:
        import torch

        api = "unavailable"
        values: dict[str, int] = {}
        try:
            if device == "cuda" and torch.cuda.is_available():
                api = "torch.cuda"
                values = {
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                }
            elif device == "xpu" and torch.xpu.is_available():
                api = "torch.xpu"
                values = {
                    "max_allocated_bytes": int(torch.xpu.max_memory_allocated()),
                    "allocated_bytes": int(torch.xpu.memory_allocated()),
                    "reserved_bytes": int(torch.xpu.memory_reserved()),
                }
        except (AttributeError, RuntimeError):
            api = "unavailable"
        return {"api": api, **values}

    def install(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False):
            raise RuntimeError("checkpoint_metric_snapshot is already installed.")
        original_update = lerobot_train.update_policy
        original_save = lerobot_train.save_checkpoint
        captured = self._captured
        snapshot_name = self._SNAPSHOT_NAME
        memory_probe = self._memory_snapshot
        context_device = context.device

        @wraps(original_update)
        def update_policy(*args: Any, **kwargs: Any) -> Any:
            result = original_update(*args, **kwargs)
            tracker = args[0] if args else kwargs.get("train_metrics")
            update_ordinal = len(captured) + 1
            captured[update_ordinal] = {
                "loss": _meter_value(tracker.loss),
                "grad_norm": _meter_value(tracker.grad_norm),
                "lr": _meter_value(tracker.lr),
                "update_s": _meter_value(tracker.update_s),
            }
            return result

        def save_checkpoint(*args: Any, **kwargs: Any) -> Any:
            step = kwargs.get("step") if "step" in kwargs else args[1]
            step = int(step)
            if step not in captured:
                raise ValueError(
                    "Checkpoint step has no exact captured metric row (finding T9): "
                    f"{step}."
                )
            result = original_save(*args, **kwargs)
            import json

            checkpoint_dir = (
                kwargs.get("checkpoint_dir") if "checkpoint_dir" in kwargs else args[0]
            )
            payload = {
                "schema_version": 1,
                "stage": "smolvla_v2_checkpoint_metric_snapshot",
                "step": step,
                "metrics": captured[step],
                "metrics_source": "update_policy_return_value",
                "memory": memory_probe(context_device),
                "run_name": context.run_name,
                "phase": context.phase,
                "experiment_id": context.experiment["experiment_id"],
            }
            (checkpoint_dir / snapshot_name).write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return result

        update_policy._rosetta_v2_original = original_update  # type: ignore[attr-defined]
        save_checkpoint._rosetta_v2_original = original_save  # type: ignore[attr-defined]
        lerobot_train.update_policy = update_policy
        lerobot_train.save_checkpoint = save_checkpoint
        setattr(lerobot_train, _marker(self.name), True)

    def restore(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False) is not True:
            raise RuntimeError("No checkpoint metric snapshot is installed.")
        current_update = lerobot_train.update_policy
        current_save = lerobot_train.save_checkpoint
        lerobot_train.update_policy = getattr(
            current_update, "_rosetta_v2_original", None
        ) or current_update
        lerobot_train.save_checkpoint = getattr(
            current_save, "_rosetta_v2_original", None
        ) or current_save
        setattr(lerobot_train, _marker(self.name), False)


class CheckpointMemoryTrimFeature(TrainingFeature):
    """Trim transient allocations after resume load and before every checkpoint.

    One device-aware implementation replacing the two historical copies
    ``vla/checkpoint_memory.py`` (XPU) and ``vla/checkpoint_accelerator_memory.py``
    (CUDA/XPU), which remain frozen for provenance.
    """

    name = "checkpoint_memory_trim"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("checkpoint_memory_trim declares no parameters.")

    def install(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False):
            raise RuntimeError("checkpoint_memory_trim is already installed.")
        device = context.device
        original_resume = lerobot_train.resume_after_prepare
        original_save = lerobot_train.save_checkpoint

        @wraps(original_resume)
        def resume_after_prepare(*args: Any, **kwargs: Any) -> Any:
            result = original_resume(*args, **kwargs)
            release_checkpoint_headroom(device)
            return result

        @wraps(original_save)
        def save_checkpoint(*args: Any, **kwargs: Any) -> Any:
            release_checkpoint_headroom(device)
            return original_save(*args, **kwargs)

        resume_after_prepare._rosetta_v2_original = original_resume  # type: ignore[attr-defined]
        save_checkpoint._rosetta_v2_original = original_save  # type: ignore[attr-defined]
        lerobot_train.resume_after_prepare = resume_after_prepare
        lerobot_train.save_checkpoint = save_checkpoint
        setattr(lerobot_train, _marker(self.name), True)

    def restore(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False) is not True:
            raise RuntimeError("No checkpoint memory trim is installed.")
        current_resume = lerobot_train.resume_after_prepare
        current_save = lerobot_train.save_checkpoint
        lerobot_train.resume_after_prepare = getattr(
            current_resume, "_rosetta_v2_original", None
        ) or current_resume
        lerobot_train.save_checkpoint = getattr(
            current_save, "_rosetta_v2_original", None
        ) or current_save
        setattr(lerobot_train, _marker(self.name), False)


class TrackioLoggingFeature(TrainingFeature):
    """Log through the sanitized local Trackio bridge instead of WandB.

    The historical ``trackio_lerobot`` module is hash-bound provenance for the
    completed runs, so this feature composes a plan-bound logger subclass
    instead of patching module state.  The runtime experiment is resolved from
    the file the launcher writes, never from an in-process overlay.
    """

    name = "trackio_logging"
    LOGGER_STANDARD = "standard"
    LOGGER_ACCELERATOR = "accelerator"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        logger = parameters.get("logger", self.LOGGER_STANDARD)
        if logger not in {self.LOGGER_STANDARD, self.LOGGER_ACCELERATOR}:
            raise ValueError("trackio_logging logger must be standard or accelerator.")
        self._logger_kind = str(logger)
        self._logger_class: type | None = None
        self._original_logger: Any = None

    def _public_extension(self, context: TrainingContext) -> dict[str, Any]:
        extension: dict[str, Any] = {
            "action_representation_adapter": context.action_space.representation_adapter,
            "action_target_projection": context.action_space.target_projection,
            "bounded_gripper_decoder": True,
            "training_harness": "v2",
            "v2_formal_plan_sha256": file_sha256(context.plan_path),
        }
        loss_contract = context.plan.get("loss_contract")
        if isinstance(loss_contract, dict):
            extension["temporal_loss_profile"] = loss_contract.get("profile")
            extension["temporal_loss_normalization"] = loss_contract.get("normalization")
        state_contract = context.plan.get("state_robustness_contract")
        if isinstance(state_contract, dict):
            extension["state_robustness_profile"] = state_contract.get("profile")
            extension["state_noise_std_normalized"] = state_contract.get(
                "normalized_standard_deviation"
            )
            extension["state_jitter_training_only"] = True
        visual_contract = context.plan.get("visual_conditioning_contract")
        if isinstance(visual_contract, dict):
            extension["visual_conditioning_profile"] = visual_contract.get("profile")
            extension["state_dropout_probability"] = visual_contract.get(
                "dropout_probability"
            )
            extension["state_dropout_granularity"] = visual_contract.get(
                "granularity"
            )
            extension["state_dropout_training_only"] = True
        vision_scope = context.plan.get("vision_front_end_contract")
        if isinstance(vision_scope, dict):
            extension["vision_front_end_profile"] = vision_scope.get("profile")
            extension["vision_front_end_scope"] = vision_scope.get("scope")
            extension["vision_front_end_language_model"] = vision_scope.get(
                "language_model"
            )
            extension["vision_front_end_training_only"] = True
        return extension

    def _build_logger_class(self, context: TrainingContext) -> type:
        import os

        import rosetta_reality.tracking.trackio_lerobot as trackio_bridge
        from rosetta_reality.tracking.public_payload import validate_public_payload

        if self._logger_kind == self.LOGGER_ACCELERATOR:
            from rosetta_reality.tracking.trackio_accelerator import (
                AcceleratorTrackioLogger as base_logger,
            )
        else:
            from rosetta_reality.tracking.trackio_lerobot import (
                TrackioLogger as base_logger,
            )

        extension = self._public_extension(context)

        class PlanBoundTrackioLogger(base_logger):
            """Base logger contract plus the plan's feature-identity fields."""

            def __init__(self, cfg: Any) -> None:
                import logging

                import trackio

                experiment = trackio_bridge._experiment_config()
                phase = os.environ.get("ROSETTA_VLA_PHASE", "")
                if phase not in {
                    "space_smoke",
                    "smoke",
                    "overfit",
                    "overfit_resume",
                    "formal",
                    "performance_benchmark",
                }:
                    raise ValueError("ROSETTA_VLA_PHASE must identify an approved VLA phase.")
                project = str(experiment["tracking"]["project"])
                name = str(cfg.job_name)
                group = f"{experiment['experiment_id']}-{phase}"
                validate_public_payload(
                    {"project": project, "name": name, "group": group},
                    context="trackio_identity",
                )
                payload = trackio_bridge._public_config(cfg, experiment, phase)
                payload.update(extension)
                validate_public_payload(payload, context="trackio_config")
                resume = "allow" if cfg.resume else "never"
                self._run = trackio.init(
                    project=project,
                    name=name,
                    group=group,
                    config=payload,
                    resume=resume,
                    embed=False,
                    auto_log_cpu=False,
                    auto_log_gpu=False,
                )
                self._trackio = trackio
                self._last_step = 0
                cfg.wandb.run_id = str(getattr(self._run, "id", name))
                logging.info(
                    "Metrics are stored in local Trackio for sanitized static-Space sync."
                )

        PlanBoundTrackioLogger.__name__ = "PlanBoundTrackioLogger"
        return PlanBoundTrackioLogger

    def install(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False):
            raise RuntimeError("trackio_logging is already installed.")
        self._original_logger = lerobot_train.WandBLogger
        self._logger_class = self._build_logger_class(context)
        lerobot_train.WandBLogger = self._logger_class
        setattr(lerobot_train, _marker(self.name), True)

    def restore(self, context: TrainingContext) -> None:
        lerobot_train = _lerobot_train_module()
        if getattr(lerobot_train, _marker(self.name), False) is not True:
            raise RuntimeError("No Trackio logging wrapper is installed.")
        if self._original_logger is not None:
            lerobot_train.WandBLogger = self._original_logger
        setattr(lerobot_train, _marker(self.name), False)


FEATURE_FACTORIES: dict[str, Callable[[Mapping[str, Any]], TrainingFeature]] = {
    TrainOnlyStatisticsFeature.name: TrainOnlyStatisticsFeature,
    MaskedCameraSkipFeature.name: MaskedCameraSkipFeature,
    ActionBoundaryProjectionFeature.name: ActionBoundaryProjectionFeature,
    FixedFrameSamplerFeature.name: FixedFrameSamplerFeature,
    HorizonWeightProfileFeature.name: HorizonWeightProfileFeature,
    StateRobustnessJitterFeature.name: StateRobustnessJitterFeature,
    StateConditioningDropoutFeature.name: StateConditioningDropoutFeature,
    VisionFrontEndUnfreezeFeature.name: VisionFrontEndUnfreezeFeature,
    CheckpointMemoryTrimFeature.name: CheckpointMemoryTrimFeature,
    GradientClipDiagnosticsFeature.name: GradientClipDiagnosticsFeature,
    CheckpointMetricSnapshotFeature.name: CheckpointMetricSnapshotFeature,
    TrackioLoggingFeature.name: TrackioLoggingFeature,
}

DECLARED_FEATURE_ORDER = tuple(FEATURE_FACTORIES)


@dataclass
class FeatureStack:
    """An ordered set of features with rollback and reverse-order restore."""

    features: tuple[TrainingFeature, ...]
    installed: list[str] = field(default_factory=list)

    @classmethod
    def from_plan(cls, plan: dict[str, Any]) -> FeatureStack:
        declarations = plan.get("features")
        if not isinstance(declarations, list):
            raise ValueError("Version-2 plans must declare an ordered features list.")
        features: list[TrainingFeature] = []
        names: set[str] = set()
        for declaration in declarations:
            if not isinstance(declaration, dict):
                raise ValueError("Each feature declaration must be a mapping.")
            name = declaration.get("name")
            factory = FEATURE_FACTORIES.get(name) if isinstance(name, str) else None
            if factory is None:
                raise ValueError(f"Unknown training feature declared: {name!r}.")
            if name in names:
                raise ValueError(f"Training feature declared twice: {name!r}.")
            names.add(name)
            parameters = {
                key: value for key, value in declaration.items() if key != "name"
            }
            features.append(factory(parameters))
        if not features:
            raise ValueError("Version-2 plans must declare at least one feature.")
        return cls(tuple(features))

    def install_all(self, context: TrainingContext) -> list[str]:
        """Install every feature in declaration order, rolling back on failure."""

        for feature in self.features:
            try:
                feature.install(context)
            except BaseException:
                self._rollback(context)
                raise
            self.installed.append(feature.name)
        return list(self.installed)

    def restore_all(self, context: TrainingContext) -> None:
        """Restore every installed feature in reverse installation order."""

        while self.installed:
            name = self.installed.pop()
            for feature in self.features:
                if feature.name == name:
                    feature.restore(context)
                    break

    def _rollback(self, context: TrainingContext) -> None:
        while self.installed:
            name = self.installed.pop()
            for feature in self.features:
                if feature.name == name:
                    feature.restore(context)
                    break


def feature_stack_from_plan(plan: dict[str, Any]) -> FeatureStack:
    """Resolve the plan's feature declarations against the registry."""

    return FeatureStack.from_plan(plan)
