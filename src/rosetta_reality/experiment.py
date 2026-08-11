"""Validated, hashable experiment configuration for auditable M2 runs."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

M2_BASE_IDENTIFIER = "Qwen/Qwen3.5-0.8B-Base"
INSTRUCT_CONTROL_IDENTIFIER = "Qwen/Qwen3.5-0.8B"


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible value with deterministic formatting."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one file incrementally."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_artifact_recipe(experiment: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen-backbone inference recipe that an artifact must preserve."""

    backbone = _mapping(_required(experiment, "backbone", "Experiment config"), "backbone")
    return {
        "experiment_id": _required(experiment, "experiment_id", "Experiment config"),
        "base_model": _required(backbone, "identifier", "backbone"),
        "base_model_family": _required(backbone, "family", "backbone"),
        "base_model_scale": _required(backbone, "scale", "backbone"),
        "adaptation": _required(backbone, "adaptation", "backbone"),
        "backbone_dtype": _required(backbone, "dtype", "backbone"),
        "processor": _mapping(_required(backbone, "processor", "backbone"), "processor"),
        "feature_layer": _required(backbone, "feature_layer", "backbone"),
        "pooling": _required(backbone, "pooling", "backbone"),
        "action_expert": _mapping(
            _required(experiment, "action_expert", "Experiment config"),
            "action_expert",
        ),
    }


def validate_frozen_artifact_recipe(
    experiment: dict[str, Any],
    artifact_config: dict[str, Any],
    *,
    context: str = "Artifact",
) -> None:
    """Reject same-ID artifacts whose feature or policy recipe drifted."""

    expected = frozen_artifact_recipe(experiment)
    for name, value in expected.items():
        if artifact_config.get(name) != value:
            raise ValueError(f"{context} config differs at {name}.")


def _branch_from_git_reference(reference: str) -> str:
    prefix = "refs/heads/"
    return reference[len(prefix) :] if reference.startswith(prefix) else reference


def workspace_code_identity(repository_root: Path) -> dict[str, Any]:
    """Content-address the current tracked and untracked, non-ignored workspace."""

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repository_root}", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    try:
        relative_files = sorted(
            path
            for path in git(
                "ls-files", "--cached", "--others", "--exclude-standard"
            ).splitlines()
            if path
        )
        revision = git("rev-parse", "HEAD")
        branch = git("branch", "--show-current")
        dirty = bool(git("status", "--short"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        roots = ("configs", "docker", "docs", "scripts", "src", "tests")
        relative_files = []
        for root_name in roots:
            root = repository_root / root_name
            if root.exists():
                relative_files.extend(
                    path.relative_to(repository_root).as_posix()
                    for path in root.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                )
        relative_files.extend(
            path.name
            for path in repository_root.iterdir()
            if path.is_file()
        )
        relative_files = sorted(set(relative_files))
        head_path = repository_root / ".git" / "HEAD"
        head = head_path.read_text(encoding="utf-8").strip() if head_path.is_file() else ""
        if head.startswith("ref: "):
            reference = head[5:]
            branch = _branch_from_git_reference(reference)
            reference_path = repository_root / ".git" / reference
            revision = (
                reference_path.read_text(encoding="utf-8").strip()
                if reference_path.is_file()
                else "unknown"
            )
        else:
            branch = "detached"
            revision = head or "unknown"
        dirty = True
    digest = hashlib.sha256()
    file_count = 0
    for relative in relative_files:
        path = repository_root / relative
        if not path.is_file():
            continue
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
        file_count += 1
    return {
        "revision": revision,
        "branch": branch,
        "dirty": dirty,
        "workspace_tree_sha256": digest.hexdigest(),
        "workspace_file_count": file_count,
    }


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return value


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ValueError(f"{context} is missing {key!r}.") from error


def _validate_episode_split(dataset: dict[str, Any], expected_episodes: set[int]) -> None:
    split = _mapping(_required(dataset, "split", "dataset"), "dataset.split")
    names = ("train", "validation", "test")
    partitions = {
        name: tuple(int(episode) for episode in _required(split, name, "dataset.split"))
        for name in names
    }
    for name, episodes in partitions.items():
        if not episodes or len(set(episodes)) != len(episodes):
            raise ValueError(f"dataset.split.{name} must be non-empty and unique.")
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = set(partitions[left_name]) & set(partitions[right_name])
            if overlap:
                raise ValueError(
                    f"Episode leakage between {left_name} and {right_name}: {sorted(overlap)}."
                )
    covered = set().union(*(set(value) for value in partitions.values()))
    if covered != expected_episodes:
        raise ValueError(
            "Episode split must cover the configured dataset exactly; "
            f"missing={sorted(expected_episodes - covered)}, "
            f"extra={sorted(covered - expected_episodes)}."
        )


def load_experiment_config(path: Path, repository_root: Path) -> dict[str, Any]:
    """Load an M2 experiment and enforce the safety-critical experiment axes."""

    raw = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "Experiment config")
    experiment_id = str(_required(raw, "experiment_id", "Experiment config"))
    if (
        not experiment_id
        or experiment_id in {".", ".."}
        or any(character.isspace() or character in "/\\" for character in experiment_id)
    ):
        raise ValueError("experiment_id must be a non-empty path-safe token.")

    backbone = _mapping(_required(raw, "backbone", "Experiment config"), "backbone")
    if _required(backbone, "scale", "backbone") != "0.8B":
        raise ValueError("M2 development experiments must use the 0.8B reference scale.")
    identifier = _required(backbone, "identifier", "backbone")
    if identifier != M2_BASE_IDENTIFIER:
        controlled_change = raw.get("controlled_change")
        is_instruct_control = (
            identifier == INSTRUCT_CONTROL_IDENTIFIER
            and raw.get("experiment_role") == "auxiliary_backbone_control"
            and raw.get("m2_completion_eligible") is False
            and isinstance(controlled_change, dict)
            and controlled_change.get("reference_identifier") == M2_BASE_IDENTIFIER
            and controlled_change.get("changed_axis")
            == "backbone.checkpoint_variant_and_native_prompt_protocol"
            and _mapping(_required(backbone, "processor", "backbone"), "processor").get(
                "prompt_mode"
            )
            == "chat_template"
        )
        if not is_instruct_control:
            raise ValueError(
                "M2 must use the explicit Qwen3.5-0.8B-Base repository identity; "
                "Instruct is allowed only as an auxiliary, M2-ineligible control."
            )
    if _required(backbone, "adaptation", "backbone") != "frozen":
        raise ValueError("This experiment is explicitly the frozen-backbone reference.")
    if _required(backbone, "manifest", "backbone") != "model_manifest.json":
        raise ValueError("M2 requires a revision-scoped model_manifest.json identity proof.")
    if _required(backbone, "pooling", "backbone") not in {
        "attention_masked_mean",
        "image_token_mean",
        "image_spatial_2x2",
        "attention_masked_mean_plus_image_spatial_2x2",
    }:
        raise ValueError("M2 backbone.pooling is not a supported frozen representation.")

    action_expert = _mapping(
        _required(raw, "action_expert", "Experiment config"), "action_expert"
    )
    if _required(action_expert, "output_projection", "action_expert") != (
        "clip_to_action_contract"
    ):
        raise ValueError("M2 policy outputs must be projected through the Action Contract.")
    if action_expert.get("prediction_parameterization", "absolute") not in {
        "absolute",
        "residual_from_current_state",
    }:
        raise ValueError("Unsupported action_expert.prediction_parameterization.")

    resources = _mapping(_required(raw, "resources", "Experiment config"), "resources")
    if _required(resources, "runtime", "resources") != "docker_linux":
        raise ValueError("This experiment must run inside the Docker Linux boundary.")
    if int(_required(resources, "feature_batch_size", "resources")) != 1:
        raise ValueError("Frozen feature extraction must start with batch_size=1.")
    training_device = str(resources.get("training_device", "cpu"))
    if training_device not in {"cpu", "xpu"}:
        raise ValueError("resources.training_device must be 'cpu' or 'xpu'.")

    benchmark = _mapping(_required(raw, "benchmark", "Experiment config"), "benchmark")
    if _required(benchmark, "required_before_training", "benchmark") is not True:
        raise ValueError("The pre-training benchmark gate cannot be disabled.")

    evaluation = _mapping(_required(raw, "evaluation", "Experiment config"), "evaluation")
    invalid_tolerance = float(
        _required(evaluation, "invalid_action_tolerance", "evaluation")
    )
    if not 0.0 <= invalid_tolerance <= 1.0:
        raise ValueError("evaluation.invalid_action_tolerance must be between 0 and 1.")

    training = _mapping(_required(raw, "training", "Experiment config"), "training")
    if int(_required(training, "optimizer_smoke_steps", "training")) != 1:
        raise ValueError("The guarded optimizer smoke must contain exactly one step.")
    if int(_required(training, "checkpoint_every_epochs", "training")) <= 0:
        raise ValueError("training.checkpoint_every_epochs must be positive.")
    first_action_loss_weight = float(training.get("first_action_loss_weight", 0.0))
    if not math.isfinite(first_action_loss_weight) or first_action_loss_weight < 0.0:
        raise ValueError(
            "training.first_action_loss_weight must be finite and non-negative."
        )
    early_phase_loss = training.get("early_phase_first_action_loss")
    if early_phase_loss is not None:
        early_phase_loss = _mapping(
            early_phase_loss, "training.early_phase_first_action_loss"
        )
        early_weight = float(early_phase_loss.get("weight", -1.0))
        maximum_frame = early_phase_loss.get("maximum_frame_index_exclusive")
        expected_samples = early_phase_loss.get("expected_selected_train_samples")
        if (
            early_weight != 1.0
            or type(maximum_frame) is not int
            or maximum_frame <= 0
            or type(expected_samples) is not int
            or expected_samples <= 0
        ):
            raise ValueError(
                "Early-phase first-action loss requires fixed weight 1.0, a positive "
                "exclusive frame bound, and a positive expected train-sample count."
            )
    state_pairing = training.get("aligned_expert_replay_state_pairing")
    if state_pairing is not None:
        state_pairing = _mapping(
            state_pairing, "training.aligned_expert_replay_state_pairing"
        )
        if state_pairing.get("enabled") is not True:
            raise ValueError("Aligned expert-replay state pairing cannot be disabled in place.")
        if float(state_pairing.get("weight", -1.0)) != 1.0:
            raise ValueError("Aligned expert-replay state-pairing weight is fixed at 1.0.")
        manifest = Path(str(state_pairing.get("manifest", "")))
        if not manifest.parts or manifest.is_absolute() or ".." in manifest.parts:
            raise ValueError(
                "Aligned expert-replay state-pairing manifest must be a safe relative path."
            )
        if early_phase_loss is not None or first_action_loss_weight != 0.0 or float(
            training.get("state_noise_std_normalized", 0.0)
        ) != 0.0:
            raise ValueError(
                "State pairing cannot be combined with scoped/global first-action "
                "weighting or state noise."
            )
    if early_phase_loss is not None and (
        first_action_loss_weight != 0.0
        or state_pairing is not None
        or float(training.get("state_noise_std_normalized", 0.0)) != 0.0
    ):
        raise ValueError(
            "Early-phase first-action loss cannot be combined with global first-action "
            "weighting, state pairing, or state noise."
        )

    dataset = _mapping(_required(raw, "dataset", "Experiment config"), "dataset")
    dataset_path = repository_root / str(_required(dataset, "config", "dataset"))
    dataset_raw = _mapping(
        yaml.safe_load(dataset_path.read_text(encoding="utf-8")), "Dataset config"
    )
    episodes = {int(episode) for episode in _required(dataset_raw, "episodes", "Dataset config")}
    _validate_episode_split(dataset, episodes)

    return raw
