"""Explicit dataset-to-SmolVLA action representation contracts."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SmolVLAActionSpace:
    """Representation choices that must remain identical across train and reload."""

    schema_version: int
    dataset_space: str
    normalization: str
    target_projection: str
    target_projection_stage: str
    reject_source_beyond_contract_tolerance: bool
    adapt_to_pi_aloha: bool
    representation_adapter: str
    representation_adapter_stage: str
    model_internal_space: str
    explicit: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON/YAML-safe identity mapping."""

        return asdict(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_smolvla_experiment(path: Path, repository_root: Path) -> dict[str, Any]:
    """Load a plain experiment or one checksum-pinned repair overlay."""

    path = path.resolve()
    repository_root = repository_root.resolve()
    if not path.is_file() or not path.is_relative_to(repository_root):
        raise ValueError("SmolVLA experiment must be a repository file.")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SmolVLA experiment must contain a mapping.")
    inheritance = value.pop("extends", None)
    if inheritance is None:
        return value
    if not isinstance(inheritance, dict):
        raise ValueError("SmolVLA repair inheritance must be a mapping.")
    relative = Path(str(inheritance.get("config", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("SmolVLA repair inheritance path is unsafe.")
    base_path = (repository_root / relative).resolve()
    if not base_path.is_file() or not base_path.is_relative_to(repository_root):
        raise FileNotFoundError("SmolVLA repair base experiment is missing.")
    expected_sha256 = inheritance.get("sha256")
    if not isinstance(expected_sha256, str) or _sha256(base_path) != expected_sha256:
        raise ValueError("SmolVLA repair base experiment checksum changed.")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or "extends" in base:
        raise ValueError("SmolVLA repair supports exactly one inheritance level.")
    merged = _deep_merge(base, value)
    merged["experiment_inheritance"] = {
        "config": relative.as_posix(),
        "sha256": expected_sha256,
    }
    return merged


def _legacy_action_space(experiment: dict[str, Any]) -> SmolVLAActionSpace:
    policy = experiment.get("model", {}).get("policy", {})
    adapt = bool(policy.get("adapt_to_pi_aloha", False)) if isinstance(policy, dict) else False
    return SmolVLAActionSpace(
        schema_version=1,
        dataset_space="standard_aloha_joint_position",
        normalization="dataset_train_only_mean_std",
        target_projection="none",
        target_projection_stage="none",
        reject_source_beyond_contract_tolerance=False,
        adapt_to_pi_aloha=adapt,
        representation_adapter="policy_pi_aloha" if adapt else "none",
        representation_adapter_stage="after_normalization" if adapt else "none",
        model_internal_space=(
            "pi_aloha_on_normalized_features" if adapt else "dataset_mean_std"
        ),
        explicit=False,
    )


def load_smolvla_action_space(
    experiment: dict[str, Any], *, require_explicit: bool = False
) -> SmolVLAActionSpace:
    """Load and validate the action representation without silently choosing an adapter."""

    model = experiment.get("model")
    if not isinstance(model, dict):
        raise ValueError("SmolVLA experiment is missing the model mapping.")
    policy = model.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("SmolVLA experiment is missing the model.policy mapping.")
    raw = model.get("action_space")
    if raw is None:
        if require_explicit:
            raise ValueError("New SmolVLA work requires an explicit model.action_space contract.")
        return _legacy_action_space(experiment)
    if not isinstance(raw, dict):
        raise ValueError("model.action_space must be a mapping.")

    adapt = policy.get("adapt_to_pi_aloha")
    if not isinstance(adapt, bool):
        raise ValueError(
            "Explicit SmolVLA action-space work requires policy.adapt_to_pi_aloha."
        )
    contract = SmolVLAActionSpace(
        schema_version=int(raw.get("schema_version", 0)),
        dataset_space=str(raw.get("dataset_space", "")),
        normalization=str(raw.get("normalization", "")),
        target_projection=str(raw.get("target_projection", "")),
        target_projection_stage=str(raw.get("target_projection_stage", "")),
        reject_source_beyond_contract_tolerance=raw.get(
            "reject_source_beyond_contract_tolerance"
        ),
        adapt_to_pi_aloha=adapt,
        representation_adapter=str(raw.get("representation_adapter", "")),
        representation_adapter_stage=str(raw.get("representation_adapter_stage", "")),
        model_internal_space=str(raw.get("model_internal_space", "")),
    )
    if (
        contract.schema_version != 1
        or contract.dataset_space != "standard_aloha_joint_position"
        or contract.normalization
        != "train_only_mean_std_after_representation_adapter"
        or contract.target_projection not in {"none", "action_contract_clip"}
        or not isinstance(contract.reject_source_beyond_contract_tolerance, bool)
    ):
        raise ValueError("The explicit SmolVLA action-space contract is invalid.")
    valid_representation = (
        contract.representation_adapter == "rosetta_pi_aloha"
        and contract.model_internal_space == "normalized_pi_aloha"
    ) or (
        contract.representation_adapter
        == "rosetta_pi_aloha_arms_bounded_sine_grippers"
        and contract.model_internal_space
        == "normalized_pi_aloha_arms_bounded_sine_grippers"
    )
    if (
        contract.adapt_to_pi_aloha
        or not valid_representation
        or contract.representation_adapter_stage
        != "after_target_projection_before_normalization"
    ):
        raise ValueError(
            "Explicit ALOHA work must use Rosetta's raw-feature adapter and disable "
            "the policy-level post-normalization adapter."
        )
    expected_stage = (
        "none" if contract.target_projection == "none" else "before_normalization"
    )
    if contract.target_projection_stage != expected_stage:
        raise ValueError("SmolVLA target projection is registered at the wrong stage.")
    if (
        contract.target_projection == "action_contract_clip"
        and not contract.reject_source_beyond_contract_tolerance
    ):
        raise ValueError(
            "Action Contract clipping must reject source values beyond registered tolerance."
        )
    return contract
