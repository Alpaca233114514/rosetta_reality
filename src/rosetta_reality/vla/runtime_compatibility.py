"""Fail-closed helpers for SmolVLA post-training runtime compatibility.

The first AutoDL Way run exposed schema and asset-layout differences between
new formal plans and historical validation/export entry points.  These helpers
provide a versioned compatibility boundary without changing hash-bound
historical runners.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rosetta_reality.experiment import file_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKENIZER_FILES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def _safe_relative_path(value: Any, *, label: str) -> Path:
    text = str(value).strip()
    candidate = Path(text)
    if (
        not text
        or "\\" in text
        or candidate.is_absolute()
        or ".." in candidate.parts
        or ":" in candidate.parts[0]
        or any(part in {"", "."} for part in candidate.parts)
    ):
        raise ValueError(f"{label} must be a safe POSIX-style relative path.")
    return candidate


def require_absolute_environment_directory(
    name: str, *, environment: Mapping[str, str] | None = None
) -> Path:
    """Return an existing absolute directory from an environment variable."""

    values = os.environ if environment is None else environment
    raw = values.get(name)
    if raw is None or not str(raw).strip():
        raise ValueError(f"{name} must be set to an absolute directory.")
    configured = Path(str(raw))
    if not configured.is_absolute():
        raise ValueError(f"{name} must be set to an absolute directory.")
    resolved = configured.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist.")
    return resolved


def resolve_runtime_evidence_path(
    raw: str,
    *,
    repository_root: Path,
    run_root: Path,
) -> Path:
    """Resolve plan evidence without coupling ignored ``runs/`` to code identity."""

    relative = _safe_relative_path(raw, label="runtime evidence path")
    repository = Path(repository_root)
    durable_runs = Path(run_root)
    if not repository.is_absolute() or not durable_runs.is_absolute():
        raise ValueError("Runtime evidence roots must be absolute.")
    repository = repository.resolve()
    durable_runs = durable_runs.resolve()
    if not repository.is_dir() or not durable_runs.is_dir():
        raise FileNotFoundError("Runtime evidence root does not exist.")
    if relative.parts[0] == "runs":
        if len(relative.parts) == 1:
            raise ValueError("A runs/ evidence path must identify a file.")
        root = durable_runs
        relative = Path(*relative.parts[1:])
    else:
        root = repository
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise FileNotFoundError("Registered runtime evidence is missing or escaped its root.")
    return candidate


def _nested_normalization(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    prerequisites = plan.get("prerequisites")
    if prerequisites is None:
        return None
    prerequisites = _mapping(prerequisites, label="plan.prerequisites")
    normalization = prerequisites.get("normalization")
    view = prerequisites.get("dataset_view_manifest")
    if normalization is None and view is None:
        return None
    normalization = _mapping(normalization, label="plan.prerequisites.normalization")
    view = _mapping(view, label="plan.prerequisites.dataset_view_manifest")
    return {
        "source_split": "train",
        "report": str(normalization.get("path", "")),
        "report_sha256": _sha256(
            normalization.get("sha256"),
            label="plan.prerequisites.normalization.sha256",
        ),
        "dataset_view_manifest": str(view.get("path", "")),
        "dataset_view_manifest_sha256": _sha256(
            view.get("sha256"),
            label="plan.prerequisites.dataset_view_manifest.sha256",
        ),
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
    }


def _legacy_normalization(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = plan.get("normalization")
    if raw is None:
        return None
    record = _mapping(raw, label="plan.normalization")
    if (
        record.get("source_split") != "train"
        or record.get("validation_episodes_loaded") is not False
        or record.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Legacy normalization must remain train-only and sealed.")
    return {
        "source_split": "train",
        "report": str(record.get("report", "")),
        "report_sha256": _sha256(
            record.get("report_sha256"), label="plan.normalization.report_sha256"
        ),
        "dataset_view_manifest": str(record.get("dataset_view_manifest", "")),
        "dataset_view_manifest_sha256": _sha256(
            record.get("dataset_view_manifest_sha256"),
            label="plan.normalization.dataset_view_manifest_sha256",
        ),
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
    }


def plan_with_normalization_alias(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with one unambiguous legacy normalization view."""

    nested = _nested_normalization(plan)
    legacy = _legacy_normalization(plan)
    if nested is None and legacy is None:
        raise ValueError("The formal plan has no normalization identity.")
    if nested is not None and legacy is not None and nested != legacy:
        raise ValueError("Nested and legacy normalization identities conflict.")
    selected = nested if nested is not None else legacy
    assert selected is not None
    for key in ("report", "dataset_view_manifest"):
        _safe_relative_path(selected[key], label=f"normalization.{key}")
    compatible = copy.deepcopy(dict(plan))
    compatible["normalization"] = copy.deepcopy(selected)
    return compatible


def _directory_tokenizer_identity(tokenizer: Path) -> tuple[dict[str, str], dict[str, Any]]:
    files = [path for path in sorted(tokenizer.rglob("*")) if path.is_file()]
    if not files:
        raise FileNotFoundError("Policy tokenizer is missing or empty.")
    resolved_root = tokenizer.resolve()
    hashes: dict[str, str] = {}
    for path in files:
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError("Policy tokenizer file escaped its directory.")
        relative = resolved.relative_to(resolved_root).as_posix()
        _safe_relative_path(relative, label="policy tokenizer file")
        hashes[relative] = file_sha256(resolved)
    if "tokenizer.json" not in hashes or "tokenizer_config.json" not in hashes:
        raise ValueError("Policy tokenizer identity is incomplete.")
    return hashes, {"source": "policy_tokenizer_directory"}


def resolve_tokenizer_identity(
    source_dir: Path,
    *,
    base_model_root: Path,
    experiment: Mapping[str, Any],
    hf_home: Path,
    expected_tokenizer_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve a policy tokenizer or the pinned base VLM dependency tokenizer."""

    source = Path(source_dir).resolve()
    base = Path(base_model_root).resolve()
    tokenizer = source / "tokenizer"
    if tokenizer.is_dir() and any(path.is_file() for path in tokenizer.rglob("*")):
        hashes, identity = _directory_tokenizer_identity(tokenizer)
        if (
            expected_tokenizer_identity is not None
            and dict(expected_tokenizer_identity) != identity
        ):
            raise ValueError("The policy tokenizer identity differs from the registered plan.")
        return hashes, identity
    if source != base:
        raise FileNotFoundError("A non-base policy tokenizer is missing or empty.")
    if not Path(hf_home).is_absolute() or not Path(hf_home).resolve().is_dir():
        raise ValueError("HF_HOME must identify an existing absolute cache root.")
    hf_home = Path(hf_home).resolve()

    model = _mapping(experiment.get("model"), label="experiment.model")
    dependency = _mapping(
        model.get("vlm_dependency"), label="experiment.model.vlm_dependency"
    )
    manifest_relative = _safe_relative_path(
        dependency.get("manifest"), label="VLM dependency manifest"
    )
    manifest_path = (base / manifest_relative).resolve()
    if not manifest_path.is_relative_to(base) or not manifest_path.is_file():
        raise FileNotFoundError("The pinned VLM dependency manifest is missing.")
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        label="VLM dependency manifest",
    )
    cache_relative = _safe_relative_path(
        manifest.get("cache_layout"), label="VLM dependency cache layout"
    )
    snapshot = (hf_home / cache_relative).resolve()
    recorded = _mapping(manifest.get("files"), label="VLM dependency files")
    if (
        not snapshot.is_relative_to(hf_home)
        or not snapshot.is_dir()
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "validated"
        or manifest.get("source") != "huggingface"
        or manifest.get("repo_id") != dependency.get("identifier")
        or manifest.get("revision") != dependency.get("revision")
    ):
        raise ValueError("The pinned VLM dependency identity is invalid.")

    hashes: dict[str, str] = {}
    for relative_name in sorted(_TOKENIZER_FILES & set(recorded)):
        relative = _safe_relative_path(relative_name, label="VLM tokenizer file")
        record = _mapping(recorded[relative_name], label=f"VLM file {relative_name}")
        candidate = (snapshot / relative).resolve()
        if (
            not candidate.is_relative_to(snapshot)
            or not candidate.is_file()
            or candidate.stat().st_size != record.get("bytes")
            or file_sha256(candidate)
            != _sha256(record.get("sha256"), label=f"VLM file {relative_name}.sha256")
        ):
            raise ValueError(f"Pinned VLM tokenizer file changed: {relative_name}.")
        hashes[relative_name] = str(record["sha256"])
    if "tokenizer.json" not in hashes or "tokenizer_config.json" not in hashes:
        raise ValueError("The pinned VLM dependency tokenizer set is incomplete.")
    identity = {
        "source": "pinned_vlm_dependency_snapshot",
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "manifest_sha256": file_sha256(manifest_path),
        "cache_layout": manifest["cache_layout"],
    }
    if expected_tokenizer_identity is not None and dict(expected_tokenizer_identity) != identity:
        raise ValueError("The VLM tokenizer identity differs from the registered plan.")
    return hashes, identity


def validate_cuda_compile_contract(
    policy: Mapping[str, Any], *, cuda_graph_smoke_accepted: bool
) -> None:
    """Reject an unproven CUDA Graph mode before an optimizer is created."""

    compile_model = policy.get("compile_model")
    mode = str(policy.get("compile_mode", "default"))
    if compile_model is not True:
        raise ValueError("The registered CUDA policy must enable torch.compile.")
    if mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError("The registered torch.compile mode is unsupported.")
    if mode == "reduce-overhead" and not cuda_graph_smoke_accepted:
        raise ValueError(
            "reduce-overhead requires a dedicated accepted two-step CUDA Graph smoke."
        )
