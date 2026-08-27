"""Post-training compatibility regressions found by the AutoDL Way run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.runtime_compatibility import (
    plan_with_normalization_alias,
    require_absolute_environment_directory,
    resolve_runtime_evidence_path,
    resolve_tokenizer_identity,
    validate_cuda_compile_contract,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _nested_plan() -> dict:
    return {
        "prerequisites": {
            "normalization": {
                "path": "runs/experiment/normalization/train.json",
                "sha256": "a" * 64,
            },
            "dataset_view_manifest": {
                "path": "runs/experiment/dataset_views/train/view_manifest.json",
                "sha256": "b" * 64,
            },
        }
    }


def test_nested_normalization_is_exposed_without_mutating_plan() -> None:
    plan = _nested_plan()
    compatible = plan_with_normalization_alias(plan)

    assert "normalization" not in plan
    assert compatible["normalization"] == {
        "source_split": "train",
        "report": "runs/experiment/normalization/train.json",
        "report_sha256": "a" * 64,
        "dataset_view_manifest": (
            "runs/experiment/dataset_views/train/view_manifest.json"
        ),
        "dataset_view_manifest_sha256": "b" * 64,
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
    }


def test_conflicting_normalization_views_fail_closed() -> None:
    plan = _nested_plan()
    plan["normalization"] = {
        "source_split": "train",
        "report": "runs/other/normalization.json",
        "report_sha256": "c" * 64,
        "dataset_view_manifest": (
            "runs/experiment/dataset_views/train/view_manifest.json"
        ),
        "dataset_view_manifest_sha256": "b" * 64,
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
    }

    with pytest.raises(ValueError, match="identities conflict"):
        plan_with_normalization_alias(plan)


def test_environment_directory_rejects_missing_and_relative_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be set"):
        require_absolute_environment_directory("HF_HOME", environment={})
    with pytest.raises(ValueError, match="absolute"):
        require_absolute_environment_directory(
            "HF_HOME", environment={"HF_HOME": "relative/cache"}
        )
    assert require_absolute_environment_directory(
        "HF_HOME", environment={"HF_HOME": str(tmp_path)}
    ) == tmp_path.resolve()


def test_runtime_evidence_routes_runs_to_durable_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    run_root = tmp_path / "durable-runs"
    repository.mkdir()
    run_root.mkdir()
    tracked = repository / "configs" / "plan.yaml"
    tracked.parent.mkdir()
    tracked.write_text("plan\n", encoding="utf-8")
    evidence = run_root / "experiment" / "report.json"
    evidence.parent.mkdir()
    evidence.write_text("{}\n", encoding="utf-8")

    assert resolve_runtime_evidence_path(
        "configs/plan.yaml", repository_root=repository, run_root=run_root
    ) == tracked.resolve()
    assert resolve_runtime_evidence_path(
        "runs/experiment/report.json",
        repository_root=repository,
        run_root=run_root,
    ) == evidence.resolve()
    with pytest.raises(ValueError, match="safe POSIX"):
        resolve_runtime_evidence_path(
            "runs/../secret", repository_root=repository, run_root=run_root
        )
    with pytest.raises(ValueError, match="safe POSIX"):
        resolve_runtime_evidence_path(
            "C:/secret", repository_root=repository, run_root=run_root
        )


def _dependency_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    base = tmp_path / "base"
    hf_home = tmp_path / "hf-home"
    snapshot_relative = Path("hub/models--test--vlm/snapshots/revision")
    snapshot = hf_home / snapshot_relative
    snapshot.mkdir(parents=True)
    base.mkdir()
    files: dict[str, dict[str, int | str]] = {}
    for name, content in {
        "tokenizer.json": "tokenizer\n",
        "tokenizer_config.json": "config\n",
        "chat_template.json": "template\n",
    }.items():
        path = snapshot / name
        path.write_text(content, encoding="utf-8")
        files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "source": "huggingface",
        "repo_id": "test/vlm",
        "revision": "revision",
        "cache_layout": snapshot_relative.as_posix(),
        "files": files,
    }
    (base / "vlm_dependency_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    experiment = {
        "model": {
            "vlm_dependency": {
                "identifier": "test/vlm",
                "revision": "revision",
                "manifest": "vlm_dependency_manifest.json",
            }
        }
    }
    return base, hf_home, experiment


def test_base_policy_can_use_pinned_dependency_tokenizer(tmp_path: Path) -> None:
    base, hf_home, experiment = _dependency_fixture(tmp_path)
    hashes, identity = resolve_tokenizer_identity(
        base,
        base_model_root=base,
        experiment=experiment,
        hf_home=hf_home,
    )

    assert set(hashes) == {
        "chat_template.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert identity["source"] == "pinned_vlm_dependency_snapshot"
    assert identity["repo_id"] == "test/vlm"
    assert identity["revision"] == "revision"
    assert identity["manifest_sha256"] == file_sha256(
        base / "vlm_dependency_manifest.json"
    )


def test_checkpoint_without_tokenizer_does_not_fall_back(tmp_path: Path) -> None:
    base, hf_home, experiment = _dependency_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(FileNotFoundError, match="non-base"):
        resolve_tokenizer_identity(
            checkpoint,
            base_model_root=base,
            experiment=experiment,
            hf_home=hf_home,
        )


def test_policy_tokenizer_is_preferred_over_dependency(tmp_path: Path) -> None:
    base, _hf_home, experiment = _dependency_fixture(tmp_path)
    tokenizer = base / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("policy\n", encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text("policy config\n", encoding="utf-8")

    hashes, identity = resolve_tokenizer_identity(
        base,
        base_model_root=base,
        experiment=experiment,
        hf_home=tmp_path / "absent-cache",
    )

    assert set(hashes) == {"tokenizer.json", "tokenizer_config.json"}
    assert identity == {"source": "policy_tokenizer_directory"}

    with pytest.raises(ValueError, match="differs from the registered plan"):
        resolve_tokenizer_identity(
            base,
            base_model_root=base,
            experiment=experiment,
            hf_home=tmp_path / "absent-cache",
            expected_tokenizer_identity={
                "source": "pinned_vlm_dependency_snapshot"
            },
        )


def test_reduce_overhead_requires_dedicated_cuda_graph_smoke() -> None:
    with pytest.raises(ValueError, match="dedicated accepted two-step"):
        validate_cuda_compile_contract(
            {"compile_model": True, "compile_mode": "reduce-overhead"},
            cuda_graph_smoke_accepted=False,
        )
    validate_cuda_compile_contract(
        {"compile_model": True, "compile_mode": "default"},
        cuda_graph_smoke_accepted=False,
    )
    validate_cuda_compile_contract(
        {"compile_model": True, "compile_mode": "reduce-overhead"},
        cuda_graph_smoke_accepted=True,
    )


def test_completed_way_formal_plan_uses_safe_default_compile_and_nested_stats() -> None:
    plan = yaml.safe_load(
        (
            REPOSITORY_ROOT
            / "configs/vla/"
            "smolvla_450m_aloha_insertion_way_cuda_batch64_default_formal_002.yaml"
        ).read_text(encoding="utf-8")
    )
    compatible = plan_with_normalization_alias(plan)

    assert compatible["normalization"]["report"] == plan["prerequisites"][
        "normalization"
    ]["path"]
    assert compatible["normalization"]["dataset_view_manifest"] == plan[
        "prerequisites"
    ]["dataset_view_manifest"]["path"]
    validate_cuda_compile_contract(
        plan["training"]["policy"], cuda_graph_smoke_accepted=False
    )


@pytest.mark.parametrize(
    "config_path, hash_mapping",
    [
        (
            "configs/vla/"
            "smolvla_450m_aloha_insertion_way_cuda_batch64_default_"
            "validation_runtime_repair_003.yaml",
            "implementation_files",
        ),
        (
            "configs/vla/"
            "smolvla_450m_aloha_insertion_way_cuda_batch64_default_"
            "export_runtime_repair_001.yaml",
            "implementation_files",
        ),
        (
            "configs/vla/"
            "smolvla_450m_aloha_insertion_way_cuda_batch64_default_sim_011.yaml",
            "simulation_code_sha256",
        ),
    ],
)
def test_completed_way_runtime_repairs_remain_hash_bound(
    config_path: str, hash_mapping: str
) -> None:
    plan = yaml.safe_load((REPOSITORY_ROOT / config_path).read_text(encoding="utf-8"))
    for relative, expected in plan[hash_mapping].items():
        assert file_sha256(REPOSITORY_ROOT / relative) == expected
