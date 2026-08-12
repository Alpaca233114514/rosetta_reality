import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.autodl_doctor import _validate_recorded_files

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPOSITORY_ROOT / "configs/runtime/autodl_rtx4090.yaml"
RUNNER_PATH = REPOSITORY_ROOT / "scripts/run_autodl.sh"
BOOTSTRAP_PATH = REPOSITORY_ROOT / "scripts/bootstrap_autodl.sh"
STAGE_PATH = REPOSITORY_ROOT / "scripts/stage_autodl_from_wsl.sh"
PREFLIGHT_PATH = REPOSITORY_ROOT / "scripts/run_autodl_preflight.py"


def test_autodl_profile_uses_platform_container_and_fail_closed_formal_gate() -> None:
    profile = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))

    assert profile["platform"] == "autodl_container_instance"
    assert profile["runtime_boundary"] == "platform_linux_container"
    assert profile["nested_docker_supported"] is False
    assert profile["accelerator"]["torch_device"] == "cuda"
    assert profile["accelerator"]["minimum_total_memory_bytes"] >= 23 * 1024**3
    assert profile["network_policy"]["optimizer"] == "disabled"
    assert profile["packages"]["lerobot_revision"] == (
        "c903b114a90e703b3f7d0c46cb38727c328c55ff"
    )
    assert profile["agent_monitoring"] == {
        "blocking_shell": "bash",
        "blocking_command": "sleep 300",
        "blocking_seconds": 300,
        "fixed_interval": True,
        "full_audit_wake_fractions": [0.25, 0.5, 0.75, 1.0],
        "short_polling_allowed": False,
    }
    assert profile["formal_training"]["enabled_by_profile"] is False
    assert profile["hidden_test_loaded"] is False


def test_autodl_runner_is_offline_and_refuses_formal_training() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "HF_HUB_OFFLINE=1" in runner
    assert "HF_DATASETS_OFFLINE=1" in runner
    assert "ROSETTA_TORCH_DEVICE=cuda" in runner
    assert "run_benchmark" in runner
    assert "formal)" in runner
    assert "formal CUDA training is locked" in runner
    assert "docker run" not in runner


def test_autodl_bootstrap_preserves_preinstalled_cuda_pytorch() -> None:
    bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")

    assert "torch.cuda.is_available()" in bootstrap
    assert "--system-site-packages" in bootstrap
    assert 'for package in ("torch", "torchvision")' in bootstrap
    assert "PIP_CONSTRAINT" in bootstrap
    assert "pip install torch" not in bootstrap
    assert "pip install cuda" not in bootstrap.lower()


def test_autodl_staging_is_versioned_and_never_deletes_remote_files() -> None:
    staging = STAGE_PATH.read_text(encoding="utf-8")

    assert "/root/autodl-tmp/rosetta/workspaces/${release_id}" in staging
    assert "remote release already exists" in staging
    assert "sha256sum \"$archive_path\"" in staging
    assert ".rosetta-workspace.sha256" in staging
    assert "status --porcelain" not in staging
    assert "--delete" not in staging
    assert 'ssh "$host" "rm ' not in staging


def test_autodl_resource_exception_is_no_optimizer_preflight_only() -> None:
    launcher = PREFLIGHT_PATH.read_text(encoding="utf-8")

    assert 'ROSETTA_AUTODL_NO_OPTIMIZER_AUTHORIZED") != "1"' in launcher
    assert 'ROSETTA_TORCH_DEVICE") != "cuda"' in launcher
    assert 'enabled_by_profile") is not False' in launcher
    assert 'preflight.get("optimizer_created") is not False' in launcher
    assert '"formal_training_authorized": False' in launcher


def test_autodl_files_preserve_historical_faust_hash_inventory() -> None:
    plan = yaml.safe_load(
        (
            REPOSITORY_ROOT
            / "configs/vla/smolvla_450m_aloha_insertion_faust_batch8_002.yaml"
        ).read_text(encoding="utf-8")
    )
    historical = {
        "scripts/smolvla_forward_check.py": (
            "ef585b3940acba34c87bd11ba4dde5176948e673bdbe7de6d2dd351e3dc82a33"
        ),
        "scripts/run_smolvla_action_repair_phase.py": (
            "8f5138b38038827663df84e1143a29740b55f42445ad17b7b73de17014cd9456"
        ),
        "src/rosetta_reality/tracking/trackio_lerobot.py": (
            "009cd1041e4a6b4c1ba1860bfba20b6166c24bdb593c7b7efb23fe02d3b02bbd"
        ),
    }
    assert {
        relative: plan["implementation_files"][relative] for relative in historical
    } == historical
    assert not {
        "scripts/autodl_doctor.py",
        "scripts/run_autodl.sh",
        "scripts/stage_autodl_from_wsl.sh",
    } & set(plan["implementation_files"])


def test_autodl_doctor_recomputes_manifest_file_records(tmp_path: Path) -> None:
    content = tmp_path / "weights.bin"
    content.write_bytes(b"immutable")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "weights.bin": {
                        "bytes": content.stat().st_size,
                        "sha256": hashlib.sha256(content.read_bytes()).hexdigest(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert _validate_recorded_files(tmp_path, manifest, label="test") == 1
    content.write_bytes(b"tampered!")
    with pytest.raises(ValueError, match="identity changed"):
        _validate_recorded_files(tmp_path, manifest, label="test")
