import hashlib
from pathlib import Path

import yaml

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
    assert "--delete" not in staging
    assert "rm -" not in staging


def test_autodl_resource_exception_is_no_optimizer_preflight_only() -> None:
    launcher = PREFLIGHT_PATH.read_text(encoding="utf-8")

    assert 'ROSETTA_AUTODL_NO_OPTIMIZER_AUTHORIZED") != "1"' in launcher
    assert 'ROSETTA_TORCH_DEVICE") != "cuda"' in launcher
    assert 'enabled_by_profile") is not False' in launcher
    assert 'preflight.get("optimizer_created") is not False' in launcher
    assert '"formal_training_authorized": False' in launcher


def test_autodl_files_do_not_change_faust_hash_bound_runtime() -> None:
    plan = yaml.safe_load(
        (
            REPOSITORY_ROOT
            / "configs/vla/smolvla_450m_aloha_insertion_faust_batch8_002.yaml"
        ).read_text(encoding="utf-8")
    )
    for relative in (
        "scripts/smolvla_forward_check.py",
        "scripts/run_smolvla_action_repair_phase.py",
        "src/rosetta_reality/tracking/trackio_lerobot.py",
    ):
        digest = hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
        assert plan["implementation_files"][relative] == digest
