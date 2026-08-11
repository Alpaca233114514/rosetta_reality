import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.run_smolvla_phase import _phase_arguments
from scripts.run_smolvla_xpu_performance import (
    _project_candidate,
    _runtime_experiment,
    _validate_performance_plan,
)
from scripts.train_smolvla_trackio import _install_masked_camera_encoder_skip

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE_PLAN = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_xpu_performance_003.yaml"
)
BASE_CONFIG = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"


def test_performance_plan_preserves_identity_split_and_memory_guardrails() -> None:
    plan, _base_path, experiment, _formal_path, _formal_plan = (
        _validate_performance_plan(PERFORMANCE_PLAN)
    )

    assert plan["protocol"]["episodes"] == experiment["dataset"]["train_episodes"]
    assert not set(plan["protocol"]["episodes"]) & set(
        experiment["dataset"]["test_episodes"]
    )
    assert plan["resources"]["memory_limit"] == "8g"
    assert plan["resources"]["memory_swap_limit"] == "8g"
    assert plan["resources"]["authorization"] == "user_explicit_2026-08-11"
    assert plan["target"]["maximum_peak_xpu_allocated_bytes"] == 7 * 1024**3
    assert [candidate["batch_size"] for candidate in plan["candidates"].values()] == [
        8,
        12,
        12,
        16,
        16,
    ]


def test_runtime_candidate_changes_only_upstream_performance_levers() -> None:
    plan = yaml.safe_load(PERFORMANCE_PLAN.read_text(encoding="utf-8"))
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    candidate = plan["candidates"]["eager-b12-skip-v3"]
    runtime = _runtime_experiment(base, plan, candidate)

    expected = copy.deepcopy(base)
    expected["model"]["policy"].update(
        {"empty_cameras": 2, "compile_model": False}
    )
    expected["resources"].update({"memory_limit": "8g", "memory_swap_limit": "8g"})
    expected["phases"]["formal"] = runtime["phases"]["formal"]
    assert runtime == expected
    assert runtime["model"]["adaptation"] == base["model"]["adaptation"]
    assert runtime["dataset"] == base["dataset"]


def test_phase_arguments_keep_old_defaults_and_expose_bounded_performance_flags(
    monkeypatch,
) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    monkeypatch.setenv("ROSETTA_TORCH_DEVICE", "xpu")
    smoke = _phase_arguments(
        base,
        "smoke",
        "smoke-defaults-001",
        Path("/model"),
        Path("/data"),
        Path("/output"),
    )
    assert "--save_checkpoint=true" in smoke
    assert "--num_workers=0" in smoke
    assert "--persistent_workers=false" in smoke
    assert not any(value.startswith("--policy.compile_model=") for value in smoke)

    plan = yaml.safe_load(PERFORMANCE_PLAN.read_text(encoding="utf-8"))
    runtime = _runtime_experiment(
        base, plan, plan["candidates"]["compile-b12-skip-v3"]
    )
    performance = _phase_arguments(
        runtime,
        "formal",
        "performance-compile-001",
        Path("/model"),
        Path("/data"),
        Path("/output"),
    )
    assert "--save_checkpoint=false" in performance
    assert "--policy.empty_cameras=2" in performance
    assert "--policy.compile_model=true" in performance
    assert "--policy.compile_mode=reduce-overhead" in performance


def test_candidate_projection_includes_startup_warmup_and_checkpoint_allowance() -> None:
    plan = yaml.safe_load(PERFORMANCE_PLAN.read_text(encoding="utf-8"))
    candidate = plan["candidates"]["eager-b12-skip-v3"]
    metrics = [
        {
            "train/step_s": 0.4,
            "train/update_s": 0.36,
            "train/dataloading_s": 0.03,
            "train/preprocessing_s": 0.01,
            "train/samples_per_s": 20.0,
            "train/loss": 1.0,
            "train/grad_norm": 2.0,
            "train/xpu_max_allocated_bytes": 2 * 1024**3,
        }
        for _ in range(30)
    ]

    result = _project_candidate(
        plan=plan,
        candidate=candidate,
        metrics=metrics,
        total_wall_seconds=40.0,
    )

    assert result["measured_startup_seconds"] == 28.0
    assert result["projected_optimizer_steps_for_one_pass"] == 1667
    assert result["projected_policy_prefix_calls_for_one_pass"] == 1667
    assert result["policy_prefix_call_reduction_fraction"] == 0.91665
    assert result["projected_one_pass_wall_seconds"] == pytest.approx(754.8)
    assert result["target_met"] is True


def test_masked_camera_skip_preserves_slots_and_calls_vision_once() -> None:
    from lerobot.policies.smolvla import modeling_smolvla

    flow_class = modeling_smolvla.VLAFlowMatching
    original = flow_class.embed_prefix
    _install_masked_camera_encoder_skip()
    optimized = flow_class.embed_prefix

    class FakeVlm:
        def __init__(self) -> None:
            self.image_calls = 0

        def embed_image(self, image: torch.Tensor) -> torch.Tensor:
            self.image_calls += 1
            return image.new_ones((image.shape[0], 4, 8))

        def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
            return torch.ones(tokens.shape[0], tokens.shape[1], 8)

    fake = SimpleNamespace(
        add_image_special_tokens=False,
        prefix_length=0,
        config=SimpleNamespace(empty_cameras=2),
        vlm_with_expert=FakeVlm(),
        state_proj=lambda state: torch.ones(state.shape[0], 8),
    )
    images = [torch.ones(1, 3, 2, 2) for _ in range(3)]
    masks = [
        torch.tensor([True]),
        torch.tensor([False]),
        torch.tensor([False]),
    ]
    try:
        embeddings, padding, attention = optimized(
            fake,
            images,
            masks,
            torch.ones(1, 2, dtype=torch.int64),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 14),
        )
    finally:
        flow_class.embed_prefix = original

    assert fake.vlm_with_expert.image_calls == 1
    assert embeddings.shape == (1, 15, 8)
    assert padding.shape == (1, 15)
    assert attention.shape == (1, 15)
    assert padding[:, 4:12].count_nonzero() == 0
