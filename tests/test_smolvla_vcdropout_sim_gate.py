"""Vcdropout simulation-gate wrapper tests (no weights, no simulator).

Covers the create-only local Gate 3/4 dispatch chain added after the frozen
gradient gate passed: the fail-closed gradient-gate entry permit, the derived
gate-facing selection record, the inventory backup record, and the rendered
simulation plan's frozen protocol fields. Nothing here loads model weights or
drives the simulator; the frozen engine (``smolvla_sim_gate``) is not touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for candidate in (str(REPOSITORY_ROOT / "src"), str(SCRIPTS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import smolvla_vcdropout_protocol as protocol  # type: ignore[import-not-found]  # noqa: E402
import smolvla_vcdropout_sim_gate as wrapper  # type: ignore[import-not-found]  # noqa: E402

FORMAL_PLAN = (
    Path(__file__).resolve().parents[1]
    / "configs/vla/smolvla_450m_aloha_insertion_vcdropout_cuda_b64_001.yaml"
)
ARTIFACT_ID = "m2-smolvla450m-vcdropout-cuda-b64-001-step0237-deploy-001-test"


def _write_manifest(artifact_dir: Path, artifact_id: str) -> None:
    manifest = {
        "plan_id": "m2-smolvla450m-vcdropout-001",
        "artifact_id": artifact_id,
        "experiment_id": protocol.EXPERIMENT_ID,
        "hidden_test_loaded": False,
        "selected_checkpoint_step": 237,
        "selected_checkpoint_model_sha256": "a" * 64,
        "selection_report_sha256": "b" * 64,
        "reload": {"verified": True, "exact_tensor_equality": True},
    }
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_gate_report(
    run_root: Path, artifact_id: str, *, passed: bool = True
) -> Path:
    directory = run_root / protocol.EXPERIMENT_ID / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "gate_passed": passed,
        "status": "passed" if passed else "failed",
        "artifact_id": artifact_id,
        "gate_criteria": [
            {"name": "normal_mean_flow_loss_ceiling", "passed": passed},
            {"name": "state_sensitivity_ceiling", "passed": passed},
            {"name": "image_sensitivity_floor", "passed": passed},
            {"name": "state_dominance_ceiling", "passed": passed},
        ],
    }
    path = directory / "vcdropout-gradient-gate-0000000000000001.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.fixture()
def gate_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    artifact_dir = tmp_path / "artifacts" / protocol.EXPERIMENT_ID / ARTIFACT_ID
    run_root = tmp_path / "runs"
    _write_manifest(artifact_dir, ARTIFACT_ID)
    _write_gate_report(run_root, ARTIFACT_ID)
    monkeypatch.setenv("ROSETTA_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(run_root))
    return {"artifact_dir": artifact_dir, "run_root": run_root}


def test_template_freezes_registered_protocol() -> None:
    evidence = {
        "run_name": "m2-smolvla450m-vcdropout-cuda-b64-001",
        "artifact_dir": Path("artifact"),
        "artifact_manifest_sha256": "c" * 64,
        "gradient_gate_report": "runs/g.json",
        "gradient_gate_sha": "d" * 64,
        "selection_report": "runs/s.json",
        "selection_sha": "e" * 64,
        "selected_step": 237,
        "model_sha": "f" * 64,
        "backup_sha": "0" * 64,
    }
    content, sim_plan_id = wrapper._build_sim_plan_content(evidence)
    assert sim_plan_id == "m2-smolvla450m-vcdropout-cuda-b64-001-sim-433"
    plan = yaml.safe_load(content)
    assert plan["status"] == "preregistered"
    assert plan["stage"] == "m2_closed_loop_simulation"
    assert plan["experiment_id"] == protocol.EXPERIMENT_ID
    assert plan["artifact_manifest_sha256"] == "c" * 64
    assert plan["gradient_gate"]["report_sha256"] == "d" * 64
    assert plan["gradient_gate"]["gate_passed"] is True
    assert plan["single_axis_change"] == {
        "field": "training.state_conditioning_dropout",
        "control": "none_uniform_flow_loss",
        "candidate": "samplewise_normalized_state_dropout_p0.5_dedicated_rng",
    }
    assert plan["prior_failure"]["failed_criterion"] == "raw_actions_within_contract"
    assert (
        plan["prior_task_failure"]["failed_criterion"] == "minimum_task_success_rate"
    )
    assert plan["action_contract"]["sha256"] == wrapper.CONTRACT_SHA
    assert len(plan["collision_policy"]["allowed_task_contacts"]) == 5
    assert plan["inference"]["noise"] == "seeded_standard_normal"
    assert plan["inference"]["policy_output_projection"] == "action_contract_clip"
    assert plan["inference"]["chunk_execution_steps"] == 1
    assert plan["resources"]["memory_limit"] == "6g"
    assert plan["resources"]["accelerator"] == "xpu"
    assert plan["resources"]["nested_docker_used"] is False
    assert plan["gate3"]["seed"] == 20260809
    assert plan["gate3"]["policy_noise_seed"] == 20260809
    assert plan["gate3"]["maximum_steps"] == 20
    assert plan["gate3"]["report_suffix"] == "433"
    assert plan["gate3"]["require_finite_actions"] is True
    assert plan["gate3"]["require_projected_policy_actions_within_contract"] is True
    assert plan["gate3"]["require_adapter_no_additional_clipping"] is True
    assert plan["gate4"]["seeds"] == [1000, 1001, 1002, 1003, 1004]
    assert plan["gate4"]["policy_noise_seeds"] == [1000, 1001, 1002, 1003, 1004]
    assert plan["gate4"]["maximum_steps"] == 500
    assert plan["gate4"]["minimum_task_success_rate"] == 0.2
    assert plan["gate4"]["require_gate3_passed"] is True
    assert plan["gate4"]["report_suffix"] == "433"
    assert plan["hidden_test_loaded"] is False
    assert set(plan["simulation_code_sha256"]) == set(wrapper.SIMULATION_CODE_FILES)


def test_gradient_gate_entry_permit_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_environment: dict[str, Path]
) -> None:
    directory = gate_environment["run_root"] / protocol.EXPERIMENT_ID / "diagnostics"
    (directory / "vcdropout-gradient-gate-0000000000000002.json").write_text(
        json.dumps(
            {
                "gate_passed": True,
                "artifact_id": ARTIFACT_ID,
                "failed_criteria": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="entry permit"):
        wrapper._prepare_gate_evidence(FORMAL_PLAN, ARTIFACT_ID)
    (directory / "vcdropout-gradient-gate-0000000000000002.json").unlink()
    failed = _write_gate_report(
        gate_environment["run_root"], ARTIFACT_ID, passed=False
    )
    failed.rename(failed.with_name("stale.json"))
    _write_gate_report(gate_environment["run_root"], ARTIFACT_ID, passed=False)
    with pytest.raises(ValueError, match="entry permit"):
        wrapper._prepare_gate_evidence(FORMAL_PLAN, ARTIFACT_ID)


def test_prepare_writes_records_and_content_binds_artifact(
    gate_environment: dict[str, Path],
) -> None:
    evidence = wrapper._prepare_gate_evidence(FORMAL_PLAN, ARTIFACT_ID)
    assert evidence["selected_step"] == 237
    assert evidence["model_sha"] == "a" * 64
    assert evidence["plan_id"] == "m2-smolvla450m-vcdropout-001"

    selection = json.loads(
        (
            gate_environment["run_root"]
            / protocol.EXPERIMENT_ID
            / "selection"
            / "m2-smolvla450m-vcdropout-cuda-b64-001-selection-gate.json"
        ).read_text(encoding="utf-8")
    )
    assert selection["status"] == "passed"
    assert selection["selected"] == {
        "step": 237,
        "model_safetensors_sha256": "a" * 64,
    }
    assert selection["hidden_test_loaded"] is False
    assert selection["derived_from"]["artifact_manifest_sha256"] == evidence[
        "artifact_manifest_sha256"
    ]

    backup = json.loads(
        (
            gate_environment["run_root"]
            / protocol.EXPERIMENT_ID
            / "artifact_backup"
            / f"{ARTIFACT_ID}-backup.json"
        ).read_text(encoding="utf-8")
    )
    assert backup["status"] == "verified"
    assert backup["artifact_manifest_sha256"] == evidence["artifact_manifest_sha256"]
    assert backup["file_count"] == 1

    first = wrapper._prepare_gate_evidence(FORMAL_PLAN, ARTIFACT_ID)
    assert first["selection_sha"] == evidence["selection_sha"]
    assert first["backup_sha"] == evidence["backup_sha"]

    content, _ = wrapper._build_sim_plan_content(evidence)
    plan = yaml.safe_load(content)
    assert plan["plan_id"] == "m2-smolvla450m-vcdropout-cuda-b64-001-sim-433"
    assert plan["artifact_id"] == ARTIFACT_ID
    assert plan["selection"]["checkpoint_step"] == 237
    assert plan["selection"]["model_safetensors_sha256"] == "a" * 64


def test_report_suffix_continues_campaign_series() -> None:
    assert wrapper.REPORT_SUFFIX == "433"
