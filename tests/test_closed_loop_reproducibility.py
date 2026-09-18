"""Tiny CPU/fake-environment counterexamples; no model weights, dataset or GPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_gate_seed3_diagnostics import TinyOnline
from test_rollout_trace import environment_factory

from rosetta_reality.eval.gate_diagnostic_capture import clone_tree
from rosetta_reality.eval.gate_diagnostic_io import load_json, seal
from rosetta_reality.eval.gate_diagnostic_protocol import validate_plan
from rosetta_reality.eval.reproducibility import compare_campaign, difference, verify_collection
from rosetta_reality.eval.reproducibility_capture import PhysicsReader, collect_reproducibility
from rosetta_reality.sim import load_action_contract
from scripts import smolvla_sim_gate as engine

ROOT = Path(__file__).resolve().parents[1]


class FakePhysics:
    def __init__(self, environment):
        self.environment = environment
        self.model_sha = "a" * 64

    def model_hash(self):
        return self.model_sha

    def __call__(self):
        return {
            "integration": self.environment.state.double().clone(),
            "spec": 8191,
            "model_sha256": self.model_sha,
            "scope": "synthetic",
        }


def collect(tmp_path, monkeypatch, role, *, fault=None, fail_at=None, mode="horizon"):
    environment, instances = environment_factory(mode=mode, fail_at=fail_at)
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    contract = load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    online = TinyOnline(contract, fault=fault)
    output = tmp_path / role
    collect_reproducibility(
        engine,
        online,
        contract,
        output,
        {"kind": "synthetic"},
        pairing={"pair_id": "synthetic-pair-001", "role": role, "seed": 1000},
        maximum_steps=4,
        maximum_bytes=64 * 1024**2,
        physics_factory=FakePhysics,
        fingerprint=lambda: {"synthetic": True},
    )
    assert engine.GymAlohaEnvironment is environment
    assert all(e.closed for e in instances)
    return output, instances[0], online


@pytest.mark.parametrize("mode", ["horizon", "success", "timeout"])
def test_three_arm_exact_parity(tmp_path, monkeypatch, mode):
    results = [
        collect(tmp_path, monkeypatch, r, mode=mode)
        for r in (
            "baseline_a",
            "baseline_b",
            "full_trace",
        )
    ]
    for path, _, _ in results:
        assert verify_collection(path)["status"] == "verified_complete"
    actual = compare_campaign(*(x[0] for x in results), tmp_path / "comparison")
    assert actual["status"] == "matched"
    assert actual["real_rollout_parity_verified"] is False
    assert actual["uninstrumented_parity_verified"] is False
    for _, env, online in results[1:]:
        assert all(
            torch.equal(a, b) for a, b in zip(env.actions, results[0][1].actions, strict=True)
        )
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                online.policy.noises,
                results[0][2].policy.noises,
                strict=True,
            )
        )


def test_first_divergence_is_processor_not_downstream_physics(tmp_path, monkeypatch):
    a, _, _ = collect(tmp_path, monkeypatch, "baseline_a")
    b, _, _ = collect(tmp_path, monkeypatch, "baseline_b", fault="scale")
    t, _, _ = collect(tmp_path, monkeypatch, "full_trace")
    result = compare_campaign(a, b, t, tmp_path / "comparison")
    assert result["status"] == "diverged"
    assert result["repeat"]["step"] == 1  # Initial fake image is zero.
    assert result["repeat"]["boundary"] == "batch"
    assert result["trace_effect"]["status"] == "matched"


def reseal(path):
    result = load_json(path / "result.json")
    (path / "result.json").unlink()
    (path / "manifest.json").unlink()
    seal(path, result)


@pytest.mark.parametrize("field", ["seed", "pair_id"])
def test_changed_pair_identity_is_not_model_divergence(tmp_path, monkeypatch, field):
    paths = [
        collect(tmp_path, monkeypatch, r)[0] for r in ("baseline_a", "baseline_b", "full_trace")
    ]
    identity = load_json(paths[1] / "identity.json")
    identity["pairing"][field] = 1001 if field == "seed" else "another-pair"
    (paths[1] / "identity.json").write_text(json.dumps(identity))
    reseal(paths[1])
    result = compare_campaign(*paths, tmp_path / "comparison")
    assert result["status"] == "incomparable"


def test_checksum_tampering_rejected(tmp_path, monkeypatch):
    path, _, _ = collect(tmp_path, monkeypatch, "baseline_a")
    (path / "runtime.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify_collection(path)


def test_resealed_missing_steps_rejected(tmp_path, monkeypatch):
    path, _, _ = collect(tmp_path, monkeypatch, "baseline_a")
    result = load_json(path / "result.json")
    result["steps_saved"] = result["predictions_saved"] = 3
    (path / "result.json").write_text(json.dumps(result))
    reseal(path)
    with pytest.raises(ValueError, match="extra rollout steps"):
        verify_collection(path)


def test_runtime_drift_is_incomparable(tmp_path, monkeypatch):
    paths = [
        collect(tmp_path, monkeypatch, r)[0]
        for r in ("baseline_a", "baseline_b", "full_trace")
    ]
    (paths[1] / "runtime.json").write_text('{"synthetic": true, "driver": "changed"}')
    reseal(paths[1])
    result = compare_campaign(*paths, tmp_path / "comparison")
    assert result["status"] == "incomparable"
    assert result["real_rollout_parity_verified"] is False


def test_old_capture_without_integration_is_not_accepted(tmp_path, monkeypatch):
    from test_gate_seed3_diagnostics import collection

    contract = load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    path, _, _, _ = collection(tmp_path, monkeypatch, contract)
    with pytest.raises(ValueError, match="scope"):
        verify_collection(path)


def test_failure_preserves_evidence_and_restores_binding(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="environment failure"):
        collect(tmp_path, monkeypatch, "baseline_a", fail_at=1)
    path = tmp_path / "baseline_a"
    result = load_json(path / "result.json")
    assert result["status"] == "incomplete" and result["task_success"] is None
    assert result["steps_saved"] == 1 and result["predictions_saved"] == 2
    assert (path / "steps/0001/before/arrays.npz").is_file()
    assert verify_collection(path)["status"] == "verified_incomplete"
    # A later separately named synthetic collection proves locks were released.
    collect(tmp_path, monkeypatch, "baseline_b")


def test_bitwise_equality_retains_dtype_and_signed_zero():
    assert difference(np.array([0.0]), np.array([-0.0]))
    assert difference(np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float64))


def test_raw_gym_numpy_observation_is_copied_without_aliasing():
    raw = np.ones((3, 4, 5), dtype=np.uint8)
    copied = clone_tree({"raw": {"pixels": raw, "flag": np.bool_(True)}})
    assert isinstance(copied["raw"]["pixels"], torch.Tensor)
    assert copied["raw"]["flag"] is True
    raw[:] = 0
    assert bool(copied["raw"]["pixels"].all())


def test_repro_schema_does_not_replace_seed3_contract():
    plan = load_json(ROOT / "configs/vla/gate_seed3_diagnostic_001.json")
    assert validate_plan(plan, ROOT, check_files=False)["execution_ready"] is False
    plan["rollout"].update(seed=1000, policy_noise_seed=1000)
    with pytest.raises(ValueError):
        validate_plan(plan, ROOT, check_files=False)
    plan["reproducibility"] = {"pair_id": "paired-001", "role": "baseline_a", "seed": 1000}
    assert validate_plan(plan, ROOT, check_files=False)["execution_ready"] is False
    plan["reproducibility"]["role"] = "untraced"
    with pytest.raises(ValueError):
        validate_plan(plan, ROOT, check_files=False)


def test_file_only_module_does_not_load_ml():
    code = (
        "import sys; import rosetta_reality.eval.reproducibility; "
        "assert not any(x in sys.modules for x in ('torch', 'lerobot', 'mujoco'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_native_state_reader_includes_warmstart_without_advancing():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="slide"/><geom size=".1" mass="1"/></body>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qacc_warmstart[:] = 0.125
    raw = SimpleNamespace(
        _env=SimpleNamespace(
            physics=SimpleNamespace(
                model=SimpleNamespace(ptr=model),
                data=SimpleNamespace(ptr=data),
            )
        )
    )
    reader = PhysicsReader(SimpleNamespace(raw_environment=raw))
    before = reader()
    assert before["spec"] & int(mujoco.mjtState.mjSTATE_WARMSTART)
    assert data.time == 0 and data.qacc_warmstart[0] == 0.125
    data.qacc_warmstart[:] = 0.25
    assert not torch.equal(before["integration"], reader()["integration"])
    assert reader.model_hash() == reader.model_sha
    model.geom_friction[0, 0] += 0.1
    assert reader.model_hash() != reader.model_sha
