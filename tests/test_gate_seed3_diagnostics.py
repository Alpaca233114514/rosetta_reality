"""Seed 3 framework tests: tiny synthetic CPU tensors, no models/data/simulation."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.lerobot_types import TransitionKey
from test_rollout_trace import environment_factory

from rosetta_reality.eval.gate_diagnostic_capture import capture_prediction, collect_episode
from rosetta_reality.eval.gate_diagnostic_io import (
    load_json,
    read_tree,
    sha,
    write_json,
    write_tree,
)
from rosetta_reality.eval.gate_diagnostic_protocol import authorize, validate_plan
from rosetta_reality.eval.gate_diagnostic_replay import replay_bundle, select_steps
from rosetta_reality.eval.gate_diagnostic_report import analyze
from rosetta_reality.eval.gate_diagnostic_training import audit_training, check_fact
from rosetta_reality.eval.gate_diagnostic_verify import arrays, verify_bundle
from rosetta_reality.sim import load_action_contract
from rosetta_reality.vla.processor import BOUNDED_SINE_ACTION_ADAPTER, PiAlohaPostprocessorStep
from scripts import smolvla_sim_gate as engine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def contract():
    return load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")


class TinyModel:
    def __init__(self, contract):
        self.config = SimpleNamespace(
            chunk_size=contract.chunk_length, max_action_dim=contract.dimension + 3
        )
        self.dimension = contract.dimension
        self.noises = []
        self.fail = False

    def predict_action_chunk(self, batch, *, noise):
        self.noises.append(noise.clone())
        if self.fail:
            torch.rand(1)
            raise RuntimeError("synthetic forward failure")
        return (
            noise[..., : self.dimension] * 0.01
            + batch["observation.state"][:, None, :] * 0.1
            + batch["observation.images.camera1"].float().mean() * 0.1
        )


class TinyOnline:
    def __init__(self, contract, fault=None):
        self.policy = TinyModel(contract)
        self._noise_generator = torch.Generator()
        self.fault = fault
        self.decoder = PiAlohaPostprocessorStep(
            lower_bounds=contract.lower_bounds.tolist(),
            upper_bounds=contract.upper_bounds.tolist(),
            dimension_names=list(contract.dimension_names),
            upstream_revision="a" * 40,
            action_representation_adapter=BOUNDED_SINE_ACTION_ADAPTER,
        )
        self.postprocessor = SimpleNamespace(steps=[self.decoder])

    def configure_noise(self, mode, seed):
        assert mode == "seeded_standard_normal"
        self._noise_generator.manual_seed(seed)

    def predict(self, observation, instruction):
        image = observation["images"]["top"].float() / 255
        batch = {
            "observation.state": observation["robot_state"][None],
            "observation.images.camera1": image[None],
            "task": [instruction],
        }
        if self.fault == "camera":
            batch["observation.images.wrong"] = batch.pop("observation.images.camera1")
            batch["observation.images.camera1"] = torch.zeros_like(image[None])
        if self.fault == "scale":
            batch["observation.images.camera1"] /= 255
        noise = torch.randn(
            1,
            self.policy.config.chunk_size,
            self.policy.config.max_action_dim,
            generator=self._noise_generator,
        )
        action = self.policy.predict_action_chunk(batch, noise=noise)
        if self.fault == "decoder":
            action = action.flip(-1)
        projected = self.decoder({TransitionKey.ACTION: action})[TransitionKey.ACTION][0]
        return self.decoder.last_unclipped_action[0], projected


def collection(tmp_path, monkeypatch, contract, *, mode="horizon", snapshot=True, steps=5):
    environment, instances = environment_factory(mode=mode, snapshot=snapshot)
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    output = tmp_path / "collection"
    online = TinyOnline(contract)
    result = collect_episode(
        engine,
        online,
        contract,
        output,
        {"kind": "synthetic"},
        maximum_steps=steps,
        maximum_bytes=64 * 1024**2,
    )
    return output, online, result, instances


@pytest.mark.parametrize("mode", ["horizon", "success", "timeout"])
def test_collect_parity_and_complete_replay(tmp_path, monkeypatch, contract, mode):
    environment, instances = environment_factory(mode=mode)
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    baseline = TinyOnline(contract)
    expected = engine._rollout(
        baseline,
        contract,
        "Insert the peg into the socket.",
        seed=3,
        policy_noise_seed=3,
        maximum_steps=5,
        noise_mode="seeded_standard_normal",
        project_policy_output=True,
    )
    treatment = TinyOnline(contract)
    before = torch.get_rng_state().clone()
    output = tmp_path / "collect"
    result = collect_episode(
        engine, treatment, contract, output, {"kind": "synthetic"}, maximum_steps=5
    )
    assert torch.equal(before, torch.get_rng_state())
    assert engine.GymAlohaEnvironment is environment
    assert all(i.closed for i in instances)
    for key, value in expected.items():
        if "seconds" not in key:
            assert result["metrics"][key] == value
    assert all(
        torch.equal(a, b)
        for a, b in zip(baseline.policy.noises, treatment.policy.noises, strict=True)
    )
    assert all(
        torch.equal(a, b) for a, b in zip(instances[0].actions, instances[1].actions, strict=True)
    )
    assert verify_bundle(output)["status"] == "verified_complete"
    new = TinyOnline(contract)
    replayed = replay_bundle(new, output, tmp_path / "replay")
    assert replayed["status"] == "complete"
    assert verify_bundle(tmp_path / "replay", source=output)["status"] == "verified_complete"


def test_probe_single_axes_metrics_and_report(tmp_path, monkeypatch, contract):
    source, _, _, _ = collection(tmp_path, monkeypatch, contract)
    result = replay_bundle(TinyOnline(contract), source, tmp_path / "probe", probe=True)
    assert result["forwards"] <= 60
    assert result["status"] == "complete"
    assert {r["kind"] for r in result["records"]} == {
        "replay",
        "self_copy",
        "image",
        "state",
        "noise",
    }
    assert verify_bundle(tmp_path / "probe", source=source)["status"] == "verified_complete"
    report = analyze(source, tmp_path / "report", probe=tmp_path / "probe")
    assert report["expert_deviation_measured"] is False
    assert (tmp_path / "report/timeline.csv").exists()
    assert (tmp_path / "report/report.md").exists()
    assert verify_bundle(tmp_path / "report")["status"] == "verified_complete"


@pytest.mark.parametrize(
    "fault,boundary", [("camera", "batch"), ("scale", "batch"), ("decoder", "internal")]
)
def test_fault_localization_stops_probes(tmp_path, monkeypatch, contract, fault, boundary):
    source, _, _, _ = collection(tmp_path, monkeypatch, contract)
    result = replay_bundle(TinyOnline(contract, fault), source, tmp_path / "probe", probe=True)
    assert result["status"] == "control_failed"
    mismatches = [r for r in result["records"] if r["status"] == "mismatch"]
    assert mismatches[0]["first_difference"] == boundary
    assert verify_bundle(tmp_path / "probe", source=source)["status"] == "verified_control_failed"


def test_noise_mismatch_localized():
    from rosetta_reality.eval.gate_diagnostic_replay import first_difference

    reference = {
        k: torch.zeros(2)
        for k in (
            "observation",
            "instruction",
            "batch",
            "noise",
            "normalized",
            "internal",
            "decoded",
            "projected",
        )
    }
    actual = copy.deepcopy(reference)
    actual["noise"][0] = 1
    assert first_difference(reference, actual) == "noise"


def test_rng_and_binding_restored_after_forward_failure(contract):
    online = TinyOnline(contract)
    online.configure_noise("seeded_standard_normal", 3)
    online.policy.fail = True
    before, own = torch.get_rng_state(), online._noise_generator.get_state()
    observation = {
        "robot_state": torch.zeros(contract.dimension),
        "images": {"top": torch.zeros(3, 4, 5, dtype=torch.uint8)},
    }
    with pytest.raises(RuntimeError, match="synthetic"):
        capture_prediction(online, observation, "task")
    assert "predict_action_chunk" not in vars(online.policy)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(own, online._noise_generator.get_state())
    online.policy.fail = False
    capture_prediction(online, observation, "task")


@pytest.mark.parametrize("snapshot", [False, "empty"])
def test_missing_physics_not_zero(tmp_path, monkeypatch, contract, snapshot):
    source, _, _, _ = collection(tmp_path, monkeypatch, contract, snapshot=snapshot)
    result = analyze(source, tmp_path / "report")
    assert any(r["check"] == "physics" for r in result["findings"])
    import csv

    with (tmp_path / "report/timeline.csv").open() as stream:
        assert all(row["object_contacts"] == "" for row in csv.DictReader(stream))


def test_resource_failure_is_incomplete_and_preserved(tmp_path, monkeypatch, contract):
    environment, instances = environment_factory()
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="budget"):
        collect_episode(
            engine,
            TinyOnline(contract),
            contract,
            output,
            {"kind": "synthetic"},
            maximum_steps=5,
            maximum_bytes=8192,
        )
    result = load_json(output / "result.json")
    assert result["status"] == "incomplete" and result["task_success"] is None
    assert instances[0].actions == [] and instances[0].closed
    assert engine.GymAlohaEnvironment is environment


def test_existing_output_rejected_before_predict(tmp_path, monkeypatch, contract):
    online = TinyOnline(contract)
    with pytest.raises(FileExistsError):
        collect_episode(engine, online, contract, tmp_path, {"kind": "synthetic"})
    assert not online.policy.noises


def test_corruption_rejected(tmp_path, monkeypatch, contract):
    source, _, _, _ = collection(tmp_path, monkeypatch, contract)
    path = source / "predictions/0000/arrays.npz"
    with path.open("ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(source)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_tensor_serialization_exact_dtype(tmp_path, dtype):
    value = {"tensor": torch.tensor([1.125, -2.5], dtype=dtype), "tuple": ("a", True)}
    output = tmp_path / "tree"
    write_tree(output, value, budget_root=tmp_path, maximum_bytes=2 * 1024**2)
    loaded = read_tree(output)
    assert loaded["tensor"].dtype == dtype and torch.equal(value["tensor"], loaded["tensor"])
    np.testing.assert_array_equal(arrays(output)["tensor"], value["tensor"].float().numpy())


def test_seed3_event_selection():
    rows = []
    for i in range(500):
        rows.append(
            {
                "reward": int(i == 20),
                "physics_after": {
                    "available": True,
                    "value": {
                        "contacts": [["vx300s_left/finger", "socket-1"]] if 10 <= i < 15 else []
                    },
                },
            }
        )
    chosen = select_steps(rows)
    assert len(chosen) == 12 and chosen == sorted(set(chosen))
    assert set([0, 9, 10, 11, 14, 15, 16, 19, 20, 21]) <= set(chosen)


@pytest.mark.parametrize(
    "name,good,bad",
    [
        (
            "loss",
            {"weights": [1, 0, 0], "is_pad": [False] * 3, "action_dim": 14, "denominator": 14},
            {"denominator": 42},
        ),
        (
            "optimizer",
            {
                "optimizer_parameters": ["expert", "state"],
                "trainable_parameters": ["expert", "state"],
            },
            {"optimizer_parameters": ["expert"]},
        ),
        (
            "time_chunk_padding",
            {
                "episode_length": 3,
                "frame": 2,
                "chunk_length": 3,
                "target_indices": [2, 2, 2],
                "is_pad": [False, True, True],
                "episode": 7,
                "target_episodes": [7, 7, 7],
                "input_timestamp": 0.04,
                "expected_timestamp": 0.04,
            },
            {"is_pad": [False, False, False]},
        ),
        (
            "scheduler",
            {
                "optimizer_updates": 2,
                "scheduler_updates": 2,
                "actual_lr": [0.1, 0.2],
                "expected_lr": [0.1, 0.2],
            },
            {"scheduler_updates": 3},
        ),
    ],
)
def test_training_fault_counterexamples(name, good, bad):
    assert check_fact(name, good)
    assert not check_fact(name, {**good, **bad})


def test_unbound_historical_evidence_never_passes(tmp_path):
    report = tmp_path / "history.json"
    write_json(report, {"status": "passed", "facts": {"loss": {"actual": 1, "expected": 1}}})
    plan = {
        "run_id": "synthetic",
        "training_evidence": [
            {
                "path": "history.json",
                "sha256": sha(report),
                "fact_pointers": {"loss": "/facts/loss"},
            }
        ],
    }
    result = audit_training(plan, tmp_path, tmp_path / "audit")
    assert all(f["category"] == "insufficient_evidence" for f in result["findings"])


def test_draft_and_seed_drift_rejected_before_ml():
    plan = load_json(ROOT / "configs/vla/gate_seed3_diagnostic_001.json")
    assert validate_plan(plan, ROOT, check_files=False)["execution_ready"] is False
    with pytest.raises(ValueError, match="Draft"):
        authorize(plan, ROOT, "collect")
    plan["rollout"]["seed"] = 1002
    with pytest.raises(ValueError, match="Seed 3"):
        validate_plan(plan, ROOT, check_files=False)


def test_cli_schema_does_not_import_model():
    code = """
import sys
from scripts.diagnose_smolvla_gate import main
args = ['validate-plan', '--plan', 'configs/vla/gate_seed3_diagnostic_001.json', '--schema-only']
assert main(args) == 0
assert not {'torch', 'lerobot', 'mujoco'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


def reseal_for_counterexample(directory):
    """Recompute hashes to prove arithmetic/structure checks do more than hashing."""
    manifest = load_json(directory / "manifest.json")
    for name in manifest["files"]:
        path = directory / name
        manifest["files"][name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    (directory / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("mutation", ["delta", "drop_record", "dimensions", "skip"])
def test_independent_verifier_rejects_resealed_probe(tmp_path, monkeypatch, contract, mutation):
    source, _, _, _ = collection(tmp_path, monkeypatch, contract)
    output = tmp_path / "probe"
    replay_bundle(TinyOnline(contract), source, output, probe=True)
    if mutation == "dimensions":
        path = output / "identity.json"
        data = load_json(path)
        data["dimension_names"] = data["dimension_names"][::-1]
    else:
        path = output / "result.json"
        data = load_json(path)
        if mutation == "drop_record":
            data["records"].pop()
        else:
            record = next(r for r in data["records"] if r["status"] == "measured")
            if mutation == "delta":
                record["deltas"]["left_joints/first_action"]["mean_absolute_delta"] += 0.25
            else:
                record["status"] = "skipped_no_distinct_donor"
    path.write_text(json.dumps(data))
    reseal_for_counterexample(output)
    with pytest.raises(ValueError):
        verify_bundle(output, source=source)


@pytest.mark.parametrize("failure", ["environment", "disk", "parameter"])
def test_failure_retains_partial_without_task_verdict(tmp_path, monkeypatch, contract, failure):
    import rosetta_reality.eval.gate_diagnostic_capture as capture

    environment, instances = environment_factory(fail_at=1 if failure == "environment" else None)
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    online = TinyOnline(contract)
    own = online._noise_generator.get_state().clone()
    before = torch.get_rng_state().clone()
    if failure == "disk":

        def failed_write(*args, **kwargs):
            raise OSError("synthetic disk full")

        monkeypatch.setattr(capture, "write_tree", failed_write)
    output = tmp_path / "partial"
    with pytest.raises((RuntimeError, OSError, ValueError)):
        collect_episode(
            engine,
            online,
            contract,
            output,
            {"kind": "synthetic"},
            maximum_steps=4,
            check_unchanged=lambda: failure != "parameter",
        )
    result = load_json(output / "result.json")
    assert result["status"] == "incomplete" and result["task_success"] is None
    assert torch.equal(own, online._noise_generator.get_state())
    assert torch.equal(before, torch.get_rng_state())
    assert engine.GymAlohaEnvironment is environment and instances[0].closed
    assert "predict_action_chunk" not in vars(online.policy)
    assert verify_bundle(output)["status"] == "verified_incomplete"


def test_one_step_probe_skips_nonexistent_donors(tmp_path, monkeypatch, contract):
    source, _, _, _ = collection(tmp_path, monkeypatch, contract, steps=1)
    result = replay_bundle(TinyOnline(contract), source, tmp_path / "probe", probe=True)
    assert result["forwards"] == 2
    assert sum(r["status"].startswith("skipped") for r in result["records"]) == 3
    verify_bundle(tmp_path / "probe", source=source)


def test_bound_training_operands_detect_bug_and_source_drift(tmp_path):
    from rosetta_reality.eval.gate_diagnostic_protocol import WEIGHT_SHA

    code = tmp_path / "source.py"
    code.write_text("# synthetic archived trainer\n")
    report = tmp_path / "historical.json"
    write_json(
        report,
        {
            "model": WEIGHT_SHA,
            "plan": "b" * 64,
            "sources": {"source.py": sha(code)},
            "loss": {
                "weights": [1, 0, 0],
                "is_pad": [False] * 3,
                "action_dim": 14,
                "denominator": 42,
            },
            "optimizer": {"trainable_parameters": ["a", "b"], "optimizer_parameters": ["a"]},
        },
    )
    plan = {
        "run_id": "synthetic",
        "training_plan": {"sha256": "b" * 64},
        "training_evidence": [
            {
                "path": "historical.json",
                "sha256": sha(report),
                "binding": {
                    "checkpoint_pointer": "/model",
                    "training_plan_pointer": "/plan",
                    "sources_pointer": "/sources",
                },
                "check_sources": {"loss": ["source.py"], "optimizer": ["source.py"]},
                "fact_pointers": {"loss": "/loss", "optimizer": "/optimizer"},
            }
        ],
    }
    result = audit_training(plan, tmp_path, tmp_path / "audit")
    bugs = [f["check"] for f in result["findings"] if f["category"] == "reproduced_code_defect"]
    assert bugs == ["loss", "optimizer"]
    code.write_text("# changed execution source\n")
    changed = audit_training(plan, tmp_path, tmp_path / "drift")
    assert all(f["category"] == "insufficient_evidence" for f in changed["findings"])


def test_missing_gradient_or_measurement_is_not_zero_or_passed():
    with pytest.raises(KeyError):
        check_fact("gradients", {"preclip_norm": 21.9, "clip_limit": 10})
    with pytest.raises(ValueError, match="Missing"):
        check_fact("processor", {"actual": None, "expected": None})


@pytest.fixture
def registered_fixture(tmp_path, monkeypatch):
    from rosetta_reality.eval import gate_diagnostic_protocol as protocol

    sources = {}
    for name in protocol.REQUIRED_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source\n")
        sources[name] = sha(path)
    monkeypatch.setattr(
        protocol,
        "REFERENCE_IMPLEMENTATIONS",
        {name: sources[name] for name in protocol.REFERENCE_IMPLEMENTATIONS},
    )
    paths = {}
    for name in ("training", "contract", "config", "weights"):
        path = tmp_path / name
        if name == "weights":
            path = tmp_path / "checkpoint/model.safetensors"
            path.parent.mkdir()
        path.write_text("synthetic " + name)
        paths[name] = {"path": path.relative_to(tmp_path).as_posix(), "sha256": sha(path)}
    weight_sha = paths["weights"]["sha256"]
    monkeypatch.setattr(protocol, "WEIGHT_SHA", weight_sha)
    monkeypatch.setattr(protocol, "CONFIG_SHA", paths["config"]["sha256"])
    authority = tmp_path / "authority.json"
    files = {"model.safetensors": weight_sha}
    write_json(
        authority,
        {
            "endpoint": {"files": files},
            "source_training_plan_sha256": paths["training"]["sha256"],
            "gate_protocol_authority": {"action_contract_sha256": paths["contract"]["sha256"]},
        },
    )
    monkeypatch.setattr(protocol, "AUTHORITY", "authority.json")
    monkeypatch.setattr(protocol, "AUTHORITY_SHA", sha(authority))
    manifest = tmp_path / "artifact.json"
    write_json(
        manifest,
        {
            "status": "verified",
            "artifact_id": "synthetic",
            "hidden_test_loaded": False,
            "selected_checkpoint_model_sha256": weight_sha,
            "reload": {"exact_tensor_equality": True},
            "files": {
                "pretrained_model/model.safetensors": weight_sha,
                "config.json": paths["config"]["sha256"],
            },
        },
    )
    gate = tmp_path / "gate3.json"
    write_json(
        gate,
        {
            "status": "passed",
            "gate": "m2_gate_3_small_policy_rollout",
            "artifact_id": "synthetic",
            "artifact_manifest_sha256": sha(manifest),
            "artifact_reload_verified": True,
            "hidden_test_loaded": False,
            "acceptance_criteria": {"synthetic": True},
        },
    )
    plan = load_json(ROOT / "configs/vla/gate_seed3_diagnostic_001.json")
    plan.update(
        status="registered",
        sources=sources,
        checkpoint={"path": "checkpoint", "files": files},
        training_plan=paths["training"],
        action_contract=paths["contract"],
        artifact_config=paths["config"],
        artifact_manifest={"path": "artifact.json", "sha256": sha(manifest)},
        gate3_report={"path": "gate3.json", "sha256": sha(gate)},
    )
    return plan, tmp_path


def test_registered_plan_file_identity_without_weights_load(registered_fixture):
    plan, root = registered_fixture
    assert validate_plan(plan, root)["status"] == "identity_valid"
    with pytest.raises(ValueError, match="stage-authorized"):
        authorize(plan, root, "collect")
    (root / "checkpoint/model.safetensors").write_text("drift")
    with pytest.raises(ValueError, match="identity differs"):
        validate_plan(plan, root)


def test_required_sources_and_output_traversal(registered_fixture):
    plan, root = registered_fixture
    first = next(iter(plan["sources"]))
    digest = plan["sources"].pop(first)
    with pytest.raises(ValueError, match="inventory"):
        validate_plan(plan, root)
    plan["sources"][first] = digest
    plan["output"] = "runs/../../outside"
    with pytest.raises(ValueError, match="escapes"):
        validate_plan(plan, root, check_files=False)


def test_old_gate_cannot_be_resealed_under_new_identity(registered_fixture):
    plan, root = registered_fixture
    name = "scripts/smolvla_sim_gate.py"
    (root / name).write_text("# changed historical gate\n")
    plan["sources"][name] = sha(root / name)
    with pytest.raises(ValueError, match="Historical Gate"):
        validate_plan(plan, root)


def test_maximum_probe_forward_budget(tmp_path, monkeypatch, contract):
    source, _, _, _ = collection(tmp_path, monkeypatch, contract, steps=13)
    result = replay_bundle(TinyOnline(contract), source, tmp_path / "probe", probe=True)
    assert result["forwards"] == 60
    verify_bundle(tmp_path / "probe", source=source)


def test_public_action_metrics_export_is_preserved():
    from rosetta_reality.eval import action_metrics
    from rosetta_reality.eval.metrics import action_metrics as original

    assert action_metrics is original
