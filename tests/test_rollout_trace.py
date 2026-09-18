"""Synthetic parity/negative evidence; never load model weights or real datasets."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lerobot.lerobot_types import TransitionKey

from rosetta_reality.eval.rollout_trace import file_sha, run_traced_rollout
from rosetta_reality.eval.rollout_trace_verify import verify_trace
from rosetta_reality.sim import load_action_contract
from rosetta_reality.vla.processor import BOUNDED_SINE_ACTION_ADAPTER, PiAlohaPostprocessorStep
from scripts import diagnose_canonical_rollout as cli
from scripts import smolvla_sim_gate as engine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def contract():
    return load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")


def options():
    return dict(seed=1000, maximum_steps=4, project_policy_output=True,
                noise_mode="seeded_standard_normal", policy_noise_seed=1000)


class FakePolicy:
    def __init__(self, contract, fail_at=None, dtype=torch.float32):
        self.generator = torch.Generator()
        self.noise = []
        self.fail_at, self.dtype = fail_at, dtype
        self.decoder = PiAlohaPostprocessorStep(
            lower_bounds=contract.lower_bounds.tolist(),
            upper_bounds=contract.upper_bounds.tolist(),
            dimension_names=list(contract.dimension_names), upstream_revision="a" * 40,
            action_representation_adapter=BOUNDED_SINE_ACTION_ADAPTER,
        )
        self.postprocessor = SimpleNamespace(steps=[self.decoder])

    def configure_noise(self, mode, seed):
        assert mode == "seeded_standard_normal"
        self.generator.manual_seed(seed)

    def predict(self, observation, instruction):
        if len(self.noise) == self.fail_at:
            raise RuntimeError("synthetic inference failure")
        noise = torch.randn(1, 50, 14, generator=self.generator, dtype=self.dtype)
        self.noise.append(noise.clone())
        action = noise * 0.01
        # Both boundary points must remain inside, including float32 rounding.
        action[:, 0, 6] = math.pi / 2
        action[:, 0, 13] = -math.pi / 2
        action[:, 7, 6] = 2.0
        action[:, 17, 13] = -2.0
        if len(self.noise) == 2:
            action[:, 0, 6], action[:, 0, 13] = -2.0, 2.0
        processed = self.decoder({TransitionKey.ACTION: action})[TransitionKey.ACTION]
        return self.decoder.last_unclipped_action[0], processed[0]


def environment_factory(mode="horizon", fail_at=None, snapshot=True):
    instances = []

    class Environment:
        def __init__(self, contract, maximum_episode_steps):
            self.state = torch.zeros(14)
            self.actions = []
            self.limit = maximum_episode_steps
            self.closed = False
            self.last_clip_mask = torch.zeros(14, dtype=torch.bool)
            instances.append(self)

        def observation(self):
            return {"robot_state": self.state.clone(),
                    "images": {"top": torch.full((3, 4, 5), len(self.actions), dtype=torch.uint8)}}

        def reset(self, seed):
            self.seed = seed
            return self.observation()

        def step(self, action):
            if len(self.actions) == fail_at:
                raise RuntimeError("synthetic environment failure")
            self.actions.append(action.clone())
            self.state = 0.75 * self.state + 0.25 * action
            success = mode == "success" and len(self.actions) == 2
            truncated = mode == "timeout" and len(self.actions) == self.limit
            return self.observation(), 4.0 if success else 0.0, success or truncated, {
                "is_success": success, "terminated": success, "truncated": truncated,
            }

        def contact_pairs(self):
            return (("table", "finger"),) if len(self.actions) == 3 else ()

        def is_unexpected_collision_pair(self, first, second):
            return True

        def state_limit_violation_count(self):
            return int(len(self.actions) == 3)

        def diagnostic_snapshot(self):
            if not snapshot:
                return None
            if snapshot == "empty":
                return {"bodies": {}, "contacts": [], "joint_limit_violations": []}
            return {"bodies": {"peg": {"position": [len(self.actions), 0., 0.]}},
                    "contacts": [list(pair) for pair in self.contact_pairs()],
                    "joint_limit_violations": [{"name": "synthetic"}]
                    if self.state_limit_violation_count() else []}

        def close(self):
            self.closed = True

    return Environment, instances


def trace(tmp_path, policy, contract, **kwargs):
    return run_traced_rollout(
        engine, policy, contract, "test", output=tmp_path,
        identity={"kind": "synthetic", "weights_loaded": False}, episode_index=0,
        **{**options(), **kwargs},
    )


def read(path, name):
    return json.loads((path / name).read_text())


@pytest.mark.parametrize("mode", ["horizon", "success", "timeout"])
@pytest.mark.parametrize("projection", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_observer_is_behavior_neutral(tmp_path, monkeypatch, contract, mode, projection, dtype):
    environment, instances = environment_factory(mode)
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    control, treatment = FakePolicy(contract, dtype=dtype), FakePolicy(contract, dtype=dtype)
    opts = {**options(), "project_policy_output": projection}
    before = torch.get_rng_state().clone()
    expected = engine._rollout(control, contract, "test", **opts)
    output = tmp_path / "trace"
    actual = trace(output, treatment, contract, project_policy_output=projection)
    assert torch.equal(before, torch.get_rng_state())
    assert engine.GymAlohaEnvironment is environment
    assert all(instance.closed for instance in instances)
    for key in expected:
        if "seconds" not in key:
            assert actual[key] == expected[key], key
    assert all(torch.equal(a, b) for a, b in zip(control.noise, treatment.noise, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(
        instances[0].actions, instances[1].actions, strict=True))
    result = verify_trace(output)
    assert result["status"] == "verified_complete"
    for side in ("left", "right"):
        assert result["support"]["first_action"][side]["outside_count"] == 1
        assert result["support"]["full_chunk"][side]["outside_count"] == len(control.noise) + 1
        assert result["executed_support"][side]["count"] == len(control.noise)
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == 3 * len(instances[1].actions)
    assert rows[2]["state_before"] == [0.] * 14
    assert rows[2]["state_after"] == rows[5]["state_before"]
    assert rows[2]["observed_grippers_after"] != [rows[2]["executed_action"][i] for i in (6, 13)]
    assert "socket" in rows[2]["physics_before"]["missing_bodies"]
    assert "images" not in rows[0] and "sha256" in rows[0]["input_identity"]["top"]


@pytest.mark.parametrize("failure", ["policy", "environment"])
def test_incomplete_execution_restores_binding(tmp_path, monkeypatch, contract, failure):
    environment, instances = environment_factory(fail_at=2 if failure == "environment" else None)
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    output = tmp_path / "trace"
    with pytest.raises(RuntimeError, match="synthetic"):
        trace(output, FakePolicy(contract, fail_at=2 if failure == "policy" else None), contract)
    assert engine.GymAlohaEnvironment is environment
    assert instances[0].closed
    summary = read(output, "summary.json")
    assert summary["status"] == "incomplete" and summary["steps_recorded"] == 2
    assert summary["steps_started"] == (3 if failure == "environment" else 2)
    assert verify_trace(output)["status"] == "verified_incomplete"
    fresh, _ = environment_factory()
    monkeypatch.setattr(engine, "GymAlohaEnvironment", fresh)
    trace(tmp_path / "next", FakePolicy(contract), contract)  # Lock also released.


@pytest.mark.parametrize("fail_write", [1, 2])
def test_logging_failure_does_not_execute_another_action(
    tmp_path, monkeypatch, contract, fail_write,
):
    environment, instances = environment_factory()
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    original = Path.open

    class FailingStream:
        def __init__(self, wrapped):
            self.wrapped, self.calls = wrapped, 0

        def write(self, value):
            if self.calls == fail_write:
                raise OSError("synthetic disk failure")
            self.calls += 1
            return self.wrapped.write(value)

        def flush(self):
            self.wrapped.flush()

        def close(self):
            self.wrapped.close()

    def open_file(path, *args, **kwargs):
        opened = original(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        return FailingStream(opened) if path.name == "trace.jsonl" and mode == "x" else opened

    monkeypatch.setattr(Path, "open", open_file)
    output = tmp_path / "trace"
    with pytest.raises(OSError):
        trace(output, FakePolicy(contract), contract)
    summary = read(output, "summary.json")
    assert summary["status"] == "incomplete"
    assert summary["steps_completed"] == fail_write - 1
    assert summary["steps_recorded"] == 0
    assert len(instances[0].actions) == fail_write - 1
    assert engine.GymAlohaEnvironment is environment
    assert verify_trace(output)["status"] == "verified_incomplete"


@pytest.mark.parametrize("snapshot", [False, "empty"])
def test_missing_physics_is_explicit_and_existing_output_untouched(
    tmp_path, monkeypatch, contract, snapshot,
):
    environment, instances = environment_factory(snapshot=snapshot)
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    output = tmp_path / "trace"
    trace(output, FakePolicy(contract), contract)
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    assert rows[2]["physics_after"]["available"] is False
    assert rows[2]["physics_after"]["value"] is None
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        trace(output, FakePolicy(contract), contract)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    assert len(instances) == 1


@pytest.mark.parametrize("tamper", ["hash", "support", "sequence", "state", "boundary"])
def test_independent_verifier_rejects_tampering(tmp_path, monkeypatch, contract, tamper):
    environment, _ = environment_factory()
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    output = tmp_path / "trace"
    trace(output, FakePolicy(contract), contract)
    if tamper == "support":
        summary = read(output, "summary.json")
        summary["support"]["first_action"]["left"]["outside_count"] = 0
        (output / "summary.json").write_text(json.dumps(summary))
    else:
        rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
        if tamper in ("hash", "sequence"):
            rows[0]["prediction"] = 7
        elif tamper == "state":
            rows[4]["state_before"][0] += 1
        else:
            rows[0]["support_limit"] = 2.1
        (output / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    if tamper != "hash":
        manifest = read(output, "manifest.json")
        manifest["files"] = {name: file_sha(output / name) for name in manifest["files"]}
        (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_trace(output)


def test_collect_rejects_draft_before_model_loading(tmp_path, monkeypatch):
    plan = tmp_path / "draft.json"
    plan.write_text(json.dumps({"execution_authorized": False}))
    with pytest.raises(ValueError, match="authorized"):
        cli.collect(plan)


def test_verify_cli_is_offline(tmp_path, monkeypatch, capsys, contract):
    environment, _ = environment_factory()
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    output = tmp_path / "trace"
    trace(output, FakePolicy(contract), contract)
    monkeypatch.setattr(cli.sys, "argv", ["trace", "verify", "--directory", str(output)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "verified_complete"


@pytest.mark.parametrize(
    "field", ["processor", "training_plan", "action_contract", "artifact_config"],
)
def test_registered_endpoint_rejects_other_processors_or_contract(field):
    authority = json.loads((ROOT / cli.AUTHORITY_PATH).read_text())
    assert file_sha(ROOT / cli.AUTHORITY_PATH) == cli.AUTHORITY_SHA
    plan = {
        "checkpoint": {"files": dict(authority["endpoint"]["files"])},
        "training_plan": {"sha256": authority["source_training_plan_sha256"]},
        "action_contract": {
            "sha256": authority["gate_protocol_authority"]["action_contract_sha256"],
        },
        "artifact_config": {"sha256": cli.ARTIFACT_CONFIG_SHA},
    }
    cli.validate_endpoint(plan, authority)
    if field == "processor":
        plan["checkpoint"]["files"]["policy_preprocessor.json"] = "0" * 64
    else:
        plan[field]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixed endpoint"):
        cli.validate_endpoint(plan, authority)


def test_environment_constructor_failure_retains_incomplete_evidence(
    tmp_path, monkeypatch, contract,
):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic constructor failure")

    monkeypatch.setattr(engine, "GymAlohaEnvironment", fail)
    with pytest.raises(RuntimeError, match="constructor"):
        trace(tmp_path / "trace", FakePolicy(contract), contract)
    assert engine.GymAlohaEnvironment is fail
    result = verify_trace(tmp_path / "trace")
    assert result["status"] == "verified_incomplete" and result["steps"] == 0


def test_invalid_output_argument_does_not_acquire_engine_lock(tmp_path, monkeypatch, contract):
    environment, _ = environment_factory()
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    with pytest.raises(TypeError):
        trace(42, FakePolicy(contract), contract)
    trace(tmp_path / "trace", FakePolicy(contract), contract)
