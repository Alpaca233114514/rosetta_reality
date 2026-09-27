"""Tiny CPU native updates only; never pretrained weights, real data or simulator."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from lerobot.scripts import lerobot_train

from rosetta_reality.diagnostics.dual_axis import EvidenceWriter, make_schedule
from rosetta_reality.diagnostics.dual_axis_torch import (
    admission_result,
    endpoint_gradient_probe,
    isolated_probe,
    module_capture,
    tensor_tree,
    tree_identity,
)
from rosetta_reality.vla.training.dual_axis import Collector, load_tensor_tree, observe_dual_axis


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 1)
        self.register_buffer("unchanged", torch.ones(1))
        self.calls = []

    def forward(self, batch):
        noise = torch.rand(())
        value = self.projection(batch["action"])
        return (value - noise).square().mean(), {"loss_parts": value.detach().square().mean()}


class Probes:
    def __init__(self, policy):
        self.state_objects = [(policy, "calls")]
        self.updates = []

    def __call__(self, policy, node):
        self.updates.append(node["update"])
        random.random()
        np.random.rand()
        torch.rand(3)
        policy.calls.append("probe")
        yield (
            {"split": "train", "episode": 2, "frame": 0, "sample_id": "train-2-0"},
            {"name": "seed_1"},
            {"prediction": policy.projection(torch.ones(1, 1)), "metrics": []},
            {k: k for k in ("input", "noise", "processor", "action_contract", "runtime")},
        )


def plan(total=2):
    return {
        "source_run": "synthetic",
        "evidence_kind": "synthetic",
        "total_updates": total,
        "maximum_bytes": 8 * 1024**2,
        "schedule": make_schedule(total),
        "modules": ["projection"],
        "panels": {"hidden": [1, 5]},
    }


def execute(tmp_path, observed):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    policy = Policy()
    opt = torch.optim.AdamW(policy.parameters(), lr=0.01)
    accelerator = Accelerator(cpu=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    policy, opt = accelerator.prepare(policy, opt)
    batches = [
        {
            "action": torch.ones(2, 1),
            "episode_index": torch.tensor([2, 2]),
            "frame_index": torch.tensor([0, 1]),
        }
        for _ in range(2)
    ]
    probes = Probes(policy)
    captured_gradients = []
    native_step = opt.optimizer.step

    def audited_step(*args, **kwargs):
        captured_gradients.append(tree_identity({n: p.grad for n, p in policy.named_parameters()}))
        return native_step(*args, **kwargs)

    opt.optimizer.step = audited_step
    losses = []
    collector = Collector(tmp_path / "capture", plan(), probes=probes) if observed else None

    def work():
        stream = lerobot_train.cycle(batches)
        try:
            for _ in range(2):
                result = lerobot_train.update_policy(
                    SimpleNamespace(update_metrics=lambda _: None),
                    policy,
                    next(stream),
                    opt,
                    10,
                    accelerator,
                    scheduler,
                )
                losses.append(result[0].loss)
        finally:
            stream.close()

    if observed:
        with observe_dual_axis(lerobot_train, collector):
            work()
    else:
        work()
    return {
        "weights": {n: p.detach().clone() for n, p in policy.named_parameters()},
        "rng": torch.get_rng_state(),
        "py": random.getstate(),
        "np": np.random.get_state(),
        "calls": policy.calls,
        "probe_updates": probes.updates,
        "gradients": captured_gradients,
        "losses": losses,
        "optimizer": tree_identity(opt.state_dict()),
        "scheduler": scheduler.state_dict(),
    }, collector


def test_native_update_next_rng_and_weights_exact(tmp_path):
    before = lerobot_train.update_policy, lerobot_train.cycle
    a, _ = execute(tmp_path / "plain", False)
    b, collector = execute(tmp_path / "observed", True)
    assert all(torch.equal(a["weights"][n], b["weights"][n]) for n in a["weights"])
    assert torch.equal(a["rng"], b["rng"]) and a["py"] == b["py"]
    assert np.array_equal(a["np"][1], b["np"][1])
    assert b["calls"] == [] and b["probe_updates"] == [0, 1, 2]
    for key in ("gradients", "losses", "optimizer", "scheduler"):
        assert a[key] == b[key]
    assert collector.update == 2 and collector.exposures == 4
    assert (lerobot_train.update_policy, lerobot_train.cycle) == before
    assert json.loads((collector.writer.root / "manifest.json").read_text())["status"] == "complete"


def test_bf16_roundtrip_and_tampered_payload(tmp_path):
    w = EvidenceWriter(tmp_path / "source", {"source_run": "s"}, maximum_bytes=65536)
    t = torch.tensor([1.0, 1.5], dtype=torch.bfloat16)
    tree = tensor_tree(t, w, chunk_bytes=2)
    assert torch.equal(load_tensor_tree(tree, w.root), t)
    (w.root / tree["chunks"][0]["path"]).write_bytes(b"aa")
    with pytest.raises(ValueError, match="identity"):
        load_tensor_tree(tree, w.root)


def test_hook_repeated_calls_and_no_graph_retained():
    p = Policy()
    with module_capture(p, ["projection"]) as rows:
        p.projection(torch.ones(1, 1))
        p.projection(torch.ones(1, 1))
    assert [r["call"] for r in rows] == [0, 1]
    assert not p.projection._forward_hooks
    json.dumps(rows, allow_nan=False)


def test_isolation_restores_on_exception_and_rejects_tensor_mutation():
    p = Policy()
    p.train()
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="fail"):
        with isolated_probe(p):
            torch.rand(1)
            raise RuntimeError("fail")
    assert torch.equal(rng, torch.get_rng_state()) and p.training
    with pytest.raises(RuntimeError, match="mutated"):
        with isolated_probe(p):
            p.unchanged.add_(1)


def test_endpoint_gradient_preserves_original_grads_and_rejects_dev(tmp_path):
    p = Policy()
    for parameter in p.parameters():
        parameter.grad = torch.ones_like(parameter)
    original = [p.grad for p in p.parameters()]
    w = EvidenceWriter(tmp_path / "out", {"source_run": "s"}, maximum_bytes=65536)
    sample = {"split": "train", "episode": 2, "frame": 0, "sample_id": "s"}

    def forward(policy, sample):
        return policy({"action": torch.ones(1, 1)})[0], {"input": torch.ones(1, 1)}

    endpoint_gradient_probe(p, [sample], forward, w, update=0)
    assert all(
        a is b.grad and torch.equal(a, torch.ones_like(a)) for a, b in zip(original, p.parameters())
    )
    with pytest.raises(ValueError, match="train-only"):
        endpoint_gradient_probe(p, [{**sample, "split": "dev"}], forward, w, update=0)


def test_admission_is_not_authorization():
    record = {
        key: "same"
        for key in (
            "outputs",
            "gradients",
            "parameters",
            "buffers",
            "rng",
            "next_batch",
            "optimizer",
            "scheduler",
        )
    }
    a = {**record, "routine_seconds": 100}
    assert admission_result(a, {**a, "routine_seconds": 110})["status"] == "passed"
    assert admission_result(a, {**a, "routine_seconds": 120})["status"] == "failed"
    assert not admission_result(a, a)["model_execution_authorized"]


def test_skipped_update_does_not_advance_successful_clock(tmp_path):
    policy = Policy()
    real = torch.optim.SGD(policy.parameters(), lr=0.1)
    optimizer = SimpleNamespace(
        param_groups=real.param_groups, state=real.state, step=lambda: None, step_was_skipped=True
    )
    accelerator = SimpleNamespace(clip_grad_norm_=lambda *a, **k: None)
    module = SimpleNamespace(cycle=lambda batches: iter(batches))

    def update(metrics, policy, batch, optimizer, clip, accelerator, scheduler=None):
        optimizer.step()
        return SimpleNamespace(loss=1.0), {}

    module.update_policy = update
    collector = Collector(tmp_path / "skipped", plan(1), probes=Probes(policy))
    with observe_dual_axis(module, collector):
        batch = {
            "action": torch.ones(1, 1),
            "episode_index": torch.tensor([2]),
            "frame_index": torch.tensor([0]),
        }
        module.update_policy(None, policy, batch, optimizer, 10, accelerator)
    assert collector.attempt == 1 and collector.update == 0 and collector.exposures == 0
    assert (
        json.loads((collector.writer.root / "manifest.json").read_text())["status"] == "incomplete"
    )


def test_failure_after_optimizer_preserves_successful_update_and_incomplete_status(tmp_path):
    p = Policy()
    optimizer = torch.optim.SGD(p.parameters(), lr=0.1)
    accelerator = SimpleNamespace(clip_grad_norm_=lambda *a, **k: None)

    def native(metrics, policy, batch, optimizer, clip, accelerator):
        optimizer.step()
        raise RuntimeError("after-step failure")

    module = SimpleNamespace(cycle=lambda x: iter(x), update_policy=native)
    collector = Collector(tmp_path / "failed", plan(1), probes=Probes(p))
    with (
        pytest.raises(RuntimeError, match="after-step failure"),
        observe_dual_axis(module, collector),
    ):
        module.update_policy(None, p, {"action": torch.ones(1, 1)}, optimizer, 10, accelerator)
    assert collector.update == 1 and collector.exposures == 0
    assert module.update_policy is native
    assert (
        json.loads((collector.writer.root / "manifest.json").read_text())["status"] == "incomplete"
    )


def test_bank_rejects_hidden_before_reading(tmp_path):
    from rosetta_reality.diagnostics.dual_axis_bank import seal_bank

    called = []
    with pytest.raises(ValueError, match="Forbidden"):
        seal_bank(
            tmp_path / "bank",
            {"extended": [{"episode": 1, "split": "train"}], "hidden": [1]},
            lambda r: called.append(r),
            lambda x: x,
            noise_shape=(1, 2, 1),
            identity={"processor": "p", "action_contract": "a", "runtime": "r"},
            maximum_bytes=8192,
        )
    assert not called


def test_registration_draft_rejects_before_model_access():
    from rosetta_reality.vla.training.dual_axis import validate_registration

    with pytest.raises(ValueError, match="not registered"):
        validate_registration({"status": "draft", "model_execution_authorized": False})


def test_sealed_bank_runs_prepared_probe_with_exact_noise_and_tail_mask(tmp_path):
    from rosetta_reality.diagnostics.dual_axis import make_panels
    from rosetta_reality.diagnostics.dual_axis_bank import seal_bank
    from rosetta_reality.vla.training.dual_axis import PreparedProbes

    noises = [{"name": "zero", "seed": None}] + [
        {"name": f"seed_{seed}", "seed": seed} for seed in (1, 2, 3)
    ]
    panels = make_panels([2, 3], [4, 5], [1], noises)

    def loader(row):
        target = torch.zeros(1, 2, 1)
        mask = torch.tensor([[True, row["frame"] != 499]])
        return (
            {
                "episode_index": torch.tensor([row["episode"]]),
                "frame_index": torch.tensor([row["frame"]]),
                "action": target.clone(),
            },
            target,
            mask,
        )

    class Processor:
        steps = []

        def __call__(self, value):
            return value

    class Model(Policy):
        def predict_action_chunk(self, batch, *, noise):
            return self.projection(noise)

    processor = Processor()
    identity = {k: k for k in ("processor", "action_contract", "runtime")}
    bank = seal_bank(
        tmp_path / "bank",
        panels,
        loader,
        processor,
        noise_shape=(1, 2, 1),
        identity=identity,
        maximum_bytes=1024**2,
    )
    p = {
        "panels": panels,
        "probe_bank": bank,
        "modules": ["projection"],
        "action_groups": {"joint": [0]},
    }
    probe = PreparedProbes(p, tmp_path / "bank", processor, processor)
    rows = list(probe(Model(), {"panel": "small", "all_noises": True}))
    assert len(rows) == 64
    assert all(len(row[2]["metrics"]) == 2 for row in rows)
    assert all(row[2]["mask"].sum().item() == 1 for row in rows if row[0]["frame"] == 499)
    assert all(m["aa_floor"] is None for row in rows for m in row[2]["metrics"])


def test_new_feature_requires_complete_source_closure(tmp_path):
    import hashlib

    from rosetta_reality.vla.training.integrity import (
        CORE_IMPLEMENTATION,
        validate_local_implementation,
    )

    members = {}
    for name in CORE_IMPLEMENTATION:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic\n")
        members[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="closure"):
        validate_local_implementation(
            {"implementation_files": members, "features": [{"name": "dual_axis_diagnostics"}]},
            tmp_path,
        )


def test_rollout_sink_records_both_axes_without_changing_actions(tmp_path, monkeypatch):
    from pathlib import Path

    from basin.dual_axis import import_dual_axis, timeline
    from basin.store import Store
    from test_closed_loop_reproducibility import FakePhysics
    from test_gate_seed3_diagnostics import TinyOnline
    from test_rollout_trace import environment_factory

    from rosetta_reality.eval.dual_axis import RolloutSink
    from rosetta_reality.eval.reproducibility_capture import collect_reproducibility
    from rosetta_reality.sim import load_action_contract
    from scripts import smolvla_sim_gate as engine

    contract = load_action_contract(
        Path(__file__).resolve().parents[1] / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    environment, instances = environment_factory(mode="horizon")
    monkeypatch.setattr(engine, "GymAlohaEnvironment", environment)
    writer = EvidenceWriter(
        tmp_path / "inputs/dual",
        {"source_run": "rollout", "evidence_kind": "synthetic"},
        maximum_bytes=16 * 1024**2,
    )
    sink = RolloutSink(
        writer,
        checkpoint="checkpoint-sha",
        update=128,
        episode=0,
        identity={k: k for k in ("processor", "action_contract", "runtime")},
        modules=["readout"],
    )

    class Model(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base, self.config = base, base.config
            self.readout = torch.nn.Identity()

        def predict_action_chunk(self, batch, *, noise):
            return self.readout(self.base.predict_action_chunk(batch, noise=noise))

    for name, active_sink in (("plain", None), ("observed", sink)):
        online = TinyOnline(contract)
        online.policy = Model(online.policy)
        collect_reproducibility(
            engine,
            online,
            contract,
            tmp_path / name,
            {"kind": "synthetic"},
            pairing={"pair_id": "p", "role": "baseline_a", "seed": 1000},
            maximum_steps=3,
            maximum_bytes=16 * 1024**2,
            physics_factory=FakePhysics,
            fingerprint=lambda: {"synthetic": True},
            diagnostic_sink=active_sink,
        )
    assert all(torch.equal(a, b) for a, b in zip(instances[0].actions, instances[1].actions))
    store = Store(tmp_path / "history")
    import_dual_axis(store, tmp_path / "inputs", "dual", "rollout")
    rows = timeline(store, "rollout", axis="execution", limit=100)["events"]["items"]
    assert rows[0]["coordinates"]["boundary"] == "reset"
    assert {r["coordinates"]["update"] for r in rows} == {128}
    assert {r["coordinates"]["rollout_step"] for r in rows} == {0, 1, 2}
    assert writer.closed
