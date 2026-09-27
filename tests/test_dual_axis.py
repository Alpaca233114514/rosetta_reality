"""Counterexamples for dual-axis evidence, independent Basin recomputation and API."""

import pytest

from rosetta_reality.diagnostics.dual_axis import (
    DeviationMonitor,
    EvidenceWriter,
    Sampling,
    Window,
    make_panels,
    make_schedule,
)

IDENTITY = {k: k + "-sealed" for k in ("input", "noise", "processor", "action_contract", "runtime")}


def bundle(tmp_path, values, *, name="source", changed=None, absolute=None):
    root = tmp_path / name
    plan = {
        "source_run": "synthetic-training",
        "evidence_kind": "synthetic",
        "schedule": [{"update": s} for s, _ in values],
    }
    w = EvidenceWriter(root, plan, maximum_bytes=1024**2, chunk_bytes=1600)
    for index, (update, value) in enumerate(values):
        m = {
            "name": "left/first/mae",
            "category": "behavior",
            "group": "left",
            "window": "first",
            "value": value,
            "aa_floor": 0.01,
        }
        if absolute is not None:
            m.update(absolute_limit=absolute, threshold_identity="sealed-train-only-rule")
        identity = dict(IDENTITY)
        if changed == index:
            identity["noise"] = "different-noise"
        w.event(
            {
                "axis": "probe",
                "update": update,
                "sample_id": "train-2-0",
                "episode": 2,
                "frame": 0,
                "noise_id": "n1",
                "boundary": "model_output",
            },
            {"metrics": [m]},
            identity=identity,
            reasons=["regular"],
        )
    w.seal()
    return root


def imported(tmp_path, values, **kwargs):
    from basin.dual_axis import import_dual_axis
    from basin.store import Store

    root = bundle(tmp_path / "inputs", values, **kwargs)
    store = Store(tmp_path / "history")
    import_dual_axis(store, root.parent, root.name, "test")
    return store


def test_schedule_deduplicates_edges_and_final():
    nodes = make_schedule(260, warmup=16, checkpoints=[128, 260], epoch_boundaries=[128])
    assert len(nodes) == len({n["update"] for n in nodes})
    assert {
        0,
        1,
        2,
        4,
        8,
        14,
        15,
        16,
        17,
        18,
        32,
        64,
        126,
        127,
        128,
        129,
        130,
        256,
        258,
        259,
        260,
    } == {n["update"] for n in nodes}
    sampling = Sampling(nodes, 260)
    sampling.trigger(70)
    assert sampling.due(70)["reasons"] == ["anomaly"]
    assert sampling.due(70) is None
    assert sampling.due(78)
    assert sampling.due(79) is None


def test_panels_split_and_first_gaussian():
    noises = [{"name": n} for n in ("zero", "seed_1", "seed_2", "seed_3")]
    p = make_panels([4, 2, 3], [9, 8, 7], [1], noises)
    assert len(p["small"]) == 16 and len(p["extended"]) == 24
    assert p["regular_noise"]["name"] == "seed_1"
    with pytest.raises(ValueError, match="overlap"):
        make_panels([2, 3], [3, 4], [1], noises)


def test_ring_eviction_has_explicit_coverage():
    ring = Window(steps=3, maximum_bytes=100)
    for i in range(8):
        ring.add(i, {"update": i})
    assert ring.snapshot()["updates"] == [5, 6, 7]
    assert ring.snapshot()["evictions"] == 5
    assert not ring.add(8, {"large": "x" * 101})


def test_train_trigger_locks_reference_and_does_not_use_dev():
    m = DeviationMonitor()
    assert m.observe("a", 0, 1, 0.01, regular=True, split="train") is None
    assert m.observe("a", 1, 2, 0.01, regular=True, split="dev") is None
    assert m.observe("a", 2, 1.5, 0.01, regular=True, split="train")["state"] == "candidate"
    assert m.observe("a", 3, 1.6, 0.01, regular=False, split="train")["state"] == "persistent"
    assert m.observe("a", 4, 1.05, 0.01, regular=False, split="train")["state"] == "recovered"


@pytest.mark.parametrize(
    "values,state",
    [
        ([(0, 1), (1, 2), (2, 2.1)], "persistent"),
        ([(0, 1), (128, 2), (129, 1)], "transient"),
        ([(0, 1), (128, 2), (129, 2), (130, 1)], "recovered_after_persistent"),
    ],
)
def test_basin_localizes_interval_and_recovery(tmp_path, values, state):
    from basin.dual_axis import locate_deviation

    store = imported(tmp_path, values)
    result = locate_deviation(store, "test")
    assert result["deviations"][0]["state"] == state
    assert result["deviations"][0]["possible_start_interval"] == [0, values[1][0]]
    assert result["causal_conclusion"] is None


def test_noise_drift_refuses_attribution_and_absolute_failure_can_start_at_zero(tmp_path):
    from basin.dual_axis import locate_deviation

    store = imported(tmp_path, [(0, 1), (128, 2)], changed=1, absolute=0.5)
    result = locate_deviation(store, "test")
    assert result["first_observed"] is None
    assert result["gaps"][0]["identity_fields"] == ["noise"]
    assert result["absolute_failures"][0]["coordinate"] == 0


def test_create_only_unpublished_and_tampering(tmp_path):
    from basin.dual_axis import import_dual_axis
    from basin.store import Store

    w = EvidenceWriter(tmp_path / "unpublished", {"source_run": "s"}, maximum_bytes=4096)
    assert not (w.root / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        EvidenceWriter(w.root, {}, maximum_bytes=4096)
    root = bundle(tmp_path / "inputs", [(0, 1), (1, 2)])
    (root / "events/000000.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="mismatch"):
        import_dual_axis(Store(tmp_path / "history"), root.parent, "source", "broken")


def test_cli_mcp_shared_api_pagination(tmp_path):
    from basin.api import BasinAPI

    store = imported(tmp_path, [(0, 1), (1, 2), (2, 3)])
    api = BasinAPI(store.root)
    result = api.call("basin_timeline", {"run_id": "test", "limit": 1})["data"]
    assert result["events"]["next_offset"] == 1
    assert api.call("basin_locate_deviation", {"run_id": "test"})["data"]["deviations_count"] == 1
    assert all(t["annotations"]["readOnlyHint"] for t in api.tools())
    assert api.call("basin_branch_compare", {"left": "test", "right": "test"})["data"]["pairs"][
        "items"
    ][0]["comparable"]


def test_interrupted_bundle_import_retains_status(tmp_path):
    from basin.dual_axis import import_dual_axis
    from basin.store import Store

    root = tmp_path / "inputs/source"
    w = EvidenceWriter(root, {"source_run": "s"}, maximum_bytes=8192)
    w.event(
        {"axis": "training", "attempt": 1, "update": 0}, {"error": "synthetic"}, status="failed"
    )
    w.seal("incomplete", error="synthetic")
    store = Store(tmp_path / "history")
    import_dual_axis(store, root.parent, "source", "failed")
    assert store.get("failed")[0]["status"] == "incomplete"
