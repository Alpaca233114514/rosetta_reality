"""Bounded synthetic and Basin contract tests; no real model or data access."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rosetta_reality.diagnostics.torchlens_capture import (
    assess_acceptance,
    compare_arms,
    envelope,
    resource_envelope,
    tree_record,
    write_new,
)

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/collect_torchlens_synthetic.py"


def worker(tmp_path, name, backend, case="normal"):
    target = tmp_path / name
    process = subprocess.run([sys.executable, str(ENTRY), "--worker", "--backend", backend,
                              "--case", case, "--output", str(target)],
                             capture_output=True, text=True, timeout=45)
    (tmp_path / f"{name}.log").write_text(process.stdout + process.stderr)
    return target, process


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    directory = tmp_path_factory.mktemp("torchlens-arms")
    values = []
    for name, backend in (("a", "plain"), ("aa", "plain"), ("b", "torchlens")):
        target, process = worker(directory, name, backend)
        assert process.returncode == 0, process.stderr
        values.append(json.loads((target / "native.json").read_text()))
    return values


def test_parity_and_repeated_modules(arms):
    comparison = compare_arms(*arms)
    assert all(comparison["plain_repeat"].values())
    # Audited 2.23.0 still advances Python RNG during trace bookkeeping. Keep
    # the negative result: passing these regression tests is NOT capture acceptance.
    assert not comparison["passed"]
    assert not comparison["torchlens_vs_plain"]["rng_after"]
    assert all(value for key, value in comparison["torchlens_vs_plain"].items()
               if key != "rng_after")
    before = arms[0]["parameters"]["rng_after"]
    after = arms[2]["parameters"]["rng_after"]
    assert before["python"] != after["python"]
    assert before["torch_cpu"] == after["torch_cpu"] and before["numpy"] == after["numpy"]
    events = arms[2]["events"]
    assert len({(e["step"], e["stage"]) for e in events}) == len(events)
    shared = [e for e in events if "shared" in e["values"].get("module", "")
              and e["values"].get("saved")]
    assert len(shared) >= 2
    assert any(e["values"].get("saved") is False for e in events)
    assert arms[2]["parameters"]["output"]["dict"]["aux"]["tuple"][1]["list"][1] is None


def test_parity_fails_on_rng_buffer_output_or_incomplete(arms):
    # The plain backend is the positive control for the acceptance machinery.
    assert compare_arms(arms[0], arms[1], arms[1])["passed"]
    for field in ("rng_after", "state_after", "output", "input_after"):
        altered = copy.deepcopy(arms[1])
        altered["parameters"][field] = None
        assert not compare_arms(arms[0], arms[1], altered)["passed"]
    altered = copy.deepcopy(arms[1])
    altered["status"] = "incomplete"
    with pytest.raises(ValueError, match="Incomplete"):
        compare_arms(arms[0], altered, arms[2])


@pytest.mark.parametrize("case", ["exception", "nonfinite"])
def test_failures_are_preserved(tmp_path, case):
    target, process = worker(tmp_path, case, "torchlens", case)
    assert process.returncode != 0
    failure = json.loads((target / "failure.json").read_text())
    assert failure["status"] == "incomplete"
    assert failure["parameters"]["error_type"] == (
        "RuntimeError" if case == "exception" else "ValueError")
    assert ("Synthetic forward failure" if case == "exception" else "Non-finite tensor") in (
        process.stderr)
    assert not (target / "native.json").exists()


def test_size_nonfinite_and_create_only(tmp_path):
    path = tmp_path / "native.json"
    with pytest.raises(ValueError, match="budget"):
        write_new(path, envelope("x"), limit=10)
    assert not path.exists()
    with pytest.raises(ValueError):
        write_new(path, {"x": float("nan")})
    write_new(path, envelope("x"))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, envelope("y"))
    assert path.read_bytes() == before


def test_tensor_limit_and_structure():
    import torch

    with pytest.raises(ValueError, match="budget"):
        tree_record(torch.zeros(4097))
    with pytest.raises(ValueError, match="Non-finite"):
        tree_record(torch.tensor(float("inf")))
    assert tree_record((1, [None])) != tree_record([1, [None]])


def test_resource_guard(tmp_path):
    (tmp_path / "memory.max").write_text(str(4 * 1024**3))
    (tmp_path / "cpu.max").write_text("200000 100000")
    assert resource_envelope(tmp_path)["cpus"] == 2
    (tmp_path / "memory.max").write_text("max")
    with pytest.raises(ValueError, match="Bounded"):
        resource_envelope(tmp_path)
    (tmp_path / "memory.max").write_text(str(8 * 1024**3))
    with pytest.raises(ValueError, match="budget"):
        resource_envelope(tmp_path)


def test_worker_timeout_retains_log(tmp_path):
    from scripts.collect_torchlens_synthetic import run_worker

    log = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        run_worker([sys.executable, "-u", "-c",
                    "import time; print('started', flush=True); time.sleep(10)"], log, 0.5)
    assert "started" in log.read_text()


def test_layered_acceptance_requires_independent_activation_evidence(tmp_path, arms):
    target, process = worker(tmp_path, "reference", "reference")
    assert process.returncode == 0, process.stderr
    reference = json.loads((target / "native.json").read_text())
    assessment = assess_acceptance(*arms, reference)
    assert assessment["reference_observer_transparent"]
    assert assessment["selected_activation_exact"]
    assert assessment["selected_activation_count"] == 6
    assert assessment["isolated_case_evidence_usable"]
    assert not assessment["strict_transparency"]["passed"]
    assert not assessment["live_integration_accepted"]
    reference["events"][0]["values"]["sha256"] = "0" * 64
    assert not assess_acceptance(*arms, reference)["isolated_case_evidence_usable"]


def test_basin_end_to_end(tmp_path, arms):
    # Only the tests know Basin's Python API. Production crosses JSON files.
    assert os.environ.get("BASIN_SOURCE"), "Basin read-only source mount is required"
    from basin.adapters import import_native
    from basin.analysis import compare
    from basin.api import BasinAPI
    from basin.store import Store

    source = tmp_path / "capture.json"
    diagnostic = copy.deepcopy(arms[2])
    diagnostic["status"] = "incomplete"
    diagnostic["parameters"]["consistency"] = compare_arms(*arms)
    write_new(source, diagnostic)
    store = Store(tmp_path / "new-history")
    assert import_native(store, source, "a")["status"] == "imported"
    assert import_native(store, source, "a")["status"] == "already_present"
    assert store.verify("a")["files"]
    record, events = store.get("a")
    assert record["evidence_kind"] == "synthetic" and record["gate"] == {}
    assert record["status"] == "incomplete"
    assert not record["parameters"]["consistency"]["passed"]
    assert (store.path("a") / "artifacts/native.json").read_bytes() == source.read_bytes()
    assert events[0]["evidence"]["pointer"] == "/events/0"
    api = BasinAPI(store.root)
    queried = api.call("basin_events", {"run_id": "a", "step": 0, "limit": 1})
    assert queried["ok"] and queried["data"]["items"]
    recovered = api.call("basin_read_artifact", {
        "run_id": "a", "name": "native.json", "pointer": "/events/0"})
    assert recovered["data"]["value"] == arms[2]["events"][0]
    cli = subprocess.run([sys.executable, "-m", "basin", "--store", str(store.root),
                          "verify", "a"], capture_output=True, text=True, timeout=10)
    assert cli.returncode == 0, cli.stderr
    initialize = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {},
        "clientInfo": {"name": "torchlens-contract", "version": "1"}}}
    ready = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "basin_verify", "arguments": {"run_id": "a"}}}
    mcp = subprocess.run([sys.executable, "-m", "basin", "--store", str(store.root), "mcp"],
                         input="".join(json.dumps(row) + "\n" for row in (initialize, ready, call)),
                         capture_output=True, text=True, timeout=10)
    assert mcp.returncode == 0, mcp.stderr
    replies = [json.loads(line) for line in mcp.stdout.splitlines()]
    response = next(row for row in replies if row.get("id") == 2)
    assert not response["result"].get("isError", False)
    changed = copy.deepcopy(diagnostic)
    changed["events"][-1]["values"]["output"] = {"different": True}
    second = tmp_path / "changed.json"
    write_new(second, changed)
    with pytest.raises(ValueError, match="different evidence"):
        import_native(store, second, "a")
    import_native(store, second, "b")
    comparison = compare(store, "a", "b", field="/output")
    assert comparison["changed_events"] == 1
    assert comparison["missing_count"] > 0
    assert comparison["causal_conclusion"] is None
    original = store.path("a") / "artifacts/native.json"
    original.write_bytes(original.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.verify("a")
