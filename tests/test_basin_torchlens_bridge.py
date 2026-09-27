"""The real isolated bridge, including strict rejection and Basin re-import."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rosetta_reality.diagnostics.torchlens_capture import assess_acceptance, digest

ENTRY = Path(__file__).resolve().parents[1] / "scripts/collect_torchlens_synthetic.py"


def test_real_bridge_receipt_and_independent_activations(tmp_path):
    from basin.adapters import import_native
    from basin.api import BasinAPI
    from basin.store import Store

    output = tmp_path / "bridge"
    process = subprocess.run([sys.executable, str(ENTRY), "--backend", "basin-torchlens",
                              "--output", str(output), "--run-id", "basin-bridge-stage2"],
                             capture_output=True, text=True, timeout=100)
    (tmp_path / "supervisor.log").write_text(process.stdout + process.stderr)
    assert process.returncode != 0
    assert "Strict parity failed" in process.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    final = json.loads((output / "basin-native.json").read_text())
    assert receipt["status"] == final["status"] == "incomplete"
    assert not final["parameters"]["consistency"]["passed"]
    assert final["parameters"]["collector"] == "basin.torchlens.v1"
    raw = output / "observed/basin-raw/native.json"
    raw_digest = receipt["parameters"]["files"]["observed/basin-raw/native.json"]
    assert raw_digest == digest(raw.read_bytes())
    reference = tmp_path / "reference"
    ref = subprocess.run([sys.executable, str(ENTRY), "--worker", "--backend", "reference",
                          "--output", str(reference)], capture_output=True, text=True, timeout=45)
    assert ref.returncode == 0, ref.stderr
    arms = [json.loads((output / name / "native.json").read_text())
            for name in ("control", "repeat", "observed")]
    assessment = assess_acceptance(*arms, json.loads((reference / "native.json").read_text()))
    assert assessment["isolated_case_evidence_usable"]
    assert assessment["selected_activation_count"] == 6
    assert not assessment["strict_transparency"]["passed"]
    (tmp_path / "assessment.json").write_text(json.dumps(assessment, indent=2))
    store = Store(tmp_path / "new-history")
    import_native(store, output / "basin-native.json", "bridge")
    api = BasinAPI(store.root)
    assert api.call("basin_verify", {"run_id": "bridge"})["data"]["integrity"] == "verified"
    record, events = store.get("bridge")
    assert record["status"] == "incomplete" and record["gate"] == {}
    assert events


@pytest.mark.parametrize("case", ["exception", "nonfinite"])
def test_bridge_retains_stage_specific_failure_records(tmp_path, case):
    output = tmp_path / case
    process = subprocess.run([sys.executable, str(ENTRY), "--worker",
                              "--backend", "basin-torchlens",
                              "--case", case, "--output", str(output)],
                             capture_output=True, text=True, timeout=45)
    (tmp_path / (case + ".log")).write_text(process.stdout + process.stderr)
    assert process.returncode != 0
    expected_type = "RuntimeError" if case == "exception" else "ValueError"
    expected_message = "Synthetic forward failure" if case == "exception" else "Non-finite"
    assert expected_message in process.stderr
    failure_paths = [output / "failure.json"]
    if case == "exception":
        failure_paths.append(output / "basin-raw/failure.json")
    else:
        # The selected head activation precedes the deliberately invalid final
        # division. Its capture can finish, but full-output validation must fail.
        raw = json.loads((output / "basin-raw/native.json").read_text())
        assert raw["status"] == "complete"
        assert raw["parameters"]["consistency"] == "not_assessed"
        assert not raw["parameters"]["live_integration_accepted"]
    for path in failure_paths:
        failure = json.loads(path.read_text())
        assert failure["status"] == "incomplete"
        assert failure["parameters"]["error_type"] == expected_type
    assert not (output / "native.json").exists()


def test_bridge_import_keeps_optional_backend_lazy():
    command = (
        "import sys; import scripts.basin_torchlens_worker; "
        "assert 'basin' not in sys.modules; assert 'torch' not in sys.modules; "
        "assert 'torchlens' not in sys.modules"
    )
    process = subprocess.run([sys.executable, "-c", command],
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
