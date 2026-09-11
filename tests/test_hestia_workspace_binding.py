from pathlib import Path

import pytest

from scripts.prepare_hestia_checkpoint_workspace import bind_existing, digest


def test_real_native_normalization_fails_missing_then_passes_bound(tmp_path, monkeypatch):
    import scripts.run_smolvla_v2 as native

    root = Path(__file__).resolve().parents[1]
    plan_path = root / "runs/hestia-recovered-20260911-001/cpu/draft/main1280.yaml"
    if not plan_path.exists():
        pytest.skip("Historical local input unavailable")
    plan, base, experiment = native._resolve_plan(plan_path)
    physical = digest(root / experiment["action_contract"]["derived"])
    monkeypatch.setattr(native, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(root / "runs"))
    with pytest.raises(ValueError, match="normalization report checksum"):
        native._validate_normalization(plan, experiment, base, physical)
    bind_existing(tmp_path, root / "runs", plan["normalization"])
    report, manifest, view = native._validate_normalization(plan, experiment, base, physical)
    assert digest(report) == plan["normalization"]["report_sha256"]
    assert manifest.parent == view
    with pytest.raises(FileExistsError):
        bind_existing(tmp_path, root / "runs", plan["normalization"])


def test_hash_drift_does_not_create_bindings(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    durable = tmp_path / "durable"
    durable.mkdir()
    (durable / "report.json").write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        bind_existing(workspace, durable, {"report": "runs/report.json", "report_sha256": "0" * 64})
    assert not (workspace / "runs").exists()
