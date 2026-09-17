"""Real non-Git code identity must stay stable across launcher stage messages."""

from rosetta_reality.experiment import workspace_code_identity


def test_durable_stage_logs_do_not_change_gate_identity(tmp_path):
    (tmp_path / "scripts").mkdir()
    source = tmp_path / "scripts/worker.py"
    source.write_text("registered source")
    (tmp_path / "runs").mkdir()
    log = tmp_path / "runs/attempt-launcher.log"
    log.write_text("render passed\n")
    gate3_identity = workspace_code_identity(tmp_path)
    with log.open("a") as stream:
        stream.write("gate3 passed\n")
    assert workspace_code_identity(tmp_path) == gate3_identity
    source.write_text("changed source")
    assert workspace_code_identity(tmp_path) != gate3_identity


def test_old_root_launcher_log_invalidates_matching_gate_report(tmp_path):
    log = tmp_path / "attempt-launcher.log"
    log.write_text("render passed\n")
    gate3_identity = workspace_code_identity(tmp_path)
    with log.open("a") as stream:
        stream.write("gate3 passed\n")
    assert workspace_code_identity(tmp_path) != gate3_identity
