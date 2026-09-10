"""Exercise real process-group termination and evidence-preserving writes."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("coverage_job", ROOT / "scripts/run_visual_coverage_job.py")
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)


def test_termination_targets_only_owned_process_group():
    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        JOB.terminate_group(owned)
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        if owned.poll() is None:
            owned.kill()
        unrelated.terminate()
        owned.wait(timeout=5)
        unrelated.wait(timeout=5)


def test_existing_evidence_cannot_be_overwritten(tmp_path):
    report = tmp_path / "report.json"
    JOB.save(report, {"status": "failed"})
    before = report.read_bytes()
    with pytest.raises(FileExistsError):
        JOB.save(report, {"status": "passed"})
    assert report.read_bytes() == before


def test_shutdown_rejects_changed_platform_wrapper(monkeypatch):
    monkeypatch.setattr(JOB, "digest", lambda _path: "unrecognized")
    with pytest.raises(AssertionError, match="wrapper changed"):
        JOB.shutdown()
