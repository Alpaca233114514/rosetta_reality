"""Offline protocol inputs are synthetic; production evidence stays private."""

import hashlib
import json

import pytest


@pytest.fixture
def synthetic_normalization(request, tmp_path, monkeypatch):
    """Bind actual bytes and digests without importing historical run reports.

    Only test-local protocol constants and loaded test plans change. The real
    checksum validator is exercised, including the existing tampering tests.
    These inputs establish no historical normalization or run acceptance.
    """
    protocol = request.module.protocol
    report = tmp_path / "synthetic-normalization.json"
    view = tmp_path / "synthetic-view.json"
    report.write_text(json.dumps({"synthetic": True, "source_split": "train"}))
    view.write_text(json.dumps({"synthetic": True, "episodes": protocol.TRAIN_EPISODES}))
    values = {
        "NORMALIZATION_REPORT_RELATIVE": str(report),
        "NORMALIZATION_REPORT_SHA256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "VIEW_MANIFEST_RELATIVE": str(view),
        "VIEW_MANIFEST_SHA256": hashlib.sha256(view.read_bytes()).hexdigest(),
    }
    for name, value in values.items():
        monkeypatch.setattr(protocol, name, value)
    load_yaml = protocol.load_yaml

    def load_test_plan(path):
        plan = load_yaml(path)
        if isinstance(plan, dict) and "normalization" in plan:
            plan["normalization"].update(
                report=str(report),
                report_sha256=values["NORMALIZATION_REPORT_SHA256"],
                dataset_view_manifest=str(view),
                dataset_view_manifest_sha256=values["VIEW_MANIFEST_SHA256"],
            )
        return plan

    monkeypatch.setattr(protocol, "load_yaml", load_test_plan)
    return values
