"""Synthetic file-boundary regressions; no model, simulator or data imports."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_audit_spec = importlib.util.spec_from_file_location("audit_paired_collection_v2", SCRIPTS / "audit_paired_collection_v2.py")
auditor = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(auditor)
_previous = sys.modules.get("audit_paired_collection_v2")
sys.modules["audit_paired_collection_v2"] = auditor
try:
    _export_spec = importlib.util.spec_from_file_location("_paired_export_test", SCRIPTS / "export_paired_cohort_v2.py")
    exporter = importlib.util.module_from_spec(_export_spec)
    _export_spec.loader.exec_module(exporter)
finally:
    if _previous is None:
        sys.modules.pop("audit_paired_collection_v2", None)
    else:
        sys.modules["audit_paired_collection_v2"] = _previous


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def export_fixture(tmp_path, *, omitted_image=False):
    root = tmp_path / "collection"
    root.mkdir()
    image = root / "scene/frame.npz"
    image.parent.mkdir()
    image.write_bytes(b"synthetic-image-identity-not-real-camera-data")
    image_sha = auditor.sha(image)
    attempts = []
    for role in ("S", "R"):
        dump(root / f"scene/{role}/labels.json", {
            "rows": [{"images": "scene/frame.npz", "images_sha256": image_sha}],
            "valid_chunk_starts": [0]})
        attempts.append({"attempt_id": "scene-" + role, "effective_input_frames": 1,
                         "valid_target_slots": 50, "task_success": True,
                         "verified_recovery_chain": role == "R", "deviation": None if role == "S" else {},
                         "scene_fingerprint": "synthetic-scene"})
    dump(root / "summary.json", {"phase": "Q2", "N": 1, "protected_scene_inventory_complete": True,
                                   "pairs": [{"pair_id": "scene", "matched": True, "S": "scene-S", "R": "scene-R"}],
                                   "attempts": attempts, "provider_status": "synthetic_fixture",
                                   "registered_attempts": 2, "candidate_matched_N": 1})
    files = [{"path": p.relative_to(root).as_posix(), "sha256": auditor.sha(p), "bytes": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file() and not (omitted_image and p == image)]
    dump(root / "manifest.json", {"files": files})
    audit_path = tmp_path / "synthetic-audit.json"
    # This is a boundary-only fixture, not an actual scientific collection audit.
    dump(audit_path, {"passed": True, "summary_sha256": auditor.sha(root / "summary.json"),
                      "manifest_sha256": auditor.sha(root / "manifest.json"),
                      "admitted_N": 1, "matched_N": 1, "registered_attempts": 2})
    return root, audit_path


def test_export_uses_original_manifest_label_digest(tmp_path):
    root, audit_path = export_fixture(tmp_path)
    result = exporter.export(root, audit_path, {})
    inventory = auditor.manifest_inventory(json.loads((root / "manifest.json").read_text()))
    assert result["runtime_inputs"][0]["labels_sha256"] == inventory["scene/S/labels.json"]["sha256"]


def test_export_rejects_labels_changed_after_audit(tmp_path):
    root, audit_path = export_fixture(tmp_path)
    labels = root / "scene/R/labels.json"
    changed = json.loads(labels.read_text())
    changed["valid_chunk_starts"] = [7]
    dump(labels, changed)
    with pytest.raises(ValueError, match="audited identity"):
        exporter.export(root, audit_path, {})


def test_export_rejects_referenced_image_changed_after_audit(tmp_path):
    root, audit_path = export_fixture(tmp_path)
    (root / "scene/frame.npz").write_bytes(b"unaudited-new-camera-payload")
    with pytest.raises(ValueError, match="audited identity"):
        exporter.export(root, audit_path, {})


def test_export_rejects_image_absent_from_bound_inventory(tmp_path):
    root, audit_path = export_fixture(tmp_path, omitted_image=True)
    with pytest.raises(ValueError, match="absent"):
        exporter.export(root, audit_path, {})


def window_rows(n=10):
    return [{"step": i, "action_source": "state_conditioned_teacher", "safe": True,
             "after": {"geometry": {"socket_grasp_contact": True,
                                     "peg_grasp_contact": True, "peg_socket_contact": True}}}
            for i in range(n)]


def recovery(start=1, end=5):
    return {"correction_start": start, "recovery": {"step": end, "sustain_steps": 5}}


def test_exact_five_step_window_entirely_in_correction():
    result = auditor.verify_recovery_window(window_rows(), recovery(), "approach")
    assert [r["step"] for r in result] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("start,end,n", [(1, 1, 10), (1, 4, 10), (1, 10, 10), (-1, 5, 10)])
def test_empty_short_or_outside_correction_window_rejected(start, end, n):
    with pytest.raises(ValueError, match="outside correction"):
        auditor.verify_recovery_window(window_rows(n), recovery(start, end), "approach")


def test_recovery_contact_loss_and_step_discontinuity_rejected():
    rows = window_rows()
    rows[3]["after"]["geometry"]["peg_socket_contact"] = False
    with pytest.raises(ValueError, match="insertion contact"):
        auditor.verify_recovery_window(rows, recovery(), "align")
    rows = window_rows()
    rows[3]["step"] = 9
    with pytest.raises(ValueError, match="consecutive"):
        auditor.verify_recovery_window(rows, recovery(), "approach")


def test_recovery_cannot_count_deliberate_deviation_as_correction():
    rows = window_rows()
    rows[1]["action_source"] = "registered_deliberate_deviation"
    with pytest.raises(ValueError, match="nonexpert"):
        auditor.verify_recovery_window(rows, recovery(), "approach")
