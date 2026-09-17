"""Read-only consistency check of the rescan report, source hashes and saved results."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path(__file__).resolve().parents[3]
report_path = root / "reports/training/m2-smolvla-training-chain-rescan-2026-09-14.json"
report = json.loads(report_path.read_text())
for name, expected in report["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
cases = ET.parse(root / report["verification"]["regression_junit"]).findall(".//testcase")
failed = [c for c in cases if c.find("failure") is not None]
skipped = [c for c in cases if c.find("skipped") is not None]
assert not any(c.find("error") is not None for c in cases)
assert len(failed) == 8 and len(skipped) == 1 and len(cases) - len(failed) - len(skipped) == 512
assert all("run_smolvla_v2.py" in c.find("failure").get("message", "") for c in failed)
data_cases = ET.parse(root / report["verification"]["data_junit"]).findall(".//testcase")
assert len(data_cases) == 1
assert all(c.find("failure") is None and c.find("skipped") is None for c in data_cases)
data = json.loads((root / report["data"]["evidence"]).read_text())
assert (data["train_rows"], data["validation_rows"]) == (20000, 2500)
assert data["decoded_image_samples"] == data["full_action_chunk_samples_verified"] == 405
assert sum(x["projected_elements"] for x in data["episodes"].values()) == 6973
assert data["control_coverage"]["unique_input_frames"] == 40
assert data["control_coverage"]["unique_target_frames"] == 2000
assert data["temporal_coverage"]["unique_input_frames"] == 5120
norm = json.loads((root / report["normalization"]["evidence"]).read_text())
assert norm["status"] == "passed"
error = max(v for row in norm["normalization"].values() for v in row["max_absolute_error"].values())
assert error == report["normalization"]["maximum_recomputed_statistic_error"]
saved = json.loads((root / report["normalization"]["saved_chain_evidence"]).read_text())
assert saved["status"] == "passed"
assert all(len(a["samples"]) == 6 for a in saved["arms"].values())
error = max(s["roundtrip_max_error"] for a in saved["arms"].values() for s in a["samples"])
assert error == report["normalization"]["maximum_action_roundtrip_error"]
assert all(
    v == 0
    for a in saved["arms"].values()
    for s in a["samples"]
    for v in s["normalization_max_error"].values()
)
result = {
    "status": "report_consistency_passed_not_full_suite_passed",
    "source_files_verified": 8,
    "test_cases": len(cases),
    "historical_pin_failures": len(failed),
    "numeric_frames": 22500,
    "chunks": 405,
    "saved_processor_samples": 12,
    "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
}
output = root / "runs/training-chain-rescan-20260914-001/report-verification.json"
with output.open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result))
