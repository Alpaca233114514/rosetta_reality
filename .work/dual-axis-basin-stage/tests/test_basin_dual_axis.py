"""Independent data-only fixtures for sharded diagnostic imports and model queries."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from basin.api import BasinAPI, ToolError
from basin.dual_axis import import_dual_axis, locate_deviation
from basin.store import Store


class DualAxisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "inputs/bundle"
        self.source.mkdir(parents=True)
        self.store = Store(self.root / "history")

    def write_bundle(self, values, *, floor=0.01, category="behavior"):
        identity = {"schema": "rosetta.dual_axis.v1", "source_run": "s", "schedule": [{"update": x} for x, _ in values]}
        payloads = {"identity.json": json.dumps(identity).encode(), "tensors/a.bin": b"abcd"}
        for i, (step, value) in enumerate(values):
            row = {"schema": "rosetta.dual_axis.v1", "sequence": i, "status": "observed", "reasons": ["regular"],
                   "coordinates": {"axis": "probe", "update": step, "sample_id": "s", "noise_id": "n"},
                   "identity": {key: key for key in ("input", "noise", "processor", "action_contract", "runtime")},
                   "values": {"metrics": [{"name": "error", "value": value, "category": category, "aa_floor": floor}]}}
            payloads[f"events/{i:06}.jsonl"] = json.dumps(row).encode() + b"\n"
        files = {}
        for name, data in payloads.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        (self.source / "manifest.json").write_text(json.dumps({"schema": identity["schema"], "status": "complete",
                                                               "events": len(values), "files": files}))
        import_dual_axis(self.store, self.source.parent, "bundle", "s")
        return BasinAPI(self.store.root, {"inputs": self.source.parent})

    def test_external_range_hash_and_tamper(self):
        api = self.write_bundle([(0, 1), (128, 2)])
        args = {"run_id": "s", "name": "tensors/a.bin", "source": "inputs", "byte_offset": 1, "byte_limit": 2}
        self.assertEqual(api.call("basin_read_artifact", args)["data"]["data"], "YmM=")
        (self.source / "tensors/a.bin").write_bytes(b"abce")
        with self.assertRaises(ToolError):
            api.call("basin_read_artifact", args)

    def test_slow_drift_descriptive_and_not_adjacent_alarm(self):
        api = self.write_bundle([(0, 1), (1, 1.1), (2, 1.21), (3, 1.33)])
        data = api.call("basin_locate_deviation", {"run_id": "s", "section": "trends"})["data"]
        self.assertEqual(data["deviations_count"], 0)
        self.assertAlmostEqual(data["trends"]["items"][0]["change"], 0.33)

    def test_missing_floor_is_not_zero(self):
        self.write_bundle([(0, 1), (1, 2)], floor=None)
        self.assertEqual(locate_deviation(self.store, "s")["result"], "insufficient_evidence")

    def test_internal_change_not_behavior_regression(self):
        self.write_bundle([(0, 1), (1, 2)], category="internal")
        self.assertIsNone(locate_deviation(self.store, "s")["first_observed"])

    def test_sharded_history_and_idempotent_import(self):
        api = self.write_bundle([(0, 1), (10, 2), (11, 2)])
        self.assertEqual(self.store.history()[0]["events"], 3)
        self.assertEqual(api.call("basin_timeline", {"run_id": "s", "offset": 1, "limit": 1})["data"]["events"]["items"][0]["coordinates"]["update"], 10)
        self.assertEqual(import_dual_axis(self.store, self.source.parent, "bundle", "s")["status"], "already_present")


if __name__ == "__main__":
    unittest.main()
