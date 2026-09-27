import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from basin.api import BasinAPI, ToolError
from basin.io import dumps, sha
from basin.mcp import Server
from basin.store import Store
from basin.torchlens_data import analyze_torchlens, import_torchlens


def op(index, label, *, module="m", parents=None, shape=None, activation=None, saved=True):
    shape = [2] if shape is None else shape
    values = {"module": module, "call_index": 1, "label": label,
              "parents": parents or [], "shape": shape, "dtype": "torch.float32",
              "saved": saved, "finite": True if saved else None}
    if saved:
        values.update(activation=[3, -4] if activation is None else activation,
                      sha256="0" * 64)
    return {"step": 0, "stage": f"module:{module}/call:1/op:{index}", "values": values}


def envelope(events, status="complete"):
    params = {"collector": "basin.torchlens.v1", "torchlens": "2.23.0",
              "consistency": "not_assessed", "live_integration_accepted": False,
              "graph_completeness": "not_independently_verified"}
    if status == "incomplete":
        params["error_type"] = "ValueError"
    return {"schema_version": 1, "source_run": "synthetic-source",
            "evidence_kind": "synthetic", "status": status, "dimensions": [],
            "gate": {}, "parameters": params, "events": events}


class TorchLensDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "history")
        self.source = self.root / "inputs" / "native.json"

    def put(self, value):
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.source.write_text(dumps(value), encoding="utf-8")
        return self.source

    def basic(self):
        return envelope([op(0, "A"), op(1, "B", parents=["A"], shape=[], activation=2),
                         op(2, "C", module="other", shape=[0], activation=[]),
                         op(3, "D", module="other", saved=False)])

    def test_independent_numeric_and_graph_expectations(self):
        source = self.put(self.basic())
        self.assertEqual(import_torchlens(self.store, source, "sample")["status"], "imported")
        self.assertEqual(import_torchlens(self.store, source, "sample")["status"], "already_present")
        result = analyze_torchlens(self.store, "sample")
        summary = result["summary"]
        self.assertEqual((summary["operations"], summary["saved_operations"],
                          summary["observed_elements"], summary["module_count"]), (4, 3, 3, 2))
        self.assertEqual((summary["minimum"], summary["maximum"]), (-4, 3))
        self.assertAlmostEqual(summary["mean"], 1 / 3)
        self.assertAlmostEqual(summary["l2_norm"], math.sqrt(29))
        self.assertAlmostEqual(summary["rms"], math.sqrt(29 / 3))
        self.assertEqual(result["modules"][0]["module"], "m")
        self.assertEqual(result["modules"][0]["observed_elements"], 3)
        self.assertIsNone(result["modules"][1]["mean"])
        self.assertIsNone(result["modules"][1]["l2_norm"])
        self.assertEqual(result["graph"]["resolved_edges"], 1)
        self.assertEqual(result["graph"]["observed_roots"], 3)
        self.assertEqual(result["graph"]["longest_observed_path_edges"], 1)
        self.assertEqual(result["edges"], [{"from_op": 0, "to_op": 1, "parent_label": "A"}])
        self.assertEqual(self.store.verify("sample")["files"]["artifacts/torchlens.json"], sha(source.read_bytes()))
        self.assertEqual(self.store.get("sample")[1][1]["evidence"]["pointer"], "/events/1")

    def test_graph_ambiguity_missing_self_cycle_and_order(self):
        cases = [
            ([op(0, "A"), op(1, "A"), op(2, "B", parents=["A"])], "ambiguous_references"),
            ([op(0, "A"), op(1, "B", parents=["missing"])], "unresolved_references"),
            ([op(0, "A", parents=["A"])], "cycle_detected"),
            ([op(0, "A", parents=["B"]), op(1, "B", parents=["A"])], "cycle_detected"),
        ]
        for index, (events, field) in enumerate(cases):
            self.put(envelope(events))
            import_torchlens(self.store, self.source, f"graph-{index}")
            graph = analyze_torchlens(self.store, f"graph-{index}")["graph"]
            self.assertTrue(graph[field])
            self.assertFalse(graph["topology_available"])
            self.assertIsNone(graph["longest_observed_path_edges"])
        self.assertIn("self_reference", [row["kind"] for row in
                      analyze_torchlens(self.store, "graph-2")["issues"]])
        reordered = [op(1, "B"), op(0, "A")]
        self.put(envelope(reordered))
        with self.assertRaisesRegex(ValueError, "index or stage"):
            import_torchlens(self.store, self.source, "reordered")

    def test_numeric_extremes_and_invalid_values(self):
        self.put(envelope([op(0, "A", activation=[1.7e308, -1.7e308])]))
        import_torchlens(self.store, self.source, "extreme")
        summary = analyze_torchlens(self.store, "extreme")["summary"]
        self.assertEqual(summary["mean"], 0)
        self.assertIsNone(summary["l2_norm"])
        self.assertEqual(summary["rms"], 1.7e308)
        self.put(envelope([op(0, "Z", shape=[2, 2], activation=[[0, 0], [0, 0]])]))
        import_torchlens(self.store, self.source, "zeros")
        zeros = analyze_torchlens(self.store, "zeros")["summary"]
        self.assertEqual((zeros["observed_elements"], zeros["mean"],
                          zeros["rms"], zeros["l2_norm"]), (4, 0, 0, 0))
        bad = self.basic()
        bad["events"][0]["values"]["activation"] = [1, float("nan")]
        with self.assertRaises(ValueError):
            dumps(bad)
        for change, error in [
            (lambda v: v["events"][0]["values"].update(activation=[1]), "shape"),
            (lambda v: v["events"][0]["values"].update(finite=False), "saved"),
            (lambda v: v["events"][0]["values"].update(sha256="broken"), "saved"),
            (lambda v: v["events"][0]["values"].update(shape=[4097]), "saved"),
            (lambda v: v["events"][0]["values"].update(activation=[10**400, 0]), "non-finite"),
            (lambda v: v["events"][3]["values"].update(activation=[0]), "Unsaved"),
        ]:
            bad = self.basic()
            change(bad)
            self.put(bad)
            with self.assertRaisesRegex(ValueError, error):
                import_torchlens(self.store, self.source, "invalid")
        self.assertFalse(self.store.path("invalid").exists())
        self.source.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            import_torchlens(self.store, self.source, "duplicate-key")

    def test_failure_and_conflict_preserve_evidence(self):
        self.put(envelope([], status="incomplete"))
        import_torchlens(self.store, self.source, "failure")
        result = analyze_torchlens(self.store, "failure")
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["summary"]["mean"])
        self.assertEqual(result["graph"]["observed_roots"], 0)
        self.put(self.basic())
        with self.assertRaisesRegex(ValueError, "different evidence"):
            import_torchlens(self.store, self.source, "failure")
        self.assertEqual(self.store.get("failure")[0]["status"], "incomplete")

    def test_budgets_and_source_boundary(self):
        self.put(envelope([op(i, str(i), saved=False) for i in range(4097)]))
        with self.assertRaisesRegex(ValueError, "operation budget"):
            import_torchlens(self.store, self.source, "many")
        self.source.write_bytes(b" " * (16 * 1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            import_torchlens(self.store, self.source, "large")
        with self.assertRaisesRegex(ValueError, "outside"):
            import_torchlens(self.store, self.store.root / "native.json", "inside")

    def test_cli_api_and_mcp_roundtrip(self):
        self.put(self.basic())
        api = BasinAPI(self.store.root, {"input": self.source.parent})
        result = api.call("basin_import_torchlens", {"source": "input", "path": "native.json", "run_id": "sample"})
        self.assertEqual(result["data"]["status"], "imported")
        page = api.call("basin_analyze_torchlens", {"run_id": "sample", "limit": 1})["data"]
        self.assertEqual(page["modules"]["total"], 2)
        self.assertEqual(page["modules"]["next_offset"], 1)
        edges = api.call("basin_analyze_torchlens", {"run_id": "sample", "section": "edges"})["data"]
        self.assertEqual(edges["edges"]["items"][0]["from_op"], 0)
        raw = api.call("basin_read_artifact", {"run_id": "sample", "name": "torchlens.json",
                                                    "pointer": "/events/1/values/activation"})["data"]
        self.assertEqual(raw["value"], 2)
        with self.assertRaises(ToolError):
            api.call("basin_import_torchlens", {"source": "input", "path": "../native.json", "run_id": "escape"})
        server = Server(BasinAPI(self.store.root))
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"}}}
        self.assertIn("result", server.handle(init))
        server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        answer = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "basin_analyze_torchlens", "arguments": {"run_id": "sample"}}})
        self.assertEqual(answer["result"]["structuredContent"]["data"]["summary"]["observed_elements"], 3)
        write_server = Server(BasinAPI(self.store.root, {"input": self.source.parent}))
        self.assertIn("result", write_server.handle(init))
        write_server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        imported = write_server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "basin_import_torchlens", "arguments": {"source": "input", "path": "native.json",
                                                         "run_id": "mcp"}}})
        self.assertEqual(imported["result"]["structuredContent"]["data"]["status"], "imported")
        cli = subprocess.run([sys.executable, "-m", "basin", "--store", str(self.store.root),
                              "analyze-torchlens", "sample"], capture_output=True, text=True, timeout=15)
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertEqual(json.loads(cli.stdout)["graph"]["resolved_edges"], 1)
        second = subprocess.run([sys.executable, "-m", "basin", "--store", str(self.store.root),
                                 "import-torchlens", str(self.source), "--id", "cli"],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(second.stdout)["status"], "imported")


if __name__ == "__main__":
    unittest.main()
