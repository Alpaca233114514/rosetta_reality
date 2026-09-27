import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from basin.adapters import import_native
from basin.api import BasinAPI, ToolError
from basin.io import dumps, sha, write_new
from basin.mcp import MAX_REQUEST_BYTES, Server, serve
from basin.store import Store


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}}
READY = {"jsonrpc": "2.0", "method": "notifications/initialized"}


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "history")
        source = self.root / "inputs" / "sample.json"
        write_new(source, dumps({"schema_version": 1, "evidence_kind": "synthetic", "status": "complete",
            "parameters": {"seed": 5}, "dimensions": ["x"], "source_run": "sample",
            "events": [{"step": i, "stage": "execution", "values": {"action": [i], "reward": i}}
                       for i in range(3)]}))
        import_native(self.store, source, "sample")
        self.api = BasinAPI(self.store.root)

    def test_default_read_only_catalog(self):
        self.assertEqual(len(self.api.tools()), 12)
        self.assertTrue(all(t["annotations"]["readOnlyHint"] for t in self.api.tools()))
        with self.assertRaises(ToolError):
            self.api.call("basin_import_native", {})

    def test_discovery_paging_and_parameter_query(self):
        self.assertEqual(self.api.call("basin_history")["data"]["total"], 1)
        self.assertNotIn("parameters", self.api.call("basin_get_run", {"run_id": "sample"})["data"])
        self.assertEqual(self.api.call("basin_get_run", {"run_id": "sample", "pointer": "/parameters/seed"})["data"], 5)
        first = self.api.call("basin_events", {"run_id": "sample", "limit": 2, "field": "/action"})["data"]
        self.assertEqual(first["next_offset"], 2)
        last = self.api.call("basin_events", {"run_id": "sample", "offset": 2})["data"]
        self.assertEqual(last["items"][0]["step"], 2)
        self.assertIsNone(last["next_offset"])

    def test_artifact_provenance_and_analyzer(self):
        data = self.api.call("basin_read_artifact", {"run_id": "sample", "name": "native.json", "pointer": "/parameters"})["data"]
        self.assertEqual(data["value"], {"seed": 5})
        self.assertEqual(len(data["evidence"]["sha256"]), 64)
        stats = self.api.call("basin_analyze", {"run_id": "sample", "prefix": "execution/action"})["data"]["statistics"]
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["items"][0]["mean"], 1)
        self.assertEqual(self.api.call("basin_verify", {"run_id": "sample"})["data"]["integrity"], "verified")

    def test_validation_and_response_budget(self):
        for args in ({"run_id": "sample", "limit": True}, {"run_id": "sample", "limit": 101},
                     {"run_id": "sample", "offset": -1}, {"run_id": "sample", "unknown": 3}):
            with self.assertRaises(ToolError):
                self.api.call("basin_events", args)
        api = BasinAPI(self.store.root, max_result_bytes=100)
        with self.assertRaisesRegex(ToolError, "narrower"):
            api.call("basin_events", {"run_id": "sample"})

    def test_explicit_import_roots(self):
        api = BasinAPI(self.store.root, {"input": self.root / "inputs"})
        result = api.call("basin_import_native", {"source": "input", "path": "sample.json", "run_id": "second"})
        self.assertTrue(result["ok"])
        with self.assertRaises(ToolError):
            api.call("basin_import_native", {"source": "input", "path": "../inputs/sample.json", "run_id": "escape"})
        with self.assertRaises(ToolError):
            api.call("basin_import_native", {"source": "unknown", "path": "sample.json", "run_id": "escape"})
        self.assertEqual(len(api.tools()), 16)

    def test_identity_claim_and_independent_source_bytes(self):
        data = b"source bytes\n"
        write_new(self.root / "inputs" / "weights.bin", data)
        source = self.root / "inputs" / "identity.json"
        write_new(source, dumps({"schema_version": 1, "evidence_kind": "synthetic",
            "parameters": {"weights_sha256": sha(data)}, "events": []}))
        import_native(self.store, source, "identity")
        api = BasinAPI(self.store.root, {"input": self.root / "inputs"})
        request = {"run_id": "identity", "pointer": "/parameters/weights_sha256",
                   "expected": sha(data), "source": "input", "path": "weights.bin"}
        result = api.call("basin_check_identity", request)["data"]
        self.assertEqual(result["status"], "match")
        self.assertEqual(result["verification"], "source_bytes_hashed")
        self.assertEqual(result["checks"]["source_file"]["bytes"], len(data))
        self.assertEqual(api.call("basin_check_identity", request | {"expected": "0" * 64})["data"]["status"], "mismatch")
        self.assertEqual(api.call("basin_check_identity", request | {"pointer": "/parameters/absent"})["data"]["status"], "missing")
        with self.assertRaises(ToolError):
            api.call("basin_check_identity", request | {"path": "../weights.bin"})
        with self.assertRaises(ToolError):
            self.api.call("basin_check_identity", {"run_id": "identity", "pointer": "/parameters/weights_sha256"})
        with self.assertRaises(ToolError):
            api.call("basin_check_identity", request | {"source": "unconfigured"})

    def test_identity_rejects_tampered_history(self):
        (self.store.path("sample") / "run.json").write_text("{}\n")
        with self.assertRaises(ToolError):
            self.api.call("basin_check_identity", {"run_id": "sample", "pointer": "/source_run", "expected": "sample"})

    def test_protocol_lifecycle_errors_and_notifications(self):
        server = Server(self.api)
        self.assertIn("error", server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
        self.assertEqual(server.handle(INIT)["result"]["protocolVersion"], "2025-11-25")
        self.assertIsNone(server.handle(READY))
        self.assertEqual(len(server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]), 12)
        linked = server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "basin_check_identity", "arguments": {"run_id": "sample",
                "pointer": "/source_run", "expected": "sample"}}})
        self.assertEqual(linked["result"]["structuredContent"]["data"]["status"], "match")
        self.assertEqual(server.handle(INIT)["error"]["code"], -32600)
        bad = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "bad"}}
        self.assertEqual(server.handle(bad)["error"]["code"], -32602)
        bad["params"] = {"name": "basin_get_run", "arguments": {"run_id": "absent"}}
        self.assertTrue(server.handle(bad)["result"]["isError"])
        bad.pop("id")
        self.assertIsNone(server.handle(bad))

    def test_stdio_recovery_and_request_bound(self):
        data = b"bad json\n" + (json.dumps(INIT) + "\n" + json.dumps(READY) + "\n").encode()
        out = io.BytesIO()
        self.assertEqual(serve(self.api, io.BytesIO(data), out), 0)
        responses = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0]["error"]["code"], -32700)
        self.assertEqual(serve(self.api, io.BytesIO(b"x" * (MAX_REQUEST_BYTES + 1)), io.BytesIO()), 2)

    def test_real_subprocess_protocol_and_cli(self):
        messages = [INIT, READY, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                        "name": "basin_get_run", "arguments": {"run_id": "sample", "pointer": "/parameters/seed"}}}]
        launcher = Path(__file__).resolve().parents[1] / "scripts/basin_mcp.py"
        result = subprocess.run([sys.executable, str(launcher), "--store", str(self.store.root)],
                                input="".join(json.dumps(m) + "\n" for m in messages), text=True,
                                capture_output=True, timeout=15, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r["id"] for r in responses], [1, 2, 3])
        self.assertEqual(responses[-1]["result"]["structuredContent"]["data"], 5)
        result = subprocess.run([sys.executable, "-m", "basin", "--store", str(self.store.root),
                                 "call", "basin_events"], input='{"run_id":"sample","step":2}',
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["data"]["items"][0]["step"], 2)


if __name__ == "__main__":
    unittest.main()
