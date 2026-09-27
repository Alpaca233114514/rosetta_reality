import copy
import tempfile
import contextlib
import io
import unittest
from pathlib import Path

from basin.adapters import import_native, import_trace
from basin.analysis import analyze, compare, differences
from basin.io import child, dumps, loads, outside, sha, write_new
from basin.report import render
from basin.store import Store
from basin.__main__ import main


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.root / "history")

    def native(self, name, values=None, **overrides):
        value = {"schema_version": 1, "evidence_kind": "synthetic", "status": "complete",
                 "dimensions": ["joint"], "parameters": {"seed": 0},
                 "events": [{"step": step, "stage": "execution", "values": row}
                            for step, row in enumerate(values or [{"action": [0.0]}, {"action": [1.0]}])]}
        value.update(overrides)
        path = self.root / (name + ".json")
        write_new(path, dumps(value))
        return path

    def trace(self, incomplete=False, corrupt_sequence=False):
        repo = self.root / "source"
        folder = repo / "trace"
        common = {"schema_version": 1, "seed": 4, "episode_index": 4}
        identity = {"schema_version": 1, "options": {"seed": 4}, "episode_index": 4,
                    "chunk_length": 1, "dimension_names": ["joint"], "gripper_indices": [0],
                    "identity": {"run_id": "source-run", "original_protocol": {"action_contract_sha256": "abc"}}}
        prediction = common | {"event": "prediction", "prediction": 0,
                               "postprocessed_first_action": [0.2], "decoded_first_action": [0.2],
                               "internal_grippers": [[0.2]], "input_identity": {"image": "hash"}}
        started = common | {"event": "step_started", "prediction": 0, "step": 0,
                            "state_before": [0.0], "executed_action": [0.2]}
        step = started | {"event": "step", "state_after": [0.1], "reward": 1.0,
                          "success": False, "done": False, "terminated": False, "truncated": False}
        if corrupt_sequence:
            step["executed_action"] = [0.9]
        rows = [prediction, started] if incomplete else [prediction, started, step]
        status = "incomplete" if incomplete else "complete"
        summary = {"schema_version": 1, "status": status, "predictions": 1, "steps_started": 1,
                   "steps_recorded": 0 if incomplete else 1, "steps_completed": 0 if incomplete else 1,
                   "error_type": "RuntimeError" if incomplete else None,
                   "metrics": {"rollout_length": 1, "maximum_reward": 1.0, "success": False}}
        payloads = {"identity.json": dumps(identity).encode(), "summary.json": dumps(summary).encode(),
                    "trace.jsonl": b"".join((dumps(r).replace("\n", "") + "\n").encode() for r in rows)}
        for name, data in payloads.items():
            write_new(folder / name, data)
        write_new(folder / "manifest.json", dumps({"schema_version": 1, "status": status,
                                                   "files": {n: sha(d) for n, d in payloads.items()}}))
        return repo

    def test_roundtrip_arbitrary_tensor(self):
        source = self.native("a", [{"activation": [[1, 2], [3, 4]], "unknown": None}])
        import_native(self.store, source, "a")
        self.assertEqual(self.store.get("a")[1][0]["values"]["activation"], [[1, 2], [3, 4]])
        self.assertEqual(analyze(self.store, "a")["statistics"]["execution/activation/1/1"]["mean"], 4)
        self.assertEqual(self.store.history()[0]["integrity"], "verified")

    def test_idempotent_import(self):
        source = self.native("a")
        import_native(self.store, source, "a")
        self.assertEqual(import_native(self.store, source, "a")["status"], "already_present")

    def test_conflict_preserves_original(self):
        import_native(self.store, self.native("a"), "same")
        with self.assertRaisesRegex(ValueError, "different evidence"):
            import_native(self.store, self.native("b", [{"action": [99]}]), "same")
        self.assertEqual(self.store.get("same")[1][0]["values"]["action"], [0])

    def test_tampering_blocks_analysis(self):
        import_native(self.store, self.native("a"), "a")
        (self.store.path("a") / "events.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "digest"):
            analyze(self.store, "a")
        self.assertEqual(self.store.history()[0]["integrity"], "invalid")

    def test_unpublished_directory_invalid(self):
        self.store.path("partial").mkdir(parents=True)
        self.assertEqual(self.store.history()[0]["integrity"], "invalid")

    def test_difference_evidence_and_max_delta(self):
        import_native(self.store, self.native("a"), "a")
        import_native(self.store, self.native("b", [{"action": [0]}, {"action": [1.5]}]), "b")
        result = compare(self.store, "a", "b")
        self.assertEqual(result["first_recorded_difference"]["step"], 1)
        self.assertEqual(result["maximum_absolute_delta"], 0.5)
        self.assertEqual(result["first_recorded_difference"]["left_evidence"]["pointer"], "/events/1")
        self.assertIsNone(result["causal_conclusion"])
        self.assertEqual(result["hypotheses"], [])

    def test_tolerance_keeps_maximum(self):
        import_native(self.store, self.native("a", [{"action": [1]}]), "a")
        import_native(self.store, self.native("b", [{"action": [1.01]}]), "b")
        result = compare(self.store, "a", "b", tolerance=0.1)
        self.assertIsNone(result["first_recorded_difference"])
        self.assertAlmostEqual(result["maximum_absolute_delta"], 0.01)

    def test_missing_not_zero(self):
        import_native(self.store, self.native("a"), "a")
        import_native(self.store, self.native("b", [{"other": 0}]), "b")
        result = compare(self.store, "a", "b")
        self.assertEqual(result["missing_count"], 2)
        self.assertFalse(result["coverage_complete"])
        self.assertIsNone(analyze(self.store, "a")["facts"]["episode_success"])
        self.assertTrue(list(differences({"x": None}, {})))

    def test_null_has_no_comparison_coverage(self):
        import_native(self.store, self.native("a", [{"action": None}]), "a")
        result = compare(self.store, "a", "a")
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["compared_events"], 0)

    def test_identical_numeric_delta_zero(self):
        import_native(self.store, self.native("a"), "a")
        self.assertEqual(compare(self.store, "a", "a")["maximum_absolute_delta"], 0.0)

    def test_cli_query_roundtrip_and_failure(self):
        source = self.native("a")
        def cli(*args):
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(["--store", str(self.store.root), *args])
            return code, loads(stdout.getvalue() or stderr.getvalue())
        self.assertEqual(cli("import-native", str(source), "--id", "a")[0], 0)
        self.assertEqual(cli("show", "a", "--pointer", "/parameters/seed"), (0, 0))
        self.assertEqual(cli("events", "a", "--step", "1")[1]["matched"], 1)
        self.assertEqual(cli("artifact", "a", "native.json", "--pointer", "/events/1/values/action")[1], [1.0])
        self.assertEqual(cli("verify", "a")[0], 0)
        self.assertEqual(cli("show", "a", "--pointer", "/absent")[0], 2)

    def test_synthetic_real_separate(self):
        import_native(self.store, self.native("a"), "a")
        import_native(self.store, self.native("b", evidence_kind="historical_observation"), "b")
        self.assertFalse(compare(self.store, "a", "b")["comparable"])

    def test_dimension_mismatch(self):
        import_native(self.store, self.native("a"), "a")
        import_native(self.store, self.native("b", dimensions=["other_units"]), "b")
        self.assertFalse(compare(self.store, "a", "b")["comparable"])

    def test_bad_tolerance(self):
        for tolerance in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                compare(self.store, "missing", "missing", tolerance=tolerance)

    def test_strict_json(self):
        for data in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                loads(data)

    def test_duplicate_event(self):
        event = {"step": 0, "stage": "x", "values": {}}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            import_native(self.store, self.native("a", events=[event, copy.deepcopy(event)]), "a")

    def test_source_output_overlap(self):
        for path in (self.root, self.root / "x"):
            with self.assertRaises(ValueError):
                outside(path, self.root)

    def test_path_traversal(self):
        for name in ("../secret", "/secret", "C:/secret", "x\\secret"):
            with self.assertRaises(ValueError):
                child(self.root, name)
        for name in ("../run", ".", "a/b", ""):
            with self.assertRaises(ValueError):
                self.store.path(name)

    def test_source_hash_mismatch(self):
        repo = self.trace()
        (repo / "trace/trace.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "digest"):
            import_trace(self.store, repo, "trace", "real")

    def test_hash_valid_sequence_invalid(self):
        repo = self.trace(corrupt_sequence=True)
        with self.assertRaisesRegex(ValueError, "changed"):
            import_trace(self.store, repo, "trace", "real")

    def test_completed_trace_source_unchanged(self):
        repo = self.trace()
        before = {p.name: p.read_bytes() for p in (repo / "trace").iterdir()}
        import_trace(self.store, repo, "trace", "real", gate="gate4")
        self.assertEqual(analyze(self.store, "real")["facts"]["completed_execution_steps"], 1)
        self.assertEqual(self.store.get("real")[0]["gate"]["acceptance"], "not_inferred_from_episode")
        self.assertEqual(before, {p.name: p.read_bytes() for p in (repo / "trace").iterdir()})

    def test_incomplete_not_fabricated(self):
        repo = self.trace(incomplete=True)
        import_trace(self.store, repo, "trace", "real")
        result = analyze(self.store, "real")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["facts"]["completed_execution_steps"], 0)
        self.assertIsNone(result["facts"]["maximum_reward"])

    def test_report_escape_no_overwrite(self):
        import_native(self.store, self.native("a", gate={"label": "<script>alert(1)</script>"}), "a")
        output = self.root / "report.html"
        render(self.store, output)
        html = output.read_text(encoding="utf-8")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        with self.assertRaises(FileExistsError):
            render(self.store, output)


if __name__ == "__main__":
    unittest.main()
