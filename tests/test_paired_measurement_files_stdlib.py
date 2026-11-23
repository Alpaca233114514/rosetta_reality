"""Pure saved-file and JSON sidecar counterexamples; no ML/physics imports."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module("audit_paired_measurement_files")
capture = module("paired_measurement_evidence")
CONTRACT = {"dimensions": [{"name": "joint", "minimum": -3.14158, "maximum": 3.14158}]}


def rows():
    return [{"event": "reset", "seed": 1000, "joint_limit_violations": 0, "unexpected_pairs": []},
            {"event": "step", "seed": 1000, "step": 1, "raw_action": [0.1], "processed_action": [0.1],
             "executed_action": [0.1], "state_before": [0.0], "state_after": [0.08], "reward": 0.0,
             "success": False, "done": False, "joint_limit_violations": 0, "unexpected_pairs": [], "raw_clipped_fields": 0}]


def physics(value=0.0):
    return {"qpos": [value], "qvel": [0.0], "ctrl": [0.1], "integration_state": [value, 0.0],
            "compiled_model_sha256": "a" * 64, "object_poses": {"peg": [0.0, 0.0, 0.0]}, "contacts": [],
            "joint_limits": [{"name": "finger", "qpos_address": 0, "lower": -1.0, "upper": 1.0,
                              "tolerance": 1e-5, "value": value, "violated": value > 1.00001}]}


def tensor(values):
    return {"values": values, "shape": [1, len(values)], "dtype": "float32"}


def initial():
    return {"episode_id": "synthetic", "source_identity": {"collector": "b" * 64},
            "processor_identity": {"effective_config": "c" * 64}, "normalization_sha256": "d" * 64,
            "action_contract_sha256": "e" * 64, "physics": physics(),
            "observation_hashes": {"camera": "f" * 64}, "noise_contract": "actual_tensor_per_call"}


def step():
    return {"step": 1, "noise": tensor([0.0]), "normalized_prediction": tensor([0.1]),
            "raw_chunk": tensor([0.1]), "projected_chunk": tensor([0.1]), "executed_slot": 0,
            "logical_command": [0.1], "physics_before": physics(), "physics_after": physics(0.08),
            "processor_identity": {"effective_config": "c" * 64}, "observation_hashes": {"camera": "f" * 64},
            "reward": 0.0, "task_success": False, "done": False}


class FileAuditTests(unittest.TestCase):
    def test_missing_physics_is_reported_not_inferred_from_state(self):
        result = audit.audit_trace(rows(), CONTRACT)
        self.assertEqual(result["coverage"]["physics_after"], {"present_nonnull": 0, "expected": 1})
        self.assertEqual(result["physical_joint_attribution"], "not_captured")
        self.assertTrue(result["logical_action_chain_exact"])

    def test_float32_boundary_recalculation(self):
        data = rows()
        data[1].update(raw_action=[4.0], processed_action=[audit.f32(3.14158)],
                       executed_action=[audit.f32(3.14158)], raw_clipped_fields=1)
        self.assertEqual(audit.audit_trace(data, CONTRACT)["raw_clipped_fields"], 1)

    def test_first_contact_does_not_mean_first_task_error(self):
        data = rows()
        data[1]["unexpected_pairs"] = [["finger", "table"]]
        result = audit.audit_trace(data, CONTRACT)
        self.assertEqual(result["first_collision"]["step"], 1)
        self.assertIsNone(result["first_limit"])
        self.assertFalse(result["success"])

    def test_reset_safety_stop_is_not_one_executed_step(self):
        data = rows()[:1]
        data[0]["joint_limit_violations"] = 1
        self.assertEqual(audit.audit_trace(data, CONTRACT)["steps"], 0)

    def test_invalid_trace_variants(self):
        for change in ("gap", "nan", "command", "false_clip", "continued", "state_chain"):
            with self.subTest(change=change):
                data = rows()
                if change == "gap": data[1]["step"] = 2
                if change == "nan": data[1]["state_after"] = [float("nan")]
                if change == "command": data[1]["executed_action"] = [0.2]
                if change == "false_clip": data[1]["raw_clipped_fields"] = 1
                if change in ("continued", "state_chain"):
                    data.append(copy.deepcopy(data[1]))
                    data[2]["step"] = 2
                    if change == "continued": data[1]["joint_limit_violations"] = 1
                with self.assertRaises(ValueError): audit.audit_trace(data, CONTRACT)

    def test_inventory_rejects_tamper_extra_duplicate_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "files"
            root.mkdir()
            (root / "a").write_text("original")
            manifest = Path(tmp) / "sha256"
            line = audit.sha(root / "a") + "  ./a\n"
            manifest.write_text(line)
            self.assertEqual(audit.verify_inventory(root, manifest)["files_verified"], 1)
            for body in (line + line, "0" * 64 + "  ./a\n", "0" * 64 + "  ../a\n"):
                manifest.write_text(body)
                with self.assertRaises(ValueError): audit.verify_inventory(root, manifest)
            manifest.write_text(line)
            (root / "extra").write_text("not inventoried")
            with self.assertRaises(ValueError): audit.verify_inventory(root, manifest)

    def test_no_science_imports(self):
        import ast
        for name in ("audit_paired_measurement_files", "audit_paired_measurement_context", "paired_measurement_evidence"):
            tree = ast.parse((Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")).read_text())
            names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(any((n or "").split(".")[0] in {"torch", "numpy", "mujoco", "gymnasium", "lerobot"} for n in names))

    def test_context_import_is_inert_and_checks_are_not_asserts(self):
        context = module("audit_paired_measurement_context")
        context.require(True)
        with self.assertRaises(ValueError): context.require(False)

    def test_bank_recovery_requires_observed_safe_streak_before_terminal(self):
        data = []
        for number in range(1, 7):
            event = {"criterion": "both_grasps", "required_consecutive_steps": 5,
                     "correction_steps_observed": number, "recovery_step": 5 if number >= 5 else None,
                     "geometric_recovery": number >= 5, "terminal_success_step": 6 if number == 6 else None,
                     "verified_recovery_chain": number == 6}
            data.append({"step": number, "executed_action": [0.1], "state_after": [0.08],
                         "raw_clips": 0, "reward": 4.0 if number == 6 else 0.0,
                         "task_success": number == 6, "joint_limit_violations": 0,
                         "unexpected_contacts": [], "geometry": {"socket_grasp_contact": True,
                         "peg_grasp_contact": True}, "recovery_event": event})
        self.assertTrue(audit.audit_trace(data, CONTRACT, bank=True)["recovery_event"]["verified_recovery_chain"])
        data[2]["geometry"]["peg_grasp_contact"] = False
        with self.assertRaises(ValueError): audit.audit_trace(data, CONTRACT, bank=True)

    def test_cli_refuses_output_inside_received_root(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sys.argv", ["audit", "--root", tmp, "--manifest", "unused",
                                   "--output", str(Path(tmp) / "output.json")]):
                with self.assertRaises(SystemExit) as failure: audit.main()
                self.assertEqual(failure.exception.code, 2)
            self.assertFalse((Path(tmp) / "output.json").exists())


class SidecarTests(unittest.TestCase):
    def test_empty_or_changed_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = capture.EvidenceSidecar(Path(tmp) / "trace", maximum_bytes=100000)
            try:
                empty = initial()
                empty["processor_identity"] = {}
                with self.assertRaises(ValueError): sink.start(empty)
                sink.start(initial())
                changed = step()
                changed["processor_identity"] = {"effective_config": "a" * 64}
                with self.assertRaises(ValueError): sink.observe(changed)
            finally: sink.close()

    def test_full_record_and_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sink = capture.EvidenceSidecar(path, maximum_bytes=100000)
            sink.start(initial())
            sink.observe(step())
            result = sink.close()
            self.assertEqual(result["steps"], 1)
            self.assertEqual(result["sha256"], audit.sha(path))
            self.assertEqual(len(path.read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError): capture.EvidenceSidecar(path, maximum_bytes=100000)

    def test_missing_full_noise_or_physics_fails(self):
        for key in ("noise", "physics_before", "physics_after", "processor_identity"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                sink = capture.EvidenceSidecar(Path(tmp) / "trace", maximum_bytes=100000)
                try:
                    sink.start(initial())
                    record = step()
                    del record[key]
                    with self.assertRaises(ValueError): sink.observe(record)
                finally: sink.close()

    def test_joint_flag_threshold_and_complete_tensor(self):
        record = physics(1.00002)
        record["joint_limits"][0]["violated"] = False
        with self.assertRaises(ValueError): capture.physics(record)
        with self.assertRaises(ValueError): capture.tensor({"shape": [50, 32], "dtype": "float32", "values": [0.0]})
        record = physics()
        record["joint_limits"][0]["tolerance"] = 0.001
        with self.assertRaises(ValueError): capture.physics(record)

    def test_no_evidence_after_safety_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = capture.EvidenceSidecar(Path(tmp) / "trace", maximum_bytes=100000)
            try:
                sink.start(initial())
                record = step()
                record["physics_after"] = physics(1.00002)
                sink.observe(record)
                record["step"] = 2
                with self.assertRaises(ValueError): sink.observe(record)
            finally: sink.close()

    def test_bytes_cap_preserves_incomplete_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            sink = capture.EvidenceSidecar(path, maximum_bytes=1)
            try:
                with self.assertRaises(ValueError): sink.start(initial())
            finally: sink.close()
            self.assertTrue(path.exists())
            self.assertEqual(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
