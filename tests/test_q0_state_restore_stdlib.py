"""Receipt lifecycle tests; no MuJoCo, model, dataset or real rollout required."""
import importlib.util
import ast
import dataclasses
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "q0_restore", Path(__file__).resolve().parents[1] / "scripts/check_q0_state_restore.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class RestoreReceiptTests(unittest.TestCase):
    def test_absolute_paths_are_hashed_without_path_content(self):
        for value in ("/private/host/assets", "C:\\private\\host\\assets"):
            receipt = CHECK.safe_string(value)
            self.assertEqual(receipt["length"], len(value))
            self.assertNotIn("private", json.dumps(receipt))
        self.assertEqual(CHECK.safe_string("configs/sim/aloha.yaml"), "configs/sim/aloha.yaml")

    def test_source_defined_slots_contract_fields_are_complete_and_shared(self):
        source = Path(__file__).resolve().parents[1] / "src/rosetta_reality/sim/action_contract.py"
        parsed = ast.parse(source.read_text(encoding="utf-8"))
        selected = [node for node in parsed.body if isinstance(node, ast.ClassDef)
                    and node.name in ("ActionDimension", "ActionContract")]
        module = types.ModuleType("rosetta_reality.sim.action_contract")
        module.dataclass = dataclasses.dataclass
        future = ast.parse("from __future__ import annotations").body[0]
        tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
        # Actual definitions, deferred annotations; no torch/yaml/simulator import.
        with patch.dict(sys.modules, {module.__name__: module}):
            exec(compile(tree, "action_contract_source_classes", "exec"), module.__dict__)
            dimension = module.ActionDimension("joint", "radian", -1.0, 1.0)
            contract = module.ActionContract(
                name="synthetic", schema_version=1, embodiment="aloha",
                environment_id="synthetic", action_type="continuous",
                semantics="absolute_joint_position_targets", control_mode="position",
                space="joint", reference_frame="joint_local", frequency_hz=50,
                timestamp_alignment="synthetic", chunk_length=1,
                chunk_execution="first_action", simulator_expansion="synthetic",
                dimensions=(dimension,))
            fields = CHECK.frozen_contract_fields(contract)
            self.assertEqual(set(fields), {field.name for field in dataclasses.fields(contract)})
            self.assertIs(fields["dimensions"][0], dimension)
            self.assertEqual(CHECK.frozen_contract_fields(dimension)["minimum"], -1.0)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                contract.name = "changed"
            object.__setattr__(contract, "dimensions", [dimension])
            with self.assertRaises(CHECK.UnsupportedRestore):
                CHECK.frozen_contract_fields(contract)

    def test_existing_receipt_is_preserved_without_running_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            output.write_text("original", encoding="utf-8")
            with patch.object(CHECK, "run") as runner:
                with self.assertRaises(FileExistsError):
                    CHECK.main(["--output", str(output)])
            runner.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "original")

    def test_unsupported_is_failclosed_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(CHECK, "run", side_effect=CHECK.UnsupportedRestore("opaque_state_owner")):
                self.assertEqual(CHECK.main(["--output", str(output)]), 2)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "unsupported")
            self.assertFalse(value["passed"])
            self.assertFalse(value["provider_qualified"])
            self.assertFalse(value["infrastructure_failure"])
            self.assertIsNone(value["task_success"])
            digest = value.pop("full_numeric_sha256")
            self.assertEqual(digest, CHECK.tree_hash(value))

    def test_exception_message_cannot_leak_host_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(CHECK, "run", side_effect=RuntimeError("/private/host/secret")):
                CHECK.main(["--output", str(output)])
            self.assertNotIn("/private/host", output.read_text(encoding="utf-8"))
            self.assertTrue(json.loads(output.read_text())["infrastructure_failure"])

    def test_safety_stop_is_distinct_from_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(CHECK, "run", side_effect=CHECK.SafetyStop("unsafe_reset_state")):
                CHECK.main(["--output", str(output)])
            value = json.loads(output.read_text())
            self.assertTrue(value["safety_stop"])
            self.assertFalse(value["infrastructure_failure"])
            self.assertEqual(value["status"], "safety_stop")

    def test_hash_covers_tiny_numeric_difference(self):
        self.assertNotEqual(CHECK.tree_hash({"state": [1.0]}),
                            CHECK.tree_hash({"state": [1.0000000000000002]}))


if __name__ == "__main__":
    unittest.main()
