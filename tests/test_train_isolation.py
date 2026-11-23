"""Real synthetic Parquet checks: protected sentinel data never leaves exporter."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_isolation", ROOT / "scripts/isolate_aloha_train.py")
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)


class TrainIsolationTests(unittest.TestCase):
    def fixture(self, root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        original = json.loads((ROOT / "configs/diagnostics/aloha_qualification_20260927_002.json").read_text())
        plan = {k: original[k] for k in ("train_episodes", "dev_episodes", "hidden_episodes", "expected_dimension", "expected_fps")}
        plan.update(authorization=isolation.AUTHORIZATION, wall_seconds=180, max_output_bytes=32*1024**2, chunk_length=50)
        source = root / "source"
        (source / "data").mkdir(parents=True)
        rows = []
        for episode in range(50):
            for frame in range(50):
                protected = episode not in plan["train_episodes"]
                rows.append({"episode_index": episode, "frame_index": frame, "timestamp": frame / 50,
                             "observation.state": [float("nan") if protected else 1.0] * 14,
                             "action": [999999.0 if protected else .25] * 14, "next.done": frame == 49})
        data = source / "data/mixed.parquet"
        pq.write_table(pa.Table.from_pylist(rows), data, row_group_size=len(rows))
        receipt = {"dataset": "test/aloha", "revision": "a"*40, "files": [{"path": "data/mixed.parquet", "bytes": data.stat().st_size, "sha256": isolation.file_sha(data)}]}
        isolation.save(source / "rosetta_download_manifest.json", receipt)
        plan["sources"] = [{"root": "source", "dataset": "test/aloha", "revision": "a"*40, "manifest_sha256": isolation.file_sha(source / "rosetta_download_manifest.json"), "already_in_D": True, "expected_episode_frames": 50}]
        plan_path = root / "plan.json"
        isolation.save(plan_path, plan)
        return plan_path, plan, data

    def test_mixed_group_exports_only_train_and_independent_verifier_passes(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, plan, _ = self.fixture(root)
            isolation.export_worker(plan_path, root, root / "export")
            # Remove access to original input; verifier must use derived files only.
            (root / "source").rename(root / "original-inaccessible")
            result = isolation.verify_export(plan_path, root / "export")
            self.assertEqual(result["sources"][0]["frames"], 2000)
            self.assertEqual(result["sources"][0]["valid_target_slots"], 40*1275)
            self.assertFalse(result["training_ready"])
            self.assertTrue(result["protected_rows_may_be_decoded_inside_worker"])
            rows = pq.read_table(root / "export/train-00.parquet").to_pylist()
            self.assertEqual({r["episode_index"] for r in rows}, set(plan["train_episodes"]))
            self.assertTrue(all(r["action"] == [.25]*14 for r in rows))

    def test_permission_and_split_fail_before_source_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, plan, _ = self.fixture(root)
            plan.pop("authorization")
            with self.assertRaises(PermissionError):
                isolation.validate_plan(plan)
            plan["authorization"] = isolation.AUTHORIZATION
            plan["train_episodes"][0] = plan["hidden_episodes"][0]
            with self.assertRaises(ValueError):
                isolation.validate_plan(plan)
            self.assertFalse((root / "export").exists())

    def test_source_drift_fails_without_complete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, _, data = self.fixture(root)
            with data.open("ab") as stream:
                stream.write(b"drift")
            with self.assertRaises(ValueError):
                isolation.export_worker(plan_path, root, root / "export")
            self.assertFalse((root / "export/manifest.json").exists())

    def test_export_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, _, _ = self.fixture(root)
            (root / "export").mkdir()
            with self.assertRaises(FileExistsError):
                isolation.export_worker(plan_path, root, root / "export")

    def test_derived_file_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, _, _ = self.fixture(root)
            isolation.export_worker(plan_path, root, root / "export")
            with (root / "export/train-00.parquet").open("ab") as stream:
                stream.write(b"drift")
            with self.assertRaises(ValueError):
                isolation.verify_export(plan_path, root / "export")


if __name__ == "__main__":
    unittest.main()
