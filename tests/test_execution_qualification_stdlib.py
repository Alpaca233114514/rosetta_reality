"""Offline synthetic regression checks using only the Python standard library."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rosetta_reality.diagnostics.cohort_qualification import file_sha, match, qualify, safe_path

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load("bounded_guard", "autodl_bounded_guard.py")
auditor = load("cohort_auditor", "audit_targeted_aloha.py")


class GuardTests(unittest.TestCase):
    def test_missing_proc_is_already_exited(self):
        with tempfile.TemporaryDirectory() as path:
            self.assertIsNone(guard.process_identity(9999, Path(path)))

    def test_stat_comm_with_spaces_and_parentheses(self):
        with tempfile.TemporaryDirectory() as path:
            proc = Path(path) / "42"
            proc.mkdir()
            fields = ["S", "1", "42"] + ["0"] * 16 + ["12345"]
            (proc / "stat").write_text("42 (a name ) tricky) " + " ".join(fields))
            self.assertEqual(guard.process_identity(42, Path(path)), ("12345", 42))

    def test_stop_does_not_signal_reused_pid(self):
        with tempfile.TemporaryDirectory() as path:
            job = Path(path)
            (job / "worker-pid.json").write_text(json.dumps({"pid": 9999, "start_ticks": "old"}))
            with patch.object(guard, "process_identity", return_value=("new", 9999)), patch.object(guard.os, "killpg") as kill:
                guard.stop_worker(job, 0)
                kill.assert_not_called()

    def test_stop_rejects_group_not_owned(self):
        with tempfile.TemporaryDirectory() as path:
            job = Path(path)
            (job / "worker-pid.json").write_text(json.dumps({"pid": 9999, "start_ticks": "old"}))
            with patch.object(guard, "process_identity", return_value=("old", 77)), patch.object(guard.os, "killpg") as kill:
                guard.stop_worker(job, 0)
                kill.assert_not_called()

    def test_exited_real_worker_and_unrelated_worker(self):
        owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as path:
                job = Path(path)
                ticks, group = guard.process_identity(owned.pid)
                self.assertEqual(group, owned.pid)
                (job / "worker-pid.json").write_text(json.dumps({"pid": owned.pid, "start_ticks": ticks}))
                guard.stop_worker(job, 0.05)
                owned.wait(timeout=5)
                self.assertIsNone(other.poll())
                self.assertEqual(guard.stop_worker(job, 0), "exited_or_identity_changed")
        finally:
            for proc in (owned, other):
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=5)

    def test_cleanup_error_still_reaches_shutdown_checks(self):
        with tempfile.TemporaryDirectory() as path:
            job = Path(path)
            (job / "registration.json").write_text(json.dumps({"started_unix": 1, "work_deadline_unix": 2, "shutdown_deadline_unix": 3, "shutdown_authorized": True}))
            with patch.object(guard, "stop_worker", side_effect=PermissionError("test")), patch.object(guard, "request_shutdown") as shutdown:
                guard.run(job, 0)
                shutdown.assert_called_once_with(job)
            self.assertTrue((job / "guard-worker-error.json").is_file())

    def test_deadline_clamping_and_bound(self):
        self.assertEqual(guard.deadlines({"started_unix": 1, "work_deadline_unix": 2, "shutdown_deadline_unix": 3}, wall=4, monotonic=10), (10, 10))
        with self.assertRaises(ValueError):
            guard.deadlines({"started_unix": 1, "work_deadline_unix": 2, "shutdown_deadline_unix": 1000000}, wall=4, monotonic=10)

    def test_missing_shutdown_authorization_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            job = Path(path)
            (job / "registration.json").write_text(json.dumps({"started_unix": 1, "work_deadline_unix": 2, "shutdown_deadline_unix": 3}))
            with patch.object(guard, "request_shutdown") as shutdown:
                with self.assertRaises(PermissionError):
                    guard.run(job, 0)
                shutdown.assert_not_called()

    def test_evidence_create_only(self):
        with tempfile.TemporaryDirectory() as path:
            target = Path(path) / "failure.json"
            guard.save(target, {"failed": True})
            with self.assertRaises(FileExistsError):
                guard.save(target, {"failed": False})
            self.assertTrue(json.loads(target.read_text())["failed"])

    def test_shutdown_wrapper_drift_rejected_without_shutdown(self):
        with patch.object(guard.Path, "read_bytes", return_value=b"changed"), patch.object(guard.os, "execv") as execute:
            with self.assertRaisesRegex(RuntimeError, "wrapper changed"):
                guard.request_shutdown(Path("/tmp"))
            execute.assert_not_called()


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.contract = "a" * 64

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, role, episode):
        record = {"dataset": "synthetic/aloha", "revision": "b" * 40, "episode": episode,
                  "role": role, "split": "train", "already_in_D": False,
                  "scene": f"scene-{episode}", "parent_trajectory": f"parent-{episode}",
                  "collection_family": "synthetic", "effective_frames": 100,
                  "content_sha256": str(episode) * 64,
                  "valid_target_slots": 4000, "padding_fraction": 0.1,
                  "duration_seconds": 2, "task": "insertion", "embodiment": "aloha",
                  "scene_stratum": "easy", "initial_pose_stratum": "fixed", "collector": "same"}
        base = {k: record[k] for k in ("dataset", "revision", "episode")}
        reports = {
            "contract": dict(base, status="passed", action_contract_sha256=self.contract,
                             semantic_equivalence_verified=True, gate1_passed=True, gate2_passed=True),
            "rows": dict(base, status="passed", hidden_loaded=False, finite_state_action=True,
                         video_alignment_verified=True, effective_frames=100, valid_target_slots=4000,
                         padding_fraction=0.1, source_records_sha256=str(episode) * 64),
            "labels": dict(base, task_outcome="success", outcome_verified=True,
                           ordinary_success_verified=True, recovery_present=role == "R",
                           deviation_frame=10, correction_start=11, correction_end=20,
                           recovery_frame=25, success_frame=90, state_conditioned_expert=True,
                           continuous_context_verified=True, action_source="human_correction", time_indexed_replay=False),
        }
        for key, value in reports.items():
            name = f"{episode}-{key}.json"
            target = self.root / name
            target.write_text(json.dumps(value))
            record[key + "_evidence"] = {"path": name, "sha256": file_sha(target)}
        return record

    def change_label(self, record, **changes):
        reference = record["labels_evidence"]
        path = self.root / reference["path"]
        value = json.loads(path.read_text())
        value.update(changes)
        path.write_text(json.dumps(value))
        reference["sha256"] = file_sha(path)

    def test_missing_labels_stays_unknown(self):
        record = self.fixture("R", 2)
        del record["labels_evidence"]
        result = qualify(record, self.root, self.contract)
        self.assertFalse(result["eligible"])
        self.assertIn("missing_labels_evidence", result["reasons"])

    def test_failed_recovery_and_replay_not_accepted(self):
        record = self.fixture("R", 2)
        self.change_label(record, task_outcome="failure", time_indexed_replay=True)
        result = qualify(record, self.root, self.contract)
        self.assertIn("outcome_not_verified_success", result["reasons"])
        self.assertIn("recovery_supervision_not_verified", result["reasons"])

    def test_ordinary_missing_intervention_not_assumed_clean(self):
        record = self.fixture("S", 3)
        self.change_label(record, recovery_present=None)
        self.assertFalse(qualify(record, self.root, self.contract)["eligible"])

    def test_event_chronology_rejected(self):
        record = self.fixture("R", 2)
        self.change_label(record, recovery_frame=5)
        self.assertIn("recovery_event_chain_missing_or_invalid", qualify(record, self.root, self.contract)["reasons"])

    def test_teacher_claim_needs_hash_bound_passed_gate(self):
        record = self.fixture("R", 2)
        self.change_label(record, action_source="qualified_teacher")
        self.assertIn("teacher_gate_not_verified", qualify(record, self.root, self.contract)["reasons"])

    def test_hash_drift_rejected(self):
        record = self.fixture("S", 3)
        (self.root / record["rows_evidence"]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            qualify(record, self.root, self.contract)

    def test_same_shape_other_contract_rejected(self):
        record = self.fixture("S", 3)
        self.assertIn("contract_not_verified", qualify(record, self.root, "c" * 64)["reasons"])

    def test_D_duplicate_not_new_data(self):
        record = self.fixture("S", 3)
        record["already_in_D"] = True
        self.assertIn("not_verified_new_trajectory", qualify(record, self.root, self.contract)["reasons"])

    def test_protected_descendant_rejected(self):
        s, r = self.fixture("S", 3), self.fixture("R", 2)
        result = match([s, r], [{"S": 0, "R": 1}], [{"parent_trajectory": "parent-2"}], self.root, self.contract)
        self.assertFalse(result["cohort_complete"])

    def test_pair_pass_does_not_authorize_training(self):
        s, r = self.fixture("S", 3), self.fixture("R", 2)
        result = match([s, r], [{"S": 0, "R": 1}], [], self.root, self.contract)
        self.assertTrue(result["cohort_complete"])
        self.assertFalse(result["training_ready"])

    def test_protected_episode_cannot_be_hidden_by_renamed_scene(self):
        s, r = self.fixture("S", 3), self.fixture("R", 2)
        protected = [{k: r[k] for k in ("dataset", "revision", "episode")}]
        result = match([s, r], [{"S": 0, "R": 1}], protected, self.root, self.contract)
        self.assertFalse(result["cohort_complete"])

    def test_protected_collection_family_rejected(self):
        s, r = self.fixture("S", 3), self.fixture("R", 2)
        result = match([s, r], [{"S": 0, "R": 1}], [{"collection_family": "synthetic"}], self.root, self.contract)
        self.assertFalse(result["cohort_complete"])

    def test_unmatched_exposure_rejected(self):
        s, r = self.fixture("S", 3), self.fixture("R", 2)
        r["duration_seconds"] = 4
        result = match([s, r], [{"S": 0, "R": 1}], [], self.root, self.contract)
        self.assertIn("unmatched_duration_seconds", result["pairs"][0]["reasons"])

    def test_pair_trajectory_reuse_rejected(self):
        s, r = self.fixture("S", 3), self.fixture("R", 2)
        with self.assertRaisesRegex(ValueError, "reused"):
            match([s, r], [{"S": 0, "R": 1}, {"S": 0, "R": 1}], [], self.root, self.contract)

    def test_path_traversal_and_symlink_escape(self):
        for name in ("../secret", "/secret", "C:/secret"):
            with self.assertRaises(ValueError):
                safe_path(self.root, name)
        os.symlink("/", self.root / "escape")
        with self.assertRaises(ValueError):
            safe_path(self.root, "escape/etc")

    def test_hidden_in_shared_row_group_excluded_before_read(self):
        stats = SimpleNamespace(has_min_max=True, null_count=0, min=0, max=3)
        self.assertFalse(auditor.allowed_group(stats, {0, 2, 3}))
        self.assertFalse(auditor.allowed_group(None, {0, 2, 3}))
        stats.min = stats.max = 2
        self.assertTrue(auditor.allowed_group(stats, {0, 2, 3}))


if __name__ == "__main__":
    unittest.main()
