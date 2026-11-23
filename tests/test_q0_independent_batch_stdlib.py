"""Pure stdlib tests for the independent batch's fail-closed boundaries."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_q0_independent_batch as batch


def plan():
    sources = {p: "0" * 64 for p in (
        "scripts/run_q0_independent_batch.py", "scripts/run_q0_screen_job.py",
        "scripts/q0_bounded_guard.py", "scripts/autodl_bounded_guard.py",
        "scripts/run_autodl.sh", "scripts/check_env.py", batch.shared.PROFILE,
        "scripts/audit_q0_screen_evidence.py", "scripts/check_q0_state_restore.py",
        "scripts/check_q0_refusal_boundaries.py", "tests/test_q0_independent_batch_stdlib.py")}
    commands = [
        ("check_env", ["python", "scripts/check_env.py"]),
        ("synthetic", ["python", "-m", "pytest", "-q", "tests/test_q0_independent_batch_stdlib.py"]),
        ("evidence_audit", ["python", "scripts/audit_q0_screen_evidence.py", "--input", "{prior_job}",
                            "--output", "{job}/evidence-audit.json"]),
        ("state_restore", ["python", "scripts/check_q0_state_restore.py", "--output", "{job}/state-restore.json"]),
        ("refusal_boundaries", ["python", "scripts/check_q0_refusal_boundaries.py", "--output", "{job}/refusal-boundaries.json"]),
    ]
    return {"schema": batch.SCHEMA, "status": "prepared", "total_cap_seconds": 600,
            "platform_target_unix": 2000000000, "shutdown_authorized": True,
            "runtime_container_image_id": "sha256:abc", "prior_job_relative": batch.PRIOR_JOB,
            "prior_manifest_sha256": batch.PRIOR_MANIFEST, "source_sha256": sources,
            "commands": [{"stage": stage, "args": argv, "seconds": 120} for stage, argv in commands]}


class BatchContractTests(unittest.TestCase):
    def test_prepared_plan_valid(self):
        batch.validate(plan())

    def test_cap_deadline_authorization_and_prior_identity(self):
        for key, value in (("total_cap_seconds", 601), ("total_cap_seconds", True),
                           ("total_cap_seconds", float("nan")), ("platform_target_unix", float("inf")),
                           ("shutdown_authorized", False), ("prior_job_relative", "runs/other"),
                           ("prior_manifest_sha256", "0" * 64), ("status", "running")):
            with self.subTest(key=key, value=value):
                candidate = plan()
                candidate[key] = value
                with self.assertRaises(ValueError):
                    batch.validate(candidate)

    def test_extra_gate_download_and_shell_argv_rejected(self):
        for argv in (["python", "scripts/run_teacher_gate.py", "g1"],
                     ["python", "scripts/prepare_q0_dependencies.py", "--job", "{job}"],
                     ["bash", "-c", "anything"],
                     ["python", "scripts/check_q0_state_restore.py", "--output", "{job}/x.json"]):
            candidate = plan()
            candidate["commands"][3]["args"] = argv
            with self.assertRaises(ValueError):
                batch.validate(candidate)

    def test_phase_order_source_coverage_and_timeout(self):
        candidate = plan()
        candidate["commands"].reverse()
        with self.assertRaises(ValueError):
            batch.validate(candidate)
        candidate = plan()
        del candidate["source_sha256"]["scripts/q0_bounded_guard.py"]
        with self.assertRaises(ValueError):
            batch.validate(candidate)
        candidate = plan()
        candidate["commands"][0]["seconds"] = 121
        with self.assertRaises(ValueError):
            batch.validate(candidate)
        candidate = plan()
        candidate["total_cap_seconds"] = 599
        with self.assertRaises(ValueError):
            batch.validate(candidate)

    def test_synthetic_whitelist(self):
        for test in ("tests/../test_escape.py", "tests/test_any.py", "-k", "tests/test_any.py;exit"):
            candidate = plan()
            candidate["commands"][1]["args"].append(test)
            with self.assertRaises(ValueError):
                batch.validate(candidate)

    def test_independent_negative_and_infrastructure_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            output = job / "state-restore.json"
            output.write_text(json.dumps({"status": "negative", "provider_qualified": False}))
            self.assertEqual(batch.classify_result(job, "state_restore", 1), "negative")
            output.write_text(json.dumps({"status": "negative", "provider_qualified": False,
                                          "infrastructure_failure": True}))
            with self.assertRaises(RuntimeError):
                batch.classify_result(job, "state_restore", 1)
            output.write_text(json.dumps({"status": "passed", "provider_qualified": False}))
            with self.assertRaises(RuntimeError):
                batch.classify_result(job, "state_restore", 1)
            output.write_text(json.dumps({"status": "passed", "provider_qualified": True}))
            with self.assertRaises(ValueError):
                batch.classify_result(job, "state_restore", 0)

    def test_sealed_overlay_inventory_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dependencies").mkdir()
            dep = root / "dependencies/dep.py"
            dep.write_bytes(b"reviewed")
            entry = {"path": "dependencies/dep.py", "bytes": 8, "sha256": batch.sha(dep)}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": [entry], "bytes": 8}))
            bound = {"prior_manifest_sha256": batch.sha(manifest)}
            self.assertEqual(batch.verify_prior(root, bound)["verified_files"], 1)
            extra = root / "dependencies/unsealed.py"
            extra.write_bytes(b"unexpected")
            with self.assertRaises(ValueError):
                batch.verify_prior(root, bound)
            extra.unlink()
            dep.write_bytes(b"drifted!")
            with self.assertRaises(ValueError):
                batch.verify_prior(root, bound)

    def test_manifest_escape_duplicate_and_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dependencies").mkdir()
            member = root / "dependencies/a.py"
            member.write_bytes(b"x")
            entry = {"path": "dependencies/a.py", "bytes": 1, "sha256": batch.sha(member)}
            for files in ([entry, entry], [dict(entry, path="../outside")]):
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({"files": files, "bytes": len(files)}))
                with self.assertRaises(ValueError):
                    batch.verify_prior(root, {"prior_manifest_sha256": batch.sha(manifest)})
            with self.assertRaises(ValueError):
                batch.verify_prior(root, {"prior_manifest_sha256": "0" * 64})

    def test_safe_error_details(self):
        message = batch.safe_message(ValueError("bad manifest member dependency/foo.py at /root/private/file\nC:\\Users\\Private\\file"))
        self.assertIn("dependency/foo.py", message)
        self.assertNotIn("/root/private", message)
        self.assertNotIn("Users", message)

    def test_prior_job_cannot_alias_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(batch.prior_root(root / "runs/new", plan()), root / batch.PRIOR_JOB)
            with self.assertRaises(ValueError):
                batch.prior_root(root / batch.PRIOR_JOB, plan())
            with self.assertRaises(ValueError):
                batch.prior_root(root / "not-runs/new", plan())

    def test_worker_negative_continues_but_safety_stop_aborts(self):
        class Child:
            def __init__(self, code):
                self.returncode = code
                self.stdout = io.BytesIO(b"synthetic process output\n")
            def poll(self):
                return self.returncode
            def wait(self, timeout):
                return self.returncode
        for safety_stop in (False, True):
            with self.subTest(safety_stop=safety_stop), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                job = root / "runs/new"
                job.mkdir(parents=True)
                plan_path = root / "plan.json"
                plan_path.write_text(json.dumps(plan()))
                batch.save(job / "registration.json", {"plan_sha256": batch.sha(plan_path),
                           "work_deadline_unix": batch.time.time() + 600})
                batch.save(job / "evidence-audit.json", {"passed": True, "provider_qualified": False})
                batch.save(job / "state-restore.json", {"status": "unsupported", "passed": False,
                           "provider_qualified": False, "safety_stop": safety_stop,
                           "infrastructure_failure": False})
                batch.save(job / "refusal-boundaries.json", {"status": "passed", "provider_qualified": False})
                calls = []
                def spawn(argv, **kwargs):
                    calls.append(argv)
                    return Child(2 if "scripts/check_q0_state_restore.py" in argv else 0)
                read_text = Path.read_text
                def safe_read(path, *args, **kwargs):
                    if str(path) == "/sys/fs/cgroup/memory.max":
                        return "8589934592"
                    return read_text(path, *args, **kwargs)
                with patch.object(batch.os, "getpgrp", return_value=batch.os.getpid()), \
                     patch.object(batch.signal, "signal"), patch.object(batch.sys, "version_info", (3, 12)), \
                     patch.object(batch.sys, "platform", "linux"), patch.object(batch.shared, "check_sources"), \
                     patch.object(batch, "verify_prior", return_value={"verified_files": 127}), \
                     patch.object(batch.shared, "resources", return_value={}), \
                     patch.object(batch.shared, "stop_children"), patch.object(batch.os, "readlink", return_value="mnt:[1]"), \
                     patch.object(batch.os, "sync", create=True), patch.object(Path, "read_text", safe_read), \
                     patch.object(batch.subprocess, "Popen", side_effect=spawn):
                    batch.worker(root, job, plan_path)
                if safety_stop:
                    self.assertEqual(len(calls), 4)
                    self.assertTrue((job / "failure.json").exists())
                    self.assertFalse((job / "result.json").exists())
                else:
                    self.assertEqual(len(calls), 5)
                    result = json.loads((job / "result.json").read_text())
                    self.assertEqual(result["logical_negatives"], ["state_restore"])
                self.assertTrue((job / "manifest.json").exists())
                self.assertTrue((job / "worker-exited.json").exists())


if __name__ == "__main__":
    unittest.main()
