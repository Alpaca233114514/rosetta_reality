"""Pure stdlib contract tests; no simulator, model or platform operation."""
from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_teacher_diagnostic_job as jobmod
import work_only_guard as guard


def plan():
    target = datetime.now(timezone(timedelta(hours=8))).replace(hour=20, minute=0, second=0, microsecond=0).timestamp()
    return {"schema": jobmod.SCHEMA, "status": "prepared", "id": "teacher-diagnostic-new",
        "total_cap_seconds": 780, "platform_target_unix": target, "shutdown_authorized": False,
        "runtime_container_image_id": "sha256:abc", "prior_job_relative": jobmod.prior_batch.PRIOR_JOB,
        "prior_manifest_sha256": jobmod.prior_batch.PRIOR_MANIFEST,
        "source_sha256": {p: "0" * 64 for p in jobmod.REQUIRED_SOURCES},
        "commands": [{"stage": "tests", "seconds": 90, "args": ["python", "-m", "pytest", "-q", *jobmod.TEST_FILES]},
                     {"stage": "diagnostic", "seconds": 600, "args": list(jobmod.DIAGNOSTIC_ARGS)}]}


class TeacherJobTests(unittest.TestCase):
    def test_valid_plan(self):
        jobmod.validate(plan())

    def test_unconfirmed_platform_timer_does_not_create_deadline(self):
        candidate = plan()
        candidate["platform_target_unix"] = None
        jobmod.validate(candidate)
        self.assertEqual(jobmod.work_deadline(candidate, 2100), 2880)

    def test_limits_authority_and_prior_fixed(self):
        for key, value in (("total_cap_seconds", 781), ("total_cap_seconds", True),
                ("total_cap_seconds", float("nan")), ("shutdown_authorized", True),
                ("platform_target_unix", float("inf")), ("prior_job_relative", "runs/other"),
                ("prior_manifest_sha256", "0" * 64), ("id", "../old")):
            with self.subTest(key=key):
                candidate = plan()
                candidate[key] = value
                with self.assertRaises(ValueError):
                    jobmod.validate(candidate)

    def test_fixed_test_and_diagnostic_commands(self):
        for argv in (["python", "-c", "anything"], ["python", "scripts/run_teacher_gate.py", "g1"],
                     ["bash", "-c", "anything"], ["python", "-m", "pytest", "-q", "tests/test_unknown.py"]):
            candidate = plan()
            candidate["commands"][0]["args"] = argv
            with self.assertRaises(ValueError):
                jobmod.validate(candidate)
        candidate = plan()
        candidate["commands"][1]["seconds"] = 601
        with self.assertRaises(ValueError):
            jobmod.validate(candidate)

    def test_source_paths_and_coverage(self):
        for raw in ("../escape", "/root/escape", "C:/escape", "scripts\\escape"):
            candidate = plan()
            candidate["source_sha256"][raw] = "0" * 64
            with self.assertRaises(ValueError):
                jobmod.validate(candidate)
        candidate = plan()
        del candidate["source_sha256"]["scripts/work_only_guard.py"]
        with self.assertRaises(ValueError):
            jobmod.validate(candidate)

    def test_budget_never_extends_target_and_requires_window(self):
        candidate = plan()
        candidate["platform_target_unix"] = 2000
        self.assertEqual(jobmod.work_deadline(candidate, 1000), 1700)
        with self.assertRaises(ValueError):
            jobmod.work_deadline(candidate, 1521)
        self.assertEqual(jobmod.work_deadline(candidate, 1520), 1700)

    def test_guard_registration_rejects_unbounded_or_authorized(self):
        reg = {"id": "new", "started_unix": 100, "work_deadline_unix": 880, "shutdown_authorized": False}
        guard.validate_registration(reg)
        for key, value in (("shutdown_authorized", True), ("work_deadline_unix", 881),
                           ("started_unix", float("nan")), ("id", "")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                guard.validate_registration(dict(reg, **{key: value}))

    def test_prior_output_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(jobmod.prior_root(root / "runs/new", plan()), root / jobmod.prior_batch.PRIOR_JOB)
            with self.assertRaises(ValueError):
                jobmod.prior_root(root / jobmod.prior_batch.PRIOR_JOB, plan())

    def test_guard_identity_drift_stops_work(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            reg = {"id": "new", "plan_sha256": "0" * 64, "work_deadline_unix": 200}
            armed = dict(reg, pid=4242, start_ticks="11", shutdown_authorized=False)
            (job / "guard-armed.json").write_text(json.dumps(armed))
            with patch.object(guard.cleanup.original, "process_identity", return_value=("12", 4242)):
                with self.assertRaises(RuntimeError):
                    jobmod.guard_alive(job, reg)

    def test_no_shutdown_callable_in_new_launch_or_guard(self):
        for module in (jobmod, guard):
            source = inspect.getsource(module)
            self.assertNotIn("request_shutdown(", source)
            self.assertNotIn("/usr/bin/shutdown", source)
            self.assertNotIn("original.run(", source)

    def test_existing_job_never_written(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "old"
            job.mkdir()
            marker = job / "keep"
            marker.write_text("original")
            with self.assertRaises(FileExistsError):
                jobmod.launch(Path(directory), job, Path(directory) / "missing.json")
            self.assertEqual(marker.read_text(), "original")
            self.assertEqual(list(job.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
