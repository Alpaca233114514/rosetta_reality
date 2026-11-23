"""Pure stdlib Q0 contract tests; never launch processes or shutdown."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_q0_screen_job.py"
SPEC = importlib.util.spec_from_file_location("q0_supervisor", SOURCE)
q0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q0)


def plan():
    candidate, module, observer = "test-candidate", "rosetta_reality.sim.candidate", "rosetta_reality.sim.observer"
    sources = ["scripts/run_q0_screen_job.py", "scripts/autodl_bounded_guard.py", "scripts/q0_bounded_guard.py", "scripts/run_autodl.sh",
               "scripts/run_teacher_gate.py", "scripts/check_env.py", "scripts/prepare_q0_dependencies.py",
               "scripts/benchmark_q0_candidate.py", q0.PROFILE,
               "src/rosetta_reality/sim/candidate.py", "src/rosetta_reality/sim/observer.py",
               "tests/test_coupled_translation_teacher.py"]
    commands = [
        ["python", "scripts/prepare_q0_dependencies.py", "--job", "{job}"],
        ["python", "-m", "pytest", "-q", "tests/test_coupled_translation_teacher.py"],
        ["python", "scripts/benchmark_q0_candidate.py", "--output", "{job}/benchmark.json"],
        ["python", "scripts/run_teacher_gate.py", "register", "--candidate-id", candidate,
         "--module", module, "--factory", "build_teacher", "--observer-module", observer, "--simulator-execution"],
    ] + [["python", "scripts/run_teacher_gate.py", stage, "--candidate-id", candidate]
         for stage in ("g0", "g1", "g2")]
    return {"total_cap_seconds": 2700, "platform_target_unix": 1900000000,
            "shutdown_authorized": True, "candidate_id": candidate, "module": module,
            "factory": "build_teacher", "observer_module": observer,
            "runtime_container_image_id": "autodl-platform-container-observed",
            "source_sha256": {source: "a" * 64 for source in sources},
            "commands": [{"stage": stage, "args": args, "seconds": 30}
                         for stage, args in zip(q0.STAGES, commands)]}


class PlanContractTests(unittest.TestCase):
    def test_valid_bounded_plan(self):
        q0.validate(plan())

    def test_unbounded_or_nonfinite_budget_is_rejected(self):
        for cap in (0, 2701, float("inf"), float("nan"), True):
            value = plan()
            value["total_cap_seconds"] = cap
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                q0.validate(value)

    def test_retry_or_later_gate_not_permitted(self):
        value = plan()
        value["commands"].append(copy.deepcopy(value["commands"][-1]))
        with self.assertRaises(ValueError):
            q0.validate(value)
        value = plan()
        value["commands"][-1]["args"][2] = "g3"
        with self.assertRaises(ValueError):
            q0.validate(value)

    def test_absolute_path_and_source_escape_rejected(self):
        for raw in ("../secret", "/root/secret", "scripts\\secret.py"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                q0.relative_path(raw)

    def test_shutdown_authorization_required(self):
        value = plan()
        value["shutdown_authorized"] = False
        with self.assertRaises(ValueError):
            q0.validate(value)


class FreshGateEvidenceTests(unittest.TestCase):
    def evidence(self, root):
        gate = root / "gate-runs" / q0.EXPERIMENT / "teacher_gate"
        (gate / "registrations").mkdir(parents=True)
        (gate / "candidate").mkdir()
        registration = gate / "registrations/candidate.json"
        q0.save(registration, {"candidate_id": "candidate"})
        report = gate / "candidate/g2-001.json"
        value = {"candidate_id": "candidate", "stage": "g2", "passed": True,
                 "registration_sha256": q0.sha(registration), "episodes": [
                     {"seed": seed, "success": True, "rollout_length": 100,
                      "joint_limit_violations": 0, "unexpected_collisions": 0, "refusals": []}
                     for seed in range(1900, 1905)]}
        return report, value

    def test_safe_five_pose_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, value = self.evidence(root)
            q0.save(report, value)
            self.assertEqual(q0.verify_gate(root, "candidate", "g2")["sha256"], q0.sha(report))

    def test_success_alone_does_not_hide_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, value = self.evidence(root)
            value["episodes"][2]["unexpected_collisions"] = 1
            q0.save(report, value)
            with self.assertRaises(ValueError):
                q0.verify_gate(root, "candidate", "g2")

    def test_create_only_and_multiple_attempts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, value = self.evidence(root)
            q0.save(report, value)
            with self.assertRaises(FileExistsError):
                q0.save(report, value)
            q0.save(report.with_name("g2-002.json"), value)
            with self.assertRaises(ValueError):
                q0.verify_gate(root, "candidate", "g2")


if __name__ == "__main__":
    unittest.main()
