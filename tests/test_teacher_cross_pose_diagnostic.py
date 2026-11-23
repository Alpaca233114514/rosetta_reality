"""Stdlib fake-harness checks; no simulator/model/data imports."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "scripts/diagnose_teacher_cross_pose.py"
SPEC = importlib.util.spec_from_file_location("teacher_cross_pose_diagnostic", SOURCE)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class Contract:
    dimension_names = ("joint",)

    def validate_tensor(self, action, *, allow_chunk):
        if len(action) != 1:
            raise ValueError("shape")

    def clip(self, action):
        return [min(1., max(-1., action[0]))], [abs(action[0]) > 1.]


class FakeEnvironment:
    def __init__(self, contract, maximum_episode_steps):
        self.steps = 0
        self.seed = None

    def reset(self, *, seed):
        self.seed = seed

    def state_limit_violation_count(self):
        return int(self.seed == 1900 and self.steps > 0 and mode == "safety")

    def unexpected_collision_count(self):
        return 0

    def step(self, action):
        self.steps += 1
        calls.append(self.seed)
        # reward 3 at seed 1900 must never substitute for is_success.
        return {}, 3., True, {"is_success": self.seed != 1900}

    def contact_pairs(self):
        return [("peg", "table")]

    def close(self):
        pass


class FakeTeacher:
    def reset(self):
        pass

    def decide(self, observation):
        if mode == "refusal" and observation["seed"] == 1900:
            return SimpleNamespace(refusal={"reason": "off_support"}, action=None)
        action = [float("nan")] if mode == "nan" else [2.]
        return SimpleNamespace(refusal=None, action=action)


calls = []
mode = "failure"


class CrossPoseTests(unittest.TestCase):
    def setUp(self):
        global calls, mode
        calls = []
        mode = "failure"
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "cross-pose"

    def run_case(self):
        return diagnostic.run_diagnostic(
            self.output, contract=Contract(), environment_factory=FakeEnvironment,
            teacher_factory=FakeTeacher, observer=lambda env, _: {"seed": env.seed},
            max_steps=2)

    def test_failure_continues_and_reward_three_is_not_success(self):
        result = self.run_case()
        self.assertEqual(calls, list(diagnostic.SEEDS))
        self.assertFalse(result["episodes"][0]["success"])
        self.assertEqual(result["successes"], 4)
        self.assertEqual(result["registered_attempts"], 5)
        self.assertFalse(result["provider_qualified"])
        self.assertIsNone(result["formal_gate_result"])
        self.assertEqual(result["episodes"][0]["action_clip_entry_count"], 1)

    def test_safety_stop_continues(self):
        global mode
        mode = "safety"
        result = self.run_case()
        self.assertEqual(result["episodes"][0]["status"], "safety_stop")
        self.assertEqual(calls, list(diagnostic.SEEDS))

    def test_refusal_continues(self):
        global mode
        mode = "refusal"
        result = self.run_case()
        self.assertEqual(result["episodes"][0]["status"], "refused")
        self.assertEqual(calls, list(diagnostic.SEEDS[1:]))

    def test_nonfinite_stops_all_before_step(self):
        global mode
        mode = "nan"
        result = self.run_case()
        self.assertEqual(calls, [])
        self.assertEqual(result["status"], "infra_failed")
        self.assertEqual(result["measured_attempts"], 1)
        self.assertEqual(len(result["episodes"]), 5)
        self.assertIsNone(result["episodes"][1]["success"])

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "preserved.txt"
        marker.write_text("evidence", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_case()
        self.assertEqual(marker.read_text(encoding="utf-8"), "evidence")
        self.assertEqual(calls, [])

    def test_internal_exception_stops_all(self):
        def fail_observer(*args):
            raise RuntimeError("internal failure")
        result = diagnostic.run_diagnostic(
            self.output, contract=Contract(), environment_factory=FakeEnvironment,
            teacher_factory=FakeTeacher, observer=fail_observer)
        self.assertEqual(result["status"], "infra_failed")
        self.assertEqual(len(result["episodes"]), 5)
        self.assertEqual(calls, [])

    def test_expired_deadline_preserves_unmeasured_seeds(self):
        ticks = iter([0., 601., 601.])
        result = diagnostic.run_diagnostic(
            self.output, contract=Contract(), environment_factory=FakeEnvironment,
            teacher_factory=FakeTeacher, observer=lambda env, _: {},
            clock=lambda: next(ticks))
        self.assertEqual(result["status"], "time_budget_exhausted")
        self.assertEqual(result["measured_attempts"], 0)
        self.assertTrue(all(item["success"] is None for item in result["episodes"]))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
