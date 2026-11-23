"""Independent batch semantics without importing model/simulator/data libraries."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/diagnose_smolvla_base_five_scene.py"
SPEC = importlib.util.spec_from_file_location("base_five_scene_diagnostic", SOURCE)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class BaseFiveSceneTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "new-run"

    def test_task_and_physical_failures_do_not_skip_independent_seeds(self):
        calls = []
        def episode(seed, path):
            calls.append(seed)
            return {"seed": seed, "outcome": "physical_safety_stop" if seed == 1001 else "task_failed",
                    "success": False, "maximum_reward": 3}
        result = diagnostic.run_batch(self.output, episode, {})
        self.assertEqual(calls, list(diagnostic.SEEDS))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["success_count"], 0)
        self.assertEqual(result["registered_seed_count"], 5)
        self.assertIsNone(result["formal_gate_result"])
        self.assertFalse(result["original_gate4_protocol_equivalent"])
        self.assertFalse(result["shutdown_requested"])

    def test_numerical_or_infrastructure_fault_keeps_unmeasured_denominator(self):
        def episode(seed, path):
            if seed == 1001:
                raise ValueError("nonfinite model output")
            return {"seed": seed, "outcome": "task_success", "success": True}
        result = diagnostic.run_batch(self.output, episode, {})
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["measured_seed_count"], 1)
        self.assertEqual(result["attempted_seeds"], [1000, 1001])
        self.assertEqual(result["incomplete_seeds"], [1001])
        self.assertEqual(result["unmeasured_seeds"], [1001, 1002, 1003, 1004])

    def test_output_is_create_only(self):
        self.output.mkdir()
        marker = self.output / "old-evidence"
        marker.write_text("preserved")
        with self.assertRaises(FileExistsError):
            diagnostic.run_batch(self.output, lambda *_: self.fail("must not execute"), {})
        self.assertEqual(marker.read_text(), "preserved")

    def test_synthetic_result_cannot_claim_passed_gate(self):
        result = diagnostic.run_batch(self.output,
            lambda seed, _: {"seed": seed, "outcome": "task_success", "success": True}, {})
        self.assertEqual(result["success_count"], 5)
        self.assertEqual(result["historical_base_gate3_status"], "failed")
        self.assertEqual(result["historical_base_gate4_status"], "not_measured")
        self.assertFalse(result["m2_complete"])

    def test_artifact_inventory_identity_is_required(self):
        with self.assertRaises(FileNotFoundError):
            diagnostic.verify_artifact(Path(self.directory.name) / "missing-artifact")

    def run_physical_case(self, mode):
        import torch
        ticks = [0.]
        resource = {"rss_bytes": 0, "cuda_allocated_bytes": 0,
                    "cuda_reserved_bytes": 0, "cuda_peak_allocated_bytes": 0,
                    "cuda_peak_reserved_bytes": 0}
        class Contract:
            dimension = 2
            def validate_tensor(self, action, allow_chunk=True):
                if action.shape[-1] != 2 or not torch.isfinite(action).all():
                    raise ValueError("invalid action")
            def clip(self, action):
                self.validate_tensor(action)
                clipped = action.clamp(-1, 1)
                return clipped, clipped.ne(action)
        instances = []
        class Environment:
            def __init__(self, *_args, **_kwargs):
                self.steps = 0
                self.last_clip_mask = torch.zeros(2, dtype=torch.bool)
                instances.append(self)
            def reset(self, seed):
                return {"robot_state": torch.zeros(2), "images": {"top": torch.zeros(3, 2, 2)}}
            def state_limit_violation_count(self):
                return int(mode == "reset_safety" or (mode == "safety" and self.steps == 1))
            def contact_pairs(self):
                return [("finger", "table")] if mode == "collision" and self.steps else []
            def is_unexpected_collision_pair(self, *_):
                return True
            def step(self, action):
                self.steps += 1
                self.executed = action
                if mode == "terminal_timeout" and self.steps == 2:
                    ticks[0] = 1800.
                return self.reset(0), 3., self.steps == 2, {"is_success": False}
            def close(self):
                self.closed = True
                if mode == "close_timeout":
                    ticks[0] = 1800.
        class Policy:
            def configure_noise(self, mode, seed):
                self.noise = (mode, seed)
            def predict(self, *_):
                if mode == "inference_timeout":
                    ticks[0] = 1800.
                elif mode == "inference_peak":
                    # Instantaneous occupancy is back to zero, peak retains the fault.
                    resource["cuda_peak_allocated_bytes"] = 8 * 1024**3 + 1
                elif mode == "inference_reserved_peak":
                    resource["cuda_peak_reserved_bytes"] = 10 * 1024**3 + 1
                value = float("nan") if mode == "nan" else 2.
                return torch.tensor([[value, 0.]]), torch.tensor([[1., 0.]])
        self.output.mkdir()
        plan = {"instruction": "test", "resources": {"maximum_seconds": 1800,
            "rss_bytes": 12 * 1024**3, "evidence_bytes": 256 * 1024**2,
            "cuda_allocated_bytes": 8 * 1024**3, "cuda_reserved_bytes": 10 * 1024**3}}
        with patch.dict("sys.modules", {"rosetta_reality.sim": SimpleNamespace(GymAlohaEnvironment=Environment)}):
            runner = diagnostic.make_episode_runner(Policy(), Contract(), plan, self.output, 0.,
                         clock=lambda: ticks[0], resource_reader=lambda: dict(resource))
            try:
                result = runner(1000, self.output / "trace.jsonl")
            except (ValueError, RuntimeError) as exc:
                self.case_error = exc
                return None, instances[0]
        return result, instances[0]

    def test_joint_limit_stops_immediately_and_preserves_clipped_command(self):
        result, env = self.run_physical_case("safety")
        self.assertEqual(env.steps, 1)
        self.assertTrue(env.closed)
        self.assertEqual(env.executed.tolist(), [1., 0.])
        self.assertEqual(result["outcome"], "physical_safety_stop")

    def test_unexpected_contact_stops_immediately(self):
        result, env = self.run_physical_case("collision")
        self.assertEqual(env.steps, 1)
        self.assertEqual(result["unexpected_collisions"], 1)

    def test_reset_safety_stops_before_any_command(self):
        result, env = self.run_physical_case("reset_safety")
        self.assertEqual(env.steps, 0)
        self.assertEqual(result["outcome"], "physical_safety_stop")

    def test_nonfinite_stops_before_any_command(self):
        result, env = self.run_physical_case("nan")
        self.assertIsNone(result)
        self.assertEqual(env.steps, 0)
        self.assertTrue(env.closed)

    def test_reward_does_not_impute_success(self):
        result, env = self.run_physical_case("normal")
        self.assertEqual(env.steps, 2)
        self.assertEqual(result["maximum_reward"], 3.)
        self.assertFalse(result["success"])
        self.assertEqual(result["outcome"], "task_failed")

    def test_inference_crossing_deadline_cannot_issue_action(self):
        result, env = self.run_physical_case("inference_timeout")
        self.assertIsNone(result)
        self.assertEqual(env.steps, 0)
        self.assertTrue(env.closed)
        self.assertIsInstance(self.case_error, RuntimeError)

    def test_transient_inference_allocation_peak_cannot_issue_action(self):
        result, env = self.run_physical_case("inference_peak")
        self.assertIsNone(result)
        self.assertEqual(env.steps, 0)
        self.assertTrue(env.closed)

    def test_transient_inference_reserved_peak_cannot_issue_action(self):
        result, env = self.run_physical_case("inference_reserved_peak")
        self.assertIsNone(result)
        self.assertEqual(env.steps, 0)

    def test_terminal_step_crossing_deadline_is_incomplete_and_trace_retained(self):
        result, env = self.run_physical_case("terminal_timeout")
        self.assertIsNone(result)
        self.assertEqual(env.steps, 2)
        self.assertTrue(env.closed)
        self.assertIn('"step": 2', (self.output / "trace.jsonl").read_text())

    def test_environment_close_crossing_deadline_cannot_finish(self):
        result, env = self.run_physical_case("close_timeout")
        self.assertIsNone(result)
        self.assertEqual(env.steps, 2)
        self.assertTrue(env.closed)

    def test_final_integrity_check_crossing_deadline_blocks_complete_batch(self):
        def final():
            raise RuntimeError("final hash crossed deadline")
        result = diagnostic.run_batch(self.output,
            lambda seed, _: {"seed": seed, "outcome": "task_failed", "success": False},
            {}, final_check=final)
        self.assertEqual(result["measured_seed_count"], 5)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(diagnostic.read(self.output / "completion.json")["status"], "blocked")

    def test_post_summary_budget_check_owns_final_completion_receipt(self):
        def episode(seed, _):
            return {"seed": seed, "outcome": "task_failed", "success": False}
        def check(**_):
            if (self.output / "summary.json").exists():
                raise RuntimeError("summary write crossed work budget")
        episode.check_budget = check
        result = diagnostic.run_batch(self.output, episode, {})
        self.assertEqual(result["status"], "blocked")
        receipt = diagnostic.read(self.output / "completion.json")
        self.assertEqual(receipt["status"], "blocked")
        self.assertEqual(receipt["summary_sha256"], diagnostic.sha(self.output / "summary.json"))

    def test_evidence_about_to_be_committed_is_budgeted(self):
        limits = {"maximum_seconds": 1800, "rss_bytes": 12 * 1024**3,
                  "evidence_bytes": 10, "cuda_allocated_bytes": 8 * 1024**3,
                  "cuda_reserved_bytes": 10 * 1024**3}
        resource = {"rss_bytes": 0, "cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0,
                    "cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0}
        checker = diagnostic.make_budget_checker({"resources": limits}, self.output, 0.,
                        clock=lambda: 0., resource_reader=lambda: resource)
        with self.assertRaises(RuntimeError):
            checker(additional_evidence_bytes=11)


if __name__ == "__main__":
    unittest.main()
