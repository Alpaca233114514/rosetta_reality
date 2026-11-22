"""Pure-stdlib unit tests for generalization controls (no ML/sim/YAML imports)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _load_controls():
    path = ROOT / "src/rosetta_reality/vla/generalization_controls.py"
    spec = importlib.util.spec_from_file_location("test_generalization_controls_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_generalization_controls_mod"] = module
    spec.loader.exec_module(module)
    return module


GC = _load_controls()


class GeneralizationControlsStdlibTest(unittest.TestCase):
    def test_clamped_bounded_sine_prevents_closing_overshoot_foldback(self) -> None:
        # At z=-2.0 sigma on right_gripper (mean=-0.7329, std=0.9266), x=-2.5861 rad:
        # unclamped (sin(x)+1)/2 folds back open to ~0.238; clamped stays 0.0 closed.
        x_overshoot_close = -2.5861
        unclamped = 0.5 * (math.sin(x_overshoot_close) + 1.0)
        self.assertGreater(unclamped, 0.20)
        decoded = GC.clamped_bounded_sine_to_gripper([x_overshoot_close, -math.pi / 2, 0.0, math.pi / 2, 2.5])
        self.assertAlmostEqual(decoded[0], 0.0, places=9)
        self.assertAlmostEqual(decoded[1], 0.0, places=9)
        self.assertAlmostEqual(decoded[2], 0.5, places=9)
        self.assertAlmostEqual(decoded[3], 1.0, places=9)
        self.assertAlmostEqual(decoded[4], 1.0, places=9)

        stats = GC.bounded_sine_fold_stats([x_overshoot_close, 0.0, 2.18])
        self.assertEqual(stats["below_neg_half_pi"], 1)
        self.assertEqual(stats["above_pos_half_pi"], 1)
        self.assertEqual(stats["total_folds"], 2)
        self.assertGreater(stats["max_decode_delta"], 0.20)

    def test_apply_clamped_override_preserves_arm_joints(self) -> None:
        raw = [0.1] * 14
        raw[6] = 0.25
        raw[13] = 0.24
        internal = [0.0] * 14
        internal[6] = 2.2   # overshoot open -> clamped 1.0
        internal[13] = -2.5  # overshoot closed -> clamped 0.0
        fixed = GC.apply_clamped_bounded_sine_override(raw, internal)
        self.assertEqual(fixed[:6], [0.1] * 6)
        self.assertAlmostEqual(fixed[6], 1.0, places=9)
        self.assertEqual(fixed[7:13], [0.1] * 6)
        self.assertAlmostEqual(fixed[13], 0.0, places=9)

    def test_temporal_ensembler_reduces_white_noise_chattering(self) -> None:
        rng = random.Random(20261122)
        chunk_size = 20
        raw_50_1 = []
        fused_te = []
        te = GC.TemporalEnsembler(chunk_size=chunk_size, decay_k=0.01, action_dim=14)
        for t in range(120):
            # Smooth linear ramp + independent per-inference Gaussian noise
            noise = [rng.gauss(0.0, 0.05) for _ in range(14)]
            chunk = [
                [0.01 * (t + slot) + noise[j] for j in range(14)]
                for slot in range(chunk_size)
            ]
            raw_50_1.append(chunk[0])
            fused_te.append(te.update(chunk))

        dyn_raw = GC.compute_trajectory_dynamics(raw_50_1)
        dyn_te = GC.compute_trajectory_dynamics(fused_te)
        # Raw 50/1 suffers near-white-noise sign flips; TemporalEnsembler drastically lowers chattering.
        self.assertGreater(dyn_raw["arm_signflip_rate"], 0.50)
        self.assertLess(dyn_te["arm_signflip_rate"], 0.25)
        self.assertLess(dyn_te["mean_arm_step_delta"], dyn_raw["mean_arm_step_delta"] * 0.5)
        self.assertGreater(dyn_te["straightness_tail"], dyn_raw["straightness_tail"] * 2.0)

    def test_stratified_4d_workspace_coverage_and_disjoint_splits(self) -> None:
        poses_40 = GC.generate_stratified_4d_poses(40, seed=1)
        poses_400 = GC.generate_stratified_4d_poses(400, seed=1)
        cov_40 = GC.audit_4d_workspace_coverage(poses_40, target_mean_nn_mm=20.0)
        cov_400 = GC.audit_4d_workspace_coverage(poses_400, target_mean_nn_mm=20.0)
        self.assertGreater(cov_40["mean_nn1_mm"], cov_400["mean_nn1_mm"])
        self.assertTrue(cov_400["meets_target"])

        splits = GC.split_episodes_disjoint(range(100), val_count=10, test_count=10, seed=7)
        self.assertEqual(len(splits["train"]), 80)
        self.assertEqual(len(splits["val"]), 10)
        self.assertEqual(len(splits["test"]), 10)
        self.assertEqual(len(set(splits["train"]) | set(splits["val"]) | set(splits["test"])), 100)

    def test_multiview_camera_contract(self) -> None:
        res = GC.validate_multiview_camera_contract(
            ["observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"],
            empty_cameras=0,
        )
        self.assertTrue(res["valid"])
        with self.assertRaises(ValueError):
            GC.validate_multiview_camera_contract(["observation.images.top"], empty_cameras=2)
        with self.assertRaises(ValueError):
            GC.validate_multiview_camera_contract(
                ["observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"],
                empty_cameras=1,
            )


if __name__ == "__main__":
    unittest.main()
