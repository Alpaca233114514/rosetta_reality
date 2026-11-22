"""Stdlib regression tests for P0-A v2 paired boundary pilot (zero ML/sim imports)."""
from __future__ import annotations

import copy
import dataclasses
import enum
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paired_boundary_pilot_v2", ROOT / "scripts/paired_boundary_pilot_v2.py"
)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class _KeyEnum(enum.Enum):
    ACTION = "action"
    OBSERVATION = "observation"


@dataclasses.dataclass
class _FakeFeature:
    shape: tuple[int, ...]


class PairedBoundaryPilotV2Test(unittest.TestCase):
    def test_source_only_import_has_no_ml_or_sim(self) -> None:
        self.assertNotIn("torch", pilot.__dict__)
        self.assertNotIn("mujoco", pilot.__dict__)
        self.assertNotIn("numpy", pilot.__dict__)

    def test_sanitize_observation_drops_raw_numpy_buffers(self) -> None:
        obs = {
            "robot_state": [0.0] * 14,
            "images": {"top": [[1, 2]]},
            "raw": {"agent_pos": object(), "pixels": {"top": object()}},
        }
        cleaned = pilot.sanitize_observation(obs)
        self.assertEqual(set(cleaned.keys()), {"robot_state", "images"})
        self.assertNotIn("raw", cleaned)

    def test_plain_config_handles_paths_sets_dataclasses_and_enums(self) -> None:
        safe = lambda s: {"hashed": True, "len": len(s)} if s.startswith("/") else s
        payload = {
            "path": Path("/root/autodl-tmp/secret"),
            "rel": Path("relative/file"),
            "feature": _FakeFeature(shape=(14,)),
            "enum": _KeyEnum.ACTION,
            "tags": {"b", "a"},
        }
        out = pilot.plain_config(payload, safe_string_fn=safe, clone_tree_fn=copy.deepcopy)
        self.assertEqual(out["path"], {"hashed": True, "len": len("/root/autodl-tmp/secret")})
        self.assertEqual(out["rel"], "relative/file")
        self.assertEqual(out["feature"], {"shape": [14]})
        self.assertEqual(out["enum"], "action")
        self.assertEqual(out["tags"], ["a", "b"])

    def test_processors_captures_live_tensor_stats(self) -> None:
        class _Step:
            _registry_name = "normalizer_processor"

            def __init__(self) -> None:
                self.stats = None
                self._tensor_stats = {"observation.state": {"mean": [0.1, 0.2]}}
                self.features = {"observation.state": _FakeFeature((14,))}
                self.norm_map = {"STATE": "MEAN_STD"}

            def get_config(self) -> dict:
                return {"eps": 1e-8}

        class _Pipe:
            def __init__(self, steps):
                self.steps = steps

        class _Online:
            preprocessor = _Pipe([_Step()])
            postprocessor = _Pipe([])

        rows = pilot.processors(_Online(), safe_string_fn=lambda s: s, clone_tree_fn=copy.deepcopy)
        self.assertEqual(rows["preprocessor"][0]["stats"], None)
        self.assertEqual(
            rows["preprocessor"][0]["tensor_stats"],
            {"observation.state": {"mean": [0.1, 0.2]}},
        )
        self.assertEqual(rows["preprocessor"][0]["norm_map"], {"STATE": "MEAN_STD"})

    def test_observe_pipeline_key_supports_enum_action_postprocessor(self) -> None:
        class _Unnorm:
            _registry_name = "unnormalizer_processor"

            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, transition):
                self.calls += 1
                out = dict(transition)
                out[_KeyEnum.ACTION] = [x * 2.0 for x in transition[_KeyEnum.ACTION]]
                return out

        step = _Unnorm()
        original = _Unnorm.__call__

        class _Pipe:
            steps = [step]

        with pilot.observe_pipeline_key(_Pipe(), copy.deepcopy, target_key="action") as rows:
            res = step({_KeyEnum.ACTION: [1.0, -2.0]})
        self.assertEqual(step.calls, 1)
        self.assertEqual(res[_KeyEnum.ACTION], [2.0, -4.0])
        self.assertEqual(rows[0]["before"], [1.0, -2.0])
        self.assertEqual(rows[0]["after"], [2.0, -4.0])
        self.assertEqual(rows[0]["registry"], "unnormalizer_processor")
        self.assertIs(_Unnorm.__call__, original)

    def test_bounded_sine_fold_report_detects_folding_above_and_below_half_pi(self) -> None:
        slot_safe = [0.0] * 14
        slot_safe[6] = 1.5
        slot_safe[13] = -1.5
        slot_fold = [0.0] * 14
        slot_fold[6] = 1.7  # > +pi/2
        slot_fold[13] = -2.1  # < -pi/2
        report = pilot.bounded_sine_fold_report([slot_safe, slot_fold])
        self.assertTrue(report["any_folded"])
        self.assertFalse(report["left_gripper"]["slot0_folded"])
        self.assertEqual(report["left_gripper"]["above_half_pi_slots"], [1])
        self.assertEqual(report["right_gripper"]["below_neg_half_pi_slots"], [1])
        self.assertEqual(report["right_gripper"]["folded_slot_count"], 1)

    def test_bounded_sine_fold_report_rejects_nonfinite(self) -> None:
        bad = [0.0] * 14
        bad[13] = math.nan
        with self.assertRaises(ValueError):
            pilot.bounded_sine_fold_report([bad])


if __name__ == "__main__":
    unittest.main()
