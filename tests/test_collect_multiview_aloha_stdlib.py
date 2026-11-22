"""Pure-stdlib unit tests for ClampedBoundedSineFeature and multi-view dataset collector."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CBS = _load_module(
    "test_clamped_bounded_sine_mod",
    "src/rosetta_reality/vla/training/clamped_bounded_sine.py",
)
COL = _load_module(
    "test_collect_multiview_aloha_mod",
    "scripts/collect_multiview_aloha_dataset.py",
)


class _FakeContract:
    def clip(self, action):
        return list(action), [False] * len(action)


class _FakeEnv:
    def __init__(self, *, steps_to_success: int = 25, collide: bool = False):
        self.step_count = 0
        self.steps_to_success = steps_to_success
        self.collide = collide
        self.closed = False

    def reset(self, *, seed: int):
        self.step_count = 0
        return {"robot_state": [0.0] * 14}

    def step(self, action):
        self.step_count += 1
        success = self.step_count >= self.steps_to_success
        reward = 4.0 if success else 0.0
        return {"robot_state": list(action)}, reward, success, {"is_success": success}

    def diagnostic_snapshot(self):
        return {"joint_limit_violations": []}

    def contact_pairs(self):
        return (("table", "vx300s_right/10_right_gripper_finger"),) if self.collide else ()

    def is_unexpected_collision_pair(self, first, second):
        return "table" in (first, second)

    def close(self):
        self.closed = True


class _SmoothTeacher:
    def __init__(self, *, chattering: bool = False):
        self.t = 0
        self.chattering = chattering

    def reset(self):
        self.t = 0

    def act(self, obs):
        self.t += 1
        if self.chattering:
            sign = 1.0 if (self.t % 2 == 0) else -1.0
            return [sign * 0.1] * 14
        return [0.01 * self.t] * 14


class ClampedBoundedSineAndMultiviewCollectorTest(unittest.TestCase):
    def test_clamped_bounded_sine_feature_install_clamp_and_fail_closed_restore(self) -> None:
        class _FakeStep:
            _registry_name = "rosetta_pi_aloha_postprocessor"

            def action(self, values):
                # Simulate sin decoding of gripper slots 6 and 13
                out = list(values)
                out[6] = 0.5 * (math.sin(out[6]) + 1.0)
                out[13] = 0.5 * (math.sin(out[13]) + 1.0)
                return out

        class _FakePost:
            def __init__(self):
                self.steps = [_FakeStep()]

        def orig_factory(*args, **kwargs):
            return object(), _FakePost()

        fake_lerobot = types.ModuleType("lerobot")
        fake_scripts = types.ModuleType("lerobot.scripts")
        fake_train = types.ModuleType("lerobot.scripts.lerobot_train")
        fake_train.make_pre_post_processors = orig_factory

        saved = {
            k: sys.modules.get(k)
            for k in ("lerobot", "lerobot.scripts", "lerobot.scripts.lerobot_train")
        }
        sys.modules["lerobot"] = fake_lerobot
        sys.modules["lerobot.scripts"] = fake_scripts
        sys.modules["lerobot.scripts.lerobot_train"] = fake_train
        try:
            feature = CBS.ClampedBoundedSineFeature({})
            feature.install(context=None)
            _, post = fake_train.make_pre_post_processors()
            vec = [0.0] * 14
            vec[6] = 2.4    # overshoot open -> clamped +pi/2 -> 1.0
            vec[13] = -2.6  # overshoot closed -> clamped -pi/2 -> 0.0 (no fold-back!)
            decoded = post.steps[0].action(vec)
            self.assertAlmostEqual(decoded[6], 1.0, places=9)
            self.assertAlmostEqual(decoded[13], 0.0, places=9)

            # Fail closed if another hook overwrote make_pre_post_processors before restore
            installed = fake_train.make_pre_post_processors
            fake_train.make_pre_post_processors = lambda *a, **k: (None, None)
            with self.assertRaises(RuntimeError):
                feature.restore(context=None)

            fake_train.make_pre_post_processors = installed
            feature.restore(context=None)
            self.assertIs(fake_train.make_pre_post_processors, orig_factory)
        finally:
            for k, old in saved.items():
                if old is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = old

    def test_resolve_verified_cameras_fails_closed_when_missing(self) -> None:
        class _FakeModel:
            ncam = 2
            _cams = ("top", "angle")

            def id2name(self, cid, kind):
                return self._cams[cid]

        class _FakePhysics:
            model = _FakeModel()

        self.assertEqual(COL.resolve_verified_cameras(_FakePhysics(), ["top", "angle"]), ["top", "angle"])
        with self.assertRaises(ValueError):
            COL.resolve_verified_cameras(_FakePhysics(), ["top", "left_wrist"])

    def test_collect_episode_accepts_smooth_safe_and_rejects_chattering_or_colliding(self) -> None:
        pose = COL.Pose4D(socket_x=-0.13, socket_y=0.49, peg_x=0.13, peg_y=0.50)

        def fake_render(env, cams, frames_dir, step_idx):
            p = frames_dir / f"step-{step_idx:04d}.json"
            p.write_text(json.dumps({"cams": cams, "step": step_idx}), encoding="utf-8")
            return f"frames/{p.name}", COL.sha256_file(p)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = COL.collect_episode(
                episode_index=0,
                seed=100,
                pose=pose,
                camera_names=["top", "angle", "front_close"],
                output_dir=root,
                maximum_steps=40,
                env_factory=lambda ms: (_FakeContract(), _FakeEnv(steps_to_success=25, collide=False)),
                teacher_factory=lambda: _SmoothTeacher(chattering=False),
                frame_renderer=fake_render,
                pose_placer=lambda env, p: None,
                observation_builder=lambda env, c: {},
            )
            self.assertTrue(good["quality"]["accepted"])
            self.assertEqual(good["steps"], 25)
            self.assertEqual(good["dynamics"]["arm_signflip_rate"], 0.0)
            self.assertTrue((root / good["trace_relpath"]).is_file())

            bad_chat = COL.collect_episode(
                episode_index=1,
                seed=101,
                pose=pose,
                camera_names=["top", "angle", "front_close"],
                output_dir=root,
                maximum_steps=40,
                env_factory=lambda ms: (_FakeContract(), _FakeEnv(steps_to_success=25, collide=False)),
                teacher_factory=lambda: _SmoothTeacher(chattering=True),
                frame_renderer=fake_render,
                pose_placer=lambda env, p: None,
                observation_builder=lambda env, c: {},
            )
            self.assertFalse(bad_chat["quality"]["accepted"])
            self.assertTrue(any("arm_signflip_rate" in r for r in bad_chat["quality"]["rejection_reasons"]))

            bad_coll = COL.collect_episode(
                episode_index=2,
                seed=102,
                pose=pose,
                camera_names=["top", "angle", "front_close"],
                output_dir=root,
                maximum_steps=40,
                env_factory=lambda ms: (_FakeContract(), _FakeEnv(steps_to_success=25, collide=True)),
                teacher_factory=lambda: _SmoothTeacher(chattering=False),
                frame_renderer=fake_render,
                pose_placer=lambda env, p: None,
                observation_builder=lambda env, c: {},
            )
            self.assertFalse(bad_coll["quality"]["accepted"])
            self.assertTrue(any("unexpected_collisions" in r for r in bad_coll["quality"]["rejection_reasons"]))


if __name__ == "__main__":
    unittest.main()
