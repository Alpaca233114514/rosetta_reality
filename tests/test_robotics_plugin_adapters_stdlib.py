"""Offline stdlib tests for Robotics plugin adapters (no torch, MuJoCo, or network)."""

from __future__ import annotations

import array
import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCORECARD = _load_module(
    "rosetta_paired_scorecard_test",
    "src/rosetta_reality/eval/paired_scorecard.py",
)
DIAG = _load_module(
    "rosetta_mujoco_diagnostics_test",
    "src/rosetta_reality/sim/mujoco_diagnostics.py",
)


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOpt:
    def __init__(self, enableflags: int = 0) -> None:
        self.enableflags = enableflags


class _FakeMjModel:
    def __init__(
        self,
        *,
        body_names: list[str],
        body_mass: list[float],
        jnt_type: list[int],
        jnt_bodyid: list[int],
        geom_names: list[str],
        geom_bodyid: list[int],
        geom_margin: list[float],
        geom_gap: list[float],
        eq_type: list[int] | None = None,
        eq_obj1id: list[int] | None = None,
        eq_obj2id: list[int] | None = None,
        actuator_names: list[str] | None = None,
        actuator_trntype: list[int] | None = None,
        actuator_trnid: list[tuple[int, int]] | None = None,
        enableflags: int = 0,
        serialized: bytes = b"aloha-v1-model-bytes",
    ) -> None:
        self._body_names = body_names
        self.nbody = len(body_names)
        self.body_mass = body_mass
        self.njnt = len(jnt_type)
        self.jnt_type = jnt_type
        self.jnt_bodyid = jnt_bodyid
        self._geom_names = geom_names
        self.ngeom = len(geom_names)
        self.geom_bodyid = geom_bodyid
        self.geom_margin = geom_margin
        self.geom_gap = geom_gap
        self.eq_type = eq_type or []
        self.eq_obj1id = eq_obj1id or []
        self.eq_obj2id = eq_obj2id or []
        self.neq = len(self.eq_type)
        self._actuator_names = actuator_names or []
        self.actuator_trntype = actuator_trntype or []
        self.actuator_trnid = actuator_trnid or []
        self.nu = len(self.actuator_trntype)
        self.opt = _FakeOpt(enableflags)
        self.serialized = serialized

    def body(self, index: int) -> _Named:
        return _Named(self._body_names[index])

    def geom(self, index: int) -> _Named:
        return _Named(self._geom_names[index])

    def actuator(self, index: int) -> _Named:
        return _Named(self._actuator_names[index])


class _FakeMjData:
    def __init__(self, state: list[float]) -> None:
        self.state = list(state)
        self.forward_calls = 0


class _FakeRenderer:
    def __init__(self, model, height: int, width: int) -> None:
        self.model = model
        self.height = height
        self.width = width
        self.closed = False
        self.depth_enabled = False
        self.last_camera = None
        self._buffer = [1, 2, 3]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    def enable_depth_rendering(self) -> None:
        self.depth_enabled = True

    def disable_depth_rendering(self) -> None:
        self.depth_enabled = False

    def update_scene(self, data, camera=None) -> None:
        self.last_camera = camera

    def render(self):
        return self._buffer


def _make_fake_mujoco(renderer_box: list[_FakeRenderer] | None = None):
    mj = types.SimpleNamespace()
    mj.mjtJoint = types.SimpleNamespace(mjJNT_FREE=0, mjJNT_HINGE=3)
    mj.mjtEq = types.SimpleNamespace(mjEQ_CONNECT=0, mjEQ_WELD=1, mjEQ_JOINT=2)
    mj.mjtTrn = types.SimpleNamespace(mjTRN_JOINT=0, mjTRN_TENDON=2)
    mj.mjtEnableBit = types.SimpleNamespace(mjENBL_OVERRIDE=1 << 0)
    mj.mjtState = types.SimpleNamespace(mjSTATE_INTEGRATION=0x7F)

    def mj_sizeModel(model: _FakeMjModel) -> int:
        return len(model.serialized)

    def mj_saveModel(model: _FakeMjModel, _filename, buf: bytearray) -> None:
        buf[:] = model.serialized

    def mj_stateSize(_model: _FakeMjModel, _spec: int) -> int:
        return 4

    def mj_getState(_model: _FakeMjModel, data: _FakeMjData, buf, _spec: int) -> None:
        for idx, value in enumerate(data.state):
            buf[idx] = value

    def mj_setState(_model: _FakeMjModel, data: _FakeMjData, buf, _spec: int) -> None:
        data.state = [float(buf[i]) for i in range(len(buf))]

    def mj_forward(_model: _FakeMjModel, data: _FakeMjData) -> None:
        data.forward_calls += 1

    def renderer_factory(model, height: int, width: int):
        renderer = _FakeRenderer(model, height=height, width=width)
        if renderer_box is not None:
            renderer_box.append(renderer)
        return renderer

    mj.mj_sizeModel = mj_sizeModel
    mj.mj_saveModel = mj_saveModel
    mj.mj_stateSize = mj_stateSize
    mj.mj_getState = mj_getState
    mj.mj_setState = mj_setState
    mj.mj_forward = mj_forward
    mj.Renderer = renderer_factory
    return mj


class _FakeGymAlohaEnv:
    def __init__(self, model: _FakeMjModel, data: _FakeMjData) -> None:
        physics = types.SimpleNamespace(
            model=types.SimpleNamespace(ptr=model),
            data=types.SimpleNamespace(ptr=data),
        )
        control_env = types.SimpleNamespace(physics=physics)
        self.raw_environment = types.SimpleNamespace(
            unwrapped=types.SimpleNamespace(_env=control_env)
        )
        self._data = data
        self._collisions = 0
        self._limits = 0

    def reset(self, *, seed: int | None = None):
        self._data.state = [float(seed or 0), 0.0, 0.0, 0.0]
        return {"robot_state": list(self._data.state)}

    def step(self, action):
        delta = float(action[0])
        self._data.state[1] += delta
        self._data.state[2] += 1.0
        reward = 1.0 if self._data.state[1] >= 2.0 else 0.0
        done = reward >= 1.0
        self._collisions = 1 if delta < 0 else 0
        return (
            {"robot_state": list(self._data.state)},
            reward,
            done,
            {"pos": self._data.state[1]},
        )

    def unexpected_collision_count(self) -> int:
        return self._collisions

    def state_limit_violation_count(self) -> int:
        return self._limits


class PairedScorecardTests(unittest.TestCase):
    def test_import_has_no_heavy_side_effects(self) -> None:
        code = "\n".join(
            [
                "import importlib.util, sys",
                "sys.path.insert(0, 'src')",
                "import rosetta_reality.eval.paired_scorecard",
                "spec = importlib.util.spec_from_file_location(",
                "    'mujoco_diag_isolated', 'src/rosetta_reality/sim/mujoco_diagnostics.py'",
                ")",
                "mod_obj = importlib.util.module_from_spec(spec)",
                "sys.modules[spec.name] = mod_obj",
                "spec.loader.exec_module(mod_obj)",
                "for mod in ('torch', 'mujoco', 'gym_aloha', 'mujoco_playground', 'lerobot'):",
                "    assert mod not in sys.modules, mod",
            ]
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_exact_mcnemar_better_and_indistinguishable(self) -> None:
        digest = "a" * 64
        # 6 discordant wins for b -> p = 1 / 64 = 0.015625 < 0.025 -> BETTER
        base = [
            SCORECARD.EpisodeScoreRecord("aloha", "T0", i, 1000 + i, digest, False)
            for i in range(6)
        ]
        cand = [
            SCORECARD.EpisodeScoreRecord("aloha", "T0", i, 1000 + i, digest, True)
            for i in range(6)
        ]
        comp = SCORECARD.compare_paired_episodes(base, cand)
        self.assertEqual(comp.verdict, "BETTER")
        self.assertAlmostEqual(comp.p_better, 1.0 / 64.0)
        self.assertEqual(comp.vetoes, 0)

        # Only 5 discordant wins -> p = 1 / 32 = 0.03125 >= 0.025 -> INDISTINGUISHABLE
        comp_small = SCORECARD.compare_paired_episodes(base[:5], cand[:5])
        self.assertEqual(comp_small.verdict, "INDISTINGUISHABLE")
        self.assertAlmostEqual(comp_small.p_better, 1.0 / 32.0)

    def test_catastrophic_safety_veto_overrides_task_gain(self) -> None:
        digest = "b" * 64
        base = [
            SCORECARD.EpisodeScoreRecord("aloha", "T0", i, 1000 + i, digest, False)
            for i in range(7)
        ]
        cand = [
            SCORECARD.EpisodeScoreRecord(
                "aloha",
                "T0",
                i,
                1000 + i,
                digest,
                True,
                unexpected_collision_count=1 if i == 0 else 0,
            )
            for i in range(7)
        ]
        comp = SCORECARD.compare_paired_episodes(base, cand)
        self.assertEqual(comp.verdict, "WORSE")
        self.assertEqual(comp.reason, "catastrophic_safety_veto")
        self.assertEqual(comp.vetoes, 1)
        self.assertEqual(comp.b_only, 6)  # episode 0 failed safe_pass due to collision

    def test_invalid_on_scene_digest_mismatch_or_missing(self) -> None:
        rec_a = [SCORECARD.EpisodeScoreRecord("aloha", "T0", 0, 1000, "1" * 64, False)]
        rec_b = [SCORECARD.EpisodeScoreRecord("aloha", "T0", 0, 1000, "2" * 64, True)]
        self.assertEqual(SCORECARD.compare_paired_episodes(rec_a, rec_b).verdict, "INVALID")

        rec_no_sha = [SCORECARD.EpisodeScoreRecord("aloha", "T0", 0, 1000, "", True)]
        self.assertEqual(
            SCORECARD.compare_paired_episodes(rec_no_sha, rec_no_sha).reason,
            "missing_scene_sha256",
        )

    def test_gate_conversion_and_json_roundtrip(self) -> None:
        rec = SCORECARD.record_from_gate_episode(
            {
                "seed": 1002,
                "task_success": False,
                "unexpected_collision_count": 3,
                "joint_limit_violation_count": 1,
                "max_reward": 2.0,
                "rollout_length": 500,
            },
            episode_index=2,
            scene_sha256="c" * 64,
        )
        self.assertFalse(rec.safe)
        self.assertIn("unexpected_collision", rec.violations)
        self.assertIn("joint_limit_violation", rec.violations)
        serialized = json.dumps(SCORECARD.records_to_json_list([rec]))
        restored = SCORECARD.records_from_json_list(json.loads(serialized))
        self.assertEqual(restored, [rec])
        summary = SCORECARD.summarize_episodes(restored)
        self.assertEqual(summary["T0/success_rate"], 0.0)
        self.assertEqual(summary["T0/safe_pass_rate"], 0.0)


class MujocoDiagnosticsTests(unittest.TestCase):
    def _clean_aloha_model(self, **overrides) -> _FakeMjModel:
        kwargs = {
            "body_names": ["world", "vx300s_left/base", "red_peg"],
            "body_mass": [0.0, 1.5, 0.08],
            "jnt_type": [3, 0],  # hinge on robot, freejoint on red_peg
            "jnt_bodyid": [1, 2],
            "geom_names": ["floor", "vx300s_left/arm_geom", "red_peg_geom"],
            "geom_bodyid": [0, 1, 2],
            "geom_margin": [0.0, 0.0, 0.0],
            "geom_gap": [0.0, 0.0, 0.0],
            "actuator_names": ["left_waist"],
            "actuator_trntype": [0],
            "actuator_trnid": [(0, 0)],
            "enableflags": 0,
        }
        kwargs.update(overrides)
        return _FakeMjModel(**kwargs)

    def test_audit_compiled_model_clean_and_defects(self) -> None:
        mj = _make_fake_mujoco()
        clean = self._clean_aloha_model()
        self.assertEqual(DIAG.audit_compiled_model(clean, mujoco_module=mj), ())

        broken = self._clean_aloha_model(
            body_mass=[0.0, 1.5, 0.001],  # too light
            geom_margin=[0.0, 0.0, 0.002],  # non-zero margin
            eq_type=[1],  # mjEQ_WELD
            eq_obj1id=[1],
            eq_obj2id=[2],  # constrains red_peg
            actuator_names=["left_waist", "cheat_peg_act"],
            actuator_trntype=[0, 0],
            actuator_trnid=[(0, 0), (1, 0)],  # drives manipuland joint
            enableflags=1,  # mjENBL_OVERRIDE
        )
        issues = DIAG.audit_compiled_model(broken, mujoco_module=mj)
        self.assertEqual(len(issues), 5)
        self.assertTrue(any("equality 0 constrains a manipuland" in i for i in issues))
        self.assertTrue(any("cheat_peg_act drives a non-robot dof" in i for i in issues))
        self.assertTrue(any("red_peg mass" in i for i in issues))
        self.assertTrue(any("red_peg_geom has margin/gap" in i for i in issues))
        self.assertTrue(any("contact override is enabled" in i for i in issues))

    def test_snapshot_restore_model_hash_guard_and_fork(self) -> None:
        mj = _make_fake_mujoco()
        model = self._clean_aloha_model()
        data = _FakeMjData([0.0, 0.0, 0.0, 0.0])
        env = _FakeGymAlohaEnv(model, data)

        # Replay 2 prefix actions ([0.5], [0.5]) -> state[1] == 1.0, then fork two candidates.
        snapshot, res_a = DIAG.fork_episode_from_actions(
            env,
            seed=1002,
            prefix_actions=([0.5], [0.5]),
            controller=lambda _obs, _step: [0.5],
            max_steps=5,
            mujoco_module=mj,
        )
        self.assertEqual(snapshot.step_index, 2)
        self.assertEqual(res_a.steps_executed, 2)  # 1.0 -> 1.5 -> 2.0 (done)
        self.assertEqual(res_a.max_reward, 1.0)
        self.assertEqual(res_a.unexpected_collision_count, 0)

        # Fork from the exact same saved snapshot with a colliding candidate.
        res_b = DIAG.fork_episode_from_snapshot(
            env,
            snapshot,
            controller=lambda _obs, _step: [-0.25],
            max_steps=3,
            mujoco_module=mj,
        )
        self.assertEqual(res_b.steps_executed, 3)
        self.assertEqual(res_b.max_reward, 0.0)
        self.assertEqual(res_b.unexpected_collision_count, 3)
        self.assertEqual(data.state[1], 0.25)  # restored to 1.0, then 3 * -0.25

        # Mutating compiled model bytes must fail closed on restore.
        model.serialized = b"mutated-model"
        with self.assertRaises(ValueError):
            DIAG.restore_integration_state(env, snapshot, mujoco_module=mj)

    def test_render_offscreen_frame_copies_and_guards_headless_linux(self) -> None:
        renderers: list[_FakeRenderer] = []
        mj = _make_fake_mujoco(renderers)
        model = self._clean_aloha_model()
        data = _FakeMjData([0.0, 0.0, 0.0, 0.0])

        with patch.dict(os.environ, {"MUJOCO_GL": "osmesa"}, clear=False):
            frame = DIAG.render_offscreen_frame(
                model, data, camera="top", width=320, height=240, depth=True, mujoco_module=mj
            )
        self.assertEqual(len(renderers), 1)
        self.assertTrue(renderers[0].closed)
        self.assertFalse(renderers[0].depth_enabled)
        self.assertEqual(renderers[0].last_camera, "top")
        self.assertIsNot(frame, renderers[0]._buffer)

        with (
            patch.object(DIAG.sys, "platform", "linux"),
            patch.dict(os.environ, {"DISPLAY": "", "MUJOCO_GL": ""}, clear=False),
            self.assertRaises(RuntimeError),
        ):
            DIAG.render_offscreen_frame(model, data, camera="top", mujoco_module=mj)

    def test_agents_md_has_no_host_absolute_paths(self) -> None:
        text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertNotIn("C:\\Users\\", text)
        self.assertNotIn("/Users/", text)
        self.assertIn("Robotics 插件配合约定", text)

    def test_float64_buffer_roundtrip(self) -> None:
        raw = array.array("d", [1.25, -3.5, 0.0, 42.0]).tobytes()
        buf = DIAG._float64_buffer_from_bytes(raw)
        self.assertEqual([float(x) for x in buf], [1.25, -3.5, 0.0, 42.0])


if __name__ == "__main__":
    unittest.main()
