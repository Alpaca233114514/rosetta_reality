"""Stdlib unit tests for P0-B CUDA horizon and noise sweep diagnostic."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_smolvla_horizon_sweep", ROOT / "scripts/diagnose_smolvla_horizon_sweep.py"
)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


@dataclass
class _Pose:
    position: list[float]


@dataclass
class _Geom:
    left_eef: _Pose
    right_eef: _Pose
    socket: _Pose
    peg: _Pose
    socket_grasp_contact: bool
    peg_grasp_contact: bool
    peg_socket_contact: bool
    pin_contact: bool


class _FakeContract:
    def clip(self, slot):
        return list(slot), [False] * len(slot)


class _FakeEnv:
    def __init__(self, *, steps_until_done: int = 10, collide_on_step: int | None = None):
        self.step_count = 0
        self.steps_until_done = steps_until_done
        self.collide_on_step = collide_on_step
        self.closed = False

    def reset(self, *, seed: int):
        self.step_count = 0
        return {"robot_state": [0.0] * 14, "images": {"top": None}}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.steps_until_done
        reward = 2.0 if self.step_count == 3 else 0.0
        return {"robot_state": list(action), "images": {"top": None}}, reward, done, {"is_success": False}

    def diagnostic_snapshot(self):
        return {"joint_limit_violations": []}

    def contact_pairs(self):
        if self.collide_on_step == self.step_count:
            return (("table", "vx300s_right/10_right_gripper_finger"),)
        return ()

    def is_unexpected_collision_pair(self, first, second):
        return "table" in (first, second)

    def close(self):
        self.closed = True


class _FakePolicy:
    def __init__(self, fold_right: bool = False):
        self.calls = 0
        self.noise_cfg = None
        self.fold_right = fold_right

        class _Decoder:
            _registry_name = "rosetta_pi_aloha_postprocessor"
            last_model_action = None

        class _Post:
            steps = [_Decoder()]

        self.postprocessor = _Post()

    def configure_noise(self, mode, seed):
        self.noise_cfg = (mode, seed)

    def predict(self, observation, instruction):
        self.calls += 1
        chunk = [[0.01 * (self.calls + i)] * 14 for i in range(50)]
        internal = [[0.0] * 14 for _ in range(50)]
        if self.fold_right:
            for row in internal:
                row[13] = -2.2  # < -pi/2 -> folded!
        self.postprocessor.steps[0].last_model_action = internal
        return chunk, chunk


class HorizonSweepTest(unittest.TestCase):
    def test_lazy_import_has_no_ml_dependencies(self) -> None:
        self.assertNotIn("torch", sweep.__dict__)
        self.assertNotIn("mujoco", sweep.__dict__)

    def test_validate_sweep_config_rejects_invalid_inputs(self) -> None:
        sweep.validate_sweep_config(
            actions_per_inference_list=[1, 5, 10, 25, 50],
            noise_modes=["seeded_standard_normal", "zeros"],
            seeds=[1000, 1001],
            maximum_steps=500,
        )
        with self.assertRaises(ValueError):
            sweep.validate_sweep_config(
                actions_per_inference_list=[0],
                noise_modes=["zeros"],
                seeds=[1000],
                maximum_steps=10,
            )
        with self.assertRaises(ValueError):
            sweep.validate_sweep_config(
                actions_per_inference_list=[1],
                noise_modes=["invalid"],
                seeds=[1000],
                maximum_steps=10,
            )

    def test_actions_per_inference_reduces_policy_calls_and_tracks_milestones_and_folds(self) -> None:
        policy = _FakePolicy(fold_right=True)
        env_holder = {}

        def make_env(contract, max_steps):
            env = _FakeEnv(steps_until_done=10)
            env_holder["env"] = env
            return env

        def read_geom(env, contract):
            step = env.step_count
            return _Geom(
                left_eef=_Pose([0.0, 0.0, 0.0]),
                right_eef=_Pose([0.0, 0.0, 0.0]),
                socket=_Pose([0.05 / step, 0.0, 0.0]),
                peg=_Pose([0.0, 0.1 / step, 0.0]),
                socket_grasp_contact=step in (3, 4, 5),
                peg_grasp_contact=step in (3, 4, 5),
                peg_socket_contact=step == 5,
                pin_contact=False,
            )

        res = sweep.run_episode(
            policy,
            _FakeContract(),
            "Insert the peg into the socket.",
            seed=1000,
            maximum_steps=10,
            actions_per_inference=5,
            noise_mode="zeros",
            env_factory=make_env,
            geometry_reader=read_geom,
        )
        self.assertTrue(env_holder["env"].closed)
        self.assertEqual(policy.noise_cfg, ("zeros", None))
        self.assertEqual(res["rollout_length"], 10)
        self.assertEqual(res["policy_inference_calls"], 2)
        self.assertEqual(res["milestones"]["max_dual_grasp_streak"], 3)
        self.assertEqual(res["milestones"]["peg_socket_contact_steps"], 1)
        self.assertAlmostEqual(res["milestones"]["min_left_eef_to_socket_m"], 0.005, places=6)
        self.assertEqual(res["bounded_sine_folds"]["right_gripper"], 10)
        self.assertEqual(res["bounded_sine_folds"]["left_gripper"], 0)

    def test_stop_on_safety_halts_immediately_on_collision(self) -> None:
        policy = _FakePolicy()

        def make_env(contract, max_steps):
            return _FakeEnv(steps_until_done=20, collide_on_step=3)

        def read_geom(env, contract):
            return _Geom(
                left_eef=_Pose([0.0, 0.0, 0.0]),
                right_eef=_Pose([0.0, 0.0, 0.0]),
                socket=_Pose([0.1, 0.0, 0.0]),
                peg=_Pose([0.0, 0.1, 0.0]),
                socket_grasp_contact=False,
                peg_grasp_contact=False,
                peg_socket_contact=False,
                pin_contact=False,
            )

        res = sweep.run_episode(
            policy,
            _FakeContract(),
            "Insert the peg into the socket.",
            seed=1001,
            maximum_steps=20,
            actions_per_inference=10,
            noise_mode="seeded_standard_normal",
            stop_on_safety=True,
            env_factory=make_env,
            geometry_reader=read_geom,
        )
        self.assertEqual(res["outcome"], "physical_safety_stop")
        self.assertEqual(res["rollout_length"], 3)
        self.assertEqual(res["unexpected_collisions"], 1)
        self.assertEqual(policy.calls, 1)

    def test_temporal_ensemble_and_clamped_bounded_sine_override_work_end_to_end(self) -> None:
        policy = _FakePolicy(fold_right=True)
        executed_actions_seen = []

        class _RecordingEnv(_FakeEnv):
            def step(self, action):
                executed_actions_seen.append(list(action))
                return super().step(action)

        def make_env(contract, max_steps):
            return _RecordingEnv(steps_until_done=12)

        def read_geom(env, contract):
            return _Geom(
                left_eef=_Pose([0.0, 0.0, 0.0]),
                right_eef=_Pose([0.0, 0.0, 0.0]),
                socket=_Pose([0.1, 0.0, 0.0]),
                peg=_Pose([0.0, 0.1, 0.0]),
                socket_grasp_contact=False,
                peg_grasp_contact=False,
                peg_socket_contact=False,
                pin_contact=False,
            )

        res = sweep.run_episode(
            policy,
            _FakeContract(),
            "Insert the peg into the socket.",
            seed=1002,
            maximum_steps=12,
            actions_per_inference=1,
            noise_mode="zeros",
            temporal_ensemble_k=0.01,
            clamp_bounded_sine=True,
            env_factory=make_env,
            geometry_reader=read_geom,
        )
        self.assertEqual(res["rollout_length"], 12)
        self.assertEqual(res["policy_inference_calls"], 12)
        self.assertEqual(res["temporal_ensemble_k"], 0.01)
        self.assertTrue(res["clamp_bounded_sine"])
        self.assertEqual(res["clamped_override_steps"], 12)
        # internal[13] = -2.2 < -pi/2 -> clamped override sets right_gripper to 0.0 closed.
        for act in executed_actions_seen:
            self.assertAlmostEqual(act[13], 0.0, places=9)
        self.assertIn("arm_signflip_rate", res["dynamics"])
        self.assertIn("grip_signflip_rate", res["dynamics"])
        self.assertIn("straightness_tail", res["dynamics"])


if __name__ == "__main__":
    unittest.main()
