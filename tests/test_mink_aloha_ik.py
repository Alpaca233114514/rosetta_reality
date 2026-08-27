"""Integration tests for the pinned upstream Mink ALOHA QP adapter."""

from __future__ import annotations

import numpy as np
import pytest

from rosetta_reality.sim.mink_aloha_ik import (
    MinkAlohaIkSettings,
    MinkAlohaIkSolver,
)


def _settings(*, joint_limit_margin_rad: float = 0.0) -> MinkAlohaIkSettings:
    return MinkAlohaIkSettings(
        integration_timestep_s=0.005,
        maximum_iterations=5,
        position_cost=1.0,
        orientation_cost=1.0,
        posture_cost=1e-4,
        frame_lm_damping=1.0,
        solver_damping=1e-5,
        maximum_joint_velocity_rad_s=np.pi,
        configuration_limit_gain=0.95,
        joint_limit_margin_rad=joint_limit_margin_rad,
    )


def test_mink_settings_reject_invalid_configuration_limit_gain() -> None:
    with pytest.raises(ValueError, match="configuration_limit_gain"):
        MinkAlohaIkSettings(
            integration_timestep_s=0.005,
            maximum_iterations=5,
            position_cost=1.0,
            orientation_cost=1.0,
            posture_cost=1e-4,
            frame_lm_damping=1.0,
            solver_damping=1e-5,
            maximum_joint_velocity_rad_s=np.pi,
            configuration_limit_gain=1.1,
        )


def test_mink_settings_reject_negative_joint_limit_margin() -> None:
    with pytest.raises(ValueError, match="joint_limit_margin_rad"):
        _settings(joint_limit_margin_rad=-0.001)


def test_mink_qp_applies_registered_arm_joint_limit_margin() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")

    solver = MinkAlohaIkSolver.from_gym_aloha(
        left_site="cali_left_site1",
        right_site="cali_right_site1",
        settings=_settings(joint_limit_margin_rad=0.01),
    )

    minimum, maximum = solver.model.joint("vx300s_right/wrist_rotate").range
    assert (minimum, maximum) == pytest.approx((-3.13158, 3.13158))


def test_mink_qp_holds_native_aloha_pose_inside_joint_limits() -> None:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    from gym_aloha.constants import START_ARM_POSE

    solver = MinkAlohaIkSolver.from_gym_aloha(
        left_site="cali_left_site1",
        right_site="cali_right_site1",
        settings=_settings(),
    )
    current = np.asarray(START_ARM_POSE, dtype=np.float64)
    data = mujoco.MjData(solver.model)
    data.qpos[:16] = current
    mujoco.mj_forward(solver.model, data)

    targets: list[tuple[np.ndarray, np.ndarray]] = []
    for site_name in ("cali_left_site1", "cali_right_site1"):
        site_id = solver.model.site(site_name).id
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
        targets.append((data.site_xpos[site_id].copy(), quaternion))

    result = solver.solve(
        current,
        left_position=targets[0][0],
        left_quaternion_wxyz=targets[0][1],
        right_position=targets[1][0],
        right_quaternion_wxyz=targets[1][1],
    )
    neutral_posture = solver._posture_task.target_q.copy()

    assert max(result.position_errors_m) <= 1e-6
    assert max(result.orientation_errors_rad) <= 1e-6
    assert np.isfinite(result.qpos).all()
    for joint_id in range(16):
        if solver.model.jnt_limited[joint_id]:
            qpos_address = solver.model.jnt_qposadr[joint_id]
            minimum, maximum = solver.model.jnt_range[joint_id]
            assert minimum <= result.qpos[qpos_address] <= maximum

    shifted_left = targets[0][0].copy()
    shifted_right = targets[1][0].copy()
    shifted_left[2] += 0.002
    shifted_right[2] += 0.002
    moved = solver.solve(
        current,
        left_position=shifted_left,
        left_quaternion_wxyz=targets[0][1],
        right_position=shifted_right,
        right_quaternion_wxyz=targets[1][1],
    )

    assert max(moved.position_errors_m) < 0.002
    assert not np.allclose(moved.qpos[:16], result.qpos[:16])

    solver.solve(
        moved.qpos[:16],
        left_position=shifted_left,
        left_quaternion_wxyz=targets[0][1],
        right_position=shifted_right,
        right_quaternion_wxyz=targets[1][1],
    )
    assert solver._posture_task.target_q == pytest.approx(neutral_posture)


def test_mink_qp_sanitizes_and_freezes_non_arm_configuration_dofs() -> None:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    from gym_aloha.constants import START_ARM_POSE

    solver = MinkAlohaIkSolver.from_gym_aloha(
        left_site="cali_left_site1",
        right_site="cali_right_site1",
        settings=_settings(),
    )
    valid = np.asarray(START_ARM_POSE, dtype=np.float64)
    current = valid.copy()
    current[[6, 7, 14, 15]] = (0.01844, -0.01844, 0.058, -0.058)
    frozen_qpos = np.asarray(solver.configuration.q, dtype=np.float64)[16:].copy()
    data = mujoco.MjData(solver.model)
    data.qpos[:16] = valid
    mujoco.mj_forward(solver.model, data)

    targets: list[tuple[np.ndarray, np.ndarray]] = []
    for site_name in ("cali_left_site1", "cali_right_site1"):
        site_id = solver.model.site(site_name).id
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
        targets.append((data.site_xpos[site_id].copy(), quaternion))

    result = solver.solve(
        current,
        left_position=targets[0][0],
        left_quaternion_wxyz=targets[0][1],
        right_position=targets[1][0],
        right_quaternion_wxyz=targets[1][1],
    )

    assert result.qpos[[6, 7, 14, 15]] == pytest.approx(
        (0.021, -0.021, 0.057, -0.057)
    )
    assert result.qpos[16:] == pytest.approx(frozen_qpos)


def test_mink_qp_sanitizes_action_contract_arm_bound_to_native_mjcf() -> None:
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    from gym_aloha.constants import START_ARM_POSE

    solver = MinkAlohaIkSolver.from_gym_aloha(
        left_site="cali_left_site1",
        right_site="cali_right_site1",
        settings=_settings(),
    )
    current = np.asarray(START_ARM_POSE, dtype=np.float64).copy()
    current[13] = -np.pi
    native = current.copy()
    native[13] = solver.model.jnt_range[13, 0]
    data = mujoco.MjData(solver.model)
    data.qpos[:16] = native
    mujoco.mj_forward(solver.model, data)

    targets: list[tuple[np.ndarray, np.ndarray]] = []
    for site_name in ("cali_left_site1", "cali_right_site1"):
        site_id = solver.model.site(site_name).id
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
        targets.append((data.site_xpos[site_id].copy(), quaternion))

    result = solver.solve(
        current,
        left_position=targets[0][0],
        left_quaternion_wxyz=targets[0][1],
        right_position=targets[1][0],
        right_quaternion_wxyz=targets[1][1],
    )

    minimum, maximum = solver.model.jnt_range[13]
    assert minimum <= result.qpos[13] <= maximum
    assert max(result.position_errors_m) <= 1e-6
    assert max(result.orientation_errors_rad) <= 1e-6
