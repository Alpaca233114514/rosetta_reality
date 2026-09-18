"""Tests for the bounded MuJoCo position-actuator feedforward boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosetta_reality.sim.mujoco_position_feedforward import (
    _affine_control_for_force,
    static_position_feedforward,
)


class _Model:
    nq = 2
    nv = 2
    nu = 2
    jnt_qposadr = np.asarray([0, 1])
    jnt_dofadr = np.asarray([0, 1])
    actuator_trnid = np.asarray([[0, -1], [1, -1]])
    actuator_trntype = np.asarray([0, 0])
    actuator_gaintype = np.asarray([0, 0])
    actuator_biastype = np.asarray([1, 1])
    actuator_dyntype = np.asarray([0, 0])
    actuator_gainprm = np.asarray([[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    actuator_biasprm = np.asarray([[0.0, -100.0, -2.0], [0.0, -200.0, -3.0]])
    actuator_forcelimited = np.asarray([True, True])
    actuator_forcerange = np.asarray([[-10.0, 10.0], [-10.0, 10.0]])
    actuator_ctrllimited = np.asarray([True, True])
    actuator_ctrlrange = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])

    @staticmethod
    def name2id(name: str, kind: str) -> int:
        assert kind == "joint"
        return {"left": 0, "right": 1}[name]


class _Physics:
    def __init__(self) -> None:
        self.model = _Model()
        self.data = SimpleNamespace(
            qpos=np.zeros(2),
            qvel=np.zeros(2),
            qacc=np.zeros(2),
            qfrc_bias=np.asarray([2.0, -4.0]),
            qfrc_passive=np.zeros(2),
            actuator_moment=np.eye(2),
            actuator_length=np.zeros(2),
            actuator_velocity=np.zeros(2),
        )

    def forward(self) -> None:
        self.data.actuator_length[:] = self.data.qpos
        self.data.actuator_velocity[:] = self.data.qvel


class _SparsePhysics(_Physics):
    def __init__(self) -> None:
        super().__init__()
        self.data.actuator_moment = np.asarray([1.0, 1.0])
        self.data.moment_rownnz = np.asarray([1, 1], dtype=np.int32)
        self.data.moment_rowadr = np.asarray([0, 1], dtype=np.int32)
        self.data.moment_colind = np.asarray([0, 1], dtype=np.int32)


def test_affine_actuator_force_inversion_matches_official_equation() -> None:
    control = _affine_control_for_force(
        required_generalized_force=5.0,
        moment=1.0,
        gain=100.0,
        bias=-20.0,
    )

    assert control == pytest.approx(0.25)


def test_static_position_feedforward_balances_bias_with_bounded_reference() -> None:
    result = static_position_feedforward(
        _Physics(),
        desired_robot_qpos=[0.2, -0.3],
        arm_joint_names=["left", "right"],
        joint_lower_rad=[-1.0, -1.0],
        joint_upper_rad=[1.0, 1.0],
        joint_limit_margin_rad=0.05,
        maximum_correction_rad=0.03,
    )

    assert result.positions == pytest.approx((0.22, -0.32))
    assert result.corrections_rad == pytest.approx((0.02, -0.02))
    assert result.maximum_correction_rad == pytest.approx(0.02)
    assert result.minimum_command_joint_limit_margin_rad == pytest.approx(0.68)


def test_static_position_feedforward_reads_official_sparse_moment_storage() -> None:
    result = static_position_feedforward(
        _SparsePhysics(),
        desired_robot_qpos=[0.2, -0.3],
        arm_joint_names=["left", "right"],
        joint_lower_rad=[-1.0, -1.0],
        joint_upper_rad=[1.0, 1.0],
        joint_limit_margin_rad=0.05,
        maximum_correction_rad=0.03,
    )

    assert result.positions == pytest.approx((0.22, -0.32))
    assert result.corrections_rad == pytest.approx((0.02, -0.02))


def test_static_position_feedforward_rejects_unregistered_correction() -> None:
    with pytest.raises(ValueError, match="exceeds its bound"):
        static_position_feedforward(
            _Physics(),
            desired_robot_qpos=[0.2, -0.3],
            arm_joint_names=["left", "right"],
            joint_lower_rad=[-1.0, -1.0],
            joint_upper_rad=[1.0, 1.0],
            joint_limit_margin_rad=0.05,
            maximum_correction_rad=0.019,
        )
