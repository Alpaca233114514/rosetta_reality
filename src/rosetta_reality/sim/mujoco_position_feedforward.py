"""Static inverse-dynamics feedforward for MuJoCo position actuators.

MuJoCo's SISO actuator force law is affine in control, actuator length and
actuator velocity.  For a static target, the generalized actuator force must
balance ``qfrc_bias - qfrc_passive``.  This module solves that registered
equation for direct, fixed-gain joint-position actuators; it does not tune a
controller gain or change the simulator's Action Contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MujocoPositionFeedforwardResult:
    """A compensated position reference and its bounded correction evidence."""

    positions: tuple[float, ...]
    corrections_rad: tuple[float, ...]
    maximum_correction_rad: float
    minimum_command_joint_limit_margin_rad: float


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _affine_control_for_force(
    *,
    required_generalized_force: float,
    moment: float,
    gain: float,
    bias: float,
) -> float:
    """Invert MuJoCo's affine SISO force law for one direct joint actuator."""

    values = (required_generalized_force, moment, gain, bias)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("MuJoCo actuator inversion received a non-finite value.")
    if abs(moment) <= 1e-12:
        raise ValueError("MuJoCo direct-joint actuator moment must be nonzero.")
    if abs(gain) <= 1e-12:
        raise ValueError("MuJoCo fixed actuator gain must be nonzero.")
    actuator_force = required_generalized_force / moment
    return (actuator_force - bias) / gain


def _actuator_moment_entries(
    data: Any,
    *,
    actuator_id: int,
    actuator_count: int,
    velocity_dimension: int,
) -> tuple[tuple[int, float], ...]:
    """Read one MuJoCo actuator-moment row from dense or official CSR storage."""

    moment = np.asarray(data.actuator_moment, dtype=np.float64)
    if moment.shape == (actuator_count, velocity_dimension):
        return tuple(
            (int(column), float(moment[actuator_id, column]))
            for column in np.flatnonzero(np.abs(moment[actuator_id]) > 1e-12)
        )
    if moment.ndim != 1:
        raise ValueError("MuJoCo actuator moment storage has an unsupported shape.")

    required_fields = ("moment_rownnz", "moment_rowadr", "moment_colind")
    if any(not hasattr(data, name) for name in required_fields):
        raise ValueError("Sparse MuJoCo actuator moments lack their CSR identity arrays.")
    row_nonzero = np.asarray(data.moment_rownnz, dtype=np.int64)
    row_address = np.asarray(data.moment_rowadr, dtype=np.int64)
    column_index = np.asarray(data.moment_colind, dtype=np.int64)
    if row_nonzero.shape != (actuator_count,) or row_address.shape != (
        actuator_count,
    ):
        raise ValueError("Sparse MuJoCo actuator-moment row metadata differs from nu.")
    if column_index.shape != moment.shape:
        raise ValueError("Sparse MuJoCo actuator-moment columns differ from values.")

    start = int(row_address[actuator_id])
    count = int(row_nonzero[actuator_id])
    stop = start + count
    if start < 0 or count < 0 or stop > moment.size:
        raise ValueError("Sparse MuJoCo actuator-moment row exits its storage.")
    columns = column_index[start:stop]
    if (
        np.any(columns < 0)
        or np.any(columns >= velocity_dimension)
        or len(set(int(value) for value in columns)) != count
    ):
        raise ValueError("Sparse MuJoCo actuator-moment columns are invalid.")
    return tuple(
        (int(column), float(value))
        for column, value in zip(columns, moment[start:stop], strict=True)
        if abs(float(value)) > 1e-12
    )


def static_position_feedforward(
    physics: Any,
    *,
    desired_robot_qpos: Sequence[float],
    arm_joint_names: Sequence[str],
    joint_lower_rad: Sequence[float],
    joint_upper_rad: Sequence[float],
    joint_limit_margin_rad: float,
    maximum_correction_rad: float,
    neutral_reference_tolerance_rad: float = 1e-9,
) -> MujocoPositionFeedforwardResult:
    """Solve official MuJoCo affine actuator equations at a static target.

    The supported boundary is deliberately narrow: every requested arm joint
    must have exactly one direct joint transmission, fixed gain, affine bias,
    no activation dynamics and no cross-joint moment arm.  Unsupported model
    identities fail closed instead of silently approximating another actuator.
    """

    names = tuple(str(name) for name in arm_joint_names)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("arm_joint_names must be unique and nonempty.")
    desired = np.asarray(desired_robot_qpos, dtype=np.float64)
    if (
        desired.ndim != 1
        or desired.size == 0
        or desired.size > int(physics.model.nq)
        or not np.isfinite(desired).all()
    ):
        raise ValueError(
            "desired_robot_qpos must be a finite, nonempty prefix of model qpos."
        )
    lower = _finite_vector(joint_lower_rad, len(names), "joint_lower_rad")
    upper = _finite_vector(joint_upper_rad, len(names), "joint_upper_rad")
    for name, value in (
        ("joint_limit_margin_rad", joint_limit_margin_rad),
        ("maximum_correction_rad", maximum_correction_rad),
        ("neutral_reference_tolerance_rad", neutral_reference_tolerance_rad),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if np.any(lower + 2.0 * joint_limit_margin_rad >= upper):
        raise ValueError("The feedforward joint-limit margin collapses a joint range.")

    data = physics.data
    model = physics.model
    full_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    full_qpos[: desired.size] = desired
    data.qpos[:] = full_qpos
    data.qvel[:] = 0.0
    if hasattr(data, "qacc"):
        data.qacc[:] = 0.0
    physics.forward()

    required_generalized_force = np.asarray(
        data.qfrc_bias,
        dtype=np.float64,
    ) - np.asarray(data.qfrc_passive, dtype=np.float64)
    positions: list[float] = []
    corrections: list[float] = []
    margins: list[float] = []
    for index, joint_name in enumerate(names):
        joint_id = int(model.name2id(joint_name, "joint"))
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        candidates = [
            actuator_id
            for actuator_id in range(int(model.nu))
            if int(model.actuator_trnid[actuator_id, 0]) == joint_id
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one direct actuator for {joint_name}, found {len(candidates)}."
            )
        actuator_id = candidates[0]
        if int(model.actuator_trntype[actuator_id]) != 0:
            raise ValueError(f"Actuator for {joint_name} is not a joint transmission.")
        if int(model.actuator_gaintype[actuator_id]) != 0:
            raise ValueError(f"Actuator for {joint_name} does not use fixed gain.")
        if int(model.actuator_biastype[actuator_id]) != 1:
            raise ValueError(f"Actuator for {joint_name} does not use affine bias.")
        if int(model.actuator_dyntype[actuator_id]) != 0:
            raise ValueError(f"Actuator for {joint_name} has unsupported activation dynamics.")

        moment_entries = _actuator_moment_entries(
            data,
            actuator_id=actuator_id,
            actuator_count=int(model.nu),
            velocity_dimension=int(model.nv),
        )
        if tuple(column for column, _value in moment_entries) != (dof_address,):
            raise ValueError(f"Actuator for {joint_name} is not a direct one-DOF mapping.")
        moment = moment_entries[0][1]
        gain = float(model.actuator_gainprm[actuator_id, 0])
        bias_parameters = np.asarray(
            model.actuator_biasprm[actuator_id, :3],
            dtype=np.float64,
        )
        actuator_length = float(data.actuator_length[actuator_id])
        actuator_velocity = float(data.actuator_velocity[actuator_id])
        bias = float(
            bias_parameters[0]
            + bias_parameters[1] * actuator_length
            + bias_parameters[2] * actuator_velocity
        )
        reference = float(desired[qpos_address])
        neutral_control = _affine_control_for_force(
            required_generalized_force=0.0,
            moment=moment,
            gain=gain,
            bias=bias,
        )
        if not math.isclose(
            neutral_control,
            reference,
            rel_tol=0.0,
            abs_tol=neutral_reference_tolerance_rad,
        ):
            raise ValueError(
                f"Actuator for {joint_name} is not a position reference in joint radians."
            )
        control = _affine_control_for_force(
            required_generalized_force=float(required_generalized_force[dof_address]),
            moment=moment,
            gain=gain,
            bias=bias,
        )
        actuator_force = float(required_generalized_force[dof_address]) / moment
        if bool(model.actuator_forcelimited[actuator_id]):
            force_lower, force_upper = model.actuator_forcerange[actuator_id]
            if not float(force_lower) <= actuator_force <= float(force_upper):
                raise ValueError(
                    f"Static force for {joint_name} exceeds the actuator force range."
                )
        if bool(model.actuator_ctrllimited[actuator_id]):
            control_lower, control_upper = model.actuator_ctrlrange[actuator_id]
            if not float(control_lower) <= control <= float(control_upper):
                raise ValueError(
                    f"Static feedforward for {joint_name} exceeds the actuator control range."
                )
        correction = control - reference
        if abs(correction) > maximum_correction_rad + 1e-12:
            raise ValueError(
                f"Static feedforward correction for {joint_name} exceeds its bound."
            )
        margin = min(control - lower[index], upper[index] - control)
        if margin < joint_limit_margin_rad - 1e-12:
            raise ValueError(
                f"Static feedforward command for {joint_name} breaches the joint margin."
            )
        positions.append(control)
        corrections.append(correction)
        margins.append(margin)

    return MujocoPositionFeedforwardResult(
        positions=tuple(positions),
        corrections_rad=tuple(corrections),
        maximum_correction_rad=max(abs(value) for value in corrections),
        minimum_command_joint_limit_margin_rad=min(margins),
    )
