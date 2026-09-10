"""Frozen-observation extraction for the scripted-teacher gate rollouts.

Builds the registered :class:`InsertionGeometry` from a live
:class:`GymAlohaEnvironment` using exactly the frozen field set: the robot
state from the environment's own qpos semantics, the calibrated end-effector
site poses, the object body poses, contact-derived task events, the
environment's own reward semantics and the adapter-owned unexpected-collision
classification.  No time, index, episode, seed or policy field is read, so a
teacher consuming only this observation cannot condition on forbidden inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rosetta_reality.sim.geometry_teacher import GeometryPose, InsertionGeometry

LEFT_EEF_SITE = "cali_left_site1"
RIGHT_EEF_SITE = "cali_right_site1"
SOCKET_BODY = "socket"
PEG_BODY = "peg"
SOCKET_GEOMETRIES = tuple(f"socket-{index}" for index in range(1, 5))
LEFT_FINGER_GEOMETRIES = (
    "vx300s_left/10_left_gripper_finger",
    "vx300s_left/10_right_gripper_finger",
)
RIGHT_FINGER_GEOMETRIES = (
    "vx300s_right/10_left_gripper_finger",
    "vx300s_right/10_right_gripper_finger",
)
TABLE_GEOMETRY = "table"
PEG_GEOMETRY = "red_peg"
PIN_GEOMETRY = "pin"


def _physics(environment: Any) -> Any:
    raw_environment = getattr(environment, "raw_environment", environment)
    unwrapped = getattr(raw_environment, "unwrapped", raw_environment)
    control_environment = getattr(unwrapped, "_env", None)
    physics = getattr(control_environment, "physics", None)
    if physics is None:
        raise RuntimeError(
            "Scripted-teacher observation requires the registered MuJoCo backend."
        )
    return physics


def _control_task(environment: Any) -> Any:
    raw_environment = getattr(environment, "raw_environment", environment)
    unwrapped = getattr(raw_environment, "unwrapped", raw_environment)
    control_environment = getattr(unwrapped, "_env", None)
    task = getattr(control_environment, "task", None)
    if task is None:
        task = getattr(control_environment, "_task", None)
    if task is None:
        raise RuntimeError(
            "Scripted-teacher observation requires the registered dm_control task."
        )
    return task


def _body_pose(physics: Any, name: str) -> GeometryPose:
    try:
        body_id = int(physics.model.name2id(name, "body"))
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"MuJoCo body {name!r} is unavailable.") from error
    if body_id < 0:
        raise RuntimeError(f"MuJoCo body {name!r} is unavailable.")
    return GeometryPose(
        position=torch.as_tensor(np.asarray(physics.data.xpos[body_id]).copy()),
        quaternion=torch.as_tensor(np.asarray(physics.data.xquat[body_id]).copy()),
    )


def _site_pose(physics: Any, name: str) -> GeometryPose:
    from dm_control.mujoco.wrapper.mjbindings import mjlib

    try:
        position = np.asarray(physics.named.data.site_xpos[name]).copy()
        matrix = np.asarray(physics.named.data.site_xmat[name]).copy()
    except KeyError as error:
        raise RuntimeError(f"MuJoCo site {name!r} is unavailable.") from error
    quaternion = np.empty(4, dtype=np.float64)
    mjlib.mju_mat2Quat(quaternion, matrix)
    return GeometryPose(
        position=torch.as_tensor(position),
        quaternion=torch.as_tensor(quaternion),
    )


def _robot_state(task: Any, physics: Any) -> torch.Tensor:
    """The environment's own 14-D logical qpos semantics (arm + normalized grippers)."""

    try:
        qpos = np.asarray(task.get_qpos(physics), dtype=np.float64).copy()
    except (AttributeError, TypeError) as error:
        raise RuntimeError(
            "Registered dm_control task exposes no get_qpos observation semantics."
        ) from error
    if qpos.shape != (14,):
        raise RuntimeError(
            f"Registered task qpos must have shape (14,), received {qpos.shape}."
        )
    return torch.as_tensor(qpos, dtype=torch.float32)


def _observed_reward(task: Any, physics: Any) -> float:
    """The environment's own registered reward for the current contact state."""

    try:
        reward = float(task.get_reward(physics))
    except (AttributeError, TypeError) as error:
        raise RuntimeError(
            "Registered dm_control task exposes no get_reward semantics."
        ) from error
    if not np.isfinite(reward):
        raise RuntimeError("Registered task reward is not finite.")
    return reward


def _has_contact(contacts: set[frozenset[str]], first: str, second: str) -> bool:
    return frozenset({first, second}) in contacts


def build_observation(environment: Any, contract: Any) -> InsertionGeometry:
    """Extract exactly the frozen registered observation field set."""

    if int(contract.dimension) != 14:
        raise ValueError(
            "Scripted-teacher observation requires the 14-D ALOHA Action Contract."
        )
    physics = _physics(environment)
    task = _control_task(environment)
    contact_pairs = environment.contact_pairs()
    contacts = {frozenset(pair) for pair in contact_pairs}
    unexpected_collision_count = sum(
        environment.is_unexpected_collision_pair(first, second)
        for first, second in contact_pairs
    )
    return InsertionGeometry(
        robot_state=_robot_state(task, physics),
        left_eef=_site_pose(physics, LEFT_EEF_SITE),
        right_eef=_site_pose(physics, RIGHT_EEF_SITE),
        socket=_body_pose(physics, SOCKET_BODY),
        peg=_body_pose(physics, PEG_BODY),
        observed_reward=_observed_reward(task, physics),
        socket_grasp_contact=any(
            _has_contact(contacts, socket_geometry, finger)
            for socket_geometry in SOCKET_GEOMETRIES
            for finger in LEFT_FINGER_GEOMETRIES
        ),
        peg_grasp_contact=any(
            _has_contact(contacts, PEG_GEOMETRY, finger)
            for finger in RIGHT_FINGER_GEOMETRIES
        ),
        socket_on_table=any(
            _has_contact(contacts, socket_geometry, TABLE_GEOMETRY)
            for socket_geometry in SOCKET_GEOMETRIES
        ),
        peg_on_table=_has_contact(contacts, PEG_GEOMETRY, TABLE_GEOMETRY),
        peg_socket_contact=any(
            _has_contact(contacts, PEG_GEOMETRY, socket_geometry)
            for socket_geometry in SOCKET_GEOMETRIES
        ),
        pin_contact=_has_contact(contacts, PEG_GEOMETRY, PIN_GEOMETRY),
        unexpected_collision_count=unexpected_collision_count,
    )
