"""Read-only diagnostics for the scripted-teacher G1 repair loop.

Modes (no gate evidence is written; nothing steps outside its own process):

- ``rollout``  replay the gate rollout for one seed with per-step phase, pose,
  contact and target traces plus single-solve residual prints.
- ``frames``   compare eef site and gripper link frames at START_ARM_POSE and
  probe constrained-IK reachability for the candidate grasp orientations.
- ``align``    per-seed spawn pose/yaw extraction plus the frozen
  socket-approach/align left-arm targets chased to convergence at the exact
  target poses and at registered orientation variants (branch diagnosis).
- ``geometry`` print the insertion MJCF object geoms (socket legs, pin, peg).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
for root in (str(SRC_ROOT),):
    if root not in sys.path:
        sys.path.insert(0, root)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from rosetta_reality.sim.mink_aloha_ik import (  # noqa: E402
    MinkAlohaIkSettings,
    MinkAlohaIkSolver,
)
from rosetta_reality.sim.scripted_teacher import (  # noqa: E402
    _axis_angle_quaternion,
    default_ik_settings,
)


def _quat_str(quaternion: np.ndarray) -> str:
    return "(" + ", ".join(f"{value:+.4f}" for value in quaternion) + ")"


def _rollout(seed: int, budget: int) -> int:
    import rosetta_reality.sim.scripted_teacher as scripted_teacher_module
    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment
    from rosetta_reality.sim.scripted_teacher import build_teacher
    from rosetta_reality.sim.scripted_teacher_observer import build_observation

    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    teacher = build_teacher()
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=budget)
    maximum_reward = 0.0
    success = False
    try:
        environment.reset(seed=seed)
        teacher.reset()
        for step in range(budget):
            geometry = build_observation(environment, contract)
            unexpected = tuple(
                (first, second)
                for first, second in environment.contact_pairs()
                if environment.is_unexpected_collision_pair(first, second)
            )
            print(
                f"step={step:03d} phase={teacher.phase.value:9s} "
                f"left=({float(geometry.left_eef.position[0]):+.3f},"
                f"{float(geometry.left_eef.position[1]):.3f},"
                f"{float(geometry.left_eef.position[2]):.3f}) "
                f"right=({float(geometry.right_eef.position[0]):+.3f},"
                f"{float(geometry.right_eef.position[1]):.3f},"
                f"{float(geometry.right_eef.position[2]):.3f}) "
                f"socket=({float(geometry.socket.position[0]):+.3f},"
                f"{float(geometry.socket.position[1]):.3f},"
                f"{float(geometry.socket.position[2]):.3f}) "
                f"peg=({float(geometry.peg.position[0]):+.3f},"
                f"{float(geometry.peg.position[1]):.3f},"
                f"{float(geometry.peg.position[2]):.3f}) "
                f"contacts=(s{int(geometry.socket_grasp_contact)}"
                f"p{int(geometry.peg_grasp_contact)}) "
                f"reward={geometry.observed_reward:.0f} "
                f"unexpected={unexpected}",
                flush=True,
            )
            violations = environment.state_limit_violation_count()
            if violations:
                physics = getattr(
                    getattr(
                        getattr(environment.raw_environment, "unwrapped"),
                        "_env",
                        None,
                    ),
                    "physics",
                    None,
                )
                named = []
                if physics is not None:
                    for joint_id in range(int(physics.model.njnt)):
                        if not bool(physics.model.jnt_limited[joint_id]):
                            continue
                        qpos_address = int(physics.model.jnt_qposadr[joint_id])
                        lower, upper = physics.model.jnt_range[joint_id]
                        value = float(physics.data.qpos[qpos_address])
                        if value < float(lower) - 1e-5 or value > float(upper) + 1e-5:
                            named.append(
                                str(physics.model.id2name(joint_id, "joint") or joint_id)
                            )
                print(f"  JV step={step}: {named}", flush=True)
            decision = teacher.decide(geometry)
            if teacher.phase.value in (
                "approach",
                "orient",
                "descend",
                "transport",
                "align",
                "insert",
            ):
                stage = teacher.phase.value
                if stage in ("transport", "align", "insert"):
                    left_target, right_target = {
                        "transport": teacher._transport_targets,
                        "align": teacher._align_targets,
                        "insert": teacher._insert_targets,
                    }[stage](geometry)
                else:
                    left_target, right_target = {
                        "approach": teacher._approach_targets,
                        "orient": teacher._approach_targets,
                        "descend": teacher._descend_targets,
                    }[stage]()
                solved = teacher._solver.solve(
                    scripted_teacher_module._expanded_robot_qpos(geometry.robot_state),
                    left_position=left_target.position.numpy(),
                    left_quaternion_wxyz=left_target.quaternion.numpy(),
                    right_position=right_target.position.numpy(),
                    right_quaternion_wxyz=right_target.quaternion.numpy(),
                )
                position_errors = solved.position_errors_m
                rotation_errors = solved.orientation_errors_rad
                print(
                    f"  target[{stage}] left=({float(left_target.position[0]):+.3f},"
                    f"{float(left_target.position[1]):.3f},"
                    f"{float(left_target.position[2]):.3f}) "
                    f"right=({float(right_target.position[0]):+.3f},"
                    f"{float(right_target.position[1]):.3f},"
                    f"{float(right_target.position[2]):.3f}) "
                    f"solve_residual pos=({position_errors[0]:.4f},"
                    f"{position_errors[1]:.4f}) "
                    f"rot=({rotation_errors[0]:.4f},{rotation_errors[1]:.4f})",
                    flush=True,
                )
            if decision.refusal is not None:
                print(f"REFUSAL at step {step}: {decision.refusal}", flush=True)
                break
            clipped, _mask = contract.clip(decision.action)
            _observation, reward, done, info = environment.step(clipped)
            maximum_reward = max(maximum_reward, float(reward))
            success = success or bool(info.get("is_success", False))
            if done:
                print(
                    f"done at step {step}: success={info.get('is_success')}",
                    flush=True,
                )
                break
        else:
            print("budget exhausted", flush=True)
    finally:
        environment.close()
    print(f"maximum_reward={maximum_reward} success={success}", flush=True)
    return 0


def _forward(
    model: mujoco.MjModel,
) -> tuple[mujoco.MjData, dict[str, tuple[np.ndarray, np.ndarray]]]:
    from gym_aloha.constants import START_ARM_POSE

    data = mujoco.MjData(model)
    data.qpos[:16] = np.asarray(START_ARM_POSE, dtype=np.float64)
    mujoco.mj_forward(model, data)
    frames = {}
    for name in (
        "cali_left_site1",
        "cali_right_site1",
        "vx300s_left/gripper_link",
        "vx300s_right/gripper_link",
        "vx300s_left/left_finger_link",
        "vx300s_left/right_finger_link",
        "vx300s_right/left_finger_link",
        "vx300s_right/right_finger_link",
    ):
        if name.endswith("_site1"):
            site_id = model.site(name).id
            position = np.asarray(data.site_xpos[site_id]).copy()
            quaternion = np.empty(4)
            site_matrix = np.asarray(data.site_xmat[site_id])
            mujoco.mju_mat2Quat(quaternion, site_matrix)
        else:
            body_id = model.body(name).id
            position = np.asarray(data.xpos[body_id]).copy()
            quaternion = np.asarray(data.xquat[body_id]).copy()
        frames[name] = (position, quaternion)
    return data, frames


def _frames() -> int:
    settings: MinkAlohaIkSettings = default_ik_settings()
    solver = MinkAlohaIkSolver.from_gym_aloha(
        left_site="cali_left_site1",
        right_site="cali_right_site1",
        settings=settings,
    )
    _data, frames = _forward(solver.model)
    print("== frames at START_ARM_POSE ==")
    for name, (position, quaternion) in frames.items():
        print(f"{name}: pos={_quat_str(position)} quat={_quat_str(quaternion)}")
    for arm in ("left", "right"):
        first = frames[f"vx300s_{arm}/left_finger_link"][0]
        second = frames[f"vx300s_{arm}/right_finger_link"][0]
        separation = second - first
        separation = separation / np.linalg.norm(separation)
        print(
            f"{arm} finger separation axis (right-left, world): {_quat_str(separation)}"
        )

    import torch

    def torch_quat(quaternion: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(quaternion, dtype=torch.float32)

    socket_position = np.asarray([-0.125, 0.500, 0.022])
    peg_position = np.asarray([0.177, 0.404, 0.010])
    current_qpos = np.asarray(_data.qpos[:16], dtype=np.float64)
    tilt_left = np.asarray(
        _axis_angle_quaternion((0.0, 1.0, 0.0), 60.0), dtype=np.float64
    )

    def chase(title: str, left_quat: np.ndarray, right_quat: np.ndarray) -> None:
        left_target = socket_position + np.asarray([0.0, 0.0, 0.08])
        right_target = peg_position + np.asarray([0.0, 0.0, 0.08])
        qpos = np.asarray(current_qpos, dtype=np.float64)
        result = None
        for _ in range(120):
            result = solver.solve(
                qpos,
                left_position=left_target,
                left_quaternion_wxyz=left_quat,
                right_position=right_target,
                right_quaternion_wxyz=right_quat,
            )
            moved = np.abs(result.qpos[:16] - qpos).max()
            qpos = np.asarray(result.qpos[:16], dtype=np.float64)
            if (
                max(result.position_errors_m) <= 1e-3
                and max(result.orientation_errors_rad) <= 1e-3
            ) or moved <= 1e-9:
                break
        position_errors = result.position_errors_m
        rotation_errors = result.orientation_errors_rad
        print(
            f"{title}: pos_err=({position_errors[0]:.4f},{position_errors[1]:.4f}) "
            f"rot_err=({rotation_errors[0]:.4f},{rotation_errors[1]:.4f})"
        )

    chase("E iterative chase, world tilt +60 both arms", tilt_left, tilt_left)

    print("== right-arm tilt matrix at transport/insert targets ==")
    stage_targets = {
        "approach": (0.177, 0.404, 0.090),
        "transport": (0.10, 0.50, 0.15715),
        "insert": (0.05, 0.50, 0.15715),
    }
    left_target = np.asarray([-0.10, 0.50, 0.15])
    for stage, right_xyz in stage_targets.items():
        for degrees in (-60.0, 0.0, 60.0, 90.0, 120.0):
            right_quat = np.asarray(
                _axis_angle_quaternion((0.0, 1.0, 0.0), degrees), dtype=np.float64
            )
            qpos = np.asarray(current_qpos, dtype=np.float64)
            result = None
            for _ in range(200):
                result = solver.solve(
                    qpos,
                    left_position=left_target,
                    left_quaternion_wxyz=tilt_left,
                    right_position=np.asarray(right_xyz),
                    right_quaternion_wxyz=right_quat,
                )
                moved = np.abs(result.qpos[:16] - qpos).max()
                qpos = np.asarray(result.qpos[:16], dtype=np.float64)
                if (
                    max(result.position_errors_m) <= 1e-3
                    and max(result.orientation_errors_rad) <= 1e-3
                ) or moved <= 1e-9:
                    break
            print(
                f"{stage} right tilt {degrees:+.0f}: "
                f"pos_err={result.position_errors_m[1]:.4f} "
                f"rot_err={result.orientation_errors_rad[1]:.4f}"
            )
    return 0


def _yaw_degrees(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _pose_text(name: str, pose) -> str:
    x, y, z = (float(value) for value in pose.position)
    w, qx, qy, qz = (float(value) for value in pose.quaternion)
    return (
        f"{name}: pos=({x:+.4f},{y:+.4f},{z:+.4f}) "
        f"quat=({w:+.4f},{qx:+.4f},{qy:+.4f},{qz:+.4f}) "
        f"yaw={_yaw_degrees(np.asarray((w, qx, qy, qz))):+.2f}deg"
    )


def _chase(
    solver: MinkAlohaIkSolver,
    qpos: np.ndarray,
    left_position: np.ndarray,
    left_quaternion: np.ndarray,
    right_position: np.ndarray,
    right_quaternion: np.ndarray,
    iterations: int = 200,
) -> tuple[Any, int]:
    current = np.asarray(qpos, dtype=np.float64)
    result = None
    used = 0
    for used in range(1, iterations + 1):
        result = solver.solve(
            current,
            left_position=left_position,
            left_quaternion_wxyz=left_quaternion,
            right_position=right_position,
            right_quaternion_wxyz=right_quaternion,
        )
        moved = np.abs(result.qpos[:16] - current).max()
        current = np.asarray(result.qpos[:16], dtype=np.float64)
        if (
            max(result.position_errors_m) <= 1e-3 and max(result.orientation_errors_rad) <= 1e-3
        ) or moved <= 1e-9:
            break
    assert result is not None
    return result, used


def _align(seeds: list[int], budget: int, align_steps: int) -> int:
    import torch

    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.sim.geometry_teacher import (
        quaternion_multiply,
    )
    from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment
    from rosetta_reality.sim.scripted_teacher import (
        ScriptedInsertionStage,
        _expanded_robot_qpos,
        build_teacher,
    )
    from rosetta_reality.sim.scripted_teacher_observer import (
        _body_pose,
        _physics,
        build_observation,
    )

    contract = load_action_contract(REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    solver = MinkAlohaIkSolver.from_gym_aloha(
        left_site="cali_left_site1",
        right_site="cali_right_site1",
        settings=default_ik_settings(),
    )

    def distance(first, second) -> float:
        delta = first.position - second.position
        return float(torch.linalg.vector_norm(delta))

    for seed in seeds:
        teacher = build_teacher()
        environment = GymAlohaEnvironment(contract, maximum_episode_steps=budget)
        try:
            environment.reset(seed=seed)
            teacher.reset()
            physics = _physics(environment)
            print(f"== seed {seed} ==", flush=True)
            socket_spawn = _body_pose(physics, "socket")
            peg_spawn = _body_pose(physics, "peg")
            print(_pose_text("socket.spawn", socket_spawn), flush=True)
            print(_pose_text("peg.spawn", peg_spawn), flush=True)
            relative_spawn_yaw = _yaw_degrees(peg_spawn.quaternion.numpy()) - _yaw_degrees(
                socket_spawn.quaternion.numpy()
            )
            print(
                f"relative spawn yaw (peg - socket) = {relative_spawn_yaw:+.2f} deg",
                flush=True,
            )

            captured = None
            refusal = None
            steps_used = 0
            while steps_used < budget:
                geometry = build_observation(environment, contract)
                decision = teacher.decide(geometry)
                steps_used += 1
                if teacher.phase is ScriptedInsertionStage.SOCKET_APPROACH and captured is None:
                    hover_left, hold_right = teacher._socket_approach_targets(geometry)
                    align_left, _right = teacher._align_targets(geometry)
                    captured = {
                        "geometry": geometry,
                        "hover_left": hover_left,
                        "hold_right": hold_right,
                        "align_left": align_left,
                        "peg_hold": teacher._peg_hold_pose,
                        "sfls": teacher._socket_from_left_site,
                        "socket": geometry.socket,
                        "left_eef": geometry.left_eef,
                    }
                    break
                if decision.refusal is not None:
                    refusal = decision.refusal
                    break
                clipped, _mask = contract.clip(decision.action)
                _observation, _reward, done, _info = environment.step(clipped)
                if done:
                    break
            if captured is None:
                print(
                    f"seed {seed}: SOCKET_APPROACH never reached "
                    f"(refusal={refusal}, steps={steps_used}); no align probe",
                    flush=True,
                )
                continue

            hover_left = captured["hover_left"]
            align_left = captured["align_left"]
            hold_right = captured["hold_right"]
            hover_distances: list[float] = []
            align_distances: list[float] = []
            final_phase = teacher.phase.value
            for _extra in range(align_steps):
                geometry = build_observation(environment, contract)
                hover_distances.append(distance(geometry.left_eef, hover_left))
                align_distances.append(distance(geometry.left_eef, align_left))
                decision = teacher.decide(geometry)
                if decision.refusal is not None:
                    refusal = decision.refusal
                    break
                final_phase = teacher.phase.value
                if teacher.phase is ScriptedInsertionStage.INSERT:
                    break
                clipped, _mask = contract.clip(decision.action)
                _observation, _reward, done, _info = environment.step(clipped)
                if done:
                    break
            print(
                f"align-window({len(align_distances)} steps): "
                f"min dist(eef->hover)={min(hover_distances):.4f} "
                f"min dist(eef->align)={min(align_distances):.4f} "
                f"final_phase={final_phase} refusal={refusal}",
                flush=True,
            )

            for name, pose in (
                ("peg.hold(frozen)", captured["peg_hold"]),
                ("socket@freeze", captured["socket"]),
                ("sfls(socket->site, frozen)", captured["sfls"]),
                ("left.eef@freeze", captured["left_eef"]),
                ("hover.left.target", hover_left),
                ("align.left.target", align_left),
            ):
                print(_pose_text(name, pose), flush=True)
            relative_frozen_yaw = _yaw_degrees(
                captured["peg_hold"].quaternion.numpy()
            ) - _yaw_degrees(captured["socket"].quaternion.numpy())
            print(
                f"relative frozen yaw (peg_hold - socket@freeze) = "
                f"{relative_frozen_yaw:+.2f} deg; "
                f"align target yaw = {_yaw_degrees(align_left.quaternion.numpy()):+.2f} deg; "
                f"left eef yaw@freeze = "
                f"{_yaw_degrees(captured['left_eef'].quaternion.numpy()):+.2f} deg",
                flush=True,
            )

            final_geometry = build_observation(environment, contract)
            qpos_live = np.asarray(
                _expanded_robot_qpos(final_geometry.robot_state).astype(np.float64)
            )
            right_position = hold_right.position.numpy().astype(np.float64)
            right_quaternion = hold_right.quaternion.numpy().astype(np.float64)
            align_position = align_left.position.numpy().astype(np.float64)
            hover_position = hover_left.position.numpy().astype(np.float64)
            align_quaternion = align_left.quaternion.numpy().astype(np.float64)

            def run_case(label: str, position, quaternion, start=None) -> None:
                result, used = _chase(
                    solver,
                    qpos_live if start is None else start,
                    np.asarray(position, dtype=np.float64),
                    np.asarray(quaternion, dtype=np.float64),
                    right_position,
                    right_quaternion,
                )
                left_position_error = result.position_errors_m[0]
                left_rotation_error = result.orientation_errors_rad[0]
                verdict = (
                    "CONVERGED"
                    if left_position_error <= 0.005 and left_rotation_error <= 0.05
                    else "STALL"
                )
                print(
                    f"chase[{label:26s}] iters={used:3d} "
                    f"left pos_err={left_position_error:.4f} "
                    f"rot_err={left_rotation_error:.4f} "
                    f"right pos_err={result.position_errors_m[1]:.4f} -> {verdict}",
                    flush=True,
                )

            meet_left = teacher._meet_left_target()
            run_case(
                "control:meet-hold",
                meet_left.position.numpy(),
                meet_left.quaternion.numpy(),
            )
            run_case("exact:hover", hover_position, align_quaternion)
            run_case("exact:align", align_position, align_quaternion)
            for degrees in (90.0, -90.0, 180.0):
                delta = _axis_angle_quaternion((0.0, 0.0, 1.0), degrees)
                variant = quaternion_multiply(delta, align_left.quaternion)
                run_case(f"align rotZ{degrees:+.0f}", align_position, variant.numpy())
            mirror = quaternion_multiply(
                _axis_angle_quaternion((0.0, 1.0, 0.0), -120.0), align_left.quaternion
            )
            run_case("align rotY-120(tilt mirror)", align_position, mirror.numpy())
            run_case(
                "align pos-only(live quat)",
                align_position,
                final_geometry.left_eef.quaternion.numpy(),
            )
            anti_hover = quaternion_multiply(
                _axis_angle_quaternion((0.0, 0.0, 1.0), 180.0), hover_left.quaternion
            )
            run_case("hover rotZ180", hover_position, anti_hover.numpy())
            run_case(
                "hover pos-only(live quat)",
                hover_position,
                final_geometry.left_eef.quaternion.numpy(),
            )

            from gym_aloha.constants import START_ARM_POSE

            run_case(
                "align from START_ARM_POSE",
                align_position,
                align_quaternion,
                start=np.asarray(START_ARM_POSE, dtype=np.float64)[:16],
            )

            from rosetta_reality.sim.geometry_teacher import (
                GeometryPose,
                compose_pose,
            )

            peg_hold = captured["peg_hold"]
            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            park_offset = GeometryPose(
                position=torch.tensor([-0.16, 0.0, 0.0], dtype=torch.float32),
                quaternion=identity_quat,
            )
            insert_offset = GeometryPose(
                position=torch.tensor([-0.09, 0.0, 0.0], dtype=torch.float32),
                quaternion=identity_quat,
            )
            flip = GeometryPose(
                position=torch.zeros(3, dtype=torch.float32),
                quaternion=_axis_angle_quaternion((0.0, 0.0, 1.0), 180.0),
            )
            park_frame = compose_pose(peg_hold, park_offset)
            insert_frame = compose_pose(peg_hold, insert_offset)
            park_flipped = compose_pose(park_frame, flip)
            insert_flipped = compose_pose(insert_frame, flip)
            hover_flipped_world = GeometryPose(
                position=park_flipped.position
                + torch.tensor([0.0, 0.0, 0.10], dtype=torch.float32),
                quaternion=park_flipped.quaternion,
            )
            for label, socket_frame in (
                ("flipped:align(exact)", park_flipped),
                ("flipped:insert(exact)", insert_flipped),
                ("flipped:hover(world-up)", hover_flipped_world),
            ):
                site = compose_pose(socket_frame, captured["sfls"])
                run_case(label, site.position.numpy(), site.quaternion.numpy())
            parallel_insert_site = compose_pose(insert_frame, captured["sfls"])
            run_case(
                "parallel:insert(control)",
                parallel_insert_site.position.numpy(),
                parallel_insert_site.quaternion.numpy(),
            )
        finally:
            environment.close()
    return 0


def _dock(seeds: list[int], budget: int) -> int:
    """Sweep the peg-park dock distance against both arms' reachability.

    For each seed frozen at the alignment freeze, the dock distance P moves
    the peg park (right arm, ``socket + P x-hat``) and with it the insert
    target (left arm, ``peg - 0.09``); the align target is P-independent. The
    sweep chases each candidate P's joint (left insert, right dock) target
    from the frozen state in both wrist branches.
    """

    import torch

    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.sim.geometry_teacher import (
        GeometryPose,
        compose_pose,
        quaternion_multiply,
    )
    from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment
    from rosetta_reality.sim.scripted_teacher import (
        ScriptedInsertionStage,
        _expanded_robot_qpos,
        build_teacher,
    )
    from rosetta_reality.sim.scripted_teacher_observer import (
        _body_pose,
        _physics,
        build_observation,
    )

    contract = load_action_contract(REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    solver = MinkAlohaIkSolver.from_gym_aloha(
        left_site="cali_left_site1",
        right_site="cali_right_site1",
        settings=default_ik_settings(),
    )

    for seed in seeds:
        teacher = build_teacher()
        environment = GymAlohaEnvironment(contract, maximum_episode_steps=budget)
        try:
            environment.reset(seed=seed)
            teacher.reset()
            physics = _physics(environment)
            print(f"== seed {seed} ==", flush=True)
            print(_pose_text("socket.spawn", _body_pose(physics, "socket")), flush=True)
            captured = None
            steps_used = 0
            while steps_used < budget:
                geometry = build_observation(environment, contract)
                decision = teacher.decide(geometry)
                steps_used += 1
                if (
                    teacher.phase
                    in (ScriptedInsertionStage.SOCKET_APPROACH, ScriptedInsertionStage.ALIGN)
                    and captured is None
                ):
                    captured = geometry
                    break
                if decision.refusal is not None:
                    break
                clipped, _mask = contract.clip(decision.action)
                _observation, _reward, done, _info = environment.step(clipped)
                if done:
                    break
            if captured is None:
                print(f"seed {seed}: freeze never reached; no dock sweep", flush=True)
                continue
            socket_freeze = captured.socket
            peg_hold = teacher._peg_hold_pose
            sfls = teacher._socket_from_left_site
            prs = teacher._peg_from_right_site
            hold_right = teacher._hold_right_site
            if peg_hold is None or sfls is None or prs is None or hold_right is None:
                print(f"seed {seed}: freeze captures incomplete", flush=True)
                continue
            identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            flip = _axis_angle_quaternion((0.0, 0.0, 1.0), 180.0)
            qpos_live = np.asarray(
                _expanded_robot_qpos(captured.robot_state).astype(np.float64)
            )
            for distance_m in (0.16, 0.14, 0.12, 0.10, 0.08):
                dock = compose_pose(
                    socket_freeze,
                    GeometryPose(
                        position=torch.tensor([distance_m, 0.0, 0.0], dtype=torch.float32),
                        quaternion=identity,
                    ),
                )
                insert_body = compose_pose(
                    dock,
                    GeometryPose(
                        position=torch.tensor([-0.09, 0.0, 0.0], dtype=torch.float32),
                        quaternion=identity,
                    ),
                )
                park_right = compose_pose(dock, prs)
                for branch, flip_quat in (("parallel", identity), ("flipped", flip)):
                    parallel_site = compose_pose(
                        GeometryPose(
                            position=insert_body.position, quaternion=peg_hold.quaternion
                        ),
                        sfls,
                    )
                    site_quat = quaternion_multiply(flip_quat, parallel_site.quaternion)
                    result, used = _chase(
                        solver,
                        qpos_live,
                        parallel_site.position.numpy().astype(np.float64),
                        site_quat.numpy().astype(np.float64),
                        park_right.position.numpy().astype(np.float64),
                        park_right.quaternion.numpy().astype(np.float64),
                    )
                    print(
                        f"P={distance_m:.2f} [{branch:8s}] iters={used:3d} "
                        f"left(insert) pos_err={result.position_errors_m[0]:.4f} "
                        f"rot_err={result.orientation_errors_rad[0]:.4f} "
                        f"right(dock) pos_err={result.position_errors_m[1]:.4f}",
                        flush=True,
                    )
        finally:
            environment.close()
    return 0


def _geometry() -> int:
    from gym_aloha.constants import ASSETS_DIR

    model = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / "bimanual_viperx_insertion.xml"))
    for name in ("socket-1", "socket-2", "socket-3", "socket-4", "pin"):
        try:
            geom_id = model.geom(name).id
        except KeyError:
            print(name, "NOT A GEOM", flush=True)
            continue
        print(
            f"{name}: type={int(model.geom_type[geom_id])} "
            f"pos={model.geom_pos[geom_id]} size={model.geom_size[geom_id]}",
            flush=True,
        )
    for body_name in ("peg", "socket"):
        body_id = model.body(body_name).id
        print(f"{body_name} body pos: {model.body_pos[body_id]}", flush=True)
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) == body_id:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                print(
                    f"  geom {geom_name!r}: type={int(model.geom_type[geom_id])} "
                    f"pos={model.geom_pos[geom_id]} size={model.geom_size[geom_id]}",
                    flush=True,
                )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("rollout", "frames", "align", "dock", "geometry"),
        default="rollout",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seeds", type=str, default="1900,1901,1902,1903,1904")
    parser.add_argument("--align-steps", type=int, default=80)
    args = parser.parse_args()
    if args.mode == "rollout":
        return _rollout(args.seed, args.budget)
    if args.mode == "geometry":
        return _geometry()
    if args.mode == "align":
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
        return _align(seeds, args.budget, args.align_steps)
    if args.mode == "dock":
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
        return _dock(seeds, args.budget)
    return _frames()


if __name__ == "__main__":
    raise SystemExit(main())
