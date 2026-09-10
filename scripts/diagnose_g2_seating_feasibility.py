"""Privileged G2 seating-feasibility probe (read-only, teacher-free verdict).

Answers the question recorded in the -003 closure: are the G2 poses 1901/
1904 seatable AT ALL under this simulator's position control, or is the
remaining wall teacher-internal?  This is a diagnostic instrument, not a
teacher and not a gate stage; it writes no gate evidence.

Method:
1. Run the registered candidate-003 teacher on a solving G2 sibling (seed
   1903) and capture the seated peg->socket relative transform at terminal
   success (privileged calibration from a solved sibling).
2. Run the teacher on the probe seed until its INSERT stall (the exact
   state where the candidate died).
3. Hand control to a privileged seater: every step it re-composes the
   ideal socket body target from the LIVE peg frame (no frozen captures),
   anchors the left-site target as ideal-position plus the live
   socket->site world offset (position-priority: the orientation target is
   the live left quaternion, so no orientation pressure), holds the right
   arm at its live pose, solves multistart constrained IK from the live
   state, and commands the solved joints with the registered 0.06 rad
   action bound.  The right arm hold and the closed grippers match the
   registered execution semantics.
4. Sanity control: the same probe on the solving seed must seat.

If the privileged seater reaches pin contact / reward 4, the wall is
teacher-internal (frozen-capture drift, exit semantics) and a live-
geometry teacher design is indicated; if it floors short, the G2 pose
demands precision beyond the position-control envelope — a protocol-level
finding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
for root in (str(SRC_ROOT),):
    if root not in sys.path:
        sys.path.insert(0, root)

from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.sim.geometric_teacher import build_teacher  # noqa: E402
from rosetta_reality.sim.geometry_teacher import (  # noqa: E402
    GeometryPose,
    compose_pose,
    relative_pose,
)
from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment  # noqa: E402
from rosetta_reality.sim.scripted_teacher import (  # noqa: E402
    ScriptedInsertionStage,
    _expanded_robot_qpos,
)
from rosetta_reality.sim.scripted_teacher_observer import (  # noqa: E402
    build_observation,
)

ARM_INDICES = (*range(6), *range(7, 13))


def _run_teacher_until(
    seed: int,
    *,
    contract: Any,
    stop_at: str,
    maximum_steps: int = 500,
) -> tuple[Any, Any, GeometryPose | None, int, float, bool]:
    """Run the registered teacher; return (env, teacher, seated_rel, step,
    max_reward, succeeded), with the environment left open at the stop
    state. ``seated_rel`` is captured at terminal success; ``stop_at`` is
    one of ``success``, ``stall`` (the INSERT deadlock) or
    ``insert_entry`` (the sanity-control handoff point)."""

    teacher = build_teacher()
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    observation = environment.reset(seed=seed)
    teacher.reset()
    seated_rel: GeometryPose | None = None
    maximum_reward = 0.0
    stop_step = -1
    for step in range(maximum_steps):
        geometry = build_observation(environment, contract)
        decision = teacher.decide(geometry)
        if decision.refusal is not None:
            stop_step = step
            break
        clipped, _mask = contract.clip(decision.action)
        observation, reward, done, info = environment.step(clipped)
        maximum_reward = max(maximum_reward, float(reward))
        if done:
            seated_rel = relative_pose(geometry.peg, geometry.socket)
            stop_step = step
            return environment, teacher, seated_rel, stop_step, maximum_reward, bool(
                info.get("is_success", False)
            )
        if stop_at == "insert_entry" and (
            teacher.phase is ScriptedInsertionStage.INSERT
        ):
            stop_step = step
            return environment, teacher, None, stop_step, maximum_reward, False
        if stop_at == "stall" and (
            teacher.phase is ScriptedInsertionStage.INSERT
            and teacher._movement_stalled()
        ):
            stop_step = step
            return (
                environment,
                teacher,
                None,
                stop_step,
                maximum_reward,
                False,
            )
    if stop_step < 0:
        stop_step = maximum_steps
    return environment, teacher, None, stop_step, maximum_reward, False


def _privileged_seater_step(
    environment: Any,
    contract: Any,
    teacher: Any,
    solver: Any,
    seated_rel: GeometryPose,
    seed_shifts: tuple[tuple[int, float], ...],
    maximum_joint_delta: float,
) -> dict[str, float | bool]:
    geometry = build_observation(environment, contract)
    ideal_socket = compose_pose(geometry.peg, seated_rel)
    left_anchor_offset = geometry.left_eef.position - geometry.socket.position
    left_target = GeometryPose(
        position=ideal_socket.position + left_anchor_offset,
        quaternion=geometry.left_eef.quaternion,
    )
    right_target = geometry.right_eef
    expanded = _expanded_robot_qpos(geometry.robot_state)
    best: tuple[float, Any] | None = None
    seeds = [expanded]
    try:
        result = solver.solve(
            expanded,
            left_position=left_target.position.numpy(),
            left_quaternion_wxyz=left_target.quaternion.numpy(),
            right_position=right_target.position.numpy(),
            right_quaternion_wxyz=right_target.quaternion.numpy(),
        )
        best = (max(result.position_errors_m), result)
    except (RuntimeError, ValueError):
        best = None
    if best is None or best[0] > 0.005:
        # Multistart only when the unshifted branch is inadequate.
        for joint_index, shift in seed_shifts:
            seed = expanded.copy()
            seed[joint_index] += shift
            try:
                result = solver.solve(
                    seed,
                    left_position=left_target.position.numpy(),
                    left_quaternion_wxyz=left_target.quaternion.numpy(),
                    right_position=right_target.position.numpy(),
                    right_quaternion_wxyz=right_target.quaternion.numpy(),
                )
            except (RuntimeError, ValueError):
                continue
            residual = max(result.position_errors_m)
            if best is None or residual < best[0]:
                best = (residual, result)
            if residual <= 0.005:
                break
    if best is None:
        raise RuntimeError("privileged seater: every IK seed failed")
    _residual, result = best
    raw = geometry.robot_state.clone()
    raw[:6] = torch.as_tensor(result.qpos[:6], dtype=torch.float32)
    raw[7:13] = torch.as_tensor(result.qpos[8:14], dtype=torch.float32)
    raw[6] = 0.09
    raw[13] = 0.09
    current = geometry.robot_state
    deltas = [float(raw[i]) - float(current[i]) for i in ARM_INDICES]
    scale = min(1.0, maximum_joint_delta / max(abs(d) for d in deltas))
    for offset, index in enumerate(ARM_INDICES):
        raw[index] = float(current[index]) + scale * deltas[offset]
    clipped, _mask = contract.clip(raw)
    observation, reward, done, info = environment.step(clipped)
    socket_gap = float(
        torch.linalg.vector_norm(geometry.socket.position - ideal_socket.position)
    )
    return {
        "reward": float(reward),
        "done": bool(done),
        "success": bool(info.get("is_success", False)),
        "socket_gap": socket_gap,
        "left_residual": max(result.position_errors_m),
        "pin_contact": bool(
            any(
                frozenset(pair) == frozenset({"red_peg", "pin"})
                for pair in environment.contact_pairs()
            )
        ),
    }


def probe_seed(
    seed: int,
    *,
    contract: Any,
    solver: Any,
    seated_rel: GeometryPose,
    seed_shifts: tuple[tuple[int, float], ...],
    maximum_joint_delta: float,
    budget: int,
    stop_at: str = "stall",
) -> dict[str, Any]:
    environment, teacher, _rel, step, max_reward, solved = _run_teacher_until(
        seed, contract=contract, stop_at=stop_at
    )
    if solved:
        environment.close()
        return {"seed": seed, "teacher_solved": True}
    try:
        minimum_gap = float("inf")
        maximum_reward = max_reward
        pinned = False
        success = False
        for _ in range(budget):
            outcome = _privileged_seater_step(
                environment,
                contract,
                teacher,
                solver,
                seated_rel,
                seed_shifts,
                maximum_joint_delta,
            )
            minimum_gap = min(minimum_gap, outcome["socket_gap"])
            maximum_reward = max(maximum_reward, outcome["reward"])
            pinned = pinned or outcome["pin_contact"]
            if outcome["done"]:
                success = outcome["success"]
                break
        return {
            "seed": seed,
            "teacher_solved": False,
            "stall_step": step,
            "teacher_max_reward": max_reward,
            "privileged_min_socket_gap": minimum_gap,
            "privileged_max_reward": maximum_reward,
            "privileged_pin_contact": pinned,
            "privileged_success": success,
        }
    finally:
        environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-seed", type=int, default=1903)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[1901, 1904])
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--skip-control", action="store_true")
    arguments = parser.parse_args()

    from rosetta_reality.sim.geometric_teacher import GeometricCommandSettings
    from rosetta_reality.sim.mink_aloha_ik import MinkAlohaIkSolver
    from rosetta_reality.sim.scripted_teacher import (
        default_ik_settings,
    )

    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    solver = MinkAlohaIkSolver.from_gym_aloha(
        left_site="cali_left_site1",
        right_site="cali_right_site1",
        settings=default_ik_settings(),
    )
    shifts = GeometricCommandSettings().ik_seed_shifts
    maximum_delta = build_teacher().settings.maximum_joint_target_delta_rad

    environment, _teacher, seated_rel, step, reward, solved = _run_teacher_until(
        arguments.calibration_seed, contract=contract, stop_at="success"
    )
    environment.close()
    if not solved or seated_rel is None:
        print(f"calibration seed {arguments.calibration_seed} did not solve; aborting")
        return 1
    print(
        f"calibration: seed {arguments.calibration_seed} solved at step {step} "
        f"(reward {reward:.0f}); seated transform captured"
    )

    if not arguments.skip_control:
        control = probe_seed(
            arguments.calibration_seed,
            contract=contract,
            solver=solver,
            seated_rel=seated_rel,
            seed_shifts=shifts,
            maximum_joint_delta=maximum_delta,
            budget=arguments.budget,
            stop_at="insert_entry",
        )
        print(f"sanity control (seater from insert entry): {control}")

    for seed in arguments.probe_seeds:
        outcome = probe_seed(
            seed,
            contract=contract,
            solver=solver,
            seated_rel=seated_rel,
            seed_shifts=shifts,
            maximum_joint_delta=maximum_delta,
            budget=arguments.budget,
        )
        print(f"probe {seed}: {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
