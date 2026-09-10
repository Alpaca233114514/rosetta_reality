"""Read-only G1 diagnostic for the geometric teacher candidate.

Replays the exact calibration episode (episode 2 / seed 10) with the
registered candidate ``mink-geometric-insertion-teacher-001`` and prints a
per-step state-conditioned trace plus the first unexpected-collision contact
pair.  Writes no gate evidence: this is a diagnosis instrument, not a gate
stage.
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
from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment  # noqa: E402
from rosetta_reality.sim.scripted_teacher import ScriptedInsertionStage  # noqa: E402
from rosetta_reality.sim.scripted_teacher_observer import (  # noqa: E402
    build_observation,
)


def _contact_pairs(environment: Any) -> list[str]:
    return ["|".join(sorted(pair)) for pair in environment.contact_pairs()]


def _violated_joints(environment: Any) -> list[str]:
    from rosetta_reality.sim.scripted_teacher_observer import _physics

    physics = _physics(environment)
    names: list[str] = []
    model, data = physics.model, physics.data
    for joint_id in range(int(model.njnt)):
        if not bool(model.jnt_limited[joint_id]):
            continue
        address = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        value = float(data.qpos[address])
        if value < float(lower) - 1e-5 or value > float(upper) + 1e-5:
            names.append(f"{model.joint(joint_id).name}={value:.4f}")
    return names


def _pin_position(environment: Any):
    """Read-only pin geom position (diagnostic only; not in the teacher contract)."""

    try:
        from rosetta_reality.sim.scripted_teacher_observer import _physics

        physics = _physics(environment)
        geom_id = int(physics.model.geom("pin").id)
        return np.asarray(physics.data.geom_xpos[geom_id]).copy()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ik-iterations", type=int, default=None)
    parser.add_argument("--align-unshifted", action="store_true")
    arguments = parser.parse_args()

    from rosetta_reality.sim.mink_aloha_ik import MinkAlohaIkSettings
    from rosetta_reality.sim.scripted_teacher import default_ik_settings

    ik_settings = default_ik_settings()
    if arguments.ik_iterations is not None:
        ik_settings = MinkAlohaIkSettings(
            **{
                **{
                    field: getattr(ik_settings, field)
                    for field in (
                        "integration_timestep_s",
                        "position_cost",
                        "orientation_cost",
                        "posture_cost",
                        "frame_lm_damping",
                        "solver_damping",
                        "maximum_joint_velocity_rad_s",
                        "configuration_limit_gain",
                        "joint_limit_margin_rad",
                    )
                },
                "maximum_iterations": arguments.ik_iterations,
            }
        )

    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    teacher = build_teacher(ik_settings=ik_settings)
    if arguments.align_unshifted:
        # Read-only experiment: restrict the branch seeds to INSERT only.
        teacher._BRANCH_SEED_STAGES = frozenset({ScriptedInsertionStage.INSERT})
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=arguments.budget)
    maximum_reward = 0.0
    cumulative_joint_limits = 0
    cumulative_collisions = 0
    try:
        environment.reset(seed=arguments.seed)
        teacher.reset()
        for step in range(arguments.budget):
            geometry = build_observation(environment, contract)
            decision = teacher.decide(geometry)
            if decision.refusal is not None:
                print(
                    f"step {step}: REFUSAL {decision.refusal.reason.value}: "
                    f"{decision.refusal.detail}"
                )
                break
            diagnostic = teacher.last_command_diagnostic
            phase = teacher.phase.value
            clipped, _mask = contract.clip(decision.action)
            observation, reward, done, info = environment.step(clipped)
            maximum_reward = max(maximum_reward, float(reward))
            unexpected = environment.unexpected_collision_count()
            joint_limits = environment.state_limit_violation_count()
            if joint_limits > 0:
                print(
                    f"step {step}: joint-limit violations={joint_limits} in phase "
                    f"{phase}: {_violated_joints(environment)}"
                )
                cumulative_joint_limits += joint_limits
            if unexpected > cumulative_collisions:
                print(
                    f"step {step}: unexpected contacts {cumulative_collisions}"
                    f"->{unexpected} in phase {phase}: {_contact_pairs(environment)}"
                )
                cumulative_collisions = unexpected
            if arguments.verbose or step % 10 == 0:
                assert diagnostic is not None
                left = geometry.left_eef.position.tolist()
                right = geometry.right_eef.position.tolist()
                socket_pos = geometry.socket.position.tolist()
                peg_pos = geometry.peg.position.tolist()
                from rosetta_reality.sim.geometry_teacher import (
                    GeometryPose,
                    compose_pose,
                )

                park = compose_pose(
                    geometry.socket,
                    GeometryPose(
                        position=torch.tensor([0.16, 0.0, 0.0], dtype=torch.float32),
                        quaternion=torch.tensor(
                            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32
                        ),
                    ),
                )
                peg_park_distance = float(
                    torch.linalg.vector_norm(geometry.peg.position - park.position)
                )
                try:
                    align_park = teacher._socket_body_target(geometry, 0.16)
                    align_gap = float(
                        torch.linalg.vector_norm(geometry.socket.position - align_park.position)
                    )
                except Exception:
                    align_gap = -1.0
                esc = int(teacher._align_escape_engaged)
                pin = _pin_position(environment)
                pin_gap = (
                    float(np.linalg.norm(pin - geometry.peg.position.numpy()))
                    if pin is not None
                    else -1.0
                )
                print(
                    f"step {step:3d} phase {phase:15s} reward {reward:.0f} "
                    f"L=({left[0]:+.3f},{left[1]:+.3f},{left[2]:+.3f}) "
                    f"R=({right[0]:+.3f},{right[1]:+.3f},{right[2]:+.3f}) "
                    f"sock=({socket_pos[0]:+.3f},{socket_pos[1]:+.3f},{socket_pos[2]:+.3f}) "
                    f"peg=({peg_pos[0]:+.3f},{peg_pos[1]:+.3f},{peg_pos[2]:+.3f}) "
                    f"peg2park={peg_park_distance:.3f} "
                    f"aligngap={align_gap:.3f} esc={esc} pingap={pin_gap:.3f} "
                    f"grip=({float(decision.action[6]):.2f},{float(decision.action[13]):.2f}) "
                    f"scale={diagnostic.command_scale:.2f} "
                    f"resP=({diagnostic.left_position_residual_m:.3f},"
                    f"{diagnostic.right_position_residual_m:.3f}) "
                    f"resO=({diagnostic.left_orientation_residual_rad:.3f},"
                    f"{diagnostic.right_orientation_residual_rad:.3f}) "
                    f"unexp={unexpected}"
                )
            if unexpected:
                print(f"step {step}: unexpected contact pairs: {_contact_pairs(environment)}")
                break
            if done:
                print(
                    f"step {step}: done success={info.get('is_success')} "
                    f"max_reward={maximum_reward}"
                )
                break
    finally:
        environment.close()
    print(f"final: max_reward={maximum_reward} phase={teacher.phase.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
