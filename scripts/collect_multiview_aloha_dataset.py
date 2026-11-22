"""CPU-ready multi-view 4D-stratified ALOHA insertion dataset collector and quality gate.

Addresses the two primary generalization bottlenecks identified in the 2026-11-22
quantitative audit:
1. Single top-down camera occlusion & depth blindness -> renders multiple verified
   MuJoCo viewpoints dynamically checked against ``physics.model.ncam``.
2. Sparse 40-episode 4D workspace coverage (mean LOO NN1 = 26.9 mm) -> generates
   Latin-Hypercube stratified 4D initial poses (socket XY x peg XY) and enforces
   per-episode safety, task-success, and trajectory-smoothness admission gates.

Runs on CPU (no GPU required; compatible with AutoDL no-GPU 128-vCPU containers).
Does not mutate any historical dataset, checkpoint, or hash-pinned module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def _load_generalization_controls():
    try:
        from rosetta_reality.vla import generalization_controls as module

        return module
    except ModuleNotFoundError:
        path = ROOT / "src/rosetta_reality/vla/generalization_controls.py"
        spec = importlib.util.spec_from_file_location("_rosetta_generalization_controls", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_rosetta_generalization_controls"] = module
        spec.loader.exec_module(module)
        return module


_GC = _load_generalization_controls()
Pose4D = _GC.Pose4D
audit_4d_workspace_coverage = _GC.audit_4d_workspace_coverage
compute_trajectory_dynamics = _GC.compute_trajectory_dynamics
generate_stratified_4d_poses = _GC.generate_stratified_4d_poses
split_episodes_disjoint = _GC.split_episodes_disjoint
validate_multiview_camera_contract = _GC.validate_multiview_camera_contract

DEFAULT_CAMERAS = ("top", "angle", "front_close")
DEFAULT_INSTRUCTION = "Insert the peg into the socket."


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def available_mujoco_cameras(physics: Any) -> list[str]:
    """Enumerate all compiled camera names directly from the live MuJoCo model."""
    model = physics.model
    names: list[str] = []
    for cam_id in range(int(model.ncam)):
        name = model.id2name(cam_id, "camera")
        if name:
            names.append(str(name))
    return names


def resolve_verified_cameras(physics: Any, requested_cameras: list[str]) -> list[str]:
    """Fail closed if any requested camera does not exist in the compiled MuJoCo model."""
    compiled = available_mujoco_cameras(physics)
    missing = [cam for cam in requested_cameras if cam not in compiled]
    if missing:
        raise ValueError(
            f"Requested cameras missing from compiled MuJoCo model: {missing}; "
            f"available={compiled}"
        )
    return list(requested_cameras)


def render_multiview_frames(
    physics: Any,
    camera_names: list[str],
    *,
    height: int = 480,
    width: int = 640,
) -> dict[str, Any]:
    """Render uint8 HWC RGB copies for each verified camera name."""
    import numpy as np

    frames: dict[str, Any] = {}
    for cam in camera_names:
        pixels = physics.render(height=height, width=width, camera_id=cam)
        arr = np.ascontiguousarray(pixels, dtype=np.uint8).copy()
        if arr.ndim != 3 or arr.shape != (height, width, 3):
            raise ValueError(f"Camera {cam} rendered unexpected shape {arr.shape}")
        frames[cam] = arr
    return frames


def place_objects_4d(physics: Any, pose: Pose4D, *, z_height: float = 0.05) -> None:
    """Position socket and peg freejoints to exact 4D tabletop coordinates before rollout.
    将 socket 与 peg 自由关节精确放置到目标 4D 分层坐标并刷新前向动力学，保证初始位姿覆盖可控。
    """
    model = physics.model
    data = physics.data
    placements = (
        ("red_peg_joint", pose.peg_x, pose.peg_y, z_height),
        ("blue_socket_joint", pose.socket_x, pose.socket_y, z_height),
    )
    for joint_name, x_m, y_m, z_m in placements:
        jnt_id = int(model.name2id(joint_name, "joint"))
        if jnt_id < 0:
            raise ValueError(f"Expected freejoint '{joint_name}' in MuJoCo model")
        adr = int(model.jnt_qposadr[jnt_id])
        data.qpos[adr : adr + 3] = [float(x_m), float(y_m), float(z_m)]
        data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        vel_adr = int(model.jnt_dofadr[jnt_id])
        data.qvel[vel_adr : vel_adr + 6] = 0.0
    physics.forward()


def evaluate_episode_quality(
    *,
    success: bool,
    joint_limit_violations: int,
    unexpected_collisions: int,
    dynamics: dict[str, float | int],
    max_arm_signflip: float = 0.15,
    min_straightness: float = 0.15,
    min_steps: int = 20,
) -> dict[str, Any]:
    """Apply strict admission gate so only safe, smooth, successful demos enter the dataset."""
    reasons: list[str] = []
    if not success:
        reasons.append("task_not_succeeded")
    if joint_limit_violations > 0:
        reasons.append(f"joint_limit_violations={joint_limit_violations}")
    if unexpected_collisions > 0:
        reasons.append(f"unexpected_collisions={unexpected_collisions}")
    if int(dynamics["steps"]) < min_steps:
        reasons.append(f"too_short={dynamics['steps']}")
    if float(dynamics["arm_signflip_rate"]) > max_arm_signflip:
        reasons.append(
            f"arm_signflip_rate={dynamics['arm_signflip_rate']:.4f}>{max_arm_signflip:.4f}"
        )
    if float(dynamics["straightness_full"]) < min_straightness:
        reasons.append(
            f"straightness_full={dynamics['straightness_full']:.4f}<{min_straightness:.4f}"
        )
    return {
        "accepted": len(reasons) == 0,
        "rejection_reasons": reasons,
    }


def collect_episode(
    *,
    episode_index: int,
    seed: int,
    pose: Pose4D,
    camera_names: list[str],
    output_dir: Path,
    maximum_steps: int = 400,
    instruction: str = DEFAULT_INSTRUCTION,
    max_arm_signflip: float = 0.15,
    min_straightness: float = 0.15,
    height: int = 480,
    width: int = 640,
    env_factory=None,
    teacher_factory=None,
    frame_renderer=None,
    pose_placer=None,
    observation_builder=None,
) -> dict[str, Any]:
    """Collect one multi-view teacher episode and gate it by safety and smoothness.

    Supports dependency injection (``env_factory``, ``teacher_factory``,
    ``frame_renderer``, ``pose_placer``, ``observation_builder``) for pure
    stdlib unit testing without MuJoCo.
    """
    if env_factory is None:
        from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract

        contract = load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
        env = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    else:
        contract, env = env_factory(maximum_steps)

    if teacher_factory is None:
        from rosetta_reality.sim.coupled_translation_teacher import build_teacher

        teacher = build_teacher()
    else:
        teacher = teacher_factory()

    if observation_builder is None:
        from rosetta_reality.sim.scripted_teacher_observer import build_observation

        observation_builder = build_observation

    ep_dir = output_dir / f"episode-{episode_index:04d}"
    frames_dir = ep_dir / "frames"
    ep_dir.mkdir(parents=True, exist_ok=False)
    frames_dir.mkdir(parents=True, exist_ok=False)

    actions: list[list[float]] = []
    states: list[list[float]] = []
    trace_rows: list[dict[str, Any]] = []
    joint_limit_violations = 0
    unexpected_collisions = 0
    success = False
    max_reward = 0.0

    try:
        obs = env.reset(seed=seed)
        physics = None
        if frame_renderer is None or pose_placer is None:
            unwrapped = getattr(env.raw_environment, "unwrapped", env.raw_environment)
            physics = getattr(getattr(unwrapped, "_env", None), "physics", None)
            if physics is None:
                raise RuntimeError("Underlying MuJoCo physics handle unavailable")
        if pose_placer is None:
            resolve_verified_cameras(physics, camera_names)
            place_objects_4d(physics, pose)
        else:
            pose_placer(env, pose)
        teacher.reset()

        for step_idx in range(maximum_steps):
            teacher_obs = observation_builder(env, contract)
            raw_action = teacher.act(teacher_obs)
            command, _ = contract.clip(raw_action)
            cmd_list = command.detach().cpu().tolist() if hasattr(command, "detach") else [float(v) for v in command]
            state_before = (
                obs["robot_state"].detach().cpu().tolist()
                if hasattr(obs["robot_state"], "detach")
                else [float(v) for v in obs["robot_state"]]
            )

            if frame_renderer is None:
                import numpy as np

                frame_dict = render_multiview_frames(physics, camera_names, height=height, width=width)
                frame_path = frames_dir / f"step-{step_idx:04d}.npz"
                np.savez_compressed(frame_path, **frame_dict)
                frame_rel = str(frame_path.relative_to(ep_dir)).replace("\\", "/")
                frame_sha = sha256_file(frame_path)
            else:
                frame_rel, frame_sha = frame_renderer(env, camera_names, frames_dir, step_idx)

            obs, reward, done, info = env.step(command)
            reward_f = float(reward)
            if not math.isfinite(reward_f):
                raise ValueError("Nonfinite reward during dataset collection")
            max_reward = max(max_reward, reward_f)
            state_after = (
                obs["robot_state"].detach().cpu().tolist()
                if hasattr(obs["robot_state"], "detach")
                else [float(v) for v in obs["robot_state"]]
            )

            diag = env.diagnostic_snapshot()
            step_jlv = len(diag.get("joint_limit_violations", []))
            joint_limit_violations += step_jlv
            step_coll = sum(
                1 for a, b in env.contact_pairs() if env.is_unexpected_collision_pair(a, b)
            )
            unexpected_collisions += step_coll
            success = success or bool(info.get("is_success", False))

            actions.append(cmd_list)
            states.append(state_after)
            trace_rows.append(
                {
                    "step": step_idx,
                    "task": instruction,
                    "state": state_before,
                    "action": cmd_list,
                    "state_after": state_after,
                    "reward": reward_f,
                    "images": frame_rel,
                    "images_sha256": frame_sha,
                    "safe": step_jlv == 0 and step_coll == 0,
                }
            )
            if done or success:
                break
    finally:
        env.close()

    dynamics = compute_trajectory_dynamics(actions, states)
    quality = evaluate_episode_quality(
        success=success,
        joint_limit_violations=joint_limit_violations,
        unexpected_collisions=unexpected_collisions,
        dynamics=dynamics,
        max_arm_signflip=max_arm_signflip,
        min_straightness=min_straightness,
    )

    trace_path = ep_dir / "trace.jsonl"
    with trace_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in trace_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    summary = {
        "episode_index": episode_index,
        "seed": seed,
        "pose_4d": {
            "socket_x": pose.socket_x,
            "socket_y": pose.socket_y,
            "peg_x": pose.peg_x,
            "peg_y": pose.peg_y,
        },
        "camera_names": list(camera_names),
        "steps": len(trace_rows),
        "success": success,
        "max_reward": max_reward,
        "joint_limit_violations": joint_limit_violations,
        "unexpected_collisions": unexpected_collisions,
        "dynamics": dynamics,
        "quality": quality,
        "trace_relpath": str(trace_path.relative_to(output_dir)).replace("\\", "/"),
        "trace_sha256": sha256_file(trace_path),
    }
    write_json_new(ep_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--val-count", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--maximum-steps", type=int, default=400)
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--empty-cameras", type=int, default=0)
    parser.add_argument("--require-wrist-pair", action="store_true")
    parser.add_argument("--target-mean-nn-mm", type=float, default=18.0)
    parser.add_argument("--max-arm-signflip", type=float, default=0.15)
    parser.add_argument("--min-straightness", type=float, default=0.15)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--inspect-cameras", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    camera_keys = [f"observation.images.{c}" for c in args.cameras]
    contract_check = validate_multiview_camera_contract(
        camera_keys,
        empty_cameras=args.empty_cameras,
        require_wrist_pair=args.require_wrist_pair,
    )
    poses = generate_stratified_4d_poses(args.episodes, seed=args.seed)
    coverage = audit_4d_workspace_coverage(poses, target_mean_nn_mm=args.target_mean_nn_mm)
    planned_splits = split_episodes_disjoint(
        range(args.episodes),
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )

    if args.check_only:
        print(
            json.dumps(
                {
                    "schema": "rosetta.multiview_collection_check.v1",
                    "valid": True,
                    "camera_contract": contract_check,
                    "planned_coverage_4d": coverage,
                    "split_sizes": {k: len(v) for k, v in planned_splits.items()},
                    "simulator_executed": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.inspect_cameras:
        from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract

        contract = load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
        env = GymAlohaEnvironment(contract, maximum_episode_steps=10)
        try:
            env.reset(seed=args.seed)
            unwrapped = getattr(env.raw_environment, "unwrapped", env.raw_environment)
            physics = getattr(getattr(unwrapped, "_env", None), "physics", None)
            compiled = available_mujoco_cameras(physics)
        finally:
            env.close()
        print(
            json.dumps(
                {
                    "schema": "rosetta.mujoco_camera_inventory.v1",
                    "compiled_cameras": compiled,
                    "requested_cameras": list(args.cameras),
                    "missing_cameras": [c for c in args.cameras if c not in compiled],
                },
                sort_keys=True,
            )
        )
        return 0

    if args.output.exists():
        raise FileExistsError("Multi-view dataset output directory must be create-only")
    args.output.mkdir(parents=True, exist_ok=False)

    started = time.monotonic()
    summaries = []
    accepted_indices = []
    rejected_indices = []
    for idx, pose in enumerate(poses):
        summary = collect_episode(
            episode_index=idx,
            seed=args.seed + idx,
            pose=pose,
            camera_names=list(args.cameras),
            output_dir=args.output,
            maximum_steps=args.maximum_steps,
            max_arm_signflip=args.max_arm_signflip,
            min_straightness=args.min_straightness,
            height=args.height,
            width=args.width,
        )
        summaries.append(summary)
        if summary["quality"]["accepted"]:
            accepted_indices.append(idx)
        else:
            rejected_indices.append(idx)

    accepted_poses = [poses[i] for i in accepted_indices]
    accepted_coverage = (
        audit_4d_workspace_coverage(accepted_poses, target_mean_nn_mm=args.target_mean_nn_mm)
        if len(accepted_poses) >= 2
        else None
    )
    accepted_splits = (
        split_episodes_disjoint(
            accepted_indices,
            val_count=min(args.val_count, max(0, len(accepted_indices) // 10)),
            test_count=min(args.test_count, max(0, len(accepted_indices) // 10)),
            seed=args.seed,
        )
        if len(accepted_indices) >= 3
        else {"train": accepted_indices, "val": [], "test": []}
    )
    manifest = {
        "schema": "rosetta.multiview_aloha_dataset.v1",
        "episodes_attempted": len(poses),
        "episodes_accepted": len(accepted_indices),
        "episodes_rejected": len(rejected_indices),
        "accepted_episode_indices": accepted_indices,
        "rejected_episode_indices": rejected_indices,
        "camera_contract": contract_check,
        "planned_coverage_4d": coverage,
        "accepted_coverage_4d": accepted_coverage,
        "splits": accepted_splits,
        "elapsed_seconds": time.monotonic() - started,
        "episodes": summaries,
    }
    write_json_new(args.output / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
