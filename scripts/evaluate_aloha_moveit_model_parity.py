"""Compare official MoveIt ALOHA FK with the registered Gym-ALOHA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract  # noqa: E402

LEFT_SITE = "cali_left_site1"
RIGHT_SITE = "cali_right_site1"
DEFAULT_REQUESTS = (
    REPOSITORY_ROOT
    / "integration/aloha_moveit2/config/fk_parity_requests.jsonl"
)
DEFAULT_ACTION_CONTRACT = (
    REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_lines(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object.")
        values.append(value)
    return values


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    return array


def _expanded_qpos(request: dict[str, Any]) -> np.ndarray:
    arm = _finite_vector(request.get("start"), 12, "start")
    fingers = _finite_vector(
        request.get("finger_positions", [0.02239, 0.02239]),
        2,
        "finger_positions",
    )
    return np.concatenate(
        (
            arm[:6],
            np.asarray([fingers[0], -fingers[0]]),
            arm[6:],
            np.asarray([fingers[1], -fingers[1]]),
        )
    )


def _physics(environment: GymAlohaEnvironment) -> Any:
    unwrapped = getattr(environment.raw_environment, "unwrapped", environment.raw_environment)
    control_environment = getattr(unwrapped, "_env", None)
    physics = getattr(control_environment, "physics", None)
    if physics is None:
        raise RuntimeError("Model parity requires the registered MuJoCo backend.")
    return physics


def _site_pose(physics: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    from dm_control.mujoco.wrapper.mjbindings import mjlib

    position = np.asarray(physics.named.data.site_xpos[name]).copy()
    quaternion = np.empty(4, dtype=physics.data.qpos.dtype)
    mjlib.mju_mat2Quat(quaternion, physics.named.data.site_xmat[name])
    return position, quaternion


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _indexed_results(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for value in _json_lines(path):
        request_id = value.get("request_id")
        if not isinstance(request_id, str):
            continue
        if request_id in indexed:
            raise ValueError(f"MoveIt returned duplicate request_id {request_id!r}.")
        indexed[request_id] = value
    return indexed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--moveit-results", type=Path, required=True)
    parser.add_argument("--action-contract", type=Path, default=DEFAULT_ACTION_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moveit-image", required=True)
    parser.add_argument("--moveit-image-id", required=True)
    parser.add_argument("--position-tolerance-m", type=float, default=2e-5)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=2e-5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.position_tolerance_m <= 0.0 or args.orientation_tolerance_rad <= 0.0:
        raise ValueError("Parity tolerances must be positive.")
    requests = [
        value for value in _json_lines(args.requests) if value.get("command") == "fk"
    ]
    if not requests:
        raise ValueError("Parity request set contains no FK commands.")
    request_ids = [value.get("request_id") for value in requests]
    if any(not isinstance(value, str) for value in request_ids):
        raise ValueError("Every FK parity request must have a string request_id.")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("FK parity request IDs must be unique.")
    results = _indexed_results(args.moveit_results)
    missing = sorted(set(request_ids).difference(results))
    if missing:
        raise ValueError(f"MoveIt results are missing request IDs: {missing}")

    contract = load_action_contract(args.action_contract)
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=1)
    samples: list[dict[str, Any]] = []
    maximum_position_error_m = 0.0
    maximum_orientation_error_rad = 0.0
    try:
        environment.reset(seed=10)
        physics = _physics(environment)
        for request in requests:
            request_id = str(request["request_id"])
            result = results[request_id]
            if result.get("status") != "ok":
                raise RuntimeError(f"MoveIt FK failed for {request_id}: {result}")
            if result.get("within_bounds") is not True:
                raise RuntimeError(f"MoveIt rejected in-bound parity sample {request_id}.")

            qpos = _expanded_qpos(request)
            physics.data.qpos[:16] = qpos
            physics.forward()
            arm_errors: dict[str, dict[str, float]] = {}
            for side, site_name in (("left", LEFT_SITE), ("right", RIGHT_SITE)):
                gym_position, gym_quaternion = _site_pose(physics, site_name)
                moveit_pose = result.get(side)
                if not isinstance(moveit_pose, dict):
                    raise ValueError(f"MoveIt result {request_id} lacks {side} pose.")
                moveit_position = _finite_vector(
                    moveit_pose.get("position"), 3, f"{request_id}.{side}.position"
                )
                moveit_quaternion = _finite_vector(
                    moveit_pose.get("quaternion_wxyz"),
                    4,
                    f"{request_id}.{side}.quaternion_wxyz",
                )
                position_error = float(np.linalg.norm(gym_position - moveit_position))
                orientation_error = _quaternion_distance(
                    gym_quaternion, moveit_quaternion
                )
                maximum_position_error_m = max(
                    maximum_position_error_m, position_error
                )
                maximum_orientation_error_rad = max(
                    maximum_orientation_error_rad, orientation_error
                )
                arm_errors[side] = {
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                }
            samples.append(
                {
                    "request_id": request_id,
                    "moveit_self_collision": bool(result.get("self_collision", False)),
                    "arms": arm_errors,
                }
            )
    finally:
        environment.close()

    passed = (
        maximum_position_error_m <= args.position_tolerance_m
        and maximum_orientation_error_rad <= args.orientation_tolerance_rad
    )
    payload = {
        "schema_version": 1,
        "report_type": "aloha_moveit_gym_model_parity",
        "status": "passed" if passed else "failed",
        "sample_count": len(samples),
        "maximum_position_error_m": maximum_position_error_m,
        "maximum_orientation_error_rad": maximum_orientation_error_rad,
        "position_tolerance_m": args.position_tolerance_m,
        "orientation_tolerance_rad": args.orientation_tolerance_rad,
        "requests_sha256": _sha256(args.requests),
        "moveit_results_sha256": _sha256(args.moveit_results),
        "action_contract_sha256": _sha256(args.action_contract),
        "moveit_image": args.moveit_image,
        "moveit_image_id": args.moveit_image_id,
        "gym_aloha_version": version("gym-aloha"),
        "mujoco_version": version("mujoco"),
        "simulator_seed": 10,
        "hidden_test_loaded": False,
        "dataset_rows_loaded": False,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
