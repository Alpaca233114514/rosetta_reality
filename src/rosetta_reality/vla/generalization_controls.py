"""Generalization controls and dataset-coverage diagnostics for SmolVLA (create-only).

Addresses the four quantitative bottlenecks identified in the 2026-11-22 closed-loop
and data-scaling audit without mutating any historical hash-pinned modules:
1. Monotonic ``bounded_sine_clamped`` gripper decoding (clamps latent values to
   [-pi/2, +pi/2] before sin() so closing overshoot never folds back open).
2. ``TemporalEnsembler`` (exponentially weighted overlapping chunk fusion) to suppress
   50/1 white-noise chattering while retaining 50 Hz closed-loop correction.
3. Standardized trajectory dynamics auditing (``arm_signflip_rate``, ``grip_signflip_rate``,
   ``mean_arm_step_delta``, ``straightness``) for both teacher data and policy rollouts.
4. Stratified 4D initial-pose sampling and nearest-neighbor workspace coverage auditing
   plus multi-view camera contract validation (top + wrist / angle views).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

HALF_PI = math.pi / 2.0
ARM_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPER_INDICES: tuple[int, int] = (6, 13)
SUPPORTED_MULTIVIEW_CAMERAS: tuple[str, ...] = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.angle",
)


def _to_float_list(values: Sequence[float]) -> list[float]:
    raw = values.detach().cpu().tolist() if hasattr(values, "detach") else list(values)
    out = [float(v) for v in raw]
    if any(not math.isfinite(v) for v in out):
        raise ValueError("Nonfinite scalar encountered in action vector")
    return out


def gripper_to_bounded_sine(values: Sequence[float]) -> list[float]:
    """Encode normalized [0, 1] gripper commands into [-pi/2, +pi/2] arcsin space."""
    result = []
    for raw in _to_float_list(values):
        clamped = min(1.0, max(0.0, raw))
        result.append(math.asin(max(-1.0, min(1.0, 2.0 * clamped - 1.0))))
    return result


def clamped_bounded_sine_to_gripper(values: Sequence[float]) -> list[float]:
    """Decode bounded-sine gripper latents monotonically by clamping to [-pi/2, +pi/2].
    先将潜变量截断在 [-pi/2, +pi/2] 再求 sin，避免闭合端过冲超过 -0.904 sigma 时反向折返张开。
    """
    result = []
    for raw in _to_float_list(values):
        clamped = max(-HALF_PI, min(HALF_PI, raw))
        result.append(0.5 * (math.sin(clamped) + 1.0))
    return result


def apply_clamped_bounded_sine_override(
    raw_action: Sequence[float],
    internal_action: Sequence[float],
    gripper_indices: tuple[int, int] = GRIPPER_INDICES,
) -> list[float]:
    """Replace gripper entries of a decoded action using monotonic clamped sin decoding."""
    out = _to_float_list(raw_action)
    internal = _to_float_list(internal_action)
    if len(out) != len(internal):
        raise ValueError("raw_action and internal_action must have matching lengths")
    decoded = clamped_bounded_sine_to_gripper([internal[i] for i in gripper_indices])
    for idx, val in zip(gripper_indices, decoded, strict=True):
        out[idx] = val
    return out


def bounded_sine_fold_stats(latents: Sequence[float]) -> dict[str, float | int]:
    """Measure how many bounded-sine gripper latents overshoot ±pi/2 and would fold back."""
    vals = _to_float_list(latents)
    below = sum(1 for v in vals if v < -HALF_PI)
    above = sum(1 for v in vals if v > HALF_PI)
    max_abs_diff = 0.0
    for v in vals:
        unclamped = 0.5 * (math.sin(v) + 1.0)
        clamped = 0.5 * (math.sin(max(-HALF_PI, min(HALF_PI, v))) + 1.0)
        max_abs_diff = max(max_abs_diff, abs(unclamped - clamped))
    return {
        "count": len(vals),
        "below_neg_half_pi": below,
        "above_pos_half_pi": above,
        "total_folds": below + above,
        "fold_rate": (below + above) / len(vals) if vals else 0.0,
        "max_decode_delta": max_abs_diff,
    }


class TemporalEnsembler:
    """ACT/SmolVLA exponentially weighted temporal ensemble over overlapping chunks.
    对多步重叠预测 chunk 按指数权重 exp(-k * age) 融合，在保留 50Hz 闭环纠偏的同时消除 50/1 白噪声抖动。
    """

    def __init__(self, chunk_size: int, decay_k: float = 0.01, action_dim: int = 14) -> None:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not math.isfinite(decay_k) or decay_k < 0.0:
            raise ValueError("decay_k must be a finite non-negative float")
        if type(action_dim) is not int or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        self.chunk_size = chunk_size
        self.decay_k = float(decay_k)
        self.action_dim = action_dim
        self._step = 0
        self._history: list[tuple[int, list[list[float]]]] = []

    def reset(self) -> None:
        self._step = 0
        self._history.clear()

    def update(self, chunk: Sequence[Sequence[float]]) -> list[float]:
        """Register the newly predicted chunk for the current step and return fused action."""
        rows = chunk.detach().cpu().tolist() if hasattr(chunk, "detach") else [list(r) for r in chunk]
        if len(rows) < self.chunk_size:
            raise ValueError(f"Expected chunk length >= {self.chunk_size}, got {len(rows)}")
        normalized_chunk: list[list[float]] = []
        for slot in rows[: self.chunk_size]:
            vec = _to_float_list(slot)
            if len(vec) != self.action_dim:
                raise ValueError(f"Expected action_dim={self.action_dim}, got {len(vec)}")
            normalized_chunk.append(vec)

        t = self._step
        self._history.append((t, normalized_chunk))
        # Drop chunks whose horizon no longer covers step t.
        self._history = [(t0, ch) for (t0, ch) in self._history if t - t0 < self.chunk_size]

        # Standard ACT weighting: older predictions get weight exp(-k * age) where age=0..m-1.
        # 标准 ACT 指数加权：按预测滞后步数计算权重并归一化，避免单次高频噪声主控当前控制步。
        fused = [0.0] * self.action_dim
        weight_sum = 0.0
        for age_idx, (t0, ch) in enumerate(self._history):
            slot_idx = t - t0
            weight = math.exp(-self.decay_k * age_idx)
            vec = ch[slot_idx]
            for j in range(self.action_dim):
                fused[j] += weight * vec[j]
            weight_sum += weight

        self._step += 1
        return [v / weight_sum for v in fused]


def compute_trajectory_dynamics(
    actions: Sequence[Sequence[float]],
    states: Sequence[Sequence[float]] | None = None,
    *,
    arm_indices: tuple[int, ...] = ARM_INDICES,
    gripper_indices: tuple[int, int] = GRIPPER_INDICES,
    tail_window: int = 100,
    delta_epsilon: float = 1e-4,
) -> dict[str, float | int]:
    """Compute sign-flip chattering rates, step velocity, and path straightness.
    计算闭环或教师轨迹的反号抖动率（对比纯白噪声理论值 66.7%）、平均步进与路径直线性。
    """
    act_rows = [_to_float_list(row) for row in actions]
    n = len(act_rows)
    if n == 0:
        return {
            "steps": 0,
            "mean_arm_step_delta": 0.0,
            "arm_signflip_rate": 0.0,
            "grip_signflip_rate": 0.0,
            "straightness_full": 0.0,
            "straightness_tail": 0.0,
        }

    arm_deltas: list[float] = []
    arm_flips = 0
    arm_pairs = 0
    grip_flips = 0
    grip_pairs = 0
    n_arm = len(arm_indices)

    for i in range(1, n):
        prev_a, curr_a = act_rows[i - 1], act_rows[i]
        rms = math.sqrt(sum((curr_a[j] - prev_a[j]) ** 2 for j in arm_indices) / n_arm)
        arm_deltas.append(rms)

    for i in range(2, n):
        a0, a1, a2 = act_rows[i - 2], act_rows[i - 1], act_rows[i]
        for j in arm_indices:
            d1, d2 = a1[j] - a0[j], a2[j] - a1[j]
            if abs(d1) > delta_epsilon and abs(d2) > delta_epsilon:
                arm_pairs += 1
                if d1 * d2 < 0.0:
                    arm_flips += 1
        for j in gripper_indices:
            d1, d2 = a1[j] - a0[j], a2[j] - a1[j]
            if abs(d1) > delta_epsilon and abs(d2) > delta_epsilon:
                grip_pairs += 1
                if d1 * d2 < 0.0:
                    grip_flips += 1

    series = [_to_float_list(row) for row in states] if states is not None and len(states) > 0 else act_rows

    def _straightness(seq: list[list[float]]) -> float:
        if len(seq) < 2:
            return 0.0
        path = sum(
            math.sqrt(sum((seq[k][j] - seq[k - 1][j]) ** 2 for j in arm_indices) / n_arm)
            for k in range(1, len(seq))
        )
        if path <= 1e-12:
            return 0.0
        net = math.sqrt(sum((seq[-1][j] - seq[0][j]) ** 2 for j in arm_indices) / n_arm)
        return net / path

    tail_seq = series[max(0, len(series) - tail_window) :]
    return {
        "steps": n,
        "mean_arm_step_delta": sum(arm_deltas) / len(arm_deltas) if arm_deltas else 0.0,
        "arm_signflip_rate": arm_flips / arm_pairs if arm_pairs else 0.0,
        "grip_signflip_rate": grip_flips / grip_pairs if grip_pairs else 0.0,
        "straightness_full": _straightness(series),
        "straightness_tail": _straightness(tail_seq),
    }


@dataclass(frozen=True)
class Pose4D:
    """Initial tabletop pose (meters) of socket (x, y) and peg (x, y)."""

    socket_x: float
    socket_y: float
    peg_x: float
    peg_y: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.socket_x, self.socket_y, self.peg_x, self.peg_y)


def generate_stratified_4d_poses(
    count: int,
    *,
    seed: int = 20261122,
    socket_x_range: tuple[float, float] = (-0.165, -0.105),
    socket_y_range: tuple[float, float] = (0.400, 0.580),
    peg_x_range: tuple[float, float] = (0.100, 0.170),
    peg_y_range: tuple[float, float] = (0.420, 0.580),
) -> list[Pose4D]:
    """Generate Latin-Hypercube stratified 4D initial poses for balanced workspace coverage.
    采用 4D 拉丁超立方分层采样覆盖双物体初始位姿空间，避免随机簇聚或单一场景重复污染。
    """
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    bounds = (socket_x_range, socket_y_range, peg_x_range, peg_y_range)
    for lo, hi in bounds:
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            raise ValueError(f"Invalid pose bound: {(lo, hi)}")
    rng = random.Random(seed)
    columns: list[list[float]] = []
    for lo, hi in bounds:
        strata = [(i + rng.random()) / count for i in range(count)]
        rng.shuffle(strata)
        columns.append([lo + u * (hi - lo) for u in strata])
    return [
        Pose4D(
            socket_x=columns[0][i],
            socket_y=columns[1][i],
            peg_x=columns[2][i],
            peg_y=columns[3][i],
        )
        for i in range(count)
    ]


def audit_4d_workspace_coverage(
    poses: Sequence[Pose4D | Sequence[float]],
    *,
    target_mean_nn_mm: float = 15.0,
) -> dict[str, float | int | bool]:
    """Compute exact leave-one-out 4D nearest-neighbor spacing (mm) across episodes."""
    pts = [p.as_tuple() if isinstance(p, Pose4D) else tuple(_to_float_list(p)) for p in poses]
    if len(pts) < 2:
        raise ValueError("At least 2 poses are required to audit nearest-neighbor coverage")
    if any(len(p) != 4 for p in pts):
        raise ValueError("Each pose must have 4 coordinates (socket_x, socket_y, peg_x, peg_y)")
    nn_mm: list[float] = []
    for i, a in enumerate(pts):
        best_sq = math.inf
        for j, b in enumerate(pts):
            if i == j:
                continue
            sq = (
                (a[0] - b[0]) ** 2
                + (a[1] - b[1]) ** 2
                + (a[2] - b[2]) ** 2
                + (a[3] - b[3]) ** 2
            )
            if sq < best_sq:
                best_sq = sq
        nn_mm.append(math.sqrt(best_sq) * 1000.0)
    nn_sorted = sorted(nn_mm)
    mean_mm = sum(nn_sorted) / len(nn_sorted)
    median_mm = nn_sorted[len(nn_sorted) // 2]
    p90_mm = nn_sorted[min(len(nn_sorted) - 1, int(0.9 * len(nn_sorted)))]
    return {
        "episode_count": len(pts),
        "mean_nn1_mm": mean_mm,
        "median_nn1_mm": median_mm,
        "p90_nn1_mm": p90_mm,
        "max_nn1_mm": nn_sorted[-1],
        "target_mean_nn_mm": float(target_mean_nn_mm),
        "meets_target": mean_mm <= float(target_mean_nn_mm),
    }


def split_episodes_disjoint(
    episode_ids: Iterable[int],
    *,
    val_count: int,
    test_count: int,
    seed: int = 20261122,
) -> dict[str, list[int]]:
    """Create leakage-free episode-level train/val/test splits."""
    ids = list(episode_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode_ids are not allowed")
    if val_count < 0 or test_count < 0 or val_count + test_count >= len(ids):
        raise ValueError("Split sizes must leave at least one training episode")
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    test = sorted(shuffled[:test_count])
    val = sorted(shuffled[test_count : test_count + val_count])
    train = sorted(shuffled[test_count + val_count :])
    return {"train": train, "val": val, "test": test}


def validate_multiview_camera_contract(
    camera_keys: Sequence[str],
    *,
    empty_cameras: int,
    require_wrist_pair: bool = True,
) -> dict[str, object]:
    """Validate multi-view camera configuration so smoke and train share identical slots."""
    keys = list(camera_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("camera_keys must be non-empty and unique")
    unknown = [k for k in keys if k not in SUPPORTED_MULTIVIEW_CAMERAS]
    if unknown:
        raise ValueError(f"Unsupported camera keys: {unknown}")
    if "observation.images.top" not in keys:
        raise ValueError("observation.images.top is required in the multi-view contract")
    if require_wrist_pair and not (
        "observation.images.left_wrist" in keys and "observation.images.right_wrist" in keys
    ):
        raise ValueError("Multi-view insertion requires both left_wrist and right_wrist cameras")
    if type(empty_cameras) is not int or empty_cameras < 0:
        raise ValueError("empty_cameras must be a non-negative integer")
    if len(keys) + empty_cameras > 3 and len(keys) >= 3 and empty_cameras > 0:
        raise ValueError("empty_cameras must be 0 when 3 real cameras are active")
    return {
        "valid": True,
        "camera_keys": keys,
        "camera_count": len(keys),
        "empty_cameras": empty_cameras,
    }
