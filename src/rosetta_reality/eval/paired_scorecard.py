"""Paired CRN scorecard with exact McNemar test and catastrophic-safety veto.

Adapted for Rosetta Reality from Google DeepMind's ``robotics`` plugin
(``robotics-antigravity-plugin`` v0.1.11,
``skills/sim2real-gauntlet/src/robosim_gauntlet/scorecard.py``,
Apache License 2.0, Copyright 2026 DeepMind Technologies Limited).

Pure standard library: importing this module never loads torch, MuJoCo,
gym-aloha, or mujoco-playground.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

# 默认一票否决的安全违规集合 / Default catastrophic violations that veto task-success gains.
DEFAULT_CATASTROPHIC_VIOLATIONS: frozenset[str] = frozenset(
    {
        "unexpected_collision",
        "joint_limit_violation",
        "invalid_action",
        "non_finite_action",
        "impact_force",
        "self_contact",
        "grasp_force",
        "spurious_abort",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeScoreRecord:
    """One paired evaluation episode record compatible with Gate 4 and sim2real tiers."""

    task_id: str
    tier: str
    episode_index: int
    seed: int
    scene_sha256: str
    task_success: bool
    violations: tuple[str, ...] = ()
    unexpected_collision_count: int = 0
    joint_limit_violation_count: int = 0
    invalid_action_count: int = 0
    aborted_ok: bool = False
    status: str = "done"
    fault: str = ""
    metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.tier:
            raise ValueError("task_id and tier must be non-empty strings.")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative.")
        if (
            self.unexpected_collision_count < 0
            or self.joint_limit_violation_count < 0
            or self.invalid_action_count < 0
        ):
            raise ValueError("Safety violation counts must be non-negative.")

        # 将数值安全计数并入规范违规集合 / Fold numeric safety counters into canonical violation tags.
        normalized: list[str] = list(self.violations)
        if self.unexpected_collision_count > 0 and "unexpected_collision" not in normalized:
            normalized.append("unexpected_collision")
        if self.joint_limit_violation_count > 0 and "joint_limit_violation" not in normalized:
            normalized.append("joint_limit_violation")
        if self.invalid_action_count > 0 and "invalid_action" not in normalized:
            normalized.append("invalid_action")
        object.__setattr__(self, "violations", tuple(dict.fromkeys(normalized)))
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in self.metrics.items()},
        )

    @property
    def key(self) -> tuple[str, str, int, int]:
        """Exact CRN pairing key: ``(task_id, tier, episode_index, seed)``."""

        return (self.task_id, self.tier, self.episode_index, self.seed)

    @property
    def safe(self) -> bool:
        """True only when zero violations occurred during the episode."""

        return len(self.violations) == 0

    @property
    def safe_pass(self) -> bool:
        """True only when the episode succeeded physically AND stayed completely safe."""

        return bool(self.task_success and self.safe)


@dataclasses.dataclass(frozen=True, slots=True)
class PairedScorecardComparison:
    """Outcome of comparing candidate ``b`` against baseline ``a`` on paired CRN seeds."""

    verdict: str
    n_pairs: int
    a_only: int
    b_only: int
    p_better: float
    p_worse: float
    vetoes: int
    reason: str = ""


def exact_mcnemar_tail(k: int, n: int) -> float:
    """Exact one-sided McNemar binomial tail P(X >= k), X ~ Binomial(n, 1/2)."""

    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)


def summarize_episodes(records: Sequence[EpisodeScoreRecord]) -> dict[str, float]:
    """Compute per-tier success/safe-pass rates and overall fault-abort precision."""

    out: dict[str, float] = {}
    if not records:
        return {"safe_pass_rate": 0.0, "fault_abort_precision": 1.0}
    for tier in sorted({record.tier for record in records}):
        group = [record for record in records if record.tier == tier]
        out[f"{tier}/success_rate"] = sum(record.task_success for record in group) / len(group)
        out[f"{tier}/safe_pass_rate"] = sum(record.safe_pass for record in group) / len(group)
    out["safe_pass_rate"] = sum(record.safe_pass for record in records) / len(records)
    true_aborts = sum(record.aborted_ok for record in records)
    spurious_aborts = sum("spurious_abort" in record.violations for record in records)
    total_aborts = true_aborts + spurious_aborts
    out["fault_abort_precision"] = true_aborts / total_aborts if total_aborts else 1.0
    return out


def compare_paired_episodes(
    baseline: Sequence[EpisodeScoreRecord],
    candidate: Sequence[EpisodeScoreRecord],
    *,
    alpha: float = 0.025,
    require_scene_digest: bool = True,
    catastrophic_violations: frozenset[str] = DEFAULT_CATASTROPHIC_VIOLATIONS,
) -> PairedScorecardComparison:
    """Compare candidate against baseline on exact CRN pairs with safety veto."""

    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5).")
    by_a = {record.key: record for record in baseline}
    by_b = {record.key: record for record in candidate}
    if not by_a or len(by_a) != len(baseline) or len(by_b) != len(candidate):
        return PairedScorecardComparison(
            "INVALID", 0, 0, 0, 1.0, 1.0, 0, reason="empty_or_duplicate_keys"
        )
    if by_a.keys() != by_b.keys():
        return PairedScorecardComparison(
            "INVALID", 0, 0, 0, 1.0, 1.0, 0, reason="mismatched_crn_keys"
        )

    # 校验场景/模型哈希一致，防止跨改动环境错误配对 / Lock scene digest to prevent invalid cross-scene pairing.
    for key in by_a:
        digest_a = by_a[key].scene_sha256
        digest_b = by_b[key].scene_sha256
        if require_scene_digest and (not digest_a or not digest_b):
            return PairedScorecardComparison(
                "INVALID", 0, 0, 0, 1.0, 1.0, 0, reason="missing_scene_sha256"
            )
        if digest_a and digest_b and digest_a != digest_b:
            return PairedScorecardComparison(
                "INVALID", 0, 0, 0, 1.0, 1.0, 0, reason="scene_sha256_mismatch"
            )

    pairs = [(by_a[key], by_b[key]) for key in sorted(by_a)]
    a_only = sum(rec_a.safe_pass and not rec_b.safe_pass for rec_a, rec_b in pairs)
    b_only = sum(rec_b.safe_pass and not rec_a.safe_pass for rec_a, rec_b in pairs)
    # 候选引入任何基线未出现的致命安全违规即触发否决 / Any new catastrophic safety violation vetoes gains.
    vetoes = sum(
        bool(set(rec_b.violations) & catastrophic_violations)
        and not bool(set(rec_a.violations) & catastrophic_violations)
        for rec_a, rec_b in pairs
    )

    discordant = a_only + b_only
    p_better = exact_mcnemar_tail(b_only, discordant)
    p_worse = exact_mcnemar_tail(a_only, discordant)
    if vetoes > 0:
        verdict, reason = "WORSE", "catastrophic_safety_veto"
    elif p_better < alpha:
        verdict, reason = "BETTER", "mcnemar_significant_gain"
    elif p_worse < alpha:
        verdict, reason = "WORSE", "mcnemar_significant_regression"
    else:
        verdict, reason = "INDISTINGUISHABLE", "insufficient_discordant_evidence"
    return PairedScorecardComparison(
        verdict=verdict,
        n_pairs=len(pairs),
        a_only=a_only,
        b_only=b_only,
        p_better=p_better,
        p_worse=p_worse,
        vetoes=vetoes,
        reason=reason,
    )


def record_from_gate_episode(
    episode: Mapping[str, Any],
    *,
    task_id: str = "gym_aloha/AlohaInsertion-v0",
    tier: str = "T0",
    episode_index: int | None = None,
    scene_sha256: str = "",
) -> EpisodeScoreRecord:
    """Convert a Rosetta Gate 4 / rollout summary dict into an EpisodeScoreRecord."""

    idx = int(episode.get("episode_index", 0) if episode_index is None else episode_index)
    seed = int(episode["seed"])
    success = bool(episode.get("task_success", episode.get("success", False)))
    collisions = int(episode.get("unexpected_collision_count", episode.get("collisions", 0)))
    joint_limits = int(
        episode.get("joint_limit_violation_count", episode.get("joint_limit_violations", 0))
    )
    invalid_actions = int(episode.get("invalid_action_count", episode.get("invalid_actions", 0)))
    raw_violations = tuple(str(v) for v in episode.get("violations", ()))
    raw_metrics = episode.get("metrics", {})
    metrics: dict[str, float] = {}
    if isinstance(raw_metrics, Mapping):
        metrics.update({str(k): float(v) for k, v in raw_metrics.items()})
    for scalar_key in ("max_reward", "rollout_length", "steps"):
        if scalar_key in episode and episode[scalar_key] is not None:
            metrics.setdefault(scalar_key, float(episode[scalar_key]))
    return EpisodeScoreRecord(
        task_id=task_id,
        tier=tier,
        episode_index=idx,
        seed=seed,
        scene_sha256=str(episode.get("scene_sha256", scene_sha256)),
        task_success=success,
        violations=raw_violations,
        unexpected_collision_count=collisions,
        joint_limit_violation_count=joint_limits,
        invalid_action_count=invalid_actions,
        aborted_ok=bool(episode.get("aborted_ok", False)),
        status=str(episode.get("status", "done")),
        fault=str(episode.get("fault", "")),
        metrics=metrics,
    )


def records_to_json_list(records: Sequence[EpisodeScoreRecord]) -> list[dict[str, Any]]:
    """Serialize EpisodeScoreRecord sequence for JSON artifacts."""

    return [
        {
            "task_id": record.task_id,
            "tier": record.tier,
            "episode_index": record.episode_index,
            "seed": record.seed,
            "scene_sha256": record.scene_sha256,
            "task_success": record.task_success,
            "safe": record.safe,
            "safe_pass": record.safe_pass,
            "aborted_ok": record.aborted_ok,
            "status": record.status,
            "fault": record.fault,
            "violations": list(record.violations),
            "unexpected_collision_count": record.unexpected_collision_count,
            "joint_limit_violation_count": record.joint_limit_violation_count,
            "invalid_action_count": record.invalid_action_count,
            "metrics": dict(record.metrics),
        }
        for record in records
    ]


def records_from_json_list(items: Sequence[Mapping[str, Any]]) -> list[EpisodeScoreRecord]:
    """Deserialize EpisodeScoreRecord objects from ``records_to_json_list`` output."""

    return [
        EpisodeScoreRecord(
            task_id=str(item["task_id"]),
            tier=str(item["tier"]),
            episode_index=int(item["episode_index"]),
            seed=int(item["seed"]),
            scene_sha256=str(item.get("scene_sha256", "")),
            task_success=bool(item["task_success"]),
            violations=tuple(str(v) for v in item.get("violations", ())),
            unexpected_collision_count=int(item.get("unexpected_collision_count", 0)),
            joint_limit_violation_count=int(item.get("joint_limit_violation_count", 0)),
            invalid_action_count=int(item.get("invalid_action_count", 0)),
            aborted_ok=bool(item.get("aborted_ok", False)),
            status=str(item.get("status", "done")),
            fault=str(item.get("fault", "")),
            metrics=dict(item.get("metrics", {})),
        )
        for item in items
    ]
