"""Fail-closed state-conditioned reference policy for recovery collection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class OracleOutOfDistributionError(RuntimeError):
    """Raised when a visited state is too far from the proven reference bank."""


def _finite_vector(value: Tensor, *, dimension: int, name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), received {tuple(tensor.shape)}.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf.")
    return tensor


@dataclass(frozen=True, slots=True)
class OracleReferenceTrajectory:
    """One successful train-only simulator trajectory used as a reference bank."""

    states: Tensor
    actions: Tensor
    source_episode: int
    source_seed: int
    first_progress_index: int
    terminal_reward: float
    terminal_success: bool

    def __post_init__(self) -> None:
        states = torch.as_tensor(self.states, dtype=torch.float32).detach().cpu().clone()
        actions = torch.as_tensor(self.actions, dtype=torch.float32).detach().cpu().clone()
        if states.ndim != 2 or actions.ndim != 2 or states.shape != actions.shape:
            raise ValueError("Oracle reference states and actions must be equal rank-two tensors.")
        if states.shape[0] < 2 or states.shape[1] < 1:
            raise ValueError(
                "Oracle reference trajectory must contain at least two non-empty rows."
            )
        if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(actions).all()):
            raise ValueError("Oracle reference trajectory contains NaN or Inf.")
        if self.source_episode < 0 or self.source_seed < 0:
            raise ValueError("Oracle source episode and simulator seed must be non-negative.")
        if not 0 <= self.first_progress_index < states.shape[0]:
            raise ValueError("Oracle first-progress index is outside the reference trajectory.")
        if not math.isfinite(self.terminal_reward):
            raise ValueError("Oracle terminal reward must be finite.")
        if not self.terminal_success:
            raise ValueError("A failed source trajectory cannot authorize a recovery oracle.")
        object.__setattr__(self, "states", states.contiguous())
        object.__setattr__(self, "actions", actions.contiguous())

    @property
    def length(self) -> int:
        """Number of pre-action reference states."""

        return int(self.states.shape[0])

    @property
    def dimension(self) -> int:
        """Shared robot-state and action width."""

        return int(self.states.shape[1])


@dataclass(frozen=True, slots=True)
class OracleDecision:
    """Auditable action selected from current state and observed task progress."""

    action: Tensor
    reference_index: int
    state_distance: float
    progress_unlocked: bool
    candidate_start: int
    candidate_stop_exclusive: int


class StateConditionedTrajectoryOracle:
    """Retrieve a monotonic successful action without using wall-clock time.

    The current robot state selects a reference row inside a bounded forward
    window.  An observed task reward is the only phase-unlock signal.  This
    deliberately refuses to label states outside the registered reference
    neighborhood instead of falling back to the same-index expert action.
    """

    def __init__(
        self,
        reference: OracleReferenceTrajectory,
        *,
        maximum_lookahead: int,
        maximum_state_distance: float,
        maximum_progress_state_distance: float,
        progress_reward_threshold: float = 1.0,
        post_progress_skip: int = 1,
        state_scale: Tensor | None = None,
    ) -> None:
        if maximum_lookahead < 1:
            raise ValueError("Oracle maximum lookahead must be positive.")
        if not math.isfinite(maximum_state_distance) or maximum_state_distance <= 0:
            raise ValueError("Oracle maximum state distance must be positive and finite.")
        if (
            not math.isfinite(maximum_progress_state_distance)
            or maximum_progress_state_distance <= 0
            or maximum_progress_state_distance > maximum_state_distance
        ):
            raise ValueError(
                "Oracle progress-state distance must be positive, finite, and no greater "
                "than the out-of-distribution distance."
            )
        if not math.isfinite(progress_reward_threshold):
            raise ValueError("Oracle progress reward threshold must be finite.")
        if post_progress_skip < 0:
            raise ValueError("Oracle post-progress skip must be non-negative.")
        if state_scale is None:
            scale = torch.ones(reference.dimension, dtype=torch.float32)
        else:
            scale = _finite_vector(
                state_scale,
                dimension=reference.dimension,
                name="Oracle state scale",
            )
        if bool(scale.le(0).any()):
            raise ValueError("Oracle state scale must be strictly positive.")
        self.reference = reference
        self.maximum_lookahead = maximum_lookahead
        self.maximum_state_distance = maximum_state_distance
        self.maximum_progress_state_distance = maximum_progress_state_distance
        self.progress_reward_threshold = progress_reward_threshold
        self.post_progress_skip = post_progress_skip
        self.state_scale = scale
        self.reset()

    def reset(self) -> None:
        """Reset only oracle history; the simulator remains caller-owned."""

        self._cursor = 0
        self._progress_unlocked = False

    @property
    def cursor(self) -> int:
        """Earliest reference index eligible for the next decision."""

        return self._cursor

    @property
    def progress_unlocked(self) -> bool:
        """Whether the simulator has observed the registered progress event."""

        return self._progress_unlocked

    def decide(self, current_state: Tensor, *, observed_reward: float) -> OracleDecision:
        """Select an action using state proximity and an observed reward event."""

        state = _finite_vector(
            current_state,
            dimension=self.reference.dimension,
            name="Oracle current state",
        )
        if not math.isfinite(observed_reward):
            raise ValueError("Oracle observed reward must be finite.")
        if observed_reward >= self.progress_reward_threshold and not self._progress_unlocked:
            self._progress_unlocked = True
            unlocked_cursor = self.reference.first_progress_index + self.post_progress_skip
            self._cursor = min(max(self._cursor, unlocked_cursor), self.reference.length - 1)

        phase_stop = (
            self.reference.length
            if self._progress_unlocked
            else self.reference.first_progress_index + 1
        )
        start = min(self._cursor, phase_stop - 1)
        stop = min(phase_stop, start + self.maximum_lookahead + 1)
        candidates = self.reference.states[start:stop]
        normalized = (candidates - state.unsqueeze(0)) / self.state_scale.unsqueeze(0)
        distances = normalized.square().mean(dim=1).sqrt()
        relative_index = int(torch.argmin(distances))
        selected = start + relative_index
        next_index = self._cursor + 1
        if (
            selected == self._cursor
            and next_index < phase_stop
            and next_index < stop
        ):
            next_distance = float(distances[next_index - start])
            if next_distance <= self.maximum_progress_state_distance:
                selected = next_index
                relative_index = selected - start
        distance = float(distances[relative_index])
        if not math.isfinite(distance) or distance > self.maximum_state_distance:
            raise OracleOutOfDistributionError(
                "Current state is outside the recovery reference neighborhood: "
                f"distance={distance:.6f}, limit={self.maximum_state_distance:.6f}."
            )
        self._cursor = max(self._cursor, selected)
        return OracleDecision(
            action=self.reference.actions[selected].clone(),
            reference_index=selected,
            state_distance=distance,
            progress_unlocked=self._progress_unlocked,
            candidate_start=start,
            candidate_stop_exclusive=stop,
        )
