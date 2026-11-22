"""Opt-in monotonic clamped bounded-sine postprocessor feature (create-only).

Prevents the non-monotonic closing-end fold-back identified in the 2026-11-22
audit (where right_gripper closed=0.0 sits at -0.904 sigma in train40 stats,
so flow-matching overshoot below -pi/2 folds back open under unclamped sin()).
Leaves historical hash-pinned ``vla/processor.py`` completely untouched.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import math
from typing import Any

HALF_PI = math.pi / 2.0
GRIPPER_INDICES: tuple[int, int] = (6, 13)


def clamp_postprocessor_step_in_place(step: Any, gripper_indices: tuple[int, int] = GRIPPER_INDICES) -> Any:
    """Wrap a Rosetta Pi-ALOHA postprocessor step so pre-sin gripper latents clamp to [-pi/2, +pi/2].
    包装后处理步骤，在 bounded_sine 解码前将夹爪潜变量截断到 [-pi/2, +pi/2]，防止闭合端过冲折返张开。
    """
    if getattr(step, "_rosetta_clamped_bounded_sine", False):
        return step
    original_action = step.action

    @wraps(original_action)
    def clamped_action(action: Any) -> Any:
        # Clamp only the gripper dimensions in pre-sin latent space when tensor-like.
        if hasattr(action, "clone"):
            clamped = action.clone()
            for idx in gripper_indices:
                clamped[..., idx] = clamped[..., idx].clamp(-HALF_PI, HALF_PI)
            return original_action(clamped)
        if isinstance(action, list):
            clamped = [list(row) if isinstance(row, list) else row for row in action]
            if clamped and isinstance(clamped[0], list):
                for row in clamped:
                    for idx in gripper_indices:
                        row[idx] = max(-HALF_PI, min(HALF_PI, float(row[idx])))
            else:
                for idx in gripper_indices:
                    clamped[idx] = max(-HALF_PI, min(HALF_PI, float(clamped[idx])))
            return original_action(clamped)
        return original_action(action)

    clamped_action._rosetta_unclamped_action = original_action  # type: ignore[attr-defined]
    step.action = clamped_action
    setattr(step, "_rosetta_clamped_bounded_sine", True)
    return step


class ClampedBoundedSineFeature:
    """Training/export feature that clamps bounded-sine gripper latents monotonically."""

    name = "clamped_bounded_sine"
    _MARKER = "_rosetta_v2_clamped_bounded_sine_installed"

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise ValueError("clamped_bounded_sine declares no parameters.")
        self._original_factory: Any = None
        self._installed_factory: Any = None
        self._wrapped_steps: int = 0

    def install(self, context: Any) -> None:
        import lerobot.scripts.lerobot_train as lerobot_train

        if getattr(lerobot_train, self._MARKER, False):
            raise RuntimeError("clamped_bounded_sine is already installed.")
        original = lerobot_train.make_pre_post_processors

        @wraps(original)
        def make_pre_post_processors(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            preprocessor, postprocessor = original(*args, **kwargs)
            steps = getattr(postprocessor, "steps", None)
            if steps is None:
                raise RuntimeError("Postprocessor exposes no steps for clamped_bounded_sine.")
            matched = [
                step
                for step in steps
                if getattr(step.__class__, "_registry_name", None)
                == "rosetta_pi_aloha_postprocessor"
            ]
            if len(matched) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 rosetta_pi_aloha_postprocessor step, found {len(matched)}."
                )
            clamp_postprocessor_step_in_place(matched[0])
            self._wrapped_steps += 1
            return preprocessor, postprocessor

        make_pre_post_processors._rosetta_v2_original = original  # type: ignore[attr-defined]
        self._original_factory = original
        self._installed_factory = make_pre_post_processors
        lerobot_train.make_pre_post_processors = make_pre_post_processors
        setattr(lerobot_train, self._MARKER, True)

    def restore(self, context: Any) -> None:
        import lerobot.scripts.lerobot_train as lerobot_train

        if getattr(lerobot_train, self._MARKER, False) is not True:
            raise RuntimeError("No clamped_bounded_sine hook is installed.")
        if lerobot_train.make_pre_post_processors is not self._installed_factory:
            raise RuntimeError("clamped_bounded_sine make_pre_post_processors hook changed before restore.")
        lerobot_train.make_pre_post_processors = self._original_factory
        setattr(lerobot_train, self._MARKER, False)
