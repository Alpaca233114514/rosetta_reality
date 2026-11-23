"""MuJoCo 3.8.1 / Gym-ALOHA preflight audit, deterministic fork, and offscreen capture.

Adapted for Rosetta Reality from Google DeepMind's ``robotics`` plugin
(``robotics-antigravity-plugin`` v0.1.11, Apache License 2.0,
Copyright 2026 DeepMind Technologies Limited):
- Compiled-model preflight audit from ``skills/sim2real-gauntlet/.../preflight.py``
- Deterministic episode-fork workflow from ``skills/sim2real-gauntlet/.../episode.py``
- Named access & offscreen ``mujoco.Renderer`` hygiene from ``skills/python`` &
  ``skills/rendering``

All heavy imports (``mujoco``, ``numpy``) are strictly lazy so importing this
module has zero side effects and requires neither ``mujoco-playground`` nor
``mujoco>=4.0.1``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

ALOHA_ROBOT_BODY_PATTERN = r"^vx300s_"


@dataclasses.dataclass(frozen=True, slots=True)
class IntegrationStateSnapshot:
    """Sealed ``mjSTATE_INTEGRATION`` snapshot bound to a compiled model digest."""

    model_sha256: str
    state_spec: int
    integration_bytes: bytes
    step_index: int = 0
    observation: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ForkEpisodeResult:
    """Outcome of a deterministic episode fork starting from a restored snapshot."""

    start_step: int
    steps_executed: int
    total_reward: float
    max_reward: float
    terminated_or_truncated: bool
    model_sha256: str
    unexpected_collision_count: int
    joint_limit_violation_count: int
    step_rewards: tuple[float, ...]
    final_info: Mapping[str, Any]


def resolve_mujoco_handles(environment: Any) -> tuple[Any, Any]:
    """Extract raw ``(MjModel, MjData)`` handles from GymAlohaEnvironment or a physics wrapper."""

    # 兼容 GymAlohaEnvironment(dm_control .ptr) 与直接暴露 mj_model/mj_data 的对象 /
    # Support both GymAlohaEnvironment (dm_control .ptr) and direct mj_model/mj_data wrappers.
    if hasattr(environment, "mj_model") and hasattr(environment, "mj_data"):
        return environment.mj_model, environment.mj_data

    raw = getattr(environment, "raw_environment", environment)
    unwrapped = getattr(raw, "unwrapped", raw)
    control_env = getattr(unwrapped, "_env", unwrapped)
    physics = getattr(control_env, "physics", None)
    if physics is None:
        raise ValueError(
            "Environment does not expose a dm_control/MuJoCo physics handle "
            "(expected raw_environment.unwrapped._env.physics or mj_model/mj_data)."
        )
    model = getattr(physics.model, "ptr", physics.model)
    data = getattr(physics.data, "ptr", physics.data)
    return model, data


def _named_entity(model: Any, kind: str, index: int) -> str:
    accessor = getattr(model, kind, None)
    if callable(accessor):
        entity = accessor(index)
        name = getattr(entity, "name", "")
        if name:
            return str(name)
    id2name = getattr(model, "id2name", None)
    if callable(id2name):
        name = id2name(index, kind)
        if name:
            return str(name)
    return f"{kind}_{index}"


def compiled_model_sha256(model: Any, *, mujoco_module: Any | None = None) -> str:
    """Hash the serialized compiled ``MjModel`` bytes without mutating simulation state."""

    mj = mujoco_module if mujoco_module is not None else _import_mujoco()
    size = int(mj.mj_sizeModel(model))
    buf = bytearray(size)
    mj.mj_saveModel(model, None, buf)
    return hashlib.sha256(bytes(buf)).hexdigest()


def audit_compiled_model(
    model: Any,
    *,
    robot_body_pattern: str = ALOHA_ROBOT_BODY_PATTERN,
    min_manipuland_mass_kg: float = 0.005,
    max_manipuland_mass_kg: float = 2.0,
    mujoco_module: Any | None = None,
) -> tuple[str, ...]:
    """Run static preflight physics checks on a compiled MuJoCo ``MjModel`` (3.8.1+).

    Checks adapted from ``robosim_gauntlet.preflight._audit``:
    1. Free manipuland bodies must not be constrained by ``CONNECT``/``WELD`` equalities.
    2. Joint actuators must not drive non-robot degrees of freedom.
    3. Manipuland body masses must stay inside physically plausible bounds.
    4. Manipuland collision geoms must not set non-zero ``margin`` or ``gap``.
    5. Global contact override (``mjENBL_OVERRIDE``) must not be enabled.
    """

    mj = mujoco_module if mujoco_module is not None else _import_mujoco()
    pattern = re.compile(robot_body_pattern)
    nbody = int(model.nbody)
    robot_mask = [bool(pattern.search(_named_entity(model, "body", b))) for b in range(nbody)]

    jnt_free = int(mj.mjtJoint.mjJNT_FREE)
    eq_connect = int(mj.mjtEq.mjEQ_CONNECT)
    eq_weld = int(mj.mjtEq.mjEQ_WELD)
    trn_joint = int(mj.mjtTrn.mjTRN_JOINT)
    enbl_override = int(mj.mjtEnableBit.mjENBL_OVERRIDE)

    manipuland_bodies = {
        int(model.jnt_bodyid[j])
        for j in range(int(model.njnt))
        if int(model.jnt_type[j]) == jnt_free and not robot_mask[int(model.jnt_bodyid[j])]
    }

    issues: list[str] = []
    for eq_idx in range(int(model.neq)):
        if int(model.eq_type[eq_idx]) in (eq_connect, eq_weld):
            obj_ids = {int(model.eq_obj1id[eq_idx]), int(model.eq_obj2id[eq_idx])}
            if obj_ids & manipuland_bodies:
                issues.append(f"equality {eq_idx} constrains a manipuland")

    for act_idx in range(int(model.nu)):
        if int(model.actuator_trntype[act_idx]) == trn_joint:
            joint_id = int(model.actuator_trnid[act_idx][0])
            body_id = int(model.jnt_bodyid[joint_id])
            if not robot_mask[body_id]:
                act_name = _named_entity(model, "actuator", act_idx)
                issues.append(f"actuator {act_name} drives a non-robot dof")

    for body_id in sorted(manipuland_bodies):
        body_name = _named_entity(model, "body", body_id)
        mass = float(model.body_mass[body_id])
        if not min_manipuland_mass_kg <= mass <= max_manipuland_mass_kg:
            issues.append(f"manipuland {body_name} mass {mass:.4f}")
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if float(model.geom_margin[geom_id]) > 0.0 or float(model.geom_gap[geom_id]) > 0.0:
                geom_name = _named_entity(model, "geom", geom_id)
                issues.append(f"manipuland geom {geom_name} has margin/gap")

    if int(model.opt.enableflags) & enbl_override:
        issues.append("contact override is enabled")

    return tuple(issues)


def capture_integration_state(
    environment: Any,
    *,
    step_index: int = 0,
    observation: Mapping[str, Any] | None = None,
    mujoco_module: Any | None = None,
) -> IntegrationStateSnapshot:
    """Capture complete ``mjSTATE_INTEGRATION`` (including warmstart) plus model SHA-256."""

    mj = mujoco_module if mujoco_module is not None else _import_mujoco()
    model, data = resolve_mujoco_handles(environment)
    spec = int(mj.mjtState.mjSTATE_INTEGRATION)
    size = int(mj.mj_stateSize(model, spec))

    # 优先使用 float64 连续缓冲区，便于在无 numpy 环境下用标准 array 测试 /
    # Prefer float64 buffer compatible with both NumPy arrays and stdlib array('d').
    state_buffer = _alloc_float64_buffer(size)
    mj.mj_getState(model, data, state_buffer, spec)
    return IntegrationStateSnapshot(
        model_sha256=compiled_model_sha256(model, mujoco_module=mj),
        state_spec=spec,
        integration_bytes=bytes(memoryview(state_buffer).cast("B")),
        step_index=step_index,
        observation=observation,
    )


def restore_integration_state(
    environment: Any,
    snapshot: IntegrationStateSnapshot,
    *,
    mujoco_module: Any | None = None,
) -> None:
    """Restore ``mjSTATE_INTEGRATION`` and run ``mj_forward`` after verifying model SHA-256."""

    mj = mujoco_module if mujoco_module is not None else _import_mujoco()
    model, data = resolve_mujoco_handles(environment)
    current_sha = compiled_model_sha256(model, mujoco_module=mj)
    # 拒绝跨改动模型恢复物理状态，避免静默错位 / Fail closed if compiled model hash changed.
    if current_sha != snapshot.model_sha256:
        raise ValueError(
            "Refusing to restore integration state onto a different compiled MuJoCo model: "
            f"expected {snapshot.model_sha256}, got {current_sha}."
        )
    expected_bytes = int(mj.mj_stateSize(model, snapshot.state_spec)) * 8
    if len(snapshot.integration_bytes) != expected_bytes:
        raise ValueError(
            "Integration state byte length mismatch: "
            f"expected {expected_bytes}, got {len(snapshot.integration_bytes)}."
        )
    state_buffer = _float64_buffer_from_bytes(snapshot.integration_bytes)
    mj.mj_setState(model, data, state_buffer, snapshot.state_spec)
    mj.mj_forward(model, data)


def fork_episode_from_snapshot(
    environment: Any,
    snapshot: IntegrationStateSnapshot,
    controller: Callable[[Mapping[str, Any], int], Any],
    *,
    max_steps: int,
    initial_observation: Mapping[str, Any] | None = None,
    mujoco_module: Any | None = None,
) -> ForkEpisodeResult:
    """Restore a saved tick snapshot and hand off control to ``controller``."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    restore_integration_state(environment, snapshot, mujoco_module=mujoco_module)
    observation = initial_observation if initial_observation is not None else snapshot.observation
    if observation is None:
        raise ValueError(
            "fork_episode_from_snapshot requires initial_observation or snapshot.observation."
        )

    rewards: list[float] = []
    collision_total = 0
    limit_total = 0
    done = False
    info: Mapping[str, Any] = {}
    current_obs = observation
    for offset in range(max_steps):
        action = controller(current_obs, snapshot.step_index + offset)
        current_obs, reward, done, info = environment.step(action)
        rewards.append(float(reward))
        if hasattr(environment, "unexpected_collision_count"):
            collision_total += int(environment.unexpected_collision_count())
        if hasattr(environment, "state_limit_violation_count"):
            limit_total += int(environment.state_limit_violation_count())
        if done:
            break

    return ForkEpisodeResult(
        start_step=snapshot.step_index,
        steps_executed=len(rewards),
        total_reward=sum(rewards),
        max_reward=max(rewards) if rewards else 0.0,
        terminated_or_truncated=bool(done),
        model_sha256=snapshot.model_sha256,
        unexpected_collision_count=collision_total,
        joint_limit_violation_count=limit_total,
        step_rewards=tuple(rewards),
        final_info=dict(info),
    )


def fork_episode_from_actions(
    environment: Any,
    *,
    seed: int,
    prefix_actions: Sequence[Any],
    controller: Callable[[Mapping[str, Any], int], Any],
    max_steps: int,
    mujoco_module: Any | None = None,
) -> tuple[IntegrationStateSnapshot, ForkEpisodeResult]:
    """Deterministically replay ``prefix_actions`` from ``seed``, snapshot, and fork."""

    observation = environment.reset(seed=seed)
    for step_idx, action in enumerate(prefix_actions):
        observation, _, done, _ = environment.step(action)
        if done:
            raise RuntimeError(
                f"Episode terminated during prefix replay at step {step_idx} before fork."
            )
    snapshot = capture_integration_state(
        environment,
        step_index=len(prefix_actions),
        observation=observation,
        mujoco_module=mujoco_module,
    )
    result = fork_episode_from_snapshot(
        environment,
        snapshot,
        controller,
        max_steps=max_steps,
        initial_observation=observation,
        mujoco_module=mujoco_module,
    )
    return snapshot, result


def render_offscreen_frame(
    model: Any,
    data: Any,
    *,
    camera: str | int,
    width: int = 640,
    height: int = 480,
    depth: bool = False,
    require_headless_gl_env: bool = True,
    mujoco_module: Any | None = None,
) -> Any:
    """Capture one copied RGB or depth frame via ``mujoco.Renderer`` context manager.

    Follows the ``mujoco-rendering`` skill contract:
    - Enforces explicit ``MUJOCO_GL`` on headless Linux when ``DISPLAY`` is unset.
    - Uses ``with mujoco.Renderer(...)`` to avoid leaking OpenGL framebuffers.
    - Calls ``.copy()`` so returned arrays never alias internal renderer buffers.
    """

    if width <= 0 or height <= 0:
        raise ValueError("Render width and height must be positive.")
    if (
        require_headless_gl_env
        and sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("MUJOCO_GL")
    ):
        raise RuntimeError(
            "Headless Linux rendering requires explicit MUJOCO_GL ('egl' or 'osmesa') "
            "before importing mujoco."
        )
    mj = mujoco_module if mujoco_module is not None else _import_mujoco()
    with mj.Renderer(model, height=height, width=width) as renderer:
        if depth:
            renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        if depth:
            renderer.disable_depth_rendering()
        return frame.copy() if hasattr(frame, "copy") else frame


def _import_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo diagnostics require the optional 'sim' dependency (mujoco==3.8.1)."
        ) from error
    return mujoco


def _alloc_float64_buffer(size: int) -> Any:
    try:
        import numpy as np

        return np.zeros(size, dtype=np.float64)
    except ImportError:
        import array

        return array.array("d", [0.0] * size)


def _float64_buffer_from_bytes(payload: bytes) -> Any:
    try:
        import numpy as np

        return np.frombuffer(payload, dtype=np.float64).copy()
    except ImportError:
        import array

        buf = array.array("d")
        buf.frombytes(payload)
        return buf
