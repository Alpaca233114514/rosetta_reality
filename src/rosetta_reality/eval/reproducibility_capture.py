"""Passive paired rollout capture. No deterministic settings or actions are changed."""

from __future__ import annotations

import hashlib
import os
import platform
import threading
import uuid
from importlib.metadata import version
from pathlib import Path

from .gate_diagnostic_capture import capture_prediction, clone_tree, rng_scope
from .gate_diagnostic_io import seal, sha, write_json, write_tree
from .rollout_trace import run_traced_rollout

_LOCK = threading.Lock()


def runtime_fingerprint():
    """Read effective settings and loaded binaries, without changing backend selection."""
    import torch

    packages = {}
    for name in ("torch", "numpy", "lerobot", "gym-aloha", "gymnasium", "mujoco", "dm-control"):
        try:
            packages[name] = version(name)
        except ModuleNotFoundError:
            packages[name] = None
    driver = Path("/proc/driver/nvidia/version")
    libraries = {}
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for path in sorted({Path(line.split()[-1]) for line in maps.read_text().splitlines()}):
            if path.is_file() and any(
                token in path.name.lower()
                for token in (
                    "libcuda",
                    "libcublas",
                    "libcudnn",
                    "libmujoco",
                    "libegl",
                    "libgl",
                )
            ):
                libraries[path.name] = sha(path)
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "packages": packages,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "driver": driver.read_text().strip() if driver.is_file() else None,
        "gpu_uuid": str(gpu.uuid) if gpu is not None else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "libraries": libraries,
        "settings": {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },
        "environment": {
            k: os.environ.get(k)
            for k in (
                "MUJOCO_GL",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
    }


class PhysicsReader:
    """Capture mjSTATE_INTEGRATION, including warmstart; never call forward/setState."""

    def __init__(self, environment):
        import mujoco
        import numpy as np

        raw = environment.raw_environment
        raw = getattr(raw, "unwrapped", raw)
        self.physics = raw._env.physics
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.size = mujoco.mj_stateSize(self.model, self.spec)
        self.buffer = np.zeros(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        self.model_sha = self.model_hash()

    def model_hash(self):
        import mujoco

        mujoco.mj_saveModel(self.model, None, self.buffer)
        return hashlib.sha256(self.buffer.tobytes()).hexdigest()

    def __call__(self):
        import mujoco
        import numpy as np
        import torch

        state = np.zeros(self.size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, self.spec)
        return {
            "spec": self.spec,
            "integration": torch.from_numpy(state),
            "model_sha256": self.model_sha,
            "scope": "mjSTATE_INTEGRATION; wrapper/callback state not restored",
        }


def collect_reproducibility(
    engine,
    online,
    contract,
    output,
    identity,
    *,
    pairing,
    maximum_steps=500,
    maximum_bytes=4 * 1024**3,
    guard=lambda: None,
    check_unchanged=lambda: True,
    physics_factory=PhysicsReader,
    fingerprint=runtime_fingerprint,
):
    """One registered arm in a fresh process; both baselines retain identical observers."""
    if (
        pairing["role"] not in ("baseline_a", "baseline_b", "full_trace")
        or pairing["seed"] not in range(1000, 1005)
        or not 1 <= maximum_steps <= 500
        or contract.frequency_hz != 50
        or contract.chunk_execution_steps != 1
        or identity.get("kind") not in ("synthetic", "registered_model")
    ):
        raise ValueError("Invalid paired rollout scope")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Concurrent reproducibility binding")
    original = engine.GymAlohaEnvironment
    output = Path(output)
    created = False
    count = 0
    completed = 0
    current = None
    result = {
        "stage": "collect-repro",
        "status": "incomplete",
        "task_success": None,
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "error_type": None,
    }

    def persist(name, value):
        write_tree(
            output / name,
            value,
            budget_root=output,
            maximum_bytes=maximum_bytes,
            reserve_bytes=min(32 * 1024**2, maximum_bytes // 4),
        )

    class Environment:
        def __init__(self, *args, **kwargs):
            nonlocal current
            if current is not None:
                raise RuntimeError("Exactly one rollout environment required")
            self.wrapped = original(*args, **kwargs)
            self.reader = None
            current = self

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def reset(self, **kwargs):
            observation = self.wrapped.reset(**kwargs)
            self.reader = physics_factory(self.wrapped)
            persist("reset", {"physics": self.reader(), "observation": clone_tree(observation)})
            write_json(output / "runtime.json", fingerprint())
            return observation

        def step(self, action):
            nonlocal completed
            index = completed
            persist(
                f"steps/{index:04d}/before",
                {
                    "physics": self.reader(),
                    "executed": clone_tree(action),
                },
            )
            returned = self.wrapped.step(action)
            observation, reward, done, info = returned
            persist(
                f"steps/{index:04d}/after",
                {
                    "physics": self.reader(),
                    "observation": clone_tree(observation),
                    "reward": float(reward),
                    "done": bool(done),
                    "success": info.get("is_success"),
                    "terminated": info.get("terminated"),
                    "truncated": info.get("truncated"),
                },
            )
            completed += 1
            return returned

    class Policy:
        postprocessor = online.postprocessor

        def configure_noise(self, *args, **kwargs):
            return online.configure_noise(*args, **kwargs)

        def predict(self, observation, instruction):
            nonlocal count
            guard()
            if count != completed:
                raise RuntimeError("Prediction/execution sequence diverged")
            returned, captured = capture_prediction(online, observation, instruction)
            persist(f"predictions/{count:04d}", captured)
            count += 1
            guard()
            return returned

    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(
            output / "identity.json",
            {
                "schema_version": 1,
                "stage": "collect-repro",
                "identity": identity,
                "pairing": pairing,
                "pid": os.getpid(),
                "execution_id": uuid.uuid4().hex,
                "maximum_steps": maximum_steps,
                "dimension_names": list(contract.dimension_names),
                "chunk_length": contract.chunk_length,
            },
        )
        engine.GymAlohaEnvironment = Environment
        options = dict(
            seed=pairing["seed"],
            policy_noise_seed=pairing["seed"],
            maximum_steps=maximum_steps,
            noise_mode="seeded_standard_normal",
            project_policy_output=True,
        )
        with rng_scope(online):
            if pairing["role"] == "full_trace":
                metrics = run_traced_rollout(
                    engine,
                    Policy(),
                    contract,
                    "Insert the peg into the socket.",
                    output=output / "trace",
                    identity=identity,
                    episode_index=0,
                    **options,
                )
            else:
                metrics = engine._rollout(
                    Policy(),
                    contract,
                    "Insert the peg into the socket.",
                    **options,
                )
            guard()
            if count != completed or completed != metrics["rollout_length"]:
                raise ValueError("Incomplete prediction/execution pairs")
            if not check_unchanged() or current.reader.model_hash() != current.reader.model_sha:
                raise ValueError("Policy or physics model changed")
        write_json(output / "runtime-final.json", fingerprint())
        result.update(
            status="complete",
            task_success=metrics["success"],
            parameters_unchanged=True,
            physics_model_unchanged=True,
        )
        return result
    except BaseException as exc:
        result.update(status="incomplete", task_success=None)
        result["error_type"] = type(exc).__name__
        raise
    finally:
        engine.GymAlohaEnvironment = original
        _LOCK.release()
        if created:
            result.update(predictions_saved=count, steps_saved=completed)
            seal(output, result)
