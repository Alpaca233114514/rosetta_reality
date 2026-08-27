"""Run Way Gate 3/4 on the AutoDL CUDA worker without nested Docker."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import smolvla_sim_gate as simulator  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla.accelerator_memory import synchronize  # noqa: E402
from rosetta_reality.vla.action_space import SmolVLAActionSpace  # noqa: E402
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402

_DELEGATED_REPOSITORY_PATH = simulator._repository_path


def _runtime_plan_path(raw: str) -> Path:
    """Resolve ignored run evidence from the durable AutoDL run root."""

    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Way simulation paths must be safe relative paths.")
    if relative.parts and relative.parts[0] == "runs":
        run_root = Path(os.environ["ROSETTA_RUN_ROOT"]).resolve()
        path = (run_root / Path(*relative.parts[1:])).resolve()
        if not path.is_relative_to(run_root) or not path.is_file():
            raise FileNotFoundError("Way durable simulation evidence is missing.")
        return path
    return _DELEGATED_REPOSITORY_PATH(raw)


def _plan_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way AutoDL simulation requires an explicit --plan path.") from error


class _AutoDLWayOnlineSmolVLA:
    def __init__(
        self,
        artifact: Path,
        config: dict[str, Any],
        normalization: dict[str, Any],
        contract: Any,
    ) -> None:
        if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Way AutoDL simulation requires CUDA.")
        self.device = torch.device("cuda")
        self.mixed_precision = str(config["mixed_precision"])
        if self.mixed_precision != "bf16":
            raise ValueError("Way simulation mixed precision differs from its artifact.")
        pretrained = artifact / "pretrained_model"
        policy_cfg = SmolVLAConfig.from_pretrained(pretrained, local_files_only=True)
        policy_cfg.device = self.device.type
        policy_cfg.pretrained_path = pretrained
        policy_cfg.pretrained_revision = None
        policy_cfg.load_vlm_weights = False
        metadata = simulator._ArtifactMetadata(config, normalization)
        dataset_state_dimension = simulator._dataset_state_dimension(metadata.features)
        self.policy = make_policy(
            cfg=policy_cfg,
            ds_meta=metadata,
            rename_map=config["rename_map"],
        )
        self.state_dimension = simulator._validate_policy_contract_shape(
            self.policy.config, contract, dataset_state_dimension
        )
        self.action_dimension = contract.dimension
        self.chunk_length = contract.chunk_length
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=pretrained,
            pretrained_revision=None,
            dataset_stats=metadata.stats,
            preprocessor_overrides={
                "device_processor": {"device": self.device.type},
                "normalizer_processor": {
                    "features": {
                        **self.policy.config.input_features,
                        **self.policy.config.output_features,
                    },
                    "norm_map": self.policy.config.normalization_mapping,
                    "stats": metadata.stats,
                },
                "rename_observations_processor": {
                    "rename_map": config["rename_map"]
                },
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "features": self.policy.config.output_features,
                    "norm_map": self.policy.config.normalization_mapping,
                    "stats": metadata.stats,
                }
            },
        )
        raw_action_space = config.get("action_space")
        if not isinstance(raw_action_space, dict):
            raise ValueError("Way artifact has no explicit action-space identity.")
        action_space = SmolVLAActionSpace(**raw_action_space)
        if (
            action_space.representation_adapter
            != "rosetta_pi_aloha_arms_bounded_sine_grippers"
            or config.get("bounded_gripper_decoder") is not True
        ):
            raise ValueError("Way artifact lost the bounded gripper decoder.")
        plan = simulator._load_yaml(_plan_path())
        contract_path = simulator._repository_path(str(plan["action_contract"]["path"]))
        if file_sha256(contract_path) != str(config["action_contract_sha256"]):
            raise ValueError("Way source and exported Action Contract checksums differ.")
        ensure_smolvla_action_boundary(
            self.preprocessor,
            self.postprocessor,
            load_action_contract(contract_path),
            action_space,
            action_contract_sha256=str(config["action_contract_sha256"]),
            upstream_revision=str(config["upstream_revision"]),
        )
        self.policy.eval()
        self._noise_mode = "zeros"
        self._noise_generator = torch.Generator(device="cpu")
        self._noise_seed: int | None = None

    def configure_noise(self, mode: str, seed: int | None) -> None:
        if mode == "zeros":
            if seed is not None:
                raise ValueError("Zero-noise inference must not register a random seed.")
        elif mode == "seeded_standard_normal":
            if seed is None or seed < 0:
                raise ValueError("Seeded Gaussian inference requires a non-negative seed.")
            self._noise_generator.manual_seed(seed)
        else:
            raise ValueError("Unsupported SmolVLA inference noise mode.")
        self._noise_mode = mode
        self._noise_seed = seed

    def predict(
        self, observation: dict[str, Any], instruction: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = observation.get("images")
        state = observation.get("robot_state")
        if not isinstance(images, dict) or "top" not in images:
            raise ValueError("Simulator observation has no registered top camera.")
        if not isinstance(state, torch.Tensor) or tuple(state.shape) != (
            self.state_dimension,
        ):
            raise ValueError("Simulator observation has an invalid ALOHA state.")
        sample = {
            "observation.images.top": images["top"],
            "observation.state": state,
            "task": instruction,
        }
        batch = self.preprocessor(sample)
        processed_state = batch.get("observation.state")
        if not isinstance(processed_state, torch.Tensor):
            raise ValueError("Processed simulator observation has no state tensor.")
        noise_shape = (
            1,
            self.policy.config.chunk_size,
            self.policy.config.max_action_dim,
        )
        if self._noise_mode == "zeros":
            noise = torch.zeros(
                noise_shape,
                device=self.device,
                dtype=processed_state.dtype,
            )
        elif self._noise_mode == "seeded_standard_normal":
            if self._noise_seed is None:
                raise RuntimeError("SmolVLA Gaussian noise was not seeded.")
            noise = torch.randn(
                noise_shape,
                generator=self._noise_generator,
                device="cpu",
                dtype=torch.float32,
            ).to(device=self.device, dtype=processed_state.dtype)
        else:
            raise RuntimeError("SmolVLA inference noise was not configured.")
        self.policy.reset()
        synchronize(torch, self.device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            action = self.policy.predict_action_chunk(batch, noise=noise)
        synchronize(torch, self.device)
        action = self.postprocessor(action)
        expected_shape = (1, self.chunk_length, self.action_dimension)
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != expected_shape:
            raise ValueError("Way simulator output differs from the Action Contract.")
        adapter_steps = [
            step
            for step in self.postprocessor.steps
            if getattr(step.__class__, "_registry_name", None)
            == "rosetta_pi_aloha_postprocessor"
        ]
        if len(adapter_steps) != 1:
            raise ValueError("Way simulator has no unique action decoder boundary.")
        raw = getattr(adapter_steps[0], "last_unclipped_action", None)
        if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != expected_shape:
            raise ValueError("Way simulator did not retain the pre-clipping action.")
        return raw[0].detach().cpu(), action[0].detach().cpu()


def main() -> int:
    plan_path = _plan_path()
    run_root = Path(os.environ["ROSETTA_RUN_ROOT"]).resolve()
    cache_root = run_root / "compiler_cache" / f"way-autodl-sim-{file_sha256(plan_path)[:12]}"
    triton_cache = cache_root / "triton"
    inductor_cache = cache_root / "inductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    simulator._repository_path = _runtime_plan_path
    simulator._OnlineSmolVLA = _AutoDLWayOnlineSmolVLA
    original_create_json = simulator.create_json

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        if payload.get("gate") in {
            "m2_gate_3_small_policy_rollout",
            "m2_gate_4_development_task_evaluation",
        }:
            payload["bounded_gripper_decoder"] = True
            payload["runtime_boundary"] = "autodl_container_instance"
            payload["nested_docker_used"] = False
            payload["autodl_way_sim_script_sha256"] = file_sha256(Path(__file__))
        original_create_json(path, payload)

    simulator.create_json = create_json
    return simulator.main()


if __name__ == "__main__":
    raise SystemExit(main())
