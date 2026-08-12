"""Run Faust simulation gates with the serialized bounded-sine action boundary."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import smolvla_sim_gate as simulator  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla.action_space import SmolVLAActionSpace  # noqa: E402
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402


def _plan_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Faust simulation requires an explicit --plan path.") from error


class _ActionRepairOnlineSmolVLA(simulator._OnlineSmolVLA):
    def __init__(
        self,
        artifact: Path,
        config: dict[str, Any],
        normalization: dict[str, Any],
        contract: Any,
    ) -> None:
        super().__init__(artifact, config, normalization, contract)
        raw_action_space = config.get("action_space")
        if not isinstance(raw_action_space, dict):
            raise ValueError("Faust artifact has no explicit action-space identity.")
        action_space = SmolVLAActionSpace(**raw_action_space)
        if (
            action_space.representation_adapter
            != "rosetta_pi_aloha_arms_bounded_sine_grippers"
            or config.get("bounded_gripper_decoder") is not True
        ):
            raise ValueError("Faust artifact does not register the bounded gripper decoder.")
        plan = simulator._load_yaml(_plan_path())
        contract_path = simulator._repository_path(str(plan["action_contract"]["path"]))
        if file_sha256(contract_path) != str(config["action_contract_sha256"]):
            raise ValueError("Faust source and exported Action Contract checksums differ.")
        ensure_smolvla_action_boundary(
            self.preprocessor,
            self.postprocessor,
            load_action_contract(contract_path),
            action_space,
            action_contract_sha256=str(config["action_contract_sha256"]),
            upstream_revision=str(config["upstream_revision"]),
        )


def main() -> int:
    plan_path = _plan_path()
    run_root = Path(os.environ["ROSETTA_RUN_ROOT"]).resolve()
    cache_root = run_root / "compiler_cache" / f"faust-sim-{file_sha256(plan_path)[:12]}"
    triton_cache = cache_root / "triton"
    inductor_cache = cache_root / "inductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)

    simulator._OnlineSmolVLA = _ActionRepairOnlineSmolVLA
    original_create_json = simulator.create_json

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        if payload.get("gate") in {
            "m2_gate_3_small_policy_rollout",
            "m2_gate_4_development_task_evaluation",
        }:
            payload["bounded_gripper_decoder"] = True
            payload["action_repair_sim_script_sha256"] = file_sha256(Path(__file__))
        original_create_json(path, payload)

    simulator.create_json = create_json
    return simulator.main()


if __name__ == "__main__":
    raise SystemExit(main())
