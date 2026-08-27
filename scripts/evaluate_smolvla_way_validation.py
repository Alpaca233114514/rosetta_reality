"""Run fixed Way validation on AutoDL CUDA with clean, unjittered inputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_smolvla_aster_validation as aster_evaluator  # noqa: E402
import evaluate_smolvla_validation as evaluator  # noqa: E402
import run_smolvla_state_robustness_cuda_formal as formal_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.vla.accelerator_memory import (  # noqa: E402
    memory_snapshot,
    reset_peak_memory_stats,
    synchronize,
)


def _plan_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way validation requires an explicit --plan path.") from error


def main() -> int:
    if (
        evaluator.file_sha256(Path(evaluator.__file__))
        != aster_evaluator.DELEGATED_EVALUATOR_SHA256
    ):
        raise ValueError("Historical evaluator changed; Way validation must be re-registered.")
    plan_path = _plan_path()
    plan, _base, _experiment = formal_runner._validate_plan(plan_path)
    if evaluator.os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Way validation requires AutoDL CUDA.")
    reset_peak_memory_stats(torch, "cuda")
    evaluator._sync = lambda device: synchronize(torch, device)
    aster_evaluator.formal_runner = formal_runner
    original_create_json = evaluator.create_json

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        payload["state_robustness_profile"] = plan["state_robustness_contract"][
            "profile"
        ]
        payload["state_jitter_active"] = False
        payload["state_jitter_training_only"] = True
        payload["accelerator_memory"] = memory_snapshot(torch, "cuda")
        payload["evaluation_script_sha256"] = file_sha256(Path(__file__))
        original_create_json(path, payload)

    evaluator.create_json = create_json
    return aster_evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
