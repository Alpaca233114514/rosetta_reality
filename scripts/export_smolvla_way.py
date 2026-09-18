"""Export and independently reload the selected Way CUDA policy artifact."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_smolvla_aster_validation as checkpoint_evaluator  # noqa: E402
import evaluate_smolvla_validation as evaluator  # noqa: E402
import export_smolvla as exporter  # noqa: E402
import run_smolvla_state_robustness_cuda_formal as formal_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla import load_smolvla_action_space  # noqa: E402
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402


def _plan_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way export requires an explicit --plan path.") from error


def _copy_policy(source: Path, destination: Path) -> None:
    """Copy exactly the serialized policy and every processor state it names."""

    pipeline_names = ("policy_preprocessor.json", "policy_postprocessor.json")
    required_names = {"config.json", "model.safetensors", *pipeline_names}
    for pipeline_name in pipeline_names:
        raw = json.loads((source / pipeline_name).read_text(encoding="utf-8"))
        steps = raw.get("steps") if isinstance(raw, dict) else None
        if not isinstance(steps, list):
            raise ValueError(f"Saved processor pipeline is invalid: {pipeline_name}.")
        for step in steps:
            state_file = step.get("state_file") if isinstance(step, dict) else None
            if state_file is None:
                continue
            state_path = Path(str(state_file))
            if (
                state_path.is_absolute()
                or ".." in state_path.parts
                or len(state_path.parts) != 1
            ):
                raise ValueError("Saved processor state path is unsafe.")
            required_names.add(state_path.name)
    destination.mkdir()
    for name in sorted(required_names):
        path = source / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Selected Way policy file is missing: {name}.")
        shutil.copy2(path, destination / name)
    tokenizer = source / "tokenizer"
    if not tokenizer.is_dir():
        raise FileNotFoundError("Selected Way policy tokenizer directory is missing.")
    shutil.copytree(tokenizer, destination / "tokenizer")


def main() -> int:
    plan_path = _plan_path()
    run_root = Path(os.environ["ROSETTA_RUN_ROOT"]).resolve()
    cache_root = run_root / "compiler_cache" / f"way-export-{file_sha256(plan_path)[:12]}"
    triton_cache = cache_root / "triton"
    inductor_cache = cache_root / "inductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    plan, _base_path, experiment = formal_runner._validate_plan(plan_path)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract = load_action_contract(contract_path)
    contract_sha256 = file_sha256(contract_path)
    upstream_revision = str(experiment["upstream"]["revision"])

    evaluator.formal_runner = formal_runner
    evaluator._checkpoint_source = checkpoint_evaluator._checkpoint_source
    evaluator._validate_checkpoint_statistics = (
        checkpoint_evaluator._validate_checkpoint_statistics
    )
    original_dataset_loader = evaluator._load_policy_and_dataset

    def load_policy_and_dataset(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = original_dataset_loader(*args, **kwargs)
        _policy, preprocessor, postprocessor, _dataset, _source, _hashes = result
        ensure_smolvla_action_boundary(
            preprocessor,
            postprocessor,
            contract,
            action_space,
            action_contract_sha256=contract_sha256,
            upstream_revision=upstream_revision,
        )
        return result

    evaluator._load_policy_and_dataset = load_policy_and_dataset
    exporter.evaluator = evaluator
    exporter.formal_runner = formal_runner
    exporter._copy_policy = _copy_policy
    original_artifact_loader = exporter._load_artifact_policy

    def load_artifact_policy(source: Path, dataset: Any) -> tuple[Any, Any, Any]:
        policy, preprocessor, postprocessor = original_artifact_loader(source, dataset)
        ensure_smolvla_action_boundary(
            preprocessor,
            postprocessor,
            contract,
            action_space,
            action_contract_sha256=contract_sha256,
            upstream_revision=upstream_revision,
        )
        return policy, preprocessor, postprocessor

    exporter._load_artifact_policy = load_artifact_policy
    original_create_json = exporter.create_json

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        if path.name in {"config.json", "manifest.json"}:
            payload.update(
                {
                    "action_space": action_space.as_dict(),
                    "bounded_gripper_decoder": True,
                    "temporal_loss_profile": plan["loss_contract"]["profile"],
                    "temporal_loss_normalization": plan["loss_contract"][
                        "normalization"
                    ],
                    "state_robustness_profile": plan["state_robustness_contract"][
                        "profile"
                    ],
                    "state_jitter_training_only": True,
                    "state_jitter_active_at_deployment": False,
                    "way_export_script_sha256": file_sha256(Path(__file__)),
                }
            )
        original_create_json(path, payload)

    exporter.create_json = create_json
    return exporter.main()


if __name__ == "__main__":
    raise SystemExit(main())
