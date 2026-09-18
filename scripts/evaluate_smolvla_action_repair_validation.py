"""Run fixed validation with the Faust plan and repaired action processors."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_smolvla_validation as evaluator  # noqa: E402
import run_smolvla_action_repair_formal as formal_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla import load_smolvla_action_space  # noqa: E402
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402

DELEGATED_EVALUATOR_SHA256 = (
    "4aff5c8c23d8194a1b4997316c904a63498a47e94f9ba695683fe41a01f7d07c"
)


def _validate_delegated_evaluator() -> None:
    path = SCRIPTS_ROOT / "evaluate_smolvla_validation.py"
    if file_sha256(path) != DELEGATED_EVALUATOR_SHA256:
        raise ValueError("Delegated SmolVLA evaluator changed; register a new validation plan.")


def _checkpoint_source(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    step: int,
    dataset_root: Path,
) -> tuple[Path, dict[str, Any]]:
    allowed = {value for value in plan["validation"]["checkpoints"] if isinstance(value, int)}
    if step not in allowed:
        raise ValueError("Checkpoint step is outside the Faust validation protocol.")
    checkpoint_root = evaluator.phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    step_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / "formal"
        / str(plan["run_name"])
        / "checkpoints"
        / f"{step:06d}"
    )
    pretrained = step_dir / "pretrained_model"
    state = step_dir / "training_state"
    required = [
        pretrained / "config.json",
        pretrained / "model.safetensors",
        pretrained / "policy_preprocessor.json",
        pretrained / "policy_postprocessor.json",
        pretrained / "policy_preprocessor_step_7_normalizer_processor.safetensors",
        pretrained / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        pretrained / "train_config.json",
        state / "training_step.json",
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        raise FileNotFoundError("Faust checkpoint files are incomplete.")
    train_config = formal_runner._load_json(pretrained / "train_config.json")
    training_step = formal_runner._load_json(state / "training_step.json")
    training = plan["training"]
    dataset = train_config.get("dataset", {})
    policy = train_config.get("policy", {})
    if (
        training_step.get("step") != step
        or dataset.get("repo_id") != experiment["dataset"]["identifier"]
        or dataset.get("revision") != experiment["dataset"]["revision"]
        or dataset.get("episodes") != training["episodes"]
        or Path(str(dataset.get("root"))).resolve() != dataset_root
        or train_config.get("job_name") != plan["run_name"]
        or train_config.get("steps") != training["steps"]
        or train_config.get("save_freq") != training["save_freq"]
        or train_config.get("batch_size") != training["batch_size"]
        or policy.get("pretrained_revision") != experiment["model"]["revision"]
    ):
        raise ValueError("Faust checkpoint identity differs from its plan.")
    formal_runner._validate_saved_optimizer_contract(train_config, training)
    return pretrained, {
        "kind": "checkpoint",
        "step": step,
        "path": step_dir.relative_to(checkpoint_root).as_posix(),
        "model_safetensors_sha256": file_sha256(pretrained / "model.safetensors"),
        "policy_config_sha256": file_sha256(pretrained / "config.json"),
        "preprocessor_config_sha256": file_sha256(
            pretrained / "policy_preprocessor.json"
        ),
        "postprocessor_config_sha256": file_sha256(
            pretrained / "policy_postprocessor.json"
        ),
    }


def _validate_checkpoint_statistics(
    pretrained: Path, normalization_report: dict[str, Any]
) -> dict[str, str]:
    from safetensors.torch import load_file

    pre_path = pretrained / "policy_preprocessor_step_7_normalizer_processor.safetensors"
    post_path = pretrained / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    pre = load_file(pre_path, device="cpu")
    post = load_file(post_path, device="cpu")
    expected = dict(normalization_report["effective_stats"])
    expected.update(
        {
            feature: normalization_report["visual_statistics"]
            for feature in normalization_report["visual_features"]
        }
    )
    for feature, statistics in expected.items():
        for statistic, value in statistics.items():
            evaluator._assert_tensor_equal(pre[f"{feature}.{statistic}"], value, feature)
    for statistic, value in normalization_report["effective_stats"]["action"].items():
        evaluator._assert_tensor_equal(post[f"action.{statistic}"], value, "action")
    return {
        "preprocessor_statistics_sha256": file_sha256(pre_path),
        "postprocessor_statistics_sha256": file_sha256(post_path),
    }


def main() -> int:
    _validate_delegated_evaluator()
    evaluator.formal_runner = formal_runner
    evaluator._checkpoint_source = _checkpoint_source
    evaluator._validate_checkpoint_statistics = _validate_checkpoint_statistics
    original_loader = evaluator._load_policy_and_dataset
    original_create_json = evaluator.create_json

    def load_policy_and_dataset(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = original_loader(*args, **kwargs)
        policy, preprocessor, postprocessor, _dataset, _source, _hashes = result
        plan = args[0]
        experiment = args[1]
        contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
        ensure_smolvla_action_boundary(
            preprocessor,
            postprocessor,
            load_action_contract(contract_path),
            load_smolvla_action_space(experiment, require_explicit=True),
            action_contract_sha256=file_sha256(contract_path),
            upstream_revision=str(experiment["upstream"]["revision"]),
        )
        del plan, policy
        return result

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        plan_path = Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
        _plan, _base, experiment = formal_runner._validate_plan(plan_path)
        payload["action_space"] = load_smolvla_action_space(
            experiment, require_explicit=True
        ).as_dict()
        payload["bounded_gripper_decoder"] = True
        payload["evaluation_script_sha256"] = file_sha256(Path(__file__))
        original_create_json(path, payload)

    evaluator._load_policy_and_dataset = load_policy_and_dataset
    evaluator.create_json = create_json
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
