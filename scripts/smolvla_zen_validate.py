"""Run the registered fixed-validation protocol against one Zen arm plan.

This is the Zen-native entry into the frozen fixed-validation engine
(``evaluate_smolvla_validation``). The engine itself stays untouched; this
wrapper binds it to the version-2 Zen plans through three explicit seams:

1. ``_validate_plan`` is replaced with the preregistered Zen identity checks
   (``smolvla_zen_protocol.validate_zen_plan``) plus an engine-compatible view
   that adds the schema-v1 keys the frozen engine reads.
2. ``_validate_prerequisites`` runs only the subset whose evidence semantics
   transfer across furnaces: the pre-training benchmark, Gate 1 and Gate 2 are
   revalidated deeply against the durable immutable reports. The remaining
   historical entries (lineage-specific smoke/overfit/trackio artifacts of the
   Faust-era campaign) are intentionally out of scope for a v2-native furnace;
   every produced report states this deviation under ``zen_protocol``.
3. The evaluation report gains a ``zen_protocol`` provenance block binding the
   wrapper checksum, theZen plan checksum and the substitution record.

Usage mirrors the frozen engine:

    python scripts/smolvla_zen_validate.py --plan <zen-plan> \
        --preflight-report <preflight.json> [--checkpoint-step 79]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_smolvla_validation as engine  # noqa: E402
import run_smolvla_formal as formal_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402
import smolvla_zen_protocol as protocol  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402

RUN_ROOT_SUBSTITUTIONS = ("benchmark", "gate1", "gate2")


_PROCESSOR_GLOBS = {
    "preprocessor": "policy_preprocessor_step_*_normalizer_processor.safetensors",
    "postprocessor": "policy_postprocessor_step_*_unnormalizer_processor.safetensors",
}


def _find_processor_file(pretrained_dir: Path, role: str) -> Path:
    """Locate a serialized processor state regardless of its pipeline-step tag."""

    matches = sorted(pretrained_dir.glob(_PROCESSOR_GLOBS[role]))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Expected exactly one {_PROCESSOR_GLOBS[role]} in {pretrained_dir.name}; "
        f"found {len(matches)}."
    )


def _zen_checkpoint_source(
    plan: dict,
    experiment: dict,
    step: int,
    dataset_root: Path,
):
    """Mirror ``evaluate_smolvla_validation._checkpoint_source`` semantics with
    pipeline-step-tag tolerance, plus an optional artifact-reload source used by
    the exporter's independent reload proof (``ROSETTA_ZEN_RELOAD_SOURCE``)."""

    import json

    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    reload_source = os.environ.get("ROSETTA_ZEN_RELOAD_SOURCE")
    if reload_source is not None:
        pretrained_dir = Path(reload_source).resolve()
        training_state_dir = None
        if not (pretrained_dir / "model.safetensors").is_file():
            raise FileNotFoundError("Zen artifact reload source is missing weights.")
    else:
        step_dir = (
            checkpoint_root
            / protocol.EXPERIMENT_ID
            / "formal"
            / plan["run_name"]
            / "checkpoints"
            / f"{step:06d}"
        )
        pretrained_dir = step_dir / "pretrained_model"
        training_state_dir = step_dir / "training_state"
    pre_norm = _find_processor_file(pretrained_dir, "preprocessor")
    post_norm = _find_processor_file(pretrained_dir, "postprocessor")
    required = [
        pretrained_dir / "config.json",
        pretrained_dir / "model.safetensors",
        pretrained_dir / "policy_preprocessor.json",
        pretrained_dir / "policy_postprocessor.json",
        pre_norm,
        post_norm,
        pretrained_dir / "train_config.json",
    ]
    if training_state_dir is not None:
        required.append(training_state_dir / "rng_state.safetensors")
        required.append(training_state_dir / "optimizer_state.safetensors")
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        raise FileNotFoundError("Formal checkpoint files are missing or empty.")
    train_config = json.loads((pretrained_dir / "train_config.json").read_text())
    training = plan["training"]
    expected_output = step_dir.parents[1] if training_state_dir is not None else None
    dataset_cfg = train_config.get("dataset", {})
    policy_cfg = train_config.get("policy", {})
    identity_ok = (
        dataset_cfg.get("repo_id") == experiment["dataset"]["identifier"]
        and dataset_cfg.get("revision") == experiment["dataset"]["revision"]
        and dataset_cfg.get("episodes") == training["episodes"]
        and Path(str(dataset_cfg.get("root"))).resolve() == dataset_root
        and train_config.get("job_name") == plan["run_name"]
        and train_config.get("seed") == experiment["seed"]
        and train_config.get("steps") == training["steps"]
        and train_config.get("save_freq") == training["save_freq"]
        and train_config.get("batch_size") == training["batch_size"]
        and policy_cfg.get("type") == "smolvla"
        and policy_cfg.get("pretrained_revision") == experiment["model"]["revision"]
        and policy_cfg.get("load_vlm_weights") is False
    )
    if training_state_dir is not None:
        identity_ok = identity_ok and (
            json.loads((training_state_dir / "training_step.json").read_text()).get("step")
            == step
            and Path(str(train_config.get("output_dir"))).resolve() == expected_output
        )
    if not identity_ok:
        raise ValueError("Formal checkpoint identity differs from the preregistered run.")
    if training_state_dir is not None:
        contract = formal_runner._validate_saved_optimizer_contract(train_config, training)
        del contract
    return pretrained_dir, {
        "kind": "checkpoint",
        "step": step,
        "path": (
            step_dir.relative_to(checkpoint_root).as_posix()
            if training_state_dir is not None
            else f"artifact:{pretrained_dir.name}"
        ),
        "model_safetensors_sha256": file_sha256(pretrained_dir / "model.safetensors"),
        "policy_config_sha256": file_sha256(pretrained_dir / "config.json"),
        "preprocessor_config_sha256": file_sha256(
            pretrained_dir / "policy_preprocessor.json"
        ),
        "postprocessor_config_sha256": file_sha256(
            pretrained_dir / "policy_postprocessor.json"
        ),
        "preprocessor_normalizer_sha256": file_sha256(pre_norm),
        "postprocessor_unnormalizer_sha256": file_sha256(post_norm),
    }


def _validate_zen_prerequisites(
    plan: dict,
    experiment: dict,
    base_path: Path,
    contract_sha256: str,
) -> dict[str, Path]:
    """Deep-validate the cross-furnace subset; document the scoped-out entries."""

    entries = {
        "benchmark": (
            protocol.EXPERIMENT_ID + "/benchmark/pre-training-af690cf13969cd22.json"
        ),
        "gate1": protocol.EXPERIMENT_ID + "/gates/gate1-e41df36c5d90.json",
        "gate2": protocol.EXPERIMENT_ID + "/gates/gate2-10fb9a9a364a.json",
    }
    resolved: dict[str, Path] = {}
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    for name, relative in entries.items():
        path = (run_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Zen prerequisite missing: {name}")
        resolved[name] = path
    phase_runner._validate_benchmark(
        resolved["benchmark"], experiment, base_path, contract_sha256
    )
    allowed_replay = [*protocol.TRAIN_EPISODES, *protocol.VALIDATION_EPISODES]
    for name, expected_gate in (
        ("gate1", "m2_gate_1_scripted_action"),
        ("gate2", "m2_gate_2_dataset_action_replay"),
    ):
        phase_runner._validate_gate(
            resolved[name],
            expected_gate=expected_gate,
            experiment_id=experiment["experiment_id"],
            contract_sha256=contract_sha256,
            dataset_revision=experiment["dataset"]["revision"],
            allowed_replay_episodes=allowed_replay,
        )
    return resolved


def _compatible_view(plan: dict) -> dict:
    view = copy.deepcopy(plan)
    spec = protocol.ZEN_SPECS[str(view["plan_id"])]
    prefix = os.environ.get("ROSETTA_ZEN_VALIDATION_PREFIX_OVERRIDE") or spec[
        "validation_prefix"
    ]
    validation = view.setdefault("validation", {})
    validation.update(
        {
            "run_name_prefix": prefix,
            "noise": "zeros",
            "flow_time": 0.5,
            "checkpoints": ["base", *protocol.CHECKPOINT_STEPS],
            "primary_selection_metric": "first_action_mae",
            "secondary_selection_metric": "fixed_flow_loss",
            "batch_size": 1,
            "samples_per_episode": len(validation.get("frame_offsets", [0])),
        }
    )
    return view


def _zen_checkpoint_statistics(pretrained_dir: Path, normalization_report: dict):
    """Mirror ``evaluate_smolvla_validation._validate_checkpoint_statistics``
    with pipeline-step-tag tolerance for serialized processor states."""

    import torch
    from safetensors.torch import load_file

    pre_path = _find_processor_file(pretrained_dir, "preprocessor")
    post_path = _find_processor_file(pretrained_dir, "postprocessor")
    pre = load_file(str(pre_path), device="cpu")
    post = load_file(str(post_path), device="cpu")
    expected = dict(normalization_report["effective_stats"])
    expected.update(
        {
            feature: normalization_report["visual_statistics"]
            for feature in normalization_report["visual_features"]
        }
    )
    for feature, statistics in expected.items():
        for statistic, value in statistics.items():
            key = f"{feature}.{statistic}"
            if key not in pre:
                raise ValueError(f"Saved processor statistic missing: {key}.")
            actual = torch.as_tensor(pre[key])
            reference = torch.tensor(value, dtype=actual.dtype)
            if actual.shape != reference.shape or not torch.equal(actual, reference):
                raise ValueError(f"Saved processor statistic differs: {key}.")
    for statistic, value in normalization_report["effective_stats"]["action"].items():
        key = f"action.{statistic}"
        if key not in post:
            raise ValueError(f"Saved processor statistic missing: {key}.")
        actual = torch.as_tensor(post[key])
        reference = torch.tensor(value, dtype=actual.dtype)
        if actual.shape != reference.shape or not torch.equal(actual, reference):
            raise ValueError(f"Saved processor statistic differs: {key}.")
    return {
        "normalizer_sha256": file_sha256(pre_path),
        "unnormalizer_sha256": file_sha256(post_path),
    }


def _patch_engine(plan_path: Path, compatible: dict) -> None:
    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla import load_smolvla_action_space
    from rosetta_reality.vla.action_space import load_smolvla_experiment
    from rosetta_reality.vla.processor import ensure_smolvla_action_boundary
    from rosetta_reality.vla.runtime_compatibility import (
        require_absolute_environment_directory,
        resolve_tokenizer_identity,
    )

    base_path = REPOSITORY_ROOT / protocol.PARENT_CONFIG
    experiment = load_smolvla_experiment(base_path, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    contract_path = REPOSITORY_ROOT / protocol.ACTION_CONTRACT_RELATIVE
    contract = load_action_contract(contract_path)

    original_processors = engine.make_pre_post_processors

    def _processors_with_boundary(*arguments, **keywords):
        pre, post = original_processors(*arguments, **keywords)
        ensure_smolvla_action_boundary(
            pre,
            post,
            contract,
            action_space,
            action_contract_sha256=file_sha256(contract_path),
            upstream_revision=str(experiment["upstream"]["revision"]),
        )
        return pre, post

    engine.make_pre_post_processors = _processors_with_boundary

    def _tokenizer_hashes_with_fallback(source_dir: Path) -> dict[str, str]:
        tokenizer_dir = source_dir / "tokenizer"
        has_files = tokenizer_dir.is_dir() and any(tokenizer_dir.rglob("*"))
        hf_home = (
            REPOSITORY_ROOT
            if has_files
            else require_absolute_environment_directory("HF_HOME")
        )
        hashes, _identity = resolve_tokenizer_identity(
            source_dir,
            base_model_root=phase_runner._model_root(experiment),
            experiment=experiment,
            hf_home=hf_home,
            expected_tokenizer_identity=None,
        )
        return hashes

    def validate_plan(active_path: Path):
        if Path(active_path).resolve() != plan_path:
            raise ValueError("The delegated Zen validation plan path changed.")
        return copy.deepcopy(compatible), base_path, experiment

    original_prerequisites = formal_runner._validate_prerequisites
    del original_prerequisites

    def validate_prerequisites(plan, experiment, base_path, contract_sha256):
        return _validate_zen_prerequisites(plan, experiment, base_path, contract_sha256)

    def validate_normalization(plan, experiment, base_path, contract_sha256):
        del experiment, base_path, contract_sha256
        run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        report_path = (
            run_root / protocol.NORMALIZATION_REPORT_UNDER_RUNROOT
        ).resolve()
        manifest_path = (
            run_root / protocol.VIEW_MANIFEST_UNDER_RUNROOT
        ).resolve()
        if (
            not report_path.is_file()
            or file_sha256(report_path) != protocol.NORMALIZATION_REPORT_SHA256
            or not manifest_path.is_file()
            or file_sha256(manifest_path) != protocol.VIEW_MANIFEST_SHA256
        ):
            raise ValueError("Zen normalization evidence changed.")
        section = plan["normalization"]
        if (
            section.get("report_sha256") != protocol.NORMALIZATION_REPORT_SHA256
            or section.get("dataset_view_manifest_sha256") != protocol.VIEW_MANIFEST_SHA256
        ):
            raise ValueError("Zen plan normalization pins differ from durable evidence.")
        return report_path, manifest_path, manifest_path.parent.resolve()

    original_create_json = engine.create_json

    def _create_json_with_zen(path: Path, payload: dict) -> None:
        payload["zen_protocol"] = {
            "schema_version": 1,
            "native_v2_posttrain": True,
            "wrapper_sha256": file_sha256(Path(__file__)),
            "protocol_module_sha256": file_sha256(Path(protocol.__file__)),
            "engine_sha256": file_sha256(Path(engine.__file__)),
            "plan_sha256": compatible["zen_protocol"]["plan_sha256"],
            "prerequisite_scope": list(RUN_ROOT_SUBSTITUTIONS),
            "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        }
        original_create_json(path, payload)

    def zen_repository_path(raw):
        relative = Path(str(raw))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Unsafe formal-plan-relative path.")
        direct = REPOSITORY_ROOT / relative
        if direct.exists():
            return direct.resolve()
        candidate = (phase_runner._absolute_root("ROSETTA_RUN_ROOT") / relative).resolve()
        run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        if not candidate.is_relative_to(run_root):
            raise ValueError("Durable evidence path escaped the run root.")
        return candidate

    formal_runner._validate_plan = validate_plan
    formal_runner._validate_prerequisites = validate_prerequisites
    formal_runner._validate_normalization = validate_normalization
    formal_runner._repository_path = zen_repository_path
    engine._tokenizer_hashes = _tokenizer_hashes_with_fallback
    engine._checkpoint_source = _zen_checkpoint_source
    engine._validate_checkpoint_statistics = _zen_checkpoint_statistics
    engine.create_json = _create_json_with_zen


def _restore_engine() -> None:
    """Restore surfaces defensively; the process exits immediately afterwards."""

    try:
        import importlib

        importlib.reload(formal_runner)
        importlib.reload(engine)
    except Exception as error:  # noqa: BLE001 - best-effort teardown only
        print(f"warning: engine restore incomplete: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    plan, plan_id = protocol.resolve_plan(plan_path)
    if args.checkpoint_step is not None and args.checkpoint_step not in protocol.CHECKPOINT_STEPS:
        raise ValueError("Checkpoint step outside the registered grid.")
    if not args.preflight_report.is_file():
        raise FileNotFoundError("Preflight report required by the frozen engine.")
    compatible = _compatible_view(plan)
    compatible["zen_protocol"] = {
        "plan_sha256": file_sha256(plan_path),
        "plan_id": plan_id,
    }
    sys.argv = [
        "smolvla_zen_validate",
        "--plan",
        str(plan_path),
        "--preflight-report",
        str(args.preflight_report),
    ]
    if args.checkpoint_step is not None:
        sys.argv += ["--checkpoint-step", str(args.checkpoint_step)]
    _patch_engine(plan_path, compatible)
    try:
        return engine.main()
    finally:
        _restore_engine()


if __name__ == "__main__":
    raise SystemExit(main())
