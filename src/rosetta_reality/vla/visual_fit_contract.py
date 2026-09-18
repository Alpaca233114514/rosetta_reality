"""Fail-closed Hestia artifact contracts; imports never load torch or datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from importlib.util import find_spec
from pathlib import Path, PurePosixPath

from rosetta_reality.vla import visual_fit as fit

HISTORY = "reports/training/m2-smolvla-native-visual-coverage40-result-2026-09-10.json"
PROCESSOR_FILES = {
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_7_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
}
MODELING_SHA = "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d"
TRAINER_SHA = "4d15d283ea54583f552b32088db0b6c195250905ca6daf06d4670383790e2059"
UPSTREAM = {
    "policies/smolvla/modeling_smolvla.py": MODELING_SHA,
    "datasets/sampler.py": "f715aaaa1118ef8901928f92975f7bc08bca412860f072bbcca84555303cc38e",
    "optim/schedulers.py": "05d57770348fbb3f412f52804a8d0f015f8128a37c55d5ce8ed9af6ed8522bc9",
    "scripts/lerobot_train.py": TRAINER_SHA,
}
FEATURES = [
    "train_only_statistics",
    "action_boundary_projection",
    "fixed_frame_sampler",
    "checkpoint_memory_trim",
    "gradient_clip_diagnostics",
    "checkpoint_metric_snapshot",
    "trackio_logging",
]
PARENT_CONFIG = "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
PARENT_SHA = "0e9dd0499d0708939ac73cc5d517849f133cf6deab072d9cde09f2880ae22210"
NORMALIZATION_SHA = "263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988"


def relative_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Artifact paths must be relative POSIX paths")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or name != relative.as_posix():
        raise ValueError("Artifact path escaped its registered root")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact symlink escaped its registered root")
    return path


def sealed_json(root: Path, reference: dict) -> dict:
    if not isinstance(reference, dict) or not fit._sha(reference.get("sha256")):
        raise ValueError("Artifact reference is not sealed")
    path = relative_file(root, reference["path"])
    if fit.file_hash(path) != reference["sha256"]:
        raise ValueError("Artifact hash changed: " + reference["path"])
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("Artifact must contain an object")
    return result


def check_inventory(directory: Path, expected: dict) -> None:
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Empty checkpoint inventory")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(expected):
        raise ValueError("Checkpoint file inventory changed")
    for name, sha in expected.items():
        path = relative_file(directory, name)
        if path.is_symlink() or not fit._sha(sha) or fit.file_hash(path) != sha:
            raise ValueError("Checkpoint file hash changed: " + name)


def processor_identity(files: dict) -> str:
    if not PROCESSOR_FILES <= set(files):
        raise ValueError("Consumed processor/tokenizer inventory is incomplete")
    values = {
        k: v
        for k, v in files.items()
        if k not in {"config.json", "train_config.json", "model.safetensors"}
    }
    if any(not fit._sha(v) for v in values.values()):
        raise ValueError("Processor file hash is missing")
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def process_start_ticks(pid: int) -> int:
    if type(pid) is not int or pid <= 0:
        raise ValueError("Invalid watchdog process")
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def check_execution(root: Path, contract: dict) -> dict:
    """Verify an active sealed comparison before importing any model runtime."""
    if (
        contract.get("protocol") != fit.PROTOCOL
        or contract.get("status") != "authorized_sealed"
        or contract.get("model_execution_authorized") is not True
        or set(contract.get("arms", {})) != {"B", "C"}
    ):
        raise ValueError("A sealed Hestia B/C execution contract is required")
    started, deadline = contract.get("started_unix"), contract.get("deadline_unix")
    if (
        type(started) not in (int, float)
        or type(deadline) not in (int, float)
        or not math.isfinite(started)
        or not math.isfinite(deadline)
        or not started <= time.time() < deadline
        or not 0 < deadline - started <= 1800
    ):
        raise ValueError("Missing, expired or excessive shared compute budget")
    watch = sealed_json(root, contract["watchdog"])
    if (
        watch.get("started_unix") != started
        or watch.get("deadline_unix") != deadline
        or watch.get("external_watchdog_verified") is not True
        or process_start_ticks(watch["pid"]) != watch.get("process_start_ticks")
    ):
        raise ValueError("The exact registered watchdog is not live")
    os.kill(watch["pid"], 0)
    for key, maximum in {
        "cuda_allocated_bytes": 8 * 1024**3,
        "cuda_reserved_bytes": 10 * 1024**3,
        "host_rss_bytes": 10 * 1024**3,
    }.items():
        value = contract.get("resource_limits", {}).get(key)
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError("Resource limit missing or exceeds registration")
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Offline execution is required")
    required = {
        "scripts/evaluate_visual_fit.py",
        "scripts/evaluate_visual_native_small.py",
        "scripts/smolvla_forward_check.py",
        "scripts/run_smolvla_v2.py",
        "scripts/train_smolvla_v2.py",
        "src/rosetta_reality/vla/visual_fit.py",
        "src/rosetta_reality/vla/visual_fit_contract.py",
        "src/rosetta_reality/vla/visual_coverage.py",
        "src/rosetta_reality/vla/vision_diagnostics.py",
        "src/rosetta_reality/vla/processor.py",
        HISTORY,
        "scripts/inspect_hestia_schedule.py",
        "scripts/seal_visual_fit_candidate.py",
        "src/rosetta_reality/vla/training/observation.py",
        "src/rosetta_reality/vla/training/observed_launch.py",
    }
    sources = contract.get("implementation_files", {})
    if not required <= set(sources):
        raise ValueError("Native collector source identity is incomplete")
    for name, sha in sources.items():
        if not fit._sha(sha) or fit.file_hash(relative_file(root, name)) != sha:
            raise ValueError("Collector implementation identity changed: " + name)
    if contract.get("upstream_files") != UPSTREAM:
        raise ValueError("Native upstream hashes differ from the registered source")
    installed = Path(next(iter(find_spec("lerobot").submodule_search_locations)))
    for name, sha in UPSTREAM.items():
        if fit.file_hash(installed / name) != sha:
            raise ValueError("Installed native upstream source changed")
    review = sealed_json(root, contract["review_plan"])
    if review.get("protocol") != fit.PROTOCOL or review.get("data") != {
        "train40": fit.TRAIN40,
        "development_validation5": fit.DEV5,
        "sealed_hidden5": fit.HIDDEN5,
    }:
        raise ValueError("Fit-strength review or fixed split changed")
    required_evidence = {
        "environment",
        "sample_contract",
        "resource_preflight",
        "two_step_smoke_reload",
        "B_training_integrity",
        "C_training_integrity",
    }
    prerequisites = contract.get("prerequisite_evidence", {})
    if not required_evidence <= set(prerequisites):
        raise ValueError("Execution prerequisite registration is incomplete")
    for reference in prerequisites.values():
        result = sealed_json(root, reference)
        if reference.get("accepted") is not True or result.get("status") != "passed":
            raise ValueError("A required prerequisite did not pass")
    return review


def expected_scheduler(arm: str) -> dict:
    return {
        "type": "cosine_decay_with_warmup",
        "num_warmup_steps": 16,
        "num_decay_steps": fit.UPDATES[arm],
        "peak_lr": 1e-4,
        "decay_lr": 2.5e-6,
    }


def validate_plan(plan: dict, arm: str) -> None:
    if arm not in fit.UPDATES:
        raise ValueError("Only B/C plans are accepted")
    steps = fit.UPDATES[arm]
    if (
        plan.get("parent_experiment", {}).get("config") != PARENT_CONFIG
        or plan["parent_experiment"].get("sha256") != PARENT_SHA
        or plan.get("normalization", {}).get("report_sha256") != NORMALIZATION_SHA
    ):
        raise ValueError("Pinned parent or train-only normalization changed")
    if (
        plan.get("scope") != "bounded_visual_overfit"
        or plan.get("run_name") != fit.RUN_NAMES[arm]
        or plan.get("initialization", {}).get("source") != "revision_pinned_base_model"
        or plan["initialization"].get("optimizer_state_reused") is not False
    ):
        raise ValueError("Hestia initialization or bounded diagnostic scope changed")
    for section in ("training", "optimizer_smoke"):
        config = plan[section]
        if (
            config.get("episodes") != fit.TRAIN40
            or type(config.get("steps")) is not int
            or config["steps"] != steps
            or type(config.get("batch_size")) is not int
            or config["batch_size"] != 4
            or config.get("num_workers") != 0
        ):
            raise ValueError("Active training or smoke recipe changed")
    training = plan["training"]
    if training.get("optimizer") != fit.OPTIMIZER or training.get(
        "scheduler"
    ) != expected_scheduler(arm):
        raise ValueError("Optimizer/scheduler differs from registered native recipe")
    checkpoints = [256] if arm == "B" else [320, 640, 960, 1280]
    if training.get("checkpoint_steps") != checkpoints or any(
        plan[k].get("save_freq") != checkpoints[0] for k in ("training", "optimizer_smoke")
    ):
        raise ValueError("Checkpoint retention grid changed")
    features = plan.get("features", [])
    if [f.get("name") for f in features] != FEATURES:
        raise ValueError("Unexpected loss/dropout/adapter feature or feature order")
    fixed = features[2]
    if fixed.get("phase") != "smoke" or fixed.get("sample_identities") != [
        {"episode": ep, "frame": 0} for ep in fit.TRAIN40
    ]:
        raise ValueError("Fixed-frame sampling identities changed")


def validate_experiment(experiment: dict) -> dict:
    revisions = {
        "base_revision": experiment["model"]["revision"],
        "vlm_revision": experiment["model"]["vlm_dependency"]["revision"],
        "upstream_revision": experiment["upstream"]["revision"],
        "data_revision": experiment["dataset"]["revision"],
    }
    data = experiment["dataset"]
    if (
        revisions != fit.REVISIONS
        or experiment.get("seed") != 20260809
        or data.get("train_episodes") != fit.TRAIN40
        or data.get("validation_episodes") != fit.DEV5
        or data.get("test_episodes") != fit.HIDDEN5
    ):
        raise ValueError("Resolved native model/data identity differs from registration")
    return revisions


def validate_saved_recipe(saved: dict, plan: dict, arm: str, identity: dict) -> dict:
    """Derive scoring metadata from the actual native saved config, never labels."""
    validate_plan(plan, arm)
    steps = fit.UPDATES[arm]
    if (
        saved.get("job_name") != fit.RUN_NAMES[arm]
        or identity.get("run_name") != fit.RUN_NAMES[arm]
        or saved.get("dataset", {}).get("episodes") != fit.TRAIN40
        or saved["dataset"].get("revision") != fit.REVISIONS["data_revision"]
        or saved.get("resume") is not False
    ):
        raise ValueError("Saved run/data/fresh-initialization identity changed")
    for key, expected in {
        "steps": steps,
        "batch_size": 4,
        "seed": 20260809,
        "num_workers": 0,
    }.items():
        if type(saved.get(key)) is not int or saved[key] != expected:
            raise ValueError("Saved native training recipe changed: " + key)
    if (
        saved.get("optimizer") != fit.OPTIMIZER
        or saved.get("scheduler") != expected_scheduler(arm)
        or saved.get("accelerator", {}).get("gradient_accumulation", {}).get("steps") != 1
    ):
        raise ValueError("Saved optimizer/scheduler/accumulation changed")
    for key, value in {
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "use_amp": False,
    }.items():
        if saved.get("policy", {}).get(key) is not value:
            raise ValueError("Saved freeze/precision contract changed")
    return {
        "optimizer_steps": steps,
        "scheduler_decay_steps": steps,
        "scheduler_warmup_steps": 16,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "seed": 20260809,
        "completed_sample_exposures": 4 * steps,
        "episodes": fit.TRAIN40,
        "frame": 0,
        "fresh_pinned_base": True,
        "optimizer_resumed": False,
        "native_loss_only": True,
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "use_amp": False,
        "optimizer": saved["optimizer"],
        "scheduler_decay_lr": 2.5e-6,
        "train_config_sha256": identity["checkpoint_files"]["train_config.json"],
    }


def check_native_recovery(checkpoint: Path, arm: str, *, expected_step: int | None = None) -> None:
    for name in (
        "optimizer_state.safetensors",
        "optimizer_param_groups.json",
        "rng_state.safetensors",
    ):
        if not (checkpoint / "training_state" / name).is_file():
            raise ValueError("Native recovery checkpoint is incomplete")
    state = json.loads((checkpoint / "training_state/scheduler_state.json").read_text())
    step = json.loads((checkpoint / "training_state/training_step.json").read_text())
    count = fit.UPDATES[arm] if expected_step is None else expected_step
    if count not in ({256} if arm == "B" else {320, 640, 960, 1280}):
        raise ValueError("Unregistered recovery checkpoint step")
    expected_lr = 1e-4 * (0.975 * 0.5 * (1 + math.cos(math.pi * count / fit.UPDATES[arm])) + 0.025)
    if (
        step.get("step") != count
        or state.get("last_epoch") != count
        or state.get("_step_count") != count + 1
        or len(state.get("_last_lr", [])) != 1
        or not math.isclose(state["_last_lr"][0], expected_lr, rel_tol=1e-12, abs_tol=1e-16)
    ):
        raise ValueError("Saved native step or scheduler state changed")


def validate_observation(report: dict, schedule: dict, resources: dict, arm: str) -> None:
    """C uses completed native updates; historical B uses its separately audited proof."""
    if arm != "C":
        raise ValueError("The completed-update observer is required only for new C")
    expected = schedule.get("sample_identities")
    if (
        schedule.get("status") != "passed"
        or not isinstance(expected, list)
        or len(expected) != 5120
        or schedule.get("native_order_verified") is not True
    ):
        raise ValueError("Native candidate schedule is not verified")
    counts = {ep: 0 for ep in fit.TRAIN40}
    for pair in expected:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or type(pair[0]) is not int
            or type(pair[1]) is not int
            or pair[0] not in counts
            or pair[1] != 0
        ):
            raise ValueError("Candidate schedule contains unexpected samples")
        counts[pair[0]] += 1
    if set(counts.values()) != {128}:
        raise ValueError("Candidate repetition allocation changed")
    digest = hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest()
    observation = report.get("observation", {})
    if (
        report.get("status") != "passed"
        or report.get("expected_schedule_sha256") != digest
        or observation.get("status") != "complete"
        or observation.get("errors") != []
        or observation.get("pending_samples") != []
    ):
        raise ValueError("Candidate native observation is incomplete or mismatched")
    for key, value in {
        "expected_samples": 5120,
        "delivered_samples": 5120,
        "completed_samples": 5120,
        "completed_updates": 1280,
        "successful_optimizer_step_calls": 1280,
    }.items():
        if type(observation.get(key)) is not int or observation[key] != value:
            raise ValueError("Candidate successful update count changed")
    rates = observation.get("learning_rates", [])
    if len(rates) != 1280:
        raise ValueError("Missing actual candidate learning rates")
    for index, values in enumerate(rates):
        factor = (
            ((1 / 17 - 1) * (1 - index / 16) + 1)
            if index < 16
            else (0.975 * 0.5 * (1 + math.cos(math.pi * index / 1280)) + 0.025)
        )
        if (
            not isinstance(values, list)
            or len(values) != 1
            or type(values[0]) not in (int, float)
            or not math.isfinite(values[0])
            or not math.isclose(values[0], 1e-4 * factor, rel_tol=1e-12, abs_tol=1e-16)
        ):
            raise ValueError("Actual candidate LR differs from the fixed 1280-step schedule")
    if resources.get("status") != "passed" or resources.get("stop_reasons") != []:
        raise ValueError("Candidate resource watchdog failed")
    for key, maximum in {
        "peak_cuda_allocated_bytes": 8 * 1024**3,
        "peak_cuda_reserved_bytes": 10 * 1024**3,
        "peak_host_rss_bytes": 10 * 1024**3,
    }.items():
        value = resources.get(key)
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError("Candidate resource evidence is missing or excessive")


def verify_arm_evidence(root: Path, contract: dict, arm: str, source: Path) -> None:
    """Follow raw proof references, including C's actual completed-update ledger."""
    identity = contract["arms"][arm]
    proof = sealed_json(root, contract["prerequisite_evidence"][arm + "_training_integrity"])
    if proof.get("status") != "passed":
        raise ValueError("Training integrity was not accepted")
    check_native_recovery(source.parent, arm)
    historical = json.loads((root / HISTORY).read_text())
    control = historical["checkpoint_identities"]["B"]
    if arm == "B":
        for key in ("run_name", "checkpoint_relative_to_root", "checkpoint_files"):
            if identity.get(key) != control[key]:
                raise ValueError("Historical B artifact identity changed")
        if identity["plan"]["sha256"] != control["plan"]["sha256"]:
            raise ValueError("Historical B plan changed")
        if (
            proof.get("historical_report_sha256") != fit.file_hash(root / HISTORY)
            or proof.get("model_sha256") != fit.CONTROL_MODEL_SHA256
            or historical["integrity"].get("status") != "passed"
            or historical["integrity"].get("consumed") != 1024
            or historical["integrity"].get("additional_optimizer_steps") != 0
        ):
            raise ValueError("Historical B proof changed")
    else:
        if (
            proof.get("model_sha256") != identity["checkpoint_files"]["model.safetensors"]
            or proof.get("plan_sha256") != identity["plan"]["sha256"]
            or proof.get("initialization") != "revision_pinned_base_model"
            or proof.get("optimizer_resumed") is not False
        ):
            raise ValueError("Candidate integrity belongs to another run")
        quarters = proof.get("quarter_checkpoints", {})
        if set(quarters) != {"320", "640", "960", "1280"}:
            raise ValueError("Missing retained candidate quarter checkpoints")
        for step, files in quarters.items():
            folder = source.parent.parent / f"{int(step):06d}"
            check_native_recovery(folder, arm, expected_step=int(step))
            check_inventory(folder, files)
        if quarters["1280"] != proof["checkpoint_files"]:
            raise ValueError("Final and quarter checkpoint inventories disagree")
        for name, sha in identity["checkpoint_files"].items():
            if proof["checkpoint_files"].get("pretrained_model/" + name) != sha:
                raise ValueError("Candidate recovery and model inventories disagree")
        observation = sealed_json(root, proof["observation"])
        schedule = sealed_json(root, proof["schedule"])
        resources = sealed_json(root, proof["resources"])
        validate_observation(observation, schedule, resources, arm)
        audit = sealed_json(root, proof["parameter_audit"])
        if (
            audit.get("status") != "passed"
            or audit.get("step") != 1280
            or audit.get("model_sha256") != proof["model_sha256"]
            or audit.get("base_revision") != fit.REVISIONS["base_revision"]
            or audit.get("recovery_state_complete") is not True
        ):
            raise ValueError("Candidate parameter audit does not bind this checkpoint")
        counts = audit.get("parameter_counts", {})
        if (
            counts.get("vlm") != {"total": 345, "changed": 0}
            or counts.get("expert", {}).get("total") != 145
            or not 0 < counts["expert"].get("changed", 0) <= 145
            or counts.get("projectors", {}).get("total") != 10
            or not 0 < counts["projectors"].get("changed", 0) <= 10
        ):
            raise ValueError("Candidate did not preserve the frozen/native update scope")
    if processor_identity(identity["checkpoint_files"]) != processor_identity(
        control["checkpoint_files"]
    ):
        raise ValueError("Consumed processor/tokenizer differs from the historical B control")
