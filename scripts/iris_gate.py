"""Iris fixed-endpoint deployment adapter for the unchanged Gate 3/4 engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from scripts.iris_runtime import budget, load_native, save, sha, verify_reload  # noqa: E402

NAME = "iris-002-gate34-20260913-003"
SOURCE_WORKSPACE = "20260913T135230Z-5d285f5c8297-3ba03ac32579"
SOURCE_RUN = "iris-k-scene-20260913-002"
MANIFEST_SHA = "8f1f80d9676c2a8f25c587d9bf7462fe41778c6fa992681096f7284dbcb9812b"
ARMS = {"control": "451", "treatment": "452"}


def load(path):
    return json.loads(Path(path).read_text())


def source_job():
    return (
        Path(os.environ["ROSETTA_AUTODL_ROOT"])
        / "workspaces"
        / SOURCE_WORKSPACE
        / "runs"
        / SOURCE_RUN
    )


def load_gate_context(*args, **kwargs):
    """Keep the native data-view root separate from the gate report root."""
    previous = os.environ.get("ROSETTA_RUN_ROOT")
    native_root = (Path(os.environ["ROSETTA_AUTODL_ROOT"]) / "runs").resolve(strict=True)
    os.environ["ROSETTA_RUN_ROOT"] = str(native_root)
    try:
        return load_native(*args, **kwargs)
    finally:
        if previous is None:
            os.environ.pop("ROSETTA_RUN_ROOT", None)
        else:
            os.environ["ROSETTA_RUN_ROOT"] = previous


def deployment_state_dimension(policy_config, contract, dataset_features):
    from scripts import smolvla_sim_gate as engine

    return engine._validate_policy_contract_shape(
        policy_config, contract, engine._dataset_state_dimension(dataset_features)
    )


def check_source(source):
    manifest = source / "handoff-manifest.json"
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError("Iris source handoff manifest changed")
    receipt = load(source / "transfer-receipt.json")
    if (
        receipt.get("manifest_sha256") != MANIFEST_SHA
        or receipt.get("all_file_sha256_matched") is not True
    ):
        raise ValueError("Verified off-worker backup receipt missing")
    for name, spec in load(manifest)["files"].items():
        path = source / name
        if path.is_symlink() or path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]:
            raise ValueError("Iris source artifact drift: " + name)
    if load(source / "worker-exited.json")["error"] is not None:
        raise ValueError("Iris training worker did not finish cleanly")


def render(job):
    import yaml

    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla.action_space import load_smolvla_experiment
    from scripts import smolvla_autodl_vfunfreeze_sim_gate as reference
    from scripts.iris_protocol import baseline, bind_inputs

    source = source_job()
    check_source(source)
    bind_inputs()
    original = baseline()
    exp = load_smolvla_experiment(ROOT / original["parent_experiment"]["config"], ROOT)
    norm_path = ROOT / original["normalization"]["report"]
    norm = load(norm_path)
    info = load(
        (ROOT / original["normalization"]["dataset_view_manifest"]).parent / "meta/info.json"
    )
    contract_path = ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    contract = load_action_contract(contract_path)
    # Link immutable old evidence for the native training-plan resolver; never edit it.
    old = ROOT / "runs" / SOURCE_RUN
    if not old.exists():
        old.symlink_to(source, target_is_directory=True)
    if old.resolve() != source.resolve():
        raise ValueError("Source job alias differs")
    for arm, suffix in ARMS.items():
        proof = verify_reload(source / (arm + "-first.npz"), source / (arm + "-reload.npz"))
        artifact = job.parent / (NAME + "-artifacts") / exp["experiment_id"] / (NAME + "-" + arm)
        artifact.mkdir(parents=True)
        pretrained = source / "exports" / arm
        files = load(source / (arm + "-main-checkpoint.json"))["files"]
        (artifact / "pretrained_model").symlink_to(pretrained, target_is_directory=True)
        cfg = {
            "artifact_id": artifact.name,
            "experiment_id": exp["experiment_id"],
            "mixed_precision": "bf16",
            "rename_map": exp["dataset"]["rename_map"],
            "action_space": {**exp["model"]["action_space"], "adapt_to_pi_aloha": False},
            "bounded_gripper_decoder": True,
            "action_contract_sha256": sha(contract_path),
            "upstream_revision": exp["upstream"]["revision"],
            "dataset_features": {
                k: v
                for k, v in info["features"].items()
                if k in ("observation.images.top", "observation.state", "action")
            },
            "dataset_fps": info["fps"],
            "selected_checkpoint_step": 1280,
            "hidden_test_loaded": False,
            "iris_arm": arm,
        }
        save(artifact / "config.json", cfg)
        save(artifact / "normalization.json", norm)
        save(artifact / "action_contract.json", asdict(contract))
        inventory = {"pretrained_model/" + k: v for k, v in files.items()}
        inventory.update(
            {
                p: sha(artifact / p)
                for p in ("config.json", "normalization.json", "action_contract.json")
            }
        )
        save(
            artifact / "manifest.json",
            {
                "status": "verified",
                "artifact_id": artifact.name,
                "experiment_id": exp["experiment_id"],
                "selected_checkpoint_step": 1280,
                "selected_checkpoint_model_sha256": files["model.safetensors"],
                "reload": {"verified": True, "exact_tensor_equality": True, "source_proof": proof},
                "files": inventory,
                "hidden_test_loaded": False,
                "scientific_acceptance": "negative_result",
                "source_manifest_sha256": MANIFEST_SHA,
            },
        )
        (job / "artifact-metadata" / arm).mkdir(parents=True)
        for name in ("config.json", "normalization.json", "action_contract.json", "manifest.json"):
            save(job / "artifact-metadata" / arm / name, load(artifact / name))
        selection = job / (arm + "-endpoint.json")
        save(
            selection,
            {
                "status": "passed",
                "meaning": "fixed endpoint identity verification only",
                "method": "preregistered_fixed_final_step_1280",
                "scientific_acceptance": "negative_result",
                "selected": {"step": 1280, "model_safetensors_sha256": files["model.safetensors"]},
                "hidden_test_loaded": False,
            },
        )
        backup = job / (arm + "-backup.json")
        save(
            backup,
            {
                "status": "verified",
                "off_host_copy_created": True,
                "source_manifest_sha256": MANIFEST_SHA,
                "receipt_sha256": sha(source / "transfer-receipt.json"),
            },
        )
        values = dict(
            sim_plan_id=NAME + "-" + arm,
            experiment_id=exp["experiment_id"],
            artifact_id=artifact.name,
            artifact_manifest_sha256=sha(artifact / "manifest.json"),
            alignment_report="not_applicable",
            alignment_sha="0" * 64,
            prior_report=reference.PRIOR_FAILURE["report"],
            prior_sha=reference.PRIOR_FAILURE["report_sha256"],
            prior_task_report=reference.PRIOR_TASK_FAILURE["report"],
            prior_task_sha=reference.PRIOR_TASK_FAILURE["report_sha256"],
            selection_report=selection.relative_to(ROOT).as_posix(),
            selection_sha=sha(selection),
            selected_step=1280,
            model_sha=files["model.safetensors"],
            backup_sha=sha(backup),
            contract_sha=sha(contract_path),
            suffix=suffix,
            sim_code_blocks="  scripts/iris_gate.py: " + sha(Path(__file__)),
        )
        plan = yaml.safe_load(reference.SIM_PLAN_TEMPLATE.format(**values))
        plan.pop("entry_permit")  # vfunfreeze-specific permit; never claim Iris passed it.
        plan["hypothesis"] = (
            "Measure fixed Iris control/treatment closed-loop outcomes "
            "after failed offline acceptance"
        )
        plan["single_axis_change"] = {
            "field": "training.image_k_scene_variance",
            "control": 0,
            "candidate": 0.01,
        }
        plan["artifact_backup"]["report"] = backup.relative_to(ROOT).as_posix()
        plan["simulation_code_sha256"] = load(job / "registration.json")["sources"]
        with (job / (arm + "-sim.yaml")).open("x") as stream:
            yaml.safe_dump(plan, stream, sort_keys=False)
    save(job / "plans-seal.json", {a: sha(job / (a + "-sim.yaml")) for a in ARMS})


def run_gate(job, arm, gate):
    import torch

    from scripts import smolvla_autodl_vfunfreeze_sim_gate as reference
    from scripts import smolvla_sim_gate as engine
    from scripts.diagnose_zen_noise_transfer import parameter_digests

    reg = load(job / "registration.json")
    source = source_job()
    plan_path = job / (arm + "-sim.yaml")
    if sha(plan_path) != load(job / "plans-seal.json")[arm]:
        raise ValueError("Rendered gate plan drift")
    for name, digest in reg["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Gate source drift")
    os.environ["ROSETTA_ARTIFACT_ROOT"] = str(job.parent / (NAME + "-artifacts"))
    os.environ["ROSETTA_RUN_ROOT"] = str(job / "results")
    Base = reference._build_online_cuda_class()
    models = []

    class Online(Base):
        def __init__(self, artifact, config, normalization, contract):
            context = load_gate_context(
                source / "plans" / (arm + "-main.yaml"),
                checkpoint=source / "exports" / arm,
                checkpoint_files=load(source / (arm + "-main-checkpoint.json"))["files"],
                split="train",
                deadline=reg["deadline"],
            )
            if asdict(context.physical) != asdict(contract):
                raise ValueError("Native and simulation Action Contracts differ")
            self.policy, self.preprocessor, self.postprocessor = (
                context.policy,
                context.pre,
                context.post,
            )
            self.device, self.mixed_precision = torch.device("cuda"), "bf16"
            self.state_dimension = deployment_state_dimension(
                self.policy.config, contract, config["dataset_features"]
            )
            self.action_dimension, self.chunk_length = contract.dimension, contract.chunk_length
            self._noise_mode, self._noise_seed = "zeros", None
            self._noise_generator = torch.Generator(device="cpu")
            self.initial_parameters = parameter_digests(self.policy)
            models.append(self)
            environment = engine.GymAlohaEnvironment(contract, maximum_episode_steps=20)
            try:
                observation = environment.reset(seed=20260809)
                self.configure_noise("zeros", None)
                actual_raw, actual = self.predict(observation, "Insert the peg into the socket.")
                sample = {
                    "observation.images.top": observation["images"]["top"],
                    "observation.state": observation["robot_state"],
                    "task": "Insert the peg into the socket.",
                }
                batch = context.pre(sample)
                noise = torch.zeros(
                    (1, self.policy.config.chunk_size, self.policy.config.max_action_dim),
                    device="cuda",
                )
                self.policy.reset()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = self.policy.predict_action_chunk(batch, noise=noise)
                    expected = context.post(prediction.clone())
                if not torch.equal(expected[0].cpu(), actual) or not torch.equal(
                    context.decoder.last_unclipped_action[0].cpu(), actual_raw
                ):
                    raise ValueError(
                        "Simulation adapter differs from native saved-processor inference"
                    )
                save(
                    job / (arm + "-" + gate + "-bridge.json"),
                    {
                        "status": "passed",
                        "exact_raw_and_projected_chunks": True,
                        "policy_forwards": 2,
                        "optimizer_steps": 0,
                    },
                )
            finally:
                environment.close()

        def predict(self, observation, instruction):
            budget(reg["deadline"])
            value = super().predict(observation, instruction)
            budget(reg["deadline"])
            return value

    # Original engine performs all manifest/contract/prior-failure and Gate 3/4 checks.
    original_path = engine._repository_path

    def evidence(raw):
        if raw in (reference.PRIOR_FAILURE["report"], reference.PRIOR_TASK_FAILURE["report"]):
            local = ROOT / raw
            if local.is_file():
                return local
            return Path(os.environ["ROSETTA_AUTODL_ROOT"]) / raw
        return original_path(raw)

    engine._repository_path = evidence
    engine._OnlineSmolVLA = Online
    engine._runtime = reference._cuda_runtime
    output = (
        job
        / "results"
        / "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
        / "gates"
    )
    report = output / f"{gate}-smolvla-sim-{ARMS[arm]}.json"
    if report.exists():
        raise FileExistsError("No gate retry or overwrite allowed")
    code = (
        engine.gate3(plan_path)
        if gate == "gate3"
        else engine.gate4(plan_path, output / f"gate3-smolvla-sim-{ARMS[arm]}.json")
    )
    for model in models:
        if parameter_digests(model.policy) != model.initial_parameters:
            raise ValueError("Gate changed parameters")
    save(
        job / (arm + "-" + gate + "-parameter-check.json"),
        {"unchanged": True, "optimizer_steps": 0},
    )
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("render", "gate3", "gate4"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    args.job = args.job.resolve()
    if args.job != ROOT / "runs" / NAME:
        raise ValueError("Unregistered Gate job path")
    from scripts.run_iris_furnace import alive

    registration = load(args.job / "registration.json")
    watchdog = load(args.job / "watchdog.json")
    if (
        registration.get("id") != NAME
        or registration.get("execution_authorized") is not True
        or not registration["started"] <= time.time() < registration["deadline"]
        or not alive(watchdog["pid"], watchdog["ticks"])
    ):
        raise ValueError("Gate registration expired or independent watchdog absent")
    if args.mode == "render":
        render(args.job)
        return 0
    return run_gate(args.job, args.arm, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
