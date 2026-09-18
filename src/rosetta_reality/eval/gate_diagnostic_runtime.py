"""Lazy real-runtime bridge, callable only with a fresh authorized registration."""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict

from .gate_diagnostic_io import load_json, relative, sha
from .gate_diagnostic_protocol import authorize


def run_registered(plan, root, stage, *, source=None):
    execution = authorize(plan, root, stage)  # All file/permission gates precede ML imports.
    output = relative(root, plan["output"]) / stage
    if output.exists():
        raise FileExistsError("Stage output already exists")
    if stage in ("replay", "probe"):
        if source is None:
            raise ValueError("Replay/probe require a collection directory")
        from .gate_diagnostic_verify import verify_bundle

        checked = verify_bundle(source)
        source_identity = load_json(source / "identity.json")
        expected = source_identity["identity"]
        if (
            checked["status"] != "verified_complete"
            or checked["stage"] != "collect"
            or expected.get("kind") != "registered_model"
            or expected.get("checkpoint_files") != plan["checkpoint"]["files"]
            or expected.get("sources") != plan["sources"]
            or expected.get("training_plan_sha256") != plan["training_plan"]["sha256"]
            or expected.get("action_contract_sha256") != plan["action_contract"]["sha256"]
        ):
            raise ValueError("Replay source must match the registered real endpoint and code")
    import resource

    import torch

    from rosetta_reality.sim import load_action_contract
    from scripts import smolvla_autodl_vfunfreeze_sim_gate as adapter
    from scripts import smolvla_sim_gate as engine
    from scripts.diagnose_zen_noise_transfer import parameter_digests
    from scripts.iris_gate import deployment_state_dimension, load_gate_context
    from scripts.run_iris_furnace import alive

    if not torch.cuda.is_available():
        raise RuntimeError("Registered CUDA runtime unavailable")
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    if str(gpu.uuid) != execution["gpu_uuid"]:
        raise ValueError("GPU identity differs")

    def guard():
        watchdog = execution["watchdog"]
        if time.time() >= execution["deadline"] or not alive(watchdog["pid"], watchdog["ticks"]):
            raise RuntimeError("Runtime deadline or watchdog expired")
        if (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 1024**3
            or torch.cuda.max_memory_allocated() > 8 * 1024**3
            or torch.cuda.max_memory_reserved() > 10 * 1024**3
        ):
            raise RuntimeError("Registered RSS/CUDA budget exceeded")
        parent = output
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < 256 * 1024**2:
            raise RuntimeError("Evidence return reserve exhausted")

    guard()
    parent = output
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < plan["maximum_evidence_bytes"] + 256 * 1024**2:
        raise RuntimeError("Insufficient disk for the registered evidence budget")
    context = load_gate_context(
        relative(root, plan["training_plan"]["path"]),
        checkpoint=relative(root, plan["checkpoint"]["path"]),
        checkpoint_files=plan["checkpoint"]["files"],
        split="train",
        deadline=execution["deadline"],
    )
    contract = load_action_contract(relative(root, plan["action_contract"]["path"]))
    if asdict(contract) != asdict(context.physical):
        raise ValueError("Native and rollout contracts differ")
    config = load_json(relative(root, plan["artifact_config"]["path"]))
    online = object.__new__(adapter._build_online_cuda_class())
    online.policy, online.preprocessor, online.postprocessor = (
        context.policy,
        context.pre,
        context.post,
    )
    online.device, online.mixed_precision = torch.device("cuda"), "bf16"
    online.state_dimension = deployment_state_dimension(
        context.policy.config, contract, config["dataset_features"]
    )
    online.action_dimension, online.chunk_length = contract.dimension, contract.chunk_length
    online._noise_generator = torch.Generator(device="cpu")
    online._noise_mode, online._noise_seed = "zeros", None
    before = parameter_digests(online.policy)

    def unchanged():
        return parameter_digests(online.policy) == before

    guard()
    if stage in ("collect", "collect-repro"):
        from .gate_diagnostic_capture import collect_episode

        identity = {
            "kind": "registered_model",
            "run_id": plan["run_id"],
            "sources": plan["sources"],
            "checkpoint_files": plan["checkpoint"]["files"],
            "training_plan_sha256": plan["training_plan"]["sha256"],
            "action_contract_sha256": plan["action_contract"]["sha256"],
            "runtime": execution,
            "optimizer_steps": 0,
            "artifact_config_sha256": sha(relative(root, plan["artifact_config"]["path"])),
        }
        if stage == "collect-repro":
            from .reproducibility_capture import collect_reproducibility

            return collect_reproducibility(
                engine,
                online,
                contract,
                output,
                identity,
                pairing=plan["reproducibility"],
                maximum_bytes=plan["maximum_evidence_bytes"],
                guard=guard,
                check_unchanged=unchanged,
            )
        return collect_episode(
            engine,
            online,
            contract,
            output,
            identity,
            maximum_bytes=plan["maximum_evidence_bytes"],
            guard=guard,
            check_unchanged=unchanged,
        )
    from .gate_diagnostic_replay import replay_bundle

    return replay_bundle(
        online,
        source,
        output,
        probe=stage == "probe",
        guard=guard,
        check_unchanged=unchanged,
        maximum_bytes=plan["maximum_evidence_bytes"],
    )
