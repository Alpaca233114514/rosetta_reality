"""Independent base measurement; never invokes or declares a formal Gate.

Imports are stdlib-only until the explicitly requested CUDA execution boundary.
Physical safety stops one episode; numerical, identity and resource faults stop
the batch. No shutdown, optimizer, download or trained-model path is present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1000, 1001, 1002, 1003, 1004)
MODEL_SHA = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
MANIFEST_SHA = "6c35ef6e294ba50dae1e1961b042eaf1f42e988d48f876009da122be3d61cc77"
CONTRACT_SHA = "fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62"
REQUIRED_SOURCES = (
    "scripts/diagnose_smolvla_base_five_scene.py", "scripts/smolvla_sim_gate.py",
    "scripts/smolvla_autodl_vfunfreeze_sim_gate.py", "src/rosetta_reality/sim/gym_aloha.py",
    "src/rosetta_reality/vla/processor.py", "tests/test_base_five_scene_diagnostic.py",
    "src/rosetta_reality/vla/action_space.py", "src/rosetta_reality/sim/action_contract.py",
)
HISTORICAL = ROOT / "runs/prepost-gate-received-20260924-001/prepost-gate4-20260924-001/artifacts/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/prepost-base-aloha-interface-001"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    json.dumps(value, allow_nan=False)
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def validate_plan(plan):
    if (plan.get("stage") != "independent_base_five_scene_measurement"
            or plan.get("status") != "preregistered"
            or plan.get("seeds") != list(SEEDS)
            or plan.get("policy_noise_seeds") != list(SEEDS)
            or plan.get("maximum_steps") != 500
            or plan.get("noise") != "seeded_standard_normal"
            or plan.get("instruction") != "Insert the peg into the socket."
            or plan.get("model_revision") != "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
            or plan.get("action_contract") != {"path": "configs/sim/aloha_insertion_smolvla.yaml"}
            or plan.get("hidden_test_loaded") is not False
            or plan.get("formal_gate_result") is not None
            or plan.get("model_sha256") != MODEL_SHA
            or plan.get("artifact_manifest_sha256") != MANIFEST_SHA
            or plan.get("action_contract_sha256") != CONTRACT_SHA
            or plan.get("shutdown_requested") is not False
            or plan.get("completion_receipt_required") is not True):
        raise ValueError("Independent base registration changed")
    limits = plan["resources"]
    if (limits.get("maximum_seconds") != 1800
            or limits.get("rss_bytes") != 12 * 1024**3
            or limits.get("cuda_allocated_bytes") != 8 * 1024**3
            or limits.get("cuda_reserved_bytes") != 10 * 1024**3
            or limits.get("evidence_bytes") != 256 * 1024**2
            or limits.get("cuda_peak_window") != "after_policy_load_and_initial_parameter_hash"):
        raise ValueError("Registered resource limits changed")
    if set(plan.get("source_sha256", {})) != set(REQUIRED_SOURCES):
        raise ValueError("Incomplete source identity inventory")
    for relative, expected in plan["source_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha(ROOT / path) != expected:
            raise ValueError("Source identity mismatch: " + relative)


def verify_artifact(artifact):
    artifact = Path(artifact)
    if sha(artifact / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Base artifact manifest identity changed")
    manifest = read(artifact / "manifest.json")
    if (manifest.get("selected_checkpoint_model_sha256") != MODEL_SHA
            or manifest.get("hidden_test_loaded") is not False
            or manifest.get("reload", {}).get("exact_tensor_equality") is not True):
        raise ValueError("Historical base identity/reload invalid")
    for relative, expected in manifest["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha(artifact / path) != expected:
            raise ValueError("Base artifact payload changed: " + relative)
    config, norm = read(artifact / "config.json"), read(artifact / "normalization.json")
    if (config.get("hidden_test_loaded") is not False
            or config.get("selected_checkpoint_model_sha256") != MODEL_SHA
            or norm.get("source_split") != "train"
            or norm.get("hidden_test_loaded") is not False):
        raise ValueError("Base config/normalization boundary invalid")
    return config, norm


def prepare_artifact(destination, canonical, base):
    """Reconstitute only the exact historical interface; existing files untouched."""
    import shutil
    manifest = read(HISTORICAL / "manifest.json")
    if sha(HISTORICAL / "manifest.json") != MANIFEST_SHA or sha(base) != MODEL_SHA:
        raise ValueError("Historical registration/base payload changed")
    sources = {}
    for relative, expected in manifest["files"].items():
        if relative == "pretrained_model/model.safetensors":
            source = Path(base)
        elif (HISTORICAL / relative).is_file():
            source = HISTORICAL / relative
        else:
            source = Path(canonical) / relative
        if sha(source) != expected:
            raise ValueError("Canonical processor payload mismatch: " + relative)
        sources[relative] = source.resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, source in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    shutil.copyfile(HISTORICAL / "manifest.json", destination / "manifest.json")
    verify_artifact(destination)


def run_batch(output, episode_runner, provenance, *, final_check=None):
    """Injectable batch lifecycle: retain denominator and never impute failures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes, attempted, error, final_checks = [], [], None, None
    started = time.monotonic()
    try:
        for seed in SEEDS:
            attempted.append(seed)
            item = episode_runner(seed, output / f"seed-{seed}.jsonl")
            if item.get("seed") != seed or item.get("outcome") not in (
                    "task_success", "task_failed", "physical_safety_stop"):
                raise ValueError("Invalid independent episode outcome")
            save(output / f"seed-{seed}.json", item)
            episodes.append(item)
        if final_check is not None:
            final_checks = final_check()
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    result = {"schema": "rosetta.base_five_scene_diagnostic.v1",
              "status": "completed" if error is None else "blocked",
              "episodes": episodes, "error": error,
              "measured_seed_count": len(episodes), "registered_seed_count": 5,
              "attempted_seeds": attempted,
              "incomplete_seeds": [s for s in attempted if s not in [e["seed"] for e in episodes]],
              "unmeasured_seeds": [s for s in SEEDS if s not in [e["seed"] for e in episodes]],
              "success_count": sum(e["success"] for e in episodes),
              "elapsed_seconds": time.monotonic() - started,
              "original_gate4_protocol_equivalent": False,
              "formal_gate_result": None, "historical_base_gate3_status": "failed",
              "historical_base_gate4_status": "not_measured",
              "optimizer_steps": 0, "m2_complete": False,
              "shutdown_requested": False, "provenance": provenance,
              "final_checks": final_checks}
    checker = getattr(episode_runner, "check_budget", None)
    if error is None and callable(checker):
        try:
            # Include the bytes about to be committed in the evidence boundary.
            checker(additional_evidence_bytes=len((json.dumps(result, sort_keys=True, indent=2,
                     allow_nan=False) + "\n").encode("utf-8")))
        except BaseException as exc:
            result.update(status="blocked", error={"type": type(exc).__name__, "message": str(exc)})
    saved_summary = dict(result)
    if result["status"] == "completed" and callable(checker):
        saved_summary["status"] = "measurements_complete_pending_closeout"
    saved_summary["closeout_report"] = "completion.json"
    save(output / "summary.json", saved_summary)
    if callable(checker):
        try:
            checker()
        except BaseException as exc:
            result.update(status="blocked", error={"type": type(exc).__name__, "message": str(exc)})
    # Summary records measurements; this post-write receipt owns process completion.
    receipt = {"status": result["status"], "error": result["error"],
               "measured_seed_count": result["measured_seed_count"],
               "summary_sha256": sha(output / "summary.json"), "shutdown_requested": False}
    if result["status"] == "completed" and callable(checker):
        try:
            checker(additional_evidence_bytes=len((json.dumps(receipt, sort_keys=True, indent=2,
                    allow_nan=False) + "\n").encode("utf-8")))
        except BaseException as exc:
            result.update(status="blocked", error={"type": type(exc).__name__, "message": str(exc)})
            receipt.update(status="blocked", error=result["error"])
    save(output / "completion.json", receipt)
    return result


def parameter_digest(policy):
    digest = hashlib.sha256()
    for name, value in list(policy.named_parameters()) + list(policy.named_buffers()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def make_budget_checker(plan, output, budget_started, *, clock, resource_reader):
    """Pure injectable boundary, including CUDA peaks during the measurement window."""
    def check_budget(*, additional_evidence_bytes=0):
        limits = plan["resources"]
        resources = resource_reader()
        evidence = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        elapsed = clock() - budget_started
        if (elapsed >= limits["maximum_seconds"]
                or resources["rss_bytes"] > limits["rss_bytes"]
                or evidence + additional_evidence_bytes > limits["evidence_bytes"]
                or max(resources["cuda_allocated_bytes"], resources["cuda_peak_allocated_bytes"])
                > limits["cuda_allocated_bytes"]
                or max(resources["cuda_reserved_bytes"], resources["cuda_peak_reserved_bytes"])
                > limits["cuda_reserved_bytes"]):
            raise RuntimeError("Registered time/memory/evidence budget exceeded")
        return {**resources, "elapsed_seconds": elapsed, "evidence_bytes": evidence,
                "cuda_peak_window": "after_policy_load_and_initial_parameter_hash"}
    return check_budget


def make_episode_runner(policy, contract, plan, output, budget_started, *,
                        clock=time.monotonic, resource_reader=None):
    import torch
    from rosetta_reality.sim import GymAlohaEnvironment

    if resource_reader is None:
        def resource_reader():
            return {"rss_bytes": int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"),
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    check_budget = make_budget_checker(plan, output, budget_started, clock=clock,
                                       resource_reader=resource_reader)

    def finite_observation(observation):
        for value in [observation["robot_state"], *observation["images"].values()]:
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite simulator observation")

    def episode(seed, trace_path):
        check_budget()
        policy.configure_noise("seeded_standard_normal", seed)
        environment = GymAlohaEnvironment(contract, maximum_episode_steps=500)
        count, success, maximum_reward, limits, collisions, raw_clips = 0, False, 0.0, 0, 0, 0
        outcome = "task_failed"
        try:
            observation = environment.reset(seed=seed)
            finite_observation(observation)
            reset_limits = environment.state_limit_violation_count()
            reset_pairs = [list(pair) for pair in environment.contact_pairs()
                           if environment.is_unexpected_collision_pair(*pair)]
            check_budget()
            with trace_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"event": "reset", "seed": seed,
                                         "joint_limit_violations": reset_limits,
                                         "unexpected_pairs": reset_pairs}, allow_nan=False) + "\n")
                if reset_limits or reset_pairs:
                    limits, collisions, outcome = reset_limits, len(reset_pairs), "physical_safety_stop"
                else:
                    for step in range(500):
                        check_budget()
                        inferred = time.perf_counter()
                        raw, projected = policy.predict(observation, plan["instruction"])
                        inference_seconds = time.perf_counter() - inferred
                        check_budget()
                        for chunk in (raw, projected):
                            contract.validate_tensor(chunk, allow_chunk=True)
                            if not bool(torch.isfinite(chunk).all()):
                                raise ValueError("Nonfinite model action")
                        # Exactly the original registered output projection/slot0.
                        action, raw_mask = contract.clip(raw[0])
                        _checked, boundary_mask = contract.clip(action)
                        if bool(boundary_mask.any()):
                            raise ValueError("Projected action outside contract")
                        raw_clips += int(raw_mask.sum())
                        state_before = observation["robot_state"].tolist()
                        stepped = time.perf_counter()
                        check_budget()
                        observation, reward, done, info = environment.step(action)
                        simulation_seconds = time.perf_counter() - stepped
                        finite_observation(observation)
                        if not math.isfinite(reward) or bool(environment.last_clip_mask.any()):
                            raise ValueError("Nonfinite reward or adapter additional clipping")
                        violation_count = environment.state_limit_violation_count()
                        pairs = [list(pair) for pair in environment.contact_pairs()
                                 if environment.is_unexpected_collision_pair(*pair)]
                        count += 1
                        limits += violation_count
                        collisions += len(pairs)
                        success = success or bool(info.get("is_success", False))
                        maximum_reward = max(maximum_reward, reward)
                        stream.write(json.dumps({"event": "step", "seed": seed, "step": count,
                            "raw_action": raw[0].tolist(), "processed_action": projected[0].tolist(),
                            "executed_action": action.tolist(), "state_before": state_before,
                            "state_after": observation["robot_state"].tolist(), "reward": reward,
                            "success": bool(info.get("is_success", False)), "done": done,
                            "raw_clipped_fields": int(raw_mask.sum()),
                            "joint_limit_violations": violation_count, "unexpected_pairs": pairs,
                            "inference_seconds": inference_seconds, "simulation_seconds": simulation_seconds},
                            allow_nan=False) + "\n")
                        stream.flush()
                        check_budget()
                        if violation_count or pairs:
                            outcome = "physical_safety_stop"
                            break
                        if done:
                            outcome = "task_success" if success else "task_failed"
                            break
                    else:
                        outcome = "task_success" if success else "task_failed"
            trace_sha = sha(trace_path)
            check_budget()
            return {"seed": seed, "policy_noise_seed": seed, "rollout_length": count,
                    "success": success, "outcome": outcome, "maximum_reward": maximum_reward,
                    "joint_limit_violations": limits, "unexpected_collisions": collisions,
                    "raw_clipped_fields": raw_clips, "trace_sha256": trace_sha}
        finally:
            environment.close()
            check_budget()
    episode.check_budget = check_budget
    return episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-artifact", action="store_true")
    parser.add_argument("--canonical-artifact", type=Path)
    parser.add_argument("--base-weights", type=Path)
    args = parser.parse_args(argv)
    if args.prepare_artifact:
        if args.canonical_artifact is None or args.base_weights is None:
            parser.error("Preparation requires canonical artifact and pinned base weights")
        prepare_artifact(args.artifact, args.canonical_artifact, args.base_weights)
        return 0
    if args.plan is None:
        parser.error("Measurement requires a preregistered plan")
    plan = read(args.plan)
    validate_plan(plan)
    config, normalization = verify_artifact(args.artifact)
    if sha(ROOT / "configs/sim/aloha_insertion_smolvla.yaml") != CONTRACT_SHA:
        raise ValueError("Action Contract changed")
    if args.check_only:
        print(json.dumps({"status": "passed", "model_loaded": False, "payload_sha_verified": True}))
        return 0
    if args.output is None:
        parser.error("Measurement requires new output directory")
    if args.output.exists():
        raise FileExistsError("Create-only measurement directory already exists")
    if (os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda"
            or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"))):
        raise RuntimeError("Offline registered AutoDL CUDA runtime required")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import torch
    import smolvla_autodl_vfunfreeze_sim_gate as reference
    from dataclasses import asdict
    from rosetta_reality.sim import load_action_contract
    contract = load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    if read(args.artifact / "action_contract.json") != json.loads(json.dumps(asdict(contract))):
        raise ValueError("Artifact/runtime Action Contracts differ")
    torch.cuda.set_per_process_memory_fraction(0.4)
    # JSON is valid YAML; loader reads the bound action_contract.path unchanged.
    reference._ACTIVE_SIM_PLAN = args.plan
    started = time.monotonic()
    policy = reference._build_online_cuda_class()(args.artifact, config, normalization, contract)
    before = parameter_digest(policy.policy)
    # Separate initialization transients from the registered measured-inference
    # peak window; current model residency remains subject to the same limits.
    torch.cuda.reset_peak_memory_stats()
    runner = make_episode_runner(policy, contract, plan, args.output, started)
    runner.check_budget()
    def finalize():
        runner.check_budget()
        after = parameter_digest(policy.policy)
        integrity = {"before": before, "after": after, "unchanged": before == after,
                     "optimizer_steps": 0}
        save(args.output / "parameter-integrity.json", integrity)
        resources = runner.check_budget()
        if before != after:
            raise ValueError("Policy parameters/buffers changed during measurement")
        return {"parameter_integrity": integrity, "resources": resources}
    result = run_batch(args.output, runner, {"plan_sha256": sha(args.plan),
                       "artifact_manifest_sha256": MANIFEST_SHA, "model_sha256": MODEL_SHA,
                       "runtime": reference._cuda_runtime(), "parameter_before_sha256": before},
                       final_check=finalize)
    print(json.dumps({"status": result["status"], "measured_seed_count": result["measured_seed_count"],
                      "success_count": result["success_count"], "formal_gate_result": None}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
