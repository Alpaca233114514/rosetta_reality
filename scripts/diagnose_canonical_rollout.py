"""Independent trace entrypoint; collect requires a new authorized registration.

No default collection plan is supplied. `verify` is offline, stdlib-only, and
does not construct a policy or simulator. Historical Gate entrypoints are intact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from rosetta_reality.eval.rollout_trace import file_sha, run_traced_rollout  # noqa: E402
from rosetta_reality.eval.rollout_trace_verify import verify_trace  # noqa: E402

CHECKPOINT_SHA = "d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef"
AUTHORITY_PATH = (
    "reports/training/m2-smolvla-canonical-fullframes-posttrain-gate-plan-004-2026-09-15.json"
)
AUTHORITY_SHA = "35511af8c7d0d9d93c827fe72ddec0205fa8932f7660735ddd38e8b7524be75e"
ARTIFACT_CONFIG_SHA = "978ec7b1b9b648749b96cf0967ae4b37e0133d65e491613e6803884d50c0984a"
REQUIRED_SOURCES = {
    "scripts/diagnose_canonical_rollout.py", "scripts/smolvla_sim_gate.py",
    "scripts/smolvla_autodl_vfunfreeze_sim_gate.py", "scripts/iris_gate.py",
    "scripts/iris_runtime.py", "scripts/evaluate_visual_native_small.py",
    "src/rosetta_reality/eval/rollout_trace.py",
    "src/rosetta_reality/eval/rollout_trace_verify.py",
    "src/rosetta_reality/vla/processor.py", "src/rosetta_reality/sim/gym_aloha.py",
    "src/rosetta_reality/sim/action_contract.py",
}


def relative_path(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Registration paths must be repository-relative")
    return ROOT / path


def validate_endpoint(plan, authority):
    if (plan["checkpoint"]["files"] != authority["endpoint"]["files"]
            or plan["training_plan"]["sha256"] != authority["source_training_plan_sha256"]
            or plan["action_contract"]["sha256"]
            != authority["gate_protocol_authority"]["action_contract_sha256"]
            or plan["artifact_config"]["sha256"] != ARTIFACT_CONFIG_SHA):
        raise ValueError("Checkpoint, processors or configuration differ from the fixed endpoint")


def validate_registration(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if (plan.get("schema_version") != 1 or plan.get("execution_authorized") is not True
            or plan.get("purpose") != "diagnostic_only_unchanged_step5000"
            or not isinstance(plan.get("run_id"), str) or not plan["run_id"]
            or plan.get("optimizer_steps") != 0):
        raise ValueError("New authorized diagnostic-only registration required")
    if not time.time() < plan["deadline"] <= time.time() + 3600:
        raise ValueError("Live bounded collection deadline required")
    if any(os.environ.get(key) != value for key, value in {
        "ROSETTA_TORCH_DEVICE": "cuda", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    }.items()):
        raise ValueError("Registered offline CUDA runtime required")
    if not REQUIRED_SOURCES <= plan["sources"].keys():
        raise ValueError("Incomplete diagnostic source inventory")
    for name, expected in plan["sources"].items():
        if file_sha(relative_path(name)) != expected:
            raise ValueError("Registered source drift: " + name)
    for name in ("training_plan", "action_contract", "artifact_config"):
        spec = plan[name]
        if file_sha(relative_path(spec["path"])) != spec["sha256"]:
            raise ValueError("Registered configuration drift: " + name)
    files = plan["checkpoint"]["files"]
    authority_path = relative_path(AUTHORITY_PATH)
    if file_sha(authority_path) != AUTHORITY_SHA:
        raise ValueError("Historical endpoint authority changed")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    validate_endpoint(plan, authority)
    if files.get("model.safetensors") != CHECKPOINT_SHA:
        raise ValueError("Only the unchanged canonical step-5000 checkpoint is allowed")
    from scripts.canonical_fullframes_contract import verify_files

    checkpoint = relative_path(plan["checkpoint"]["path"])
    verify_files(checkpoint, files)
    required = {"config.json", "train_config.json", "policy_preprocessor.json",
                "policy_postprocessor.json", "model.safetensors"}
    if not required <= files.keys() or any(not any(name.endswith(suffix) for name in files)
                                         for suffix in ("_normalizer_processor.safetensors",
                                                        "_unnormalizer_processor.safetensors")):
        raise ValueError("Saved policy and processor inventory required")
    options = plan["rollout"]
    if (set(options) != {"seed", "maximum_steps", "project_policy_output", "noise_mode",
                         "policy_noise_seed"}
            or options["seed"] not in range(1000, 1005) or options["maximum_steps"] != 500
            or options["policy_noise_seed"] != options["seed"]
            or options["noise_mode"] != "seeded_standard_normal"
            or options["project_policy_output"] is not True):
        raise ValueError("Original five-seed first-action rollout protocol required")
    output = relative_path(plan["output"])
    if output.exists():
        raise FileExistsError("Trace output already exists")
    # A real collection must use the existing guarded worker lifecycle.
    from scripts.run_iris_furnace import alive

    watchdog = plan["watchdog"]
    if not alive(watchdog["pid"], watchdog["ticks"]):
        raise ValueError("Live registered shutdown watchdog required")
    return plan


def collect(path):
    plan = validate_registration(path)  # Before policy/data imports and construction.
    import torch

    from rosetta_reality.sim import load_action_contract
    from scripts import smolvla_autodl_vfunfreeze_sim_gate as adapter
    from scripts import smolvla_sim_gate as engine
    from scripts.iris_gate import deployment_state_dimension, load_gate_context
    from scripts.iris_runtime import budget

    context = load_gate_context(
        relative_path(plan["training_plan"]["path"]),
        checkpoint=relative_path(plan["checkpoint"]["path"]),
        checkpoint_files=plan["checkpoint"]["files"], split="train", deadline=plan["deadline"],
    )
    from dataclasses import asdict

    contract = load_action_contract(relative_path(plan["action_contract"]["path"]))
    if asdict(contract) != asdict(context.physical):
        raise ValueError("Saved processor and rollout action contracts differ")
    config = json.loads(relative_path(plan["artifact_config"]["path"]).read_text())
    base = adapter._build_online_cuda_class()

    class Online(base):
        def predict(self, observation, instruction):
            budget(plan["deadline"])
            result = super().predict(observation, instruction)
            budget(plan["deadline"])
            return result

    policy = object.__new__(Online)
    policy.policy, policy.preprocessor, policy.postprocessor = (
        context.policy, context.pre, context.post,
    )
    policy.device, policy.mixed_precision = torch.device("cuda"), "bf16"
    policy.state_dimension = deployment_state_dimension(
        context.policy.config, contract, config["dataset_features"])
    policy.action_dimension, policy.chunk_length = contract.dimension, contract.chunk_length
    policy._noise_generator = torch.Generator(device="cpu")
    identity = {**plan, "kind": "registered_model", "registration_sha256": file_sha(path),
                "endpoint_authority_sha256": AUTHORITY_SHA}
    run_traced_rollout(
        engine, policy, contract, "Insert the peg into the socket.",
        output=relative_path(plan["output"]), identity=identity,
        episode_index=plan["rollout"]["seed"] - 1000, **plan["rollout"],
    )
    return verify_trace(relative_path(plan["output"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("verify").add_argument("--directory", type=Path, required=True)
    modes.add_parser("collect").add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = verify_trace(args.directory) if args.mode == "verify" else collect(args.plan)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
