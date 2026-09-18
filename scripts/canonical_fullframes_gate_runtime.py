"""Sealed canonical adapter for the unchanged native Gate engine."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from scripts import smolvla_autodl_vfunfreeze_sim_gate as reference  # noqa: E402
from scripts import smolvla_sim_gate as engine  # noqa: E402
from scripts.canonical_fullframes_contract import validate_gate_protocol, verify_files  # noqa: E402
from scripts.canonical_fullframes_runtime import (  # noqa: E402
    active,
    load,
    load_context,  # noqa: E402
)
from scripts.iris_runtime import budget, save, sha  # noqa: E402

original_load = engine._load_artifact


def load_artifact(path):
    plan, job, _ = active()
    seal = load(job / "gate-seal.json")
    if Path(path).resolve() != job / "gate.yaml" or sha(path) != seal["plan_sha256"]:
        raise ValueError("Rendered Gate plan seal drift")
    actual = engine._load_yaml(path)
    validate_gate_protocol(actual, plan)
    for relative, digest in seal["evidence"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError("Gate prerequisite evidence changed: " + relative)
    result = original_load(path)
    verify_files(result[1] / "pretrained_model", plan["endpoint"]["files"])
    expected = plan["endpoint"]["model_safetensors_sha256"]
    if result[2]["files"].get("pretrained_model/model.safetensors") != expected:
        raise ValueError("Manifest weight entry differs from registered endpoint")
    return result


def main():
    import torch

    from scripts.diagnose_zen_noise_transfer import parameter_digests
    from scripts.iris_gate import deployment_state_dimension

    plan, job, reg = active()
    gate = sys.argv[1]
    if gate not in ("gate3", "gate4"):
        raise ValueError("Unknown Gate stage")
    models = []
    base = reference._build_online_cuda_class()

    class Online(base):
        def __init__(self, artifact, config, normalization, contract):
            context = load_context(artifact)
            if asdict(context.physical) != asdict(contract):
                raise ValueError("Native and simulation contracts differ")
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
                instruction = plan["executed_inference"]["instruction"]
                raw, projected = self.predict(observation, instruction)
                batch = context.pre(
                    {
                        "observation.images.top": observation["images"]["top"],
                        "observation.state": observation["robot_state"],
                        "task": instruction,
                    }
                )
                noise = torch.zeros(
                    (1, self.policy.config.chunk_size, self.policy.config.max_action_dim),
                    device="cuda",
                )
                self.policy.reset()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    action = self.policy.predict_action_chunk(batch, noise=noise)
                    expected = context.post(action.clone())
                if not torch.equal(projected, expected[0].cpu()) or not torch.equal(
                    raw, context.decoder.last_unclipped_action[0].cpu()
                ):
                    raise ValueError("Gate adapter differs from native complete action chunks")
                save(
                    job / (gate + "-bridge.json"),
                    {
                        "status": "passed",
                        "full_chunks_exact": True,
                        "policy_forwards": 2,
                        "optimizer_steps": 0,
                    },
                )
            finally:
                environment.close()

        def predict(self, observation, instruction):
            budget(reg["deadline"])
            result = super().predict(observation, instruction)
            budget(reg["deadline"])
            return result

    old_path = engine._repository_path

    def evidence(raw):
        for entry in (reference.PRIOR_FAILURE, reference.PRIOR_TASK_FAILURE):
            if raw == entry["report"]:
                import os

                local = ROOT / raw
                return local if local.is_file() else Path(os.environ["ROSETTA_AUTODL_ROOT"]) / raw
        return old_path(raw)

    engine._repository_path = evidence
    engine._load_artifact = load_artifact
    engine._OnlineSmolVLA = Online
    engine._runtime = reference._cuda_runtime
    code = engine.main()
    if not models or any(parameter_digests(m.policy) != m.initial_parameters for m in models):
        raise ValueError("Gate parameters changed or no real model executed")
    save(job / (gate + "-parameter-check.json"), {"unchanged": True, "optimizer_steps": 0})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
