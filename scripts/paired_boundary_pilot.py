"""Bounded one-step restore/observer diagnostic; historical runners stay immutable.

Runtime imports are lazy. Execution requires a new sealed plan, the existing
AutoDL container, an idle registered GPU and the round supervisor's guard.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import enum
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
PAIR = "bank-003b-scene-2445"
INSTRUCTION = "Insert the peg into the socket."
CASES = (("teacher", None, False), ("a1", "A", False),
         ("a2", "A", False), ("a3", "A", True),
         ("b", "B", True), ("c", "C", True))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def resolve(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Relative POSIX path required")
    raw = Path(name)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("Input escapes registered root")
    target = (Path(root) / raw).resolve(strict=True)
    if not target.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError("Resolved input escapes root")
    return target


def validate(plan):
    if (plan.get("schema") != "rosetta.paired_boundary_pilot.v1"
            or plan.get("execution_authorized") is not True
            or plan.get("pair_id") != PAIR or plan.get("seed") != 2445
            or plan.get("cases") != [row[0] for row in CASES]
            or plan.get("maximum_forward_calls") != 5
            or plan.get("maximum_physics_steps") != 6
            or plan.get("optimizer_updates") != 0
            or plan.get("selected_checkpoint_step") != 5000
            or plan.get("prediction_horizon") != 50
            or plan.get("execution_horizon") != 1
            or plan.get("physical_limit_tolerance") != 1e-5
            or plan.get("shutdown_authorized") is not False
            or plan.get("hidden_test_loaded") is not False):
        raise ValueError("Pilot scientific/safety scope changed")
    limits = plan["resources"]
    ceilings = {"maximum_seconds": 900, "rss_bytes": 8 * 1024**3,
                "evidence_bytes": 256 * 1024**2, "cuda_allocated_bytes": 8 * 1024**3,
                "cuda_reserved_bytes": 10 * 1024**3}
    for key, ceiling in ceilings.items():
        if type(limits.get(key)) is not int or not 0 < limits[key] <= ceiling:
            raise ValueError("Missing or excessive runtime budget")
    if not plan.get("source_sha256"):
        raise ValueError("Sealed source inventory required")
    for name, expected in plan["source_sha256"].items():
        if sha(resolve(ROOT, name)) != expected:
            raise ValueError("Pilot source identity drift: " + name)
    required = {"scripts/paired_boundary_pilot.py", "scripts/restore_paired_branch_v2.py",
                "scripts/check_q0_state_restore.py", "scripts/evaluate_paired_policy_v2.py",
                "scripts/smolvla_autodl_vfunfreeze_sim_gate.py",
                "src/rosetta_reality/eval/gate_diagnostic_capture.py",
                "src/rosetta_reality/eval/gate_diagnostic_io.py",
                "src/rosetta_reality/eval/reproducibility_capture.py",
                "src/rosetta_reality/sim/gym_aloha.py",
                "src/rosetta_reality/vla/processor.py"}
    if not required <= plan["source_sha256"].keys():
        raise ValueError("Pilot observation and execution identity incomplete")


def compare_arrays(left, right):
    """No broadcasting, tolerance relaxation, sample dropping or tensor imports."""
    if len(left) != len(right):
        return {"exact": False, "reason": "length_mismatch"}
    for x, y in zip(left, right):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Nonfinite comparison")
    differences = [abs(x - y) for x, y in zip(left, right)]
    return {"exact": all(x == y for x, y in zip(left, right)),
            "maximum_absolute_difference": max(differences, default=0.0),
            "different_values": sum(x != y for x, y in zip(left, right)),
            "values": len(left)}


def state_tensor(value):
    if isinstance(value, dict):
        if "observation.state" in value:
            return value["observation.state"]
        for child in value.values():
            found = state_tensor(child)
            if found is not None:
                return found
    return None


@contextmanager
def observe_processor_states(pipeline, clone):
    """Observe actual calls; never invoke a processor a second time for logging."""
    traces, installed = [], []
    targets = {id(step): i for i, step in enumerate(pipeline.steps)}
    if len(targets) != len(pipeline.steps):
        raise ValueError("Aliased processor instances")
    classes = list(dict.fromkeys(type(step) for step in pipeline.steps))
    originals = [(cls, cls.__call__, "__call__" in vars(cls)) for cls in classes]
    try:
        for cls, original, had_own in originals:
            def wrapper(step, transition, *args, _original=original, **kwargs):
                if id(step) not in targets:
                    return _original(step, transition, *args, **kwargs)
                before = clone(state_tensor(transition))
                returned = _original(step, transition, *args, **kwargs)
                traces.append({"index": targets[id(step)], "before": before,
                               "after": clone(state_tensor(returned))})
                return returned
            cls.__call__ = wrapper
            installed.append((cls, original, wrapper, had_own))
        yield traces
    finally:
        conflicts = []
        for cls, original, wrapper, had_own in reversed(installed):
            if cls.__call__ is not wrapper:
                conflicts.append(cls.__name__)
                continue
            if had_own:
                cls.__call__ = original
            else:
                delattr(cls, "__call__")
        if conflicts:
            raise RuntimeError("Processor hook ownership changed: " + ",".join(conflicts))


def capture_physics(env, reader):
    import mujoco
    from rosetta_reality.sim.scripted_teacher_observer import build_observation
    model, data = reader.model, reader.data
    joints = []
    for index in range(model.njnt):
        if model.jnt_limited[index]:
            address = int(model.jnt_qposadr[index])
            lo, hi = map(float, model.jnt_range[index])
            value = float(data.qpos[address])
            joints.append({"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
                           "qpos_address": address, "lower": lo, "upper": hi,
                           "value": value, "tolerance": 1e-5,
                           "margin": min(value - lo, hi - value),
                           "violated": value < lo - 1e-5 or value > hi + 1e-5})
    geometry = dataclasses.asdict(build_observation(env, env.contract))
    return {"integration": reader()["integration"], "integration_spec": reader.spec,
            "compiled_model_sha256": reader.model_sha,
            "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy(),
            "qacc": data.qacc.copy(), "qfrc_actuator": data.qfrc_actuator.copy(),
            "qfrc_constraint": data.qfrc_constraint.copy(), "joint_limits": joints,
            "contacts": [{"pair": list(pair), "unexpected": env.is_unexpected_collision_pair(*pair)}
                         for pair in env.contact_pairs()], "geometry": geometry,
            "diagnostic": env.diagnostic_snapshot()}


def safe(snapshot):
    return not any(j["violated"] for j in snapshot["joint_limits"]) and not any(
        c["unexpected"] for c in snapshot["contacts"])


def plain_config(value):
    """Retain effective features/stats; host paths are represented only by SHA."""
    from check_q0_state_restore import safe_string
    from rosetta_reality.eval.gate_diagnostic_capture import clone_tree
    if dataclasses.is_dataclass(value):
        return plain_config(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return plain_config(value.value)
    if isinstance(value, dict):
        return {str(k.value if isinstance(k, enum.Enum) else k): plain_config(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(v) for v in value]
    if isinstance(value, str):
        return safe_string(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if hasattr(value, "detach") or hasattr(value, "dtype") and hasattr(value, "shape"):
        return clone_tree(value)
    raise TypeError("Unserializable effective processor configuration: " + type(value).__name__)


def processors(online):
    rows = {}
    for name in ("preprocessor", "postprocessor"):
        rows[name] = [{"class": type(step).__module__ + "." + type(step).__qualname__,
                       "registry": getattr(type(step), "_registry_name", None),
                       "config": plain_config(step.get_config()),
                       "stats": plain_config(getattr(step, "stats", None)),
                       "features": plain_config(getattr(step, "features", None))}
                      for step in getattr(online, name).steps]
    return rows


def execute(plan, output, guard_job):
    validate(plan)
    admission_name = plan.get("runtime_admission_receipt")
    if not admission_name:
        raise ValueError("Synthetic/container admission is still unverified")
    admission_path = resolve(ROOT, admission_name)
    if sha(admission_path) != plan.get("runtime_admission_sha256"):
        raise ValueError("Runtime admission receipt changed")
    admission = read(admission_path)
    if (admission.get("passed") is not True
            or admission.get("source_sha256") != plan["source_sha256"]
            or admission.get("no_model_or_dataset_loaded") is not True):
        raise ValueError("Exact-source synthetic/container admission required")
    if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or any(
            os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Offline registered AutoDL CUDA environment required")
    import numpy as np
    import torch
    from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract
    from rosetta_reality.eval.reproducibility_capture import PhysicsReader, runtime_fingerprint
    from rosetta_reality.eval.gate_diagnostic_capture import capture_prediction, clone_tree
    from rosetta_reality.eval.gate_diagnostic_io import write_tree, write_json, seal
    from check_q0_state_restore import PythonState, tree_hash
    from rosetta_reality.sim.scripted_teacher_observer import _robot_state, _control_task
    from restore_paired_branch_v2 import restore_branch
    from diagnose_smolvla_base_five_scene import parameter_digest
    from evaluate_paired_policy_v2 import verify as verify_artifact
    import smolvla_autodl_vfunfreeze_sim_gate as reference
    import run_round_experiment_job as supervisor
    durable = Path(os.environ["ROSETTA_AUTODL_ROOT"]).resolve(strict=True)
    bank = resolve(durable, plan["bank_root"])
    if sha(bank / "manifest.json") != plan["bank_manifest_sha256"]:
        raise ValueError("Bank identity changed")
    manifest = read(bank / "manifest.json")
    for row in manifest["files"]:
        if sha(resolve(bank, row["path"])) != row["sha256"]:
            raise ValueError("Bank file identity changed")
    source = [json.loads(line) for line in (bank / PAIR / "R/trace.jsonl").read_text().splitlines()]
    correction = [row for row in source if row["step"] == 178]
    if len(correction) != 1 or correction[0]["action_source"] != "state_conditioned_teacher":
        raise ValueError("Registered saved teacher first correction missing")
    contract = load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "runtime.json", runtime_fingerprint())
    started = time.monotonic()
    counts = {"forward": 0, "physics": 0}
    results, online, current_arm = {}, None, None
    result = {"status": "incomplete", "diagnostic_only": True, "optimizer_updates": 0,
              "formal_gate_result": None, "m2_complete": False}

    def guard():
        limits = plan["resources"]
        supervisor.guard_alive(guard_job, supervisor.registration(guard_job))
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if (time.monotonic() - started >= limits["maximum_seconds"] or rss > limits["rss_bytes"]
                or size > limits["evidence_bytes"]
                or torch.cuda.max_memory_allocated() > limits["cuda_allocated_bytes"]
                or torch.cuda.max_memory_reserved() > limits["cuda_reserved_bytes"]):
            raise RuntimeError("Pilot resource budget exceeded")
        if counts["forward"] > 5 or counts["physics"] > 6:
            raise RuntimeError("Pilot call budget exceeded")

    def persist(name, value):
        write_tree(output / name, value, budget_root=output,
                   maximum_bytes=plan["resources"]["evidence_bytes"], reserve_bytes=4*1024**2)
        guard()

    def compare_current_observation(env, reader, saved):
        """Read live state/pixels once, then prove inspection did not change restore."""
        integration = reader()["integration"].clone()
        python = tree_hash(PythonState([env], reader.physics).receipt())
        live = {"robot_state": _robot_state(_control_task(env), reader.physics), "images": {}}
        for camera, expected in saved["images"].items():
            if expected.ndim != 3 or expected.shape[0] != 3:
                raise ValueError("Saved camera shape unsupported")
            pixels = reader.physics.render(height=int(expected.shape[1]), width=int(expected.shape[2]), camera_id=camera)
            live["images"][camera] = torch.as_tensor(pixels).permute(2,0,1).to(torch.float32).div(255)
        unchanged = (torch.equal(integration, reader()["integration"])
                     and python == tree_hash(PythonState([env], reader.physics).receipt()))
        equal = (torch.equal(saved["robot_state"], live["robot_state"])
                 and all(torch.equal(saved["images"][k],live["images"][k]) for k in saved["images"]))
        return live, unchanged, equal

    try:
        torch.cuda.set_per_process_memory_fraction(.4)
        torch.cuda.reset_peak_memory_stats()
        for case, arm, observe in CASES:
            guard()
            if arm is not None and current_arm != arm:
                if online is not None:
                    del online
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                artifact = resolve(durable, plan["artifacts"][arm])
                evaluation = read(resolve(durable, plan["evaluation_plans"][arm]))
                config, norm = verify_artifact(evaluation, artifact)
                if config["selected_checkpoint_model_sha256"] != plan["model_sha256"][arm]:
                    raise ValueError("Fixed endpoint model changed")
                reference._ACTIVE_SIM_PLAN = resolve(durable, plan["evaluation_plans"][arm])
                online = reference._build_online_cuda_class()(artifact, config, norm, contract)
                current_arm = arm
                persist("processors-" + arm, processors(online))
                guard()
            env = GymAlohaEnvironment(contract, maximum_episode_steps=500)
            try:
                env.reset(seed=2445)
                reader = PhysicsReader(env)
                restored = restore_branch(env, reader, bank / PAIR)
                observation = restored["observation"]
                before = capture_physics(env, reader)
                python_before = PythonState([env], reader.physics).receipt()
                persist(case + "/initial", {"physics": before, "observation": observation,
                                             "python": python_before})
                live, inspection_unchanged, observations_equal = compare_current_observation(env, reader, observation)
                persist(case + "/observation-check", {"live": live, "saved": observation,
                        "inspection_state_unchanged": inspection_unchanged, "exact": observations_equal})
                if not inspection_unchanged or not observations_equal:
                    raise ValueError("Restored observation/live state or inspection parity failed")
                if not safe(before):
                    raise ValueError("Unsafe initial restore")
                if case == "teacher":
                    raw = torch.tensor(correction[0]["action"], dtype=torch.float32).reshape(1, 14)
                    projected, clip = contract.clip(raw[0])
                    if bool(clip.any()):
                        raise ValueError("Saved teacher command requires clipping")
                    projected = projected.reshape(1,14)
                else:
                    online.configure_noise("seeded_standard_normal", 2445)
                    parameters = parameter_digest(online.policy)
                    noise_before = online._noise_generator.get_state().clone()
                    guard()
                    counts["forward"] += 1
                    if observe:
                        with observe_processor_states(online.preprocessor, clone_tree) as transitions:
                            (raw, projected), captured = capture_prediction(online, observation, INSTRUCTION)
                        captured["processor_state_transitions"] = transitions
                        persist(case + "/prediction", captured)
                    else:
                        raw, projected = online.predict(observation, INSTRUCTION)
                    if parameter_digest(online.policy) != parameters:
                        raise ValueError("Policy changed during no-optimizer diagnostic")
                    persist(case + "/actions", {"raw": raw, "projected": projected,
                        "noise_rng_before": noise_before, "noise_rng_after": online._noise_generator.get_state()})
                guard()
                for chunk in (raw, projected):
                    contract.validate_tensor(chunk, allow_chunk=True)
                    if not bool(torch.isfinite(chunk).all()):
                        raise ValueError("Nonfinite pilot output")
                command, clips = contract.clip(raw[0])
                if not torch.equal(command, projected[0]):
                    raise ValueError("Raw projection and returned projected slot differ")
                counts["physics"] += 1
                returned, reward, done, info = env.step(command)
                after = capture_physics(env, reader)
                python_after = PythonState([env], reader.physics).receipt()
                if not math.isfinite(reward) or bool(env.last_clip_mask.any()):
                    raise ValueError("Nonfinite reward or adapter clipping")
                persist(case + "/after", {"physics": after, "observation": returned,
                                          "python": python_after, "reward": float(reward),
                                          "done": bool(done), "success": bool(info.get("is_success",False)),
                                          "command": command, "raw_clips": int(clips.sum())})
                results[case] = {"raw": raw.clone(), "projected": projected.clone(),
                                 "integration_before": before["integration"], "integration_after": after["integration"],
                                 "python_before": tree_hash(python_before), "python_after": tree_hash(python_after),
                                 "state_after": returned["robot_state"].clone(),
                                 "images_after": clone_tree(returned["images"]), "safe_after": safe(after)}
                if arm is not None:
                    results[case]["noise_rng_before"] = noise_before
                    results[case]["noise_rng_after"] = online._noise_generator.get_state().clone()
                if case == "teacher" and not safe(after):
                    raise ValueError("Teacher one-step command positive control failed")
                if case in ("a2", "a3"):
                    left, right = results["a1"], results[case]
                    for key in ("raw","projected","integration_before","integration_after","state_after",
                                "noise_rng_before","noise_rng_after"):
                        if not torch.equal(left[key], right[key]):
                            raise ValueError("A/A or observer parity failed: " + key)
                    if (left["python_before"] != right["python_before"] or left["python_after"] != right["python_after"]
                            or left["images_after"].keys() != right["images_after"].keys()
                            or any(not torch.equal(left["images_after"][k], right["images_after"][k]) for k in left["images_after"])):
                        raise ValueError("A/A or observer Python/image parity failed")
            finally:
                env.close()
        result.update(status="completed", aa_and_observer_parity=True,
                      post_step_safe={k:v["safe_after"] for k,v in results.items()},
                      causal_attribution_proven=False)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(forward_calls=counts["forward"], physics_steps=counts["physics"],
                      elapsed_seconds=time.monotonic()-started, shutdown_requested=False)
        seal(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--guard-job', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    validate(plan)
    if args.check_only:
        print(json.dumps({"source_contract_verified": True, "model_or_physics_executed": False}))
    else:
        if args.output is None or args.guard_job is None:
            parser.error('Execution requires a create-only output and existing supervisor guard')
        execute(plan, args.output, args.guard_job.resolve(strict=True))
