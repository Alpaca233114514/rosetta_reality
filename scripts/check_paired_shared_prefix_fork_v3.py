"""Bounded real shared-prefix fork engineering; no model/optimizer or admission.

Native solver state is reconstructed by exact decision replay, never deepcopied.
New post-fork renders are retained separately and differences are disclosed.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compare_branches(first, second):
    keys = ("action", "phase", "before", "after", "integration", "reward", "done", "info", "safe")
    differences = []
    for i in range(min(len(first), len(second))):
        for key in keys:
            if first[i][key] != second[i][key]:
                differences.append({"step": i, "field": key})
    if len(first) != len(second):
        differences.append({"field": "branch_length", "A": len(first), "B": len(second)})
    return {"physical_action_event_exact": not differences, "differences": differences}


def run(plan, output):
    import mujoco
    import numpy as np
    from check_q0_state_restore import PythonState, UnsupportedRestore
    from diagnose_teacher_cross_pose import finite_json, validate_state_finite
    from rosetta_reality.eval.reproducibility_capture import PhysicsReader
    from rosetta_reality.sim.action_contract import load_action_contract
    from rosetta_reality.sim.coupled_translation_teacher import build_teacher
    from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment
    from rosetta_reality.sim.scripted_teacher_observer import build_observation
    started = time.monotonic()
    physical_steps = decision_calls = 0
    history = []
    prefix = []
    branches = []
    contract = load_action_contract(ROOT/plan["contract"])
    env = GymAlohaEnvironment(contract, maximum_episode_steps=plan["max_steps"])
    teacher = build_teacher()
    teacher.reset()

    def write(path, value):
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")

    def guard():
        if time.monotonic()-started > plan["work_cap_seconds"]:
            raise RuntimeError("work_cap")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > plan["evidence_byte_cap"]:
            raise RuntimeError("evidence_cap")

    def signature(owner, observation):
        nonlocal decision_calls
        guard()
        decision = owner.decide(observation)
        decision_calls += 1
        if decision.refusal is not None:
            raise RuntimeError("teacher_refusal")
        return decision.action.clone(), str(owner._phase.value)

    def step(action, before, phase, directory, index):
        nonlocal physical_steps
        guard()
        validate_state_finite(action)
        contract.validate_tensor(action, allow_chunk=False)
        if bool(contract.clip(action)[1].any()):
            raise RuntimeError("action_requires_clipping")
        current, reward, done, info = env.step(action)
        physical_steps += 1
        validate_state_finite(current)
        if not np.isfinite(reward):
            raise RuntimeError("nonfinite_reward")
        safe = not (env.state_limit_violation_count() or env.unexpected_collision_count() or bool(env.last_clip_mask.any()))
        pixels = {str(k): np.asarray(v).copy() for k,v in current["raw"].get("pixels", current["raw"].get("images", {})).items()}
        if not pixels or any(v.dtype != np.uint8 or v.ndim != 3 for v in pixels.values()):
            raise RuntimeError("native_pixel_schema")
        np.savez_compressed(directory/f"frame-{index:04d}.npz", **pixels)
        row = {"action": finite_json(action), "phase": phase, "before": finite_json(before),
               "after": finite_json(build_observation(env, contract)), "integration": finite_json(reader()["integration"]),
               "reward": float(reward), "done": bool(done), "info": finite_json(info), "safe": bool(safe),
               "native_images": {k:{"sha256":hashlib.sha256(v.tobytes()).hexdigest(), "shape":list(v.shape), "dtype":v.dtype.str}
                                 for k,v in pixels.items()}}
        write(directory/f"row-{index:04d}.json", row)
        if not safe:
            raise RuntimeError("physical_safety_stop")
        return current, row

    try:
        current = env.reset(seed=plan["seed"])
        reader = PhysicsReader(env)
        prefix_dir = output/"shared-prefix"
        prefix_dir.mkdir()
        for logical_step in range(plan["max_steps"]):
            observation = build_observation(env, contract)
            # Tensor-only immutable geometry copy; no C solver object is copied.
            captured = copy.deepcopy(observation)
            action, phase = signature(teacher, observation)
            history.append((captured, finite_json(action), phase))
            if phase == plan["trigger"]:
                break
            current, row = step(action, observation, phase, prefix_dir, logical_step)
            prefix.append(row)
            if row["done"]:
                raise RuntimeError("event_not_reached")
        else:
            raise RuntimeError("event_not_reached")
        if logical_step + plan["branch_steps"] > plan["max_steps"]:
            raise RuntimeError("insufficient_registered_branch_budget")
        if any(getattr(mujoco,name)() is not None for name in dir(mujoco) if name.startswith("get_mjcb_")) or int(reader.model.nplugin):
            raise UnsupportedRestore("native_external_state")
        saved_integration = reader()["integration"].numpy().copy()
        saved_python = PythonState([env], reader.physics)
        cached_observation = copy.deepcopy(current)
        cached_native = {str(k):np.asarray(v).copy() for k,v in current["raw"].get("pixels", current["raw"].get("images", {})).items()}
        if not cached_native or any(v.dtype != np.uint8 or v.ndim != 3 for v in cached_native.values()):
            raise RuntimeError("fork_native_pixel_schema")
        np.savez_compressed(output/"fork-native-observation.npz", **cached_native)
        np.savez_compressed(output/"fork-integration.npz", state=saved_integration)
        np.savez_compressed(output/"fork-observation.npz", robot_state=current["robot_state"].detach().cpu().numpy(),
                            **{"image::"+k:v.detach().cpu().numpy() for k,v in current["images"].items()})
        with (output/"fork-python.json.gz").open("xb") as handle:
            raw = json.dumps({"python_state":saved_python.receipt(), "model_sha256":reader.model_sha,
                              "integration_spec":reader.spec}, sort_keys=True, allow_nan=False).encode()
            handle.write(gzip.compress(raw, compresslevel=6, mtime=0))
        write(output/"teacher-reconstruction-history.json", [{"geometry":finite_json(g), "action":a, "phase":p} for g,a,p in history])
        reconstructed = []
        cached_hash = digest({"robot_state":finite_json(cached_observation["robot_state"]),
                              "images":{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in cached_observation["images"].items()},
                              "native_images":{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in cached_native.items()}})
        for label in ("A", "B"):
            saved_python.restore()
            mujoco.mj_setState(reader.model, reader.data, saved_integration, reader.spec)
            mujoco.mj_forward(reader.model, reader.data)
            mujoco.mj_setState(reader.model, reader.data, saved_integration, reader.spec)
            if not np.array_equal(reader()["integration"].numpy(), saved_integration):
                raise UnsupportedRestore("integration_restore_mismatch")
            if saved_python.receipt() != PythonState([env], reader.physics).receipt():
                raise UnsupportedRestore("python_restore_mismatch")
            fresh = build_teacher()
            fresh.reset()
            for replay_step,(g, expected_action, expected_phase) in enumerate(history):
                replay_action, replay_phase = signature(fresh, g)
                if finite_json(replay_action) != expected_action or replay_phase != expected_phase:
                    raise UnsupportedRestore("teacher_decision_replay_mismatch_at_"+str(replay_step))
            reconstructed.append({"branch":label,"calls":len(history),"all_decisions_exact":True,"cached_observation_sha256":cached_hash})
            branch_dir = output/label
            branch_dir.mkdir()
            rows = []
            branch_action, branch_phase = replay_action, replay_phase
            branch_geometry = build_observation(env, contract)
            if finite_json(branch_geometry) != finite_json(history[-1][0]):
                raise UnsupportedRestore("fork_geometry_restore_mismatch")
            for offset in range(plan["branch_steps"]):
                if offset:
                    branch_geometry = build_observation(env, contract)
                    branch_action, branch_phase = signature(fresh, branch_geometry)
                _, row = step(branch_action, branch_geometry, branch_phase, branch_dir, offset)
                rows.append(row)
                if row["done"]:
                    break
            branches.append(rows)
        comparison = compare_branches(*branches)
        render_differences = []
        for i,(a,b) in enumerate(zip(*branches)):
            for camera in a["native_images"]:
                with np.load(output/"A"/f"frame-{i:04d}.npz", allow_pickle=False) as xa, np.load(output/"B"/f"frame-{i:04d}.npz", allow_pickle=False) as xb:
                    av,bv = xa[camera],xb[camera]
                    delta = np.abs(av.astype(np.int16)-bv.astype(np.int16))
                    render_differences.append({"step":i,"camera":camera,"different_channels":int(np.count_nonzero(delta)),
                                               "max_abs_uint8":int(delta.max()),"A":a["native_images"][camera],"B":b["native_images"][camera]})
        return {"schema":"rosetta.shared_prefix_fork_engineering.v1", "passed":comparison["physical_action_event_exact"],
                "comparison":comparison,"reconstruction":reconstructed,"render_differences":render_differences,
                "post_fork_render_bitwise_exact":all(r["different_channels"]==0 for r in render_differences),
                "shared_prefix_physical_steps":len(prefix),"fork_logical_step":logical_step,
                "physical_steps":physical_steps,"logical_branch_frames":[len(r) for r in branches],
                "teacher_decision_calls":decision_calls,"engineering_replay_decision_calls":2*len(history),
                "model_sha256":reader.model_sha,"no_deliberate_deviation_used":True,"N":0,
                "teacher_qualified":False,"legacy_object_scene_nonoverlap":"not_established"}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if (plan["seed"] != 2411 or plan["trigger"] != "grasp" or plan["branch_steps"] != 16
            or plan["max_steps"] != 500 or plan["work_cap_seconds"] != 600 or plan["evidence_byte_cap"] != 67108864):
        raise ValueError("engineering registration differs")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"registration.json").write_text(json.dumps(plan, indent=2, sort_keys=True)+"\n")
    try:
        result = run(plan, args.output)
    except Exception as error:
        result = {"schema":"rosetta.shared_prefix_fork_engineering.v1", "passed":False,"error_type":type(error).__name__,"reason":str(error),"N":0}
    result["plan_sha256"] = digest(plan)
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/"result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    files = [{"path":p.relative_to(args.output).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(args.output.rglob("*")) if p.is_file()]
    (args.output/"manifest.json").write_text(json.dumps({"files":files,"plan_sha256":digest(plan)},indent=2,sort_keys=True)+"\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
