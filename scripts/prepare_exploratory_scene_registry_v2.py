"""Capture fresh registered reset states, never legacy observations or model calls."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value) + b"\n")


def registration(plan_paths):
    groups, parents, plans = {}, {}, {}
    for group, path in zip(("Q1", "Q2", "bank"), plan_paths, strict=True):
        plan = json.loads(path.read_text(encoding="utf-8-sig"))
        if plan["phase"] != group or plan.get("sampling_rule") != "symmetric_common_legal_chunk_grid_v1":
            raise ValueError("Approved 003-B collection plans required")
        pairs = plan["pairs"]
        groups[group] = [p["seed"] for p in pairs]
        parents.update({p["seed"]: p["pair_id"] for p in pairs})
        family = {key: plan[key] for key in ("phase", "pairs", "contract", "sampling_rule", "minimum_common_chunk_starts")}
        # Capture-source SHA is historical provenance, not a cyclic requirement
        # on a later collection plan that will bind this registry's manifest.
        plans[group] = {"id": plan["id"], "capture_source_plan_sha256": sha(path),
                         "family_definition": family, "family_definition_sha256": hashlib.sha256(encoded(family)).hexdigest()}
    groups.update(teacher_tuning=[10, *range(1900, 1905)],
                  policy_gate=list(range(1000, 1005)), engineering=[2400])
    seeds = [seed for values in groups.values() for seed in values]
    if any(type(s) is not int or s < 0 for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Registered source families must have unique nonnegative seeds")
    if any(not 1 <= len(groups[g]) <= n for g, n in (("Q1", 4), ("Q2", 20), ("bank", 10))):
        raise ValueError("Registered collection-family budget exceeded")
    return groups, parents, plans


def prepare(args):
    import numpy as np
    import mujoco
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    from check_q0_state_restore import PythonState
    from diagnose_teacher_cross_pose import finite_json
    from rosetta_reality.eval.reproducibility_capture import PhysicsReader
    from rosetta_reality.sim import load_action_contract, GymAlohaEnvironment
    from rosetta_reality.sim.scripted_teacher_observer import build_observation

    groups, parents, plans = registration([args.q1_plan, args.q2_plan, args.bank_plan])
    if not 0 < args.max_seconds <= 600 or not 0 < args.max_bytes <= 256 * 1024**2:
        raise ValueError("Registry requires bounded 600 second / 256 MiB limits")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ["scripts/prepare_exploratory_scene_registry_v2.py",
               "scripts/audit_exploratory_scene_registry_v2.py", "scripts/check_q0_state_restore.py",
               "scripts/diagnose_teacher_cross_pose.py", "src/rosetta_reality/eval/reproducibility_capture.py",
               "src/rosetta_reality/sim/gym_aloha.py", "src/rosetta_reality/sim/action_contract.py",
               "src/rosetta_reality/sim/scripted_teacher_observer.py", "src/rosetta_reality/sim/geometry_teacher.py"]
    source_identity = {name: sha(ROOT / name) for name in sources}
    contract_path = ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    contract = load_action_contract(contract_path)
    start = time.monotonic()
    entries, failure = [], None
    save(args.output / "registration.json", {"groups": groups, "plans": plans,
         "source_sha256": source_identity, "action_contract_sha256": sha(contract_path),
         "maximum_seconds": args.max_seconds, "maximum_bytes": args.max_bytes,
         "legacy_observations_read": False, "reference_groups_are_regenerated_resets": True})

    def budget():
        if time.monotonic() - start >= args.max_seconds:
            raise TimeoutError("Registered reset-capture work budget exceeded")
        if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) >= args.max_bytes - 1024**2:
            raise OSError("Registered reset-capture evidence budget exceeded")

    try:
        for group, seeds in groups.items():
            for seed in seeds:
                budget()
                env = GymAlohaEnvironment(contract, maximum_episode_steps=500)
                try:
                    env.reset(seed=seed)
                    reader = PhysicsReader(env)
                    if int(getattr(reader.model, "nplugin", 0)) or any(
                        getattr(mujoco, name)() is not None for name in dir(mujoco)
                        if name.startswith("get_mjcb_")):
                        raise ValueError("External native state prevents complete reset capture")
                    physics = reader()["integration"].numpy().copy()
                    python = PythonState([env], reader.physics).receipt()
                    geometry = finite_json(build_observation(env, contract))
                    scene = {k: geometry[k] for k in ("socket", "peg", "robot_state", "left_eef", "right_eef")}
                    name = f"{group}-{seed}"
                    physics_path, python_path = name + "-physics.npz", name + "-python.json.gz"
                    with (args.output / physics_path).open("xb") as stream:
                        np.savez_compressed(stream, integration=physics)
                    python_bytes = encoded(python)
                    with (args.output / python_path).open("xb") as stream:
                        stream.write(gzip.compress(python_bytes, mtime=0))
                    entries.append({"seed": seed, "group": group,
                        "parent_id": parents.get(seed, f"registered-reset-{group}-{seed}"),
                        "scene_fingerprint": hashlib.sha256(encoded(scene)).hexdigest(), "geometry": scene,
                        "reset_state_sha256": hashlib.sha256(physics.tobytes()).hexdigest(),
                        "integration_spec": reader.spec, "integration_dtype": str(physics.dtype),
                        "integration_shape": list(physics.shape), "model_sha256": reader.model_sha,
                        "complete_state_captured": True, "physics_path": physics_path,
                        "python_state_path": python_path, "python_state_sha256": hashlib.sha256(python_bytes).hexdigest(),
                        "source_sha256": source_identity})
                    budget()
                finally:
                    env.close()
    except Exception as exc:
        failure = type(exc).__name__
    finally:
        save(args.output / "registry.json", {"schema": "rosetta.exploratory_scene_registry.v1",
             "status": "completed" if failure is None else "incomplete", "error_type": failure,
             "groups": groups, "entries": entries, "plans": plans,
             "legacy_object_scene_nonoverlap": "not_established", "legacy_observations_read": False,
             "reference_groups_are_regenerated_resets": True, "action_steps": 0,
             "model_forward_calls": 0, "optimizer_updates": 0, "elapsed_seconds": time.monotonic() - start})
        files = [{"path": p.relative_to(args.output).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(args.output.rglob("*")) if p.is_file()]
        save(args.output / "manifest.json", {"schema": "rosetta.exploratory_scene_registry_manifest.v1", "files": files})
    return 0 if failure is None else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("q1-plan", "q2-plan", "bank-plan", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    raise SystemExit(prepare(p.parse_args()))
