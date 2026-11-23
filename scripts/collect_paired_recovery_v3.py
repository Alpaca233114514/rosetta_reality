"""Shared real prefix causal fork S/R collection; old v2 evidence is untouched.

No policy, weights, optimizer, downloads or power control. AutoDL orchestration
must supply the registered dependency overlay and an external work-only guard.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import gzip
import importlib.util
import json
import sys
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# File-only plan parsing must not execute data/__init__.py's historical eager
# Torch type exports. The fixed local source path is identified in every receipt.
_spec = importlib.util.spec_from_file_location(
    "_paired_collection_rules", ROOT / "src/rosetta_reality/data/paired_collection_v2.py")
_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rules)
SCHEMA, digest, match_pair = _rules.SCHEMA, _rules.digest, _rules.match_pair
sha_file, valid_starts, validate_plan = _rules.sha_file, _rules.valid_starts, _rules.validate_plan
protected_inventory = _rules.protected_inventory
symmetric_sampler = _rules.symmetric_sampler
exploratory_registry = _rules.exploratory_registry


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_receipt(path, value, *, compressed):
    if not compressed:
        write(path, value)
        return path
    target = path.with_suffix(path.suffix + ".gz")
    with target.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as archive:
            # The full JSON object and every bytes_hex value are preserved.
            archive.write((json.dumps(value, sort_keys=True, allow_nan=False,
                                      separators=(",", ":")) + "\n").encode())
    return target


class BudgetStop(RuntimeError):
    pass


def run(plan, output, *, scene_registry=None, scene_registry_audit=None):
    import mujoco
    import numpy as np
    import torch
    from check_q0_state_restore import PythonState, UnsupportedRestore
    from diagnose_teacher_cross_pose import finite_json, validate_state_finite
    from rosetta_reality.eval.reproducibility_capture import PhysicsReader
    from rosetta_reality.sim.action_contract import load_action_contract
    from rosetta_reality.sim.coupled_translation_teacher import build_teacher
    from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment
    from rosetta_reality.sim.scripted_teacher_observer import build_observation

    contract_path = ROOT / plan["contract"]
    contract = load_action_contract(contract_path)
    if contract.semantics != "absolute_joint_position_targets":
        raise ValueError("absolute action contract required")
    if "left_waist" not in contract.dimension_names:
        raise ValueError("registered joint name unavailable")
    joint = contract.dimension_names.index("left_waist")
    started = time.monotonic()
    written = 0
    simulation_steps = 0
    simulation_steps_attempted = 0
    attempts, pair_records, bank, forks = [], [], [], []
    decision_calls = 0
    shared_prefix_physical_steps = 0
    if plan.get("collection_protocol") != "shared_real_prefix_causal_fork_v3":
        raise ValueError("v3 requires a new explicitly registered fork protocol")
    protection = protected_inventory(ROOT, plan)
    exploratory = plan.get("research_scope") == "fresh_lineage_fixed_development_exploration_v1"
    fresh = exploratory_registry(scene_registry, scene_registry_audit, plan) if exploratory else {
        "complete": False, "forbidden": [], "entries": {}}
    if exploratory and plan["phase"] != "Q1" and not fresh["complete"]:
        raise ValueError("formal exploratory collection requires actually audited new scene families")
    if fresh["complete"]:
        destination = output / "scene-registry"
        destination.mkdir()
        for item in fresh["payloads"]:
            target = destination / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                with (fresh["registry_root"] / item["path"]).open("rb") as source:
                    shutil.copyfileobj(source, handle)
            if sha_file(target) != item["sha256"]:
                raise ValueError("new scene registry changed during immutable capture")
            written += target.stat().st_size
        for name, raw in (("manifest.json", fresh["manifest_bytes"]), ("audit.json", fresh["audit_bytes"])):
            with (destination / name).open("xb") as handle:
                handle.write(raw)
            written += len(raw)
    if protection["complete"]:
        for name in ("inventory", "audit"):
            with (output / ("protected-scene-" + name + ".json")).open("xb") as handle:
                handle.write(protection[name + "_bytes"])
            written += len(protection[name + "_bytes"])
    seen = set(plan["forbidden_scene_fingerprints"]) | set(protection["fingerprints"]) | set(fresh["forbidden"])
    stop_all = None
    compressed_receipts = plan.get("receipt_encoding") == "lossless_json_gzip_v1"

    def guard():
        if stop_all:
            raise BudgetStop(stop_all)
        if time.monotonic() - started >= plan["work_cap_seconds"]:
            raise BudgetStop("work_cap")
        # Reserve label/manifest closeout space before another image is written.
        if written >= plan["evidence_byte_cap"] - 16 * 1024 * 1024:
            raise BudgetStop("evidence_cap")

    def snapshot(env, reader):
        if any(getattr(mujoco, name)() is not None for name in dir(mujoco)
               if name.startswith("get_mjcb_")) or int(getattr(reader.model, "nplugin", 0)):
            raise UnsupportedRestore("native_external_state")
        return reader()["integration"].numpy().copy(), PythonState([env], reader.physics)

    def restore(reader, saved):
        physics, python = saved
        python.restore()
        mujoco.mj_setState(reader.model, reader.data, physics, reader.spec)
        mujoco.mj_forward(reader.model, reader.data)
        mujoco.mj_setState(reader.model, reader.data, physics, reader.spec)
        if not np.array_equal(reader()["integration"].numpy(), physics):
            raise UnsupportedRestore("physics_restore_mismatch")

    def checked_step(env, reader, action):
        nonlocal simulation_steps, simulation_steps_attempted
        guard()
        validate_state_finite(action)
        contract.validate_tensor(action, allow_chunk=False)
        if bool(contract.clip(action)[1].any()):
            raise ValueError("action_requires_clipping")
        simulation_steps_attempted += 1
        state, reward, done, info = env.step(action)
        simulation_steps += 1
        validate_state_finite(state)
        if not np.isfinite(reward):
            raise ValueError("nonfinite_reward")
        limits, contacts = env.state_limit_violation_count(), env.unexpected_collision_count()
        safe = not (limits or contacts or bool(env.last_clip_mask.any()))
        record = finite_json({"integration": reader()["integration"],
                              "geometry": build_observation(env, contract),
                              "reward": reward, "done": done, "info": info,
                              "contacts": env.contact_pairs(), "joint_limits": limits,
                              "unexpected_collisions": contacts})
        return state, record, safe

    def parity(env, reader, current, label, directory):
        nonlocal written
        saved = snapshot(env, reader)
        action = current["robot_state"].clone()
        traces = []
        replays = []
        for _ in range(2):
            restore(reader, saved)
            state, result, safe = checked_step(env, reader, action)
            if not safe or result["done"]:
                restore(reader, saved)
                return False
            # Includes images bytewise and wrapper state, not geometry alone.
            images = {k: {"sha256": hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest(),
                          "dtype": str(v.dtype), "shape": list(v.shape)}
                      for k, v in state["images"].items()}
            replay = {"step": result, "images": images,
                      "python": PythonState([env], reader.physics).receipt()}
            replays.append(replay)
            traces.append(digest(replay))
        restore(reader, saved)
        receipt_path = directory / (label + "-restore.json")
        receipt_path = write_receipt(receipt_path, {
            "integration_spec": reader.spec, "physics_model_sha256": reader.model_sha,
            "python_state": saved[1].receipt(), "integration_sha256":
                hashlib.sha256(saved[0].tobytes()).hexdigest(),
            "exact_replays": traces, "replays": replays, "passed": traces[0] == traces[1],
            "engineering_steps": 2}, compressed=compressed_receipts)
        written += receipt_path.stat().st_size
        return traces[0] == traces[1]

    def teacher_decision(teacher, geometry):
        nonlocal decision_calls
        decision = teacher.decide(geometry)
        decision_calls += 1
        return decision, str(teacher._phase.value)

    def reconstruct(history):
        teacher = build_teacher()
        teacher.reset()
        last = None
        for index, (geometry, expected_action, expected_phase) in enumerate(history):
            decision, phase = teacher_decision(teacher, geometry)
            if decision.refusal is not None or finite_json(decision.action) != expected_action or phase != expected_phase:
                raise UnsupportedRestore("teacher_history_replay_mismatch_at_" + str(index))
            last = decision
        return teacher, last

    for registered in plan["pairs"]:
        pair_id = registered["pair_id"]
        directory = output / pair_id
        directory.mkdir()
        env = None
        summaries = []
        reset_saved = None
        current_reset = None
        scene = None
        reset_exact = False
        try:
            if stop_all:
                raise BudgetStop(stop_all)
            guard()
            env = GymAlohaEnvironment(contract, maximum_episode_steps=500)
            current_reset = env.reset(seed=registered["seed"])
            reader = PhysicsReader(env)
            reset_saved = snapshot(env, reader)
            geometry = finite_json(build_observation(env, contract))
            # Object+robot poses form a scene fingerprint independent of seed IDs.
            scene = digest({key: geometry[key] for key in
                            ("socket", "peg", "robot_state", "left_eef", "right_eef")})
            if fresh["complete"]:
                registered_reset = fresh["entries"][registered["seed"]]
                if (registered_reset["parent_id"] != pair_id or registered_reset["scene_fingerprint"] != scene
                        or registered_reset["model_sha256"] != reader.model_sha
                        or registered_reset["reset_state_sha256"] != hashlib.sha256(reset_saved[0].tobytes()).hexdigest()):
                    raise UnsupportedRestore("actual_reset_differs_from_registered_new_family")
            overlap = scene in seen
            seen.add(scene)
            reset_exact = parity(env, reader, current_reset, "reset", directory)
            if not reset_exact:
                raise UnsupportedRestore("reset_parity_failed")
            np.savez_compressed(directory / "reset-integration.npz", state=reset_saved[0])
            written += (directory / "reset-integration.npz").stat().st_size
            # One actual common history is observed, never substitute independently
            # rendered R images after the fact. Its files are referenced by both roles.
            prefix_rows, history = [], []
            prefix_dir = directory / "shared-prefix"
            prefix_dir.mkdir()
            prefix_teacher = build_teacher()
            prefix_teacher.reset()
            current = current_reset
            restore(reader, reset_saved)
            fork_step, fork_saved, fork_current = None, None, None
            prefix_status, prefix_safety, prefix_success = "event_not_reached", False, False
            with (prefix_dir / "commands.jsonl").open("x", encoding="utf-8") as prefix_commands:
                if not overlap and not (env.state_limit_violation_count() or env.unexpected_collision_count()):
                    for step in range(plan["max_steps"]):
                        guard()
                        geometry = build_observation(env, contract)
                        decision, phase = teacher_decision(prefix_teacher, geometry)
                        if decision.refusal is not None:
                            prefix_status = "refused"
                            break
                        history.append((copy.deepcopy(geometry), finite_json(decision.action), phase))
                        if phase == registered["trigger"]:
                            fork_step = step
                            fork_saved = snapshot(env, reader)
                            fork_current = copy.deepcopy(current)
                            break
                        pixels = current["raw"].get("pixels", current["raw"].get("images", {}))
                        arrays = {str(k):np.asarray(v).copy() for k,v in pixels.items()}
                        if not arrays or any(a.dtype != np.uint8 or a.ndim != 3 for a in arrays.values()):
                            raise ValueError("native_prefix_pixel_schema_drift")
                        image_path = prefix_dir / f"frame-{step:04d}.npz"
                        np.savez_compressed(image_path, **arrays)
                        written += image_path.stat().st_size
                        before = finite_json(geometry)
                        command = {"step":step,"action":finite_json(decision.action),"state":before["robot_state"],
                                   "action_source":"state_conditioned_teacher","images":image_path.relative_to(output).as_posix()}
                        encoded = json.dumps(command,sort_keys=True,allow_nan=False)+"\n"
                        prefix_commands.write(encoded)
                        prefix_commands.flush()
                        written += len(encoded.encode())
                        current, after, safe = checked_step(env, reader, decision.action)
                        shared_prefix_physical_steps += 1
                        row = {**command,"pts_seconds":step/50,"images_sha256":sha_file(image_path),
                               "task":"Insert the peg into the socket.","positive_target":False,
                               "before":before,"after":after,"phase":phase,"safe":safe}
                        prefix_rows.append(row)
                        if not safe:
                            prefix_status, prefix_safety = "safety_stop", True
                            break
                        if after["info"].get("is_success",False):
                            prefix_status, prefix_success = "success_before_registered_event", True
                            break
                        if after["done"]:
                            prefix_status = "task_failed"
                            break
            write(prefix_dir/"rows.json", {"rows":prefix_rows,"actual_physical_steps":len(prefix_rows)})
            written += (prefix_dir/"rows.json").stat().st_size
            fork_record = {"pair_id":pair_id,"protocol":plan["collection_protocol"],
                           "fork_step":fork_step,"shared_prefix_physical_steps":len(prefix_rows),
                           "teacher_history_calls":len(history),"teacher_reconstruction":[]}
            if fork_saved is not None:
                np.savez_compressed(directory/"fork-integration.npz",state=fork_saved[0])
                fork_python = write_receipt(directory/"fork-python.json",{
                    "python_state":fork_saved[1].receipt(),"model_sha256":reader.model_sha,
                    "source_sha256":fork_saved[1].sources,"integration_spec":reader.spec},compressed=compressed_receipts)
                write(directory/"teacher-history.json",[{"geometry":finite_json(g),"action":a,"phase":p} for g,a,p in history])
                fork_arrays = {str(k):np.asarray(v).copy() for k,v in fork_current["raw"].get("pixels",fork_current["raw"].get("images",{})).items()}
                np.savez_compressed(directory/"fork-native-observation.npz", **fork_arrays)
                fork_record["cached_state"] = finite_json(fork_current["robot_state"])
                fork_record["cached_native_images"] = {k:{"sha256":hashlib.sha256(v.tobytes()).hexdigest(),"dtype":v.dtype.str,"shape":list(v.shape)} for k,v in fork_arrays.items()}
                for file in (directory/"fork-integration.npz",fork_python,directory/"teacher-history.json",directory/"fork-native-observation.npz"):
                    written += file.stat().st_size
            forks.append(fork_record)
            for kind in ("S", "R"):
                summary = {"schema": SCHEMA, "attempt_id": pair_id + "-" + kind,
                           "pair_id": pair_id, "parent_id": pair_id, "kind": kind,
                           "seed": registered["seed"], "provider_id": plan["provider_id"],
                           "provider_qualified": False, "source_kind": "experimental_teacher",
                           "research_scope": plan.get("research_scope", "strict_legacy_scene_disjoint"),
                           "fresh_lineage_verified": fresh["complete"],
                           "new_scene_disjoint_verified": fresh["complete"] and not overlap,
                           "legacy_object_scene_nonoverlap": plan.get("legacy_object_scene_nonoverlap", "strict_inventory_required"),
                           "split": "pilot" if plan["phase"] == "Q1" else
                                    "evaluation" if plan["phase"] == "bank" else "train_candidate",
                           "contract_sha256": sha_file(contract_path),
                           "model_sha256": reader.model_sha, "scene_fingerprint": scene,
                           "reset_state_sha256": hashlib.sha256(reset_saved[0].tobytes()).hexdigest(),
                           "scene_overlap": overlap, "restore_exact": reset_exact,
                           "status": "not_started", "task_success": False, "safety_stop": False,
                           "deviation": None, "correction_start": None, "recovery": None,
                           "verified_recovery_chain": False, "rows": 0,
                           "effective_input_frames": 0, "valid_target_slots": 0,
                           "positive_target_policy": "safe_expert_actions_in_verified_success_only"}
                summaries.append(summary)
                rows = copy.deepcopy(prefix_rows)
                image_dir = directory / kind
                image_dir.mkdir()
                with (image_dir/"commands.jsonl").open("x",encoding="utf-8") as commands, (image_dir/"trace.jsonl").open("x",encoding="utf-8") as trace:
                    for shared in rows:
                        commands.write(json.dumps({key:shared[key] for key in ("step","action","state","action_source","images")},sort_keys=True,allow_nan=False)+"\n")
                        trace.write(json.dumps(shared,sort_keys=True,allow_nan=False)+"\n")
                written += (image_dir/"commands.jsonl").stat().st_size + (image_dir/"trace.jsonl").stat().st_size
                try:
                    guard()
                    summary["rows"] = len(rows)
                    if fork_saved is None:
                        summary.update(status=prefix_status,safety_stop=prefix_safety,task_success=prefix_success)
                        if prefix_success:
                            summary["terminal_success_step"] = len(rows)-1
                        continue
                    restore(reader, fork_saved)
                    if fork_saved[1].receipt() != PythonState([env],reader.physics).receipt():
                        raise UnsupportedRestore("fork_python_restore_mismatch")
                    current = copy.deepcopy(fork_current)
                    teacher, fork_decision = reconstruct(history)
                    fork_record["teacher_reconstruction"].append({"role":kind,"calls":len(history),"all_decisions_exact":True})
                    if finite_json(current["robot_state"]) != fork_record["cached_state"]:
                        raise UnsupportedRestore("cached_fork_state_changed")
                    if finite_json(build_observation(env,contract)) != finite_json(history[-1][0]):
                        raise UnsupportedRestore("fork_physics_geometry_changed")
                    if env.state_limit_violation_count() or env.unexpected_collision_count():
                        summary.update(status="reset_safety_stop", safety_stop=True)
                        continue
                    if overlap:
                        summary["status"] = "scene_overlap_excluded"
                        continue
                    sustain = 0
                    with ((image_dir / "trace.jsonl").open("a", encoding="utf-8") as trace,
                          (image_dir / "commands.jsonl").open("a", encoding="utf-8") as commands):
                        for step in range(fork_step, plan["max_steps"]):
                            guard()
                            geometry = build_observation(env, contract)
                            if step == fork_step:
                                decision, phase = fork_decision, history[-1][2]
                            else:
                                decision, phase = teacher_decision(teacher, geometry)
                            if decision.refusal is not None:
                                summary.update(status="refused", refusal=finite_json(decision.refusal))
                                break
                            action = decision.action.clone()
                            source = "state_conditioned_teacher"
                            if kind == "R" and summary["deviation"] is None and phase == registered["trigger"]:
                                if step + 1 >= plan["max_steps"]:
                                    summary["status"] = "insufficient_correction_budget"
                                    break
                                if not parity(env, reader, current, "loaded", directory):
                                    summary.update(status="loaded_restore_failed", restore_exact=False)
                                    break
                                # Perturb one registered absolute command, no reward/object mutation.
                                action = geometry.robot_state.clone()
                                action[joint] += registered["joint_delta_rad"]
                                summary["deviation"] = {"step": step, "trigger": phase,
                                                         "joint": registered["joint_name"],
                                                         "delta_rad": registered["joint_delta_rad"],
                                                         "duration_steps": 1,
                                                         "pre_state": finite_json(geometry.robot_state)}
                                source = "registered_deliberate_deviation"
                            if (kind == "R" and summary["correction_start"] is not None
                                    and step - summary["correction_start"] >= plan["correction_max_steps"]):
                                summary["status"] = "correction_timeout"
                                break
                            # Lossless original native camera pixels; no resized/interpolated images.
                            pixels = current["raw"].get("pixels", current["raw"].get("images", {}))
                            if not isinstance(pixels, dict) or not pixels:
                                raise ValueError("native_camera_pixels_missing")
                            image_path = image_dir / f"frame-{step:04d}.npz"
                            arrays = {str(k): np.asarray(v) for k, v in pixels.items()}
                            if any(a.dtype != np.uint8 or a.ndim != 3 for a in arrays.values()):
                                raise ValueError("native_pixel_schema_drift")
                            np.savez_compressed(image_path, **arrays)
                            if step == fork_step:
                                actual_images = {k:{"sha256":hashlib.sha256(v.tobytes()).hexdigest(),"dtype":v.dtype.str,"shape":list(v.shape)} for k,v in arrays.items()}
                                if actual_images != fork_record["cached_native_images"]:
                                    raise UnsupportedRestore("fork_cached_native_input_changed")
                            written += image_path.stat().st_size
                            before = finite_json(geometry)
                            command = json.dumps({"step": step, "action": finite_json(action),
                                                  "state": before["robot_state"], "action_source": source,
                                                  "images": image_path.relative_to(output).as_posix()},
                                                 sort_keys=True, allow_nan=False) + "\n"
                            commands.write(command)
                            commands.flush()
                            written += len(command.encode())
                            current, after, safe = checked_step(env, reader, action)
                            row = {"step": step, "pts_seconds": step / 50,
                                   "images": str(image_path.relative_to(output).as_posix()),
                                   "images_sha256": sha_file(image_path),
                                   "state": before["robot_state"], "action": finite_json(action),
                                   "task": "Insert the peg into the socket.", "action_source": source,
                                   "positive_target": False, "before": before, "after": after,
                                   "phase": phase, "safe": safe}
                            rows.append(row)
                            encoded = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                            trace.write(encoded)
                            trace.flush()
                            written += len(encoded.encode())
                            summary["rows"] = len(rows)
                            if not safe:
                                summary.update(status="safety_stop", safety_stop=True)
                                break
                            if source == "registered_deliberate_deviation":
                                change = abs(before["robot_state"][joint] - after["geometry"]["robot_state"][joint])
                                summary["deviation"]["observed_joint_change_rad"] = change
                                summary["deviation"]["verified"] = change >= plan["deviation_min_joint_change_rad"]
                                summary["correction_start"] = step + 1
                                if plan["phase"] == "bank":
                                    saved_branch = snapshot(env, reader)
                                    np.savez_compressed(directory / "branch-integration.npz", state=saved_branch[0])
                                    branch_arrays = {"robot_state": current["robot_state"].detach().cpu().numpy()}
                                    branch_arrays.update({"image::" + k: v.detach().cpu().numpy()
                                                          for k, v in current["images"].items()})
                                    np.savez_compressed(directory / "branch-observation.npz", **branch_arrays)
                                    branch_python_path = write_receipt(directory / "branch-python.json", {
                                        "python_state": saved_branch[1].receipt(),
                                        "source_sha256": saved_branch[1].sources,
                                        "model_sha256": reader.model_sha,
                                        "integration_spec": reader.spec}, compressed=compressed_receipts)
                                    written += branch_python_path.stat().st_size
                                    written += (directory / "branch-integration.npz").stat().st_size
                                    written += (directory / "branch-observation.npz").stat().st_size
                                    if not parity(env, reader, current, "branch", directory):
                                        summary.update(status="branch_restore_failed", restore_exact=False)
                                        break
                            elif kind == "R" and summary["deviation"] is not None:
                                g = after["geometry"]
                                recovered = bool(g["socket_grasp_contact"] and g["peg_grasp_contact"])
                                if registered["trigger"] == "align":
                                    recovered = recovered and bool(g["peg_socket_contact"])
                                sustain = sustain + 1 if recovered else 0
                                if sustain >= plan["recovery_sustain_steps"] and summary["recovery"] is None:
                                    summary["recovery"] = {"step": step, "sustain_steps": sustain,
                                                           "criterion": "both_grasps" if registered["trigger"] != "align"
                                                           else "both_grasps_and_peg_socket_contact"}
                            success = bool(after["info"].get("is_success", False))
                            summary["task_success"] |= success
                            if success:
                                summary["terminal_success_step"] = step
                                summary["status"] = "success"
                                break
                            if after["done"]:
                                summary["status"] = "task_failed"
                                break
                        else:
                            summary["status"] = "episode_timeout"
                    if summary["status"] == "not_started":
                        summary["status"] = "event_not_reached" if kind == "R" and summary["deviation"] is None else "task_failed"
                except BudgetStop as error:
                    stop_all = str(error)
                    summary["status"] = "resource_stop_" + stop_all
                except Exception as error:
                    stop_all = "infrastructure_failure"
                    summary.update(status=stop_all, error_type=type(error).__name__)
                finally:
                    summary["verified_recovery_chain"] = bool(
                        kind == "R" and summary["task_success"] and not summary["safety_stop"]
                        and summary["deviation"] and summary["deviation"].get("verified")
                        and summary["correction_start"] is not None and summary["recovery"]
                        and summary["recovery"]["step"] < summary.get("terminal_success_step", -1))
                    eligible = summary["task_success"] and not summary["safety_stop"] and (
                        kind == "S" or summary["verified_recovery_chain"])
                    for row in rows:
                        row["positive_target"] = bool(eligible and row["safe"] and row["action_source"] == "state_conditioned_teacher")
                    starts = valid_starts(rows)
                    summary["effective_input_frames"] = len(starts)
                    summary["valid_target_slots"] = len(starts) * 50
                    summary["raw_effective_input_frames"] = len(starts)
                    summary["raw_valid_target_slots"] = len(starts)*50
                    summary["padding_slots"] = 0
                    write(image_dir / "labels.json", {"rows": rows, "valid_chunk_starts": starts,
                                                       "native_valid_chunk_starts": starts,
                                                       "chunk_length": 50, "native_loss_changed": False})
                    write(image_dir / "summary.json", summary)
                    written += (image_dir / "labels.json").stat().st_size
                    written += (image_dir / "summary.json").stat().st_size
                    attempts.append(summary)
            sampler = None
            if plan.get("sampling_rule") == "symmetric_common_legal_chunk_grid_v1":
                s_rows = json.loads((directory / "S/labels.json").read_text())["rows"]
                r_rows = json.loads((directory / "R/labels.json").read_text())["rows"]
                deviation = summaries[1]["deviation"]
                sampler = symmetric_sampler(s_rows, r_rows, deviation["step"] if deviation else None,
                                            minimum_common_starts=plan["minimum_common_chunk_starts"])
                write(directory / "pair-sampler.json", sampler)
                written += (directory / "pair-sampler.json").stat().st_size
                for item in summaries:
                    item["sampler_effective_input_frames"] = sampler["sampler_effective_input_frames_per_role"]
                    item["sampler_valid_target_slots"] = sampler["sampler_valid_target_slots_per_role"]
                    item["common_grid_sha256"] = sampler["common_starts_sha256"]
            pair_records.append(match_pair(*summaries, sampler=sampler))
            if plan["phase"] == "bank":
                branch_created = (summaries[1]["restore_exact"] and
                                  all((directory / name).is_file() for name in (
                                      "branch-integration.npz", "branch-python.json.gz" if compressed_receipts else "branch-python.json",
                                      "branch-observation.npz", "branch-restore.json.gz" if compressed_receipts else "branch-restore.json")))
                bank.append({"pair_id": pair_id, "feasible": summaries[1]["verified_recovery_chain"],
                             "status": summaries[1]["status"], "parent_id": pair_id,
                             "branch_state": pair_id + "/branch-integration.npz" if branch_created else None})
            write(directory/"fork-protocol.json",fork_record)
            written += (directory/"fork-protocol.json").stat().st_size
        except BudgetStop as error:
            stop_all = str(error)
        except Exception as error:
            stop_all = "infrastructure_failure"
            write(directory / "failure.json", {"error_type": type(error).__name__, "status": stop_all})
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as error:
                    stop_all = "infrastructure_failure"
                    write(directory / "close-failure.json", {"error_type": type(error).__name__})
            # Missing/unstarted attempts stay in the registered denominator.
            for kind in ("S", "R"):
                if not any(x["attempt_id"] == pair_id + "-" + kind for x in attempts):
                    attempts.append({"attempt_id": pair_id + "-" + kind, "pair_id": pair_id,
                                     "kind": kind, "status": "not_measured_" + (stop_all or "failure"),
                                     "task_success": None, "rows": 0})
            if not any(x["pair_id"] == pair_id for x in pair_records):
                pair_records.append({"pair_id": pair_id, "matched": False,
                                     "reasons": ["pair_incomplete"]})
    candidate_n = sum(p["matched"] for p in pair_records)
    # Pilot and bank never enter training; unresolved protected fingerprint inventory
    # is an admission boundary, never a reason to suppress the physical measurement.
    strict_n = candidate_n if plan["phase"] == "Q2" and protection["complete"] and not exploratory else 0
    exploratory_n = candidate_n if plan["phase"] == "Q2" and exploratory and fresh["complete"] else 0
    admitted_n = exploratory_n if exploratory else strict_n
    summary = {"schema": SCHEMA, "phase": plan["phase"], "plan_id": plan["id"],
               "collection_protocol":plan["collection_protocol"],"forks":forks,
               "status": stop_all or ("completed" if candidate_n else "completed_negative_result"),
               "provider_id": plan["provider_id"], "provider_qualified": False,
               "provider_status": "experimental_real_success_labels_only",
               "registered_pairs": len(plan["pairs"]), "registered_attempts": len(plan["pairs"]) * 2,
               "attempts": attempts, "pairs": pair_records, "candidate_matched_N": candidate_n,
               "N": admitted_n, "formal_training_eligible": strict_n > 0,
               "exploratory_training_eligible": exploratory_n > 0,
               "N_strict": strict_n,
               "N_exploratory": exploratory_n,
               "research_scope": plan.get("research_scope", "strict_legacy_scene_disjoint"),
               "legacy_object_scene_nonoverlap": plan.get("legacy_object_scene_nonoverlap", "strict_inventory_required"),
               "fresh_lineage_verified": fresh["complete"],
               "new_scene_disjoint_verified": fresh["complete"],
               "protected_scene_inventory_complete": protection["complete"],
               "protected_scene_inventory_reference": protection.get("reference"),
               "selection_bias": "success_and_matching_conditioned_subset_all_failures_retained",
               "sampling_rule": plan.get("sampling_rule", "all_native_legal_chunks"),
               "receipt_encoding": plan.get("receipt_encoding", "plain_json_v1"),
               "environment_steps": simulation_steps, "environment_steps_attempted": simulation_steps_attempted,
               "shared_prefix_physical_steps":shared_prefix_physical_steps,
               "logical_trajectory_frames":sum(a["rows"] for a in attempts),
               "teacher_decision_calls":decision_calls,
               "teacher_reconstruction_decision_calls":sum(r["calls"] for f in forks for r in f["teacher_reconstruction"]),
               "engineering_restore_steps_cap": len(plan["pairs"]) * (6 if plan["phase"] == "bank" else 4),
               "main_environment_steps_cap": len(plan["pairs"]) * 1000,
               "evidence_bytes_accounted": written, "elapsed_seconds": time.monotonic() - started,
               "bank": bank, "model_forward_calls": 0, "optimizer_steps": 0,
               "shutdown_requested": False,
               "code_sha256": {path: sha_file(ROOT / path) for path in (
                   "scripts/collect_paired_recovery_v3.py", "scripts/check_q0_state_restore.py",
                   "scripts/restore_paired_branch_v2.py", "scripts/audit_paired_collection_v2.py",
                   "scripts/audit_paired_collection_v3.py", "scripts/verify_paired_numerics_v2.py",
                   "src/rosetta_reality/data/paired_collection_v2.py",
                   "src/rosetta_reality/sim/coupled_translation_teacher.py",
                   "src/rosetta_reality/sim/geometric_teacher.py",
                   "src/rosetta_reality/sim/scripted_teacher.py",
                   "src/rosetta_reality/sim/scripted_teacher_observer.py",
                   "src/rosetta_reality/sim/gym_aloha.py")}}
    write(output / "summary.json", summary)
    files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size,
              "sha256": sha_file(p)} for p in sorted(output.rglob("*")) if p.is_file()]
    write(output / "manifest.json", {"schema": "rosetta.paired_files.v1", "files": files,
                                     "plan_sha256": digest(plan)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-registry", type=Path)
    parser.add_argument("--scene-registry-audit", type=Path)
    args = parser.parse_args()
    plan = validate_plan(json.loads(args.plan.read_text(encoding="utf-8-sig")))
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "registration.json", plan)
    try:
        summary = run(plan, args.output, scene_registry=args.scene_registry,
                      scene_registry_audit=args.scene_registry_audit)
    except Exception as error:
        # Backend bootstrap failures also preserve the entire registration denominator.
        if (args.output / "summary.json").exists():
            raise
        summary = {"schema": SCHEMA, "phase": plan["phase"], "plan_id": plan["id"],
                   "status": "infrastructure_failure", "error_type": type(error).__name__,
                   "registered_attempts": len(plan["pairs"])*2,
                   "attempts": [{"attempt_id": p["pair_id"]+"-"+k, "pair_id": p["pair_id"],
                                 "kind": k, "status": "not_measured_infrastructure_failure",
                                 "task_success": None, "rows": 0} for p in plan["pairs"] for k in ("S", "R")],
                   "candidate_matched_N": 0, "N": 0, "environment_steps": None,
                   "formal_training_eligible": False, "provider_qualified": False,
                   "shutdown_requested": False}
        write(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "registered_attempts", "candidate_matched_N", "N", "environment_steps")}))
    return 2 if summary["status"] in ("infrastructure_failure", "work_cap", "evidence_cap") else 0


if __name__ == "__main__":
    raise SystemExit(main())
