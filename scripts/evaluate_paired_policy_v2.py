"""Independent fixed-five-scene measurement of one actual paired-data endpoint."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from math import isfinite as math_isfinite
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_bank_selection(plan, bank):
    registered = "bank_manifest_sha256" in plan
    if registered != (bank is not None):
        raise ValueError("Registered recovery bank and CLI selection must agree")


class RecoveryEventTracker:
    """Measure the registered five consecutive safe correction steps."""

    def __init__(self, trigger, sustain_steps=5):
        if trigger not in ("approach", "grasp", "align") or sustain_steps != 5:
            raise ValueError("Registered approach/grasp/align five-step recovery required")
        self.trigger = trigger
        self.last_step = 0
        self.streak = 0
        self.recovery_step = None
        self.terminal_success_step = None
        self.safety_stop = False

    def observe(self, step, geometry, *, safe, task_success):
        if type(step) is not int or step != self.last_step + 1:
            raise ValueError("Recovery observations must be consecutive correction steps")
        self.last_step = step
        contacts = ("socket_grasp_contact", "peg_grasp_contact")
        if self.trigger == "align":
            contacts += ("peg_socket_contact",)
        if any(type(geometry.get(key)) is not bool for key in contacts):
            raise ValueError("Recovery contact observations must be explicit booleans")
        self.safety_stop |= not safe
        recovered = safe and all(geometry[key] for key in contacts)
        self.streak = self.streak + 1 if recovered else 0
        if self.streak >= 5 and self.recovery_step is None:
            self.recovery_step = step
        if task_success and self.terminal_success_step is None:
            self.terminal_success_step = step
        return self.result()

    def result(self):
        return {"criterion": "both_grasps_and_peg_socket_contact" if self.trigger == "align" else "both_grasps",
                "required_consecutive_steps": 5, "correction_steps_observed": self.last_step,
                "recovery_step": self.recovery_step,
                "geometric_recovery": self.recovery_step is not None,
                "terminal_success_step": self.terminal_success_step,
                "verified_recovery_chain": bool(not self.safety_stop and self.recovery_step is not None
                    and self.terminal_success_step is not None and self.recovery_step < self.terminal_success_step)}


def verify(plan, artifact):
    if plan.get("status")!="preregistered" or plan.get("seeds")!=[1000,1001,1002,1003,1004] or plan.get("maximum_steps")!=500:
        raise ValueError("Unchanged five-scene protocol required")
    if plan.get("optimizer_updates")!=0 or plan.get("hidden_test_loaded") is not False:
        raise ValueError("Independent evaluation cannot update parameters or open hidden")
    if plan.get("instruction")!="Insert the peg into the socket." or plan.get("noise")!="seeded_standard_normal":
        raise ValueError("Instruction or inference noise drift")
    if plan.get("arm") not in ("A","B","C") or plan.get("action_contract")!={"path":"configs/sim/aloha_insertion_smolvla.yaml"}:
        raise ValueError("Actual arm and original Action Contract path required")
    required={"scripts/evaluate_paired_policy_v2.py","scripts/diagnose_smolvla_base_five_scene.py","scripts/smolvla_autodl_vfunfreeze_sim_gate.py","scripts/smolvla_sim_gate.py","src/rosetta_reality/sim/gym_aloha.py","src/rosetta_reality/vla/action_space.py","src/rosetta_reality/vla/processor.py","src/rosetta_reality/sim/action_contract.py"}
    if "bank_manifest_sha256" in plan:
        required.update({"scripts/restore_paired_branch_v2.py", "scripts/check_q0_state_restore.py",
                         "src/rosetta_reality/eval/reproducibility_capture.py",
                         "src/rosetta_reality/sim/scripted_teacher_observer.py",
                         "src/rosetta_reality/sim/geometry_teacher.py"})
    if not required <= plan["source_sha256"].keys():raise ValueError("Evaluation identity inventory incomplete")
    resources=plan["resources"]
    for key in ("maximum_seconds","rss_bytes","evidence_bytes","cuda_allocated_bytes","cuda_reserved_bytes"):
        if type(resources.get(key)) is not int or resources[key]<=0:raise ValueError("Explicit positive evaluation budgets required")
    if resources["maximum_seconds"]>3600 or resources["rss_bytes"]>12*1024**3 or resources["cuda_allocated_bytes"]>8*1024**3 or resources["cuda_reserved_bytes"]>10*1024**3:
        raise ValueError("Evaluation exceeds registered implementation limits")
    for name,expected in plan["source_sha256"].items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(ROOT/relative)!=expected:
            raise ValueError("Evaluation source identity changed")
    if sha(artifact/"manifest.json")!=plan["artifact_manifest_sha256"]:
        raise ValueError("Actual endpoint artifact identity differs")
    manifest=read(artifact/"manifest.json")
    if manifest.get("selected_checkpoint_model_sha256")!=plan["model_sha256"] or manifest.get("hidden_test_loaded") is not False or manifest.get("reload",{}).get("exact_tensor_equality") is not True:
        raise ValueError("Independent full tensor reload proof required")
    for name,expected in manifest["files"].items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(artifact/relative)!=expected:
            raise ValueError("Endpoint payload changed")
    configuration,normalization=read(artifact/"config.json"),read(artifact/"normalization.json")
    if normalization.get("source_split")!="train" or normalization.get("hidden_test_loaded") is not False or sha(artifact/"normalization.json")!=plan["D_normalization_sha256"]:
        raise ValueError("Common D-only normalization identity differs")
    return configuration,normalization


def measure_bank(policy,contract,plan,bank_root,output,started):
    import torch
    from rosetta_reality.sim import GymAlohaEnvironment
    from rosetta_reality.eval.reproducibility_capture import PhysicsReader
    from restore_paired_branch_v2 import restore_branch
    from rosetta_reality.sim.scripted_teacher_observer import build_observation
    from rosetta_reality.vla.training.paired_mixture import save_bundle
    manifest=read(bank_root/"manifest.json")
    if sha(bank_root/"manifest.json")!=plan["bank_manifest_sha256"]:
        raise ValueError("Common recovery bank manifest changed")
    for item in manifest["files"]:
        path=Path(item["path"])
        if path.is_absolute() or ".." in path.parts or sha(bank_root/path)!=item["sha256"]:
            raise ValueError("Recovery bank payload changed")
    summary=read(bank_root/"summary.json")
    audit_path=Path(plan["bank_audit_report"])
    if audit_path.is_absolute() or ".." in audit_path.parts or sha(bank_root/audit_path)!=plan["bank_audit_sha256"]:
        raise ValueError("Independent bank audit identity changed")
    audit=read(bank_root/audit_path)
    if audit.get("passed") is not True or audit.get("manifest_sha256")!=plan["bank_manifest_sha256"] or audit.get("summary_sha256")!=sha(bank_root/"summary.json"):
        raise ValueError("Bank has no complete independent integrity/event audit")
    registration=read(bank_root/"registration.json")
    states=summary.get("bank",[])
    registered=registration["pairs"]
    if registration["phase"] != "bank" or registration["recovery_sustain_steps"] != 5:
        raise ValueError("Registered recovery-bank protocol required")
    if len(registered)>10 or len({p["pair_id"] for p in registered}) != len(registered):
        raise ValueError("Recovery bank exceeds registered unique-state budget")
    state_by_id={entry["pair_id"]:entry for entry in states}
    if len(state_by_id) != len(states) or not state_by_id.keys() <= {p["pair_id"] for p in registered}:
        raise ValueError("Recovery bank denominator or identity mismatch")
    rows=[]
    def budget():
        limits=plan["resources"]
        rss=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
        if (time.monotonic()-started>=limits["maximum_seconds"] or rss>limits["rss_bytes"] or torch.cuda.max_memory_allocated()>limits["cuda_allocated_bytes"] or torch.cuda.max_memory_reserved()>limits["cuda_reserved_bytes"] or sum(p.stat().st_size for p in output.parent.rglob("*") if p.is_file())>limits["evidence_bytes"]):
            raise RuntimeError("Registered recovery evaluation resource cap")
    for registered_state in registered:
        budget()
        entry=state_by_id.get(registered_state["pair_id"], {"pair_id":registered_state["pair_id"],"feasible":False})
        tracker=RecoveryEventTracker(registered_state["trigger"])
        record={"pair_id":entry["pair_id"],"branch_task_success":None,"measured_steps":0,"teacher_feasible":entry["feasible"],"recovery_event":None}
        if not entry.get("branch_state"):
            record["status"]="unmeasured_branch_not_created"
            rows.append(record)
            continue
        if time.monotonic()-started>=plan["resources"]["maximum_seconds"]:
            raise TimeoutError("Shared evaluation work deadline")
        env=GymAlohaEnvironment(contract,maximum_episode_steps=500)
        try:
            env.reset(seed=registered_state["seed"])
            directory=(bank_root/entry["branch_state"]).parent
            if not directory.resolve().is_relative_to(bank_root.resolve()):raise ValueError("Branch escaped bank")
            restored=restore_branch(env,PhysicsReader(env),directory)
            observation=restored["observation"]
            policy.configure_noise("seeded_standard_normal",registered_state["seed"])
            success=False
            record["status"]="timeout"
            with (output/(entry["pair_id"]+".jsonl")).open("x") as stream:
                for step in range(150):
                    budget()
                    if time.monotonic()-started>=plan["resources"]["maximum_seconds"]:raise TimeoutError("Shared evaluation work deadline")
                    if any(not torch.isfinite(v).all() for v in [observation["robot_state"],*observation["images"].values()]):raise ValueError("Nonfinite restored/current observation")
                    if env.state_limit_violation_count() or env.unexpected_collision_count():
                        record["status"]="physical_safety_stop"
                        break
                    raw,projected=policy.predict(observation,plan["instruction"])
                    budget()
                    for chunk in (raw,projected):
                        contract.validate_tensor(chunk,allow_chunk=True)
                        if not torch.isfinite(chunk).all():raise ValueError("Nonfinite recovery action")
                    action,clip=contract.clip(raw[0])
                    observation,reward,done,info=env.step(action)
                    if not math_isfinite(reward) or bool(env.last_clip_mask.any()):raise ValueError("Invalid reward or adapter clipping")
                    violation=env.state_limit_violation_count()
                    contacts=[list(pair) for pair in env.contact_pairs() if env.is_unexpected_collision_pair(*pair)]
                    observed_geometry=build_observation(env,contract)
                    geometry={key: bool(getattr(observed_geometry,key)) for key in
                              ("socket_grasp_contact","peg_grasp_contact","peg_socket_contact")}
                    event=tracker.observe(step+1,geometry,safe=not (violation or contacts),task_success=bool(info.get("is_success",False)))
                    success=success or bool(info.get("is_success",False))
                    record["measured_steps"]=step+1
                    stream.write(json.dumps({"step":step+1,"executed_action":action.tolist(),"state_after":observation["robot_state"].tolist(),"reward":reward,"task_success":bool(info.get("is_success",False)),"joint_limit_violations":violation,"unexpected_contacts":contacts,"raw_clips":int(clip.sum()),"geometry":geometry,"recovery_event":event},allow_nan=False)+"\n")
                    stream.flush()
                    budget()
                    if violation or contacts:
                        record["status"]="physical_safety_stop";break
                    if done:
                        record["status"]="task_success" if success else "task_failed";break
            record["branch_task_success"]=success
            record["recovery_event"]=tracker.result() if record["measured_steps"] else None
            budget()
        finally:env.close()
        rows.append(record)
    save_bundle(output/"bank-summary.json",{"registered_states":len(registered),"measured_states":sum(row["measured_steps"]>0 for row in rows),"rows":rows,"branch_task_success_count":sum(row["branch_task_success"] is True for row in rows),"metric":"task_success_from_exact_registered_deviation_state","geometric_recovery_count":sum(bool((row["recovery_event"] or {}).get("geometric_recovery")) for row in rows),"verified_recovery_chain_count":sum(bool((row["recovery_event"] or {}).get("verified_recovery_chain")) for row in rows),"geometric_recovery_event":"registered_five_consecutive_safe_correction_steps","optimizer_updates":0,"shutdown_requested":False})


def main():
    p=argparse.ArgumentParser()
    for name in ("plan","artifact","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--bank",type=Path)
    args=p.parse_args()
    plan=read(args.plan)
    validate_bank_selection(plan,args.bank)
    configuration,normalization=verify(plan,args.artifact)
    if os.environ.get("ROSETTA_TORCH_DEVICE")!="cuda" or any(os.environ.get(k)!="1" for k in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE")):
        raise ValueError("Registered offline AutoDL CUDA worker required")
    import torch
    import smolvla_autodl_vfunfreeze_sim_gate as reference
    from rosetta_reality.sim import load_action_contract
    from diagnose_smolvla_base_five_scene import make_episode_runner,parameter_digest
    from rosetta_reality.vla.training.paired_mixture import save_bundle
    contract=load_action_contract(ROOT/"configs/sim/aloha_insertion_smolvla.yaml")
    if sha(ROOT/"configs/sim/aloha_insertion_smolvla.yaml")!=plan["action_contract_sha256"] or read(args.artifact/"action_contract.json")!=json.loads(json.dumps(asdict(contract))):
        raise ValueError("Physical action contract changed")
    args.output.mkdir(parents=True,exist_ok=False)
    save_bundle(args.output/"registration.json",plan)
    reference._ACTIVE_SIM_PLAN=args.plan
    torch.cuda.set_per_process_memory_fraction(.4)
    started=time.monotonic()
    policy=reference._build_online_cuda_class()(args.artifact,configuration,normalization,contract)
    before=parameter_digest(policy.policy)
    torch.cuda.reset_peak_memory_stats()
    runner=make_episode_runner(policy,contract,plan,args.output,started)
    episodes=[]
    error=None
    try:
        for seed in plan["seeds"]:
            episode=runner(seed,args.output/f"seed-{seed}.jsonl")
            save_bundle(args.output/f"seed-{seed}.json",episode)
            episodes.append(episode)
    except Exception as exc:
        error=type(exc).__name__
    if args.bank is not None and error is None:
        bank_output=args.output/"recovery-bank"
        bank_output.mkdir(exist_ok=False)
        try:measure_bank(policy,contract,plan,args.bank,bank_output,started)
        except Exception as exc:error=type(exc).__name__
    after=parameter_digest(policy.policy)
    try:runner.check_budget()
    except Exception as exc:error=error or type(exc).__name__
    save_bundle(args.output/"parameter-integrity.json",{"before":before,"after":after,"unchanged":before==after,"optimizer_updates":0})
    save_bundle(args.output/"summary.json",{"schema":"rosetta.paired_policy_measurement.v1","status":"completed" if error is None and before==after else "incomplete","arm":plan["arm"],"episodes":episodes,"registered_scenes":5,"measured_scenes":len(episodes),"success_count":sum(e["success"] for e in episodes),"unmeasured_seeds":[s for s in plan["seeds"] if s not in [e["seed"] for e in episodes]],"error_type":error,"formal_gate_result":None,"original_gate4_protocol_equivalent":False,"shutdown_requested":False,"optimizer_updates":0,"plan_sha256":sha(args.plan),"elapsed_seconds":time.monotonic()-started})
    return 0 if error is None and before==after else 1


if __name__=="__main__":raise SystemExit(main())
