"""Seal three independent fresh-base native-v2 plans and their exposure tables."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rosetta_reality.vla.training.paired_mixture import build_schedules, save_bundle, verify_matched_cohort


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    import yaml
    from rosetta_reality.vla.training.features import FEATURE_FACTORIES
    from rosetta_reality.vla.training.plan import load_v2_plan, validate_plan_structure
    from rosetta_reality.vla.training.integrity import CORE_IMPLEMENTATION, validate_local_implementation
    from rosetta_reality.vla import load_smolvla_experiment

    template = load_v2_plan(Path(args.template), ROOT)
    if template.get("initialization",{}).get("source") != "revision_pinned_base_model" or template.get("initialization",{}).get("optimizer_state_reused") is not False:
        raise ValueError("Independent fresh pinned-base initialization is required")
    if any(key in template for key in ("loss_contract", "image_key_scene_contract", "state_robustness_contract", "visual_conditioning_contract", "vision_front_end_contract")):
        raise ValueError("Template must preserve native uniform loss and frozen VLM")
    original = json.loads(Path(args.original).read_text())
    cohort = json.loads(Path(args.cohort).read_text())
    if original.get("hidden_test_loaded") is not False or original.get("source_split") != "train":
        raise ValueError("Original inventory must be audited D train only")
    parent = template["parent_experiment"]
    parent_path = ROOT / parent["config"]
    if sha(parent_path) != parent["sha256"]:
        raise ValueError("Parent experiment identity changed")
    experiment = load_smolvla_experiment(parent_path, ROOT)
    parent_episodes = list(experiment["dataset"]["train_episodes"])
    inventory_episodes = {row["episode"] for row in original["inputs"]}
    if inventory_episodes != set(parent_episodes):
        raise ValueError("D inventory must cover exactly every parent train episode")
    research_scope = cohort.get("research_scope", "strict_scene_disjoint_v1")
    if research_scope == "fresh_lineage_fixed_development_exploration_v1":
        if cohort.get("N_strict") != 0 or cohort.get("legacy_object_scene_nonoverlap") != "not_established":
            raise ValueError("Exploratory cohort must disclose unestablished legacy overlap")
        if cohort.get("N_exploratory") != len(cohort.get("pairs", [])):
            raise ValueError("Exploratory denominator differs from admitted pair inventory")
    N = verify_matched_cohort(cohort["pairs"], research_scope=research_scope) if cohort.get("pairs") else 0
    schedules = build_schedules(original["inputs"], cohort.get("R_inputs") if N else None, cohort.get("S_inputs") if N else None)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    saved = []
    excluded = {"fixed_frame_sampler", "explicit_sample_schedule", "horizon_weight_profile", "image_key_scene_regularization", "state_robustness_jitter", "state_conditioning_dropout", "vision_front_end_unfreeze", "dual_axis_diagnostics"}
    for arm in "ABC":
        save_bundle(output / f"schedule-{arm}.json", schedules[arm])
        if schedules[arm]["status"] != "sealed":
            continue
        for phase, steps in (("smoke", 2), ("train", 5000)):
            plan = copy.deepcopy(template)
            plan["plan_id"] = f"paired-mixture-{arm.lower()}-{phase}-"+getattr(args,"identity","20260927-001")
            plan["run_name"] = plan["plan_id"]
            plan["arm"] = arm
            plan["scope"] = "paired_recovery_data_mixture"
            plan["hypothesis"] = "Compare fixed-budget D, D+verified recovery R, and D+ordinary success S under the same pinned-base initialization, D-only statistics and native uniform objective."
            plan["formal_training_authorized"] = phase == "train"
            plan["authorization"] = "explicit_user_three_experiments_and_disclosed_fresh_lineage_exploration_20260927"
            plan["paired_cuda_budget"] = {"maximum_allocated_bytes":12*1024**3,"maximum_reserved_bytes":12*1024**3}
            plan["stop_conditions"] = ["identity or split mismatch","nonfinite loss, gradient or output",
                "CUDA allocation or reservation above registered 12 GiB","host RSS above outer registered bound",
                "free disk below outer registered minimum","registered per-stage wall deadline",
                "failed no-optimizer benchmark, two-step smoke or independent exact smoke reload"]
            plan["research_scope"] = research_scope
            if research_scope == "fresh_lineage_fixed_development_exploration_v1":
                plan["legacy_object_scene_nonoverlap"] = "not_established"
                plan["N_strict"] = 0
                plan["N_exploratory"] = N
            active = "optimizer_smoke" if phase == "smoke" else "training"
            plan[active].update(batch_size=4, steps=steps, run_name=plan["run_name"])
            plan[active].update(episodes=list(parent_episodes), num_workers=0, persistent_workers=False)
            plan["training"]["episodes"] = list(parent_episodes)
            plan["training"].update(batch_size=4,steps=5000,checkpoint_steps=[1250,2500,3750,5000],save_freq=1250,log_freq=50,num_workers=0,persistent_workers=False)
            plan["training"]["optimizer"] = {"type":"adamw", "lr":1e-4, "betas":[0.9,0.95], "eps":1e-8, "weight_decay":1e-10, "grad_clip_norm":10}
            plan["training"]["scheduler"] = {"type":"cosine_decay_with_warmup", "num_warmup_steps":16, "num_decay_steps":5000, "peak_lr":1e-4,"decay_lr":2.5e-6}
            plan["optimizer_smoke"].update(steps=2,batch_size=4,save_freq=2,log_freq=1,
                save_checkpoint=True,episodes=list(parent_episodes),num_workers=0,persistent_workers=False)
            # Native-v2 deliberately has one shared optimizer contract. Phase
            # overlays are rejected by the original schema and never reach CLI.
            for key in ("optimizer","scheduler"):
                plan["optimizer_smoke"].pop(key,None)
            plan["smoke_optimizer_configuration_source"] = "training.optimizer_and_scheduler"
            if phase == "smoke":
                plan[active].update(save_freq=2, log_freq=1, save_checkpoint=True)
            if phase == "train":
                plan[active].update(checkpoint_steps=[1250,2500,3750,5000], save_freq=1250, log_freq=50)
                plan[active]["optimizer"] = {"type":"adamw", "lr":1e-4, "betas":[0.9,0.95], "eps":1e-8, "weight_decay":1e-10, "grad_clip_norm":10}
                plan[active]["scheduler"] = {"type":"cosine_decay_with_warmup", "num_warmup_steps":16, "num_decay_steps":5000,"peak_lr":1e-4, "decay_lr":2.5e-6}
            schedule = copy.deepcopy(schedules[arm])
            schedule["entries"] = schedule["entries"][:steps*4]
            from rosetta_reality.vla.training.paired_mixture import digest
            schedule.update(updates=steps, entries_sha256=digest(schedule["entries"]), common_D_exposures=steps*3, auxiliary_exposures=steps,total_exposures=steps*4)
            schedule_name = f"schedule-{arm}-{phase}.json"
            save_bundle(output / schedule_name, schedule)
            inventory = {"hidden_test_loaded":False,"normalization_source":"D_train_only","inputs":cohort.get("runtime_inputs",[]) if arm != "A" else []}
            inventory_name = f"auxiliary-{arm}.json"
            if not (output / inventory_name).exists():
                save_bundle(output / inventory_name, inventory)
            plan["features"] = [x for x in plan["features"] if x["name"] not in excluded]
            plan["features"].append({"name":"paired_mixture","schedule":schedule_name,"schedule_sha256":sha(output/schedule_name),"auxiliary_root":args.auxiliary_root,"auxiliary_inventory":inventory_name,"auxiliary_inventory_sha256":sha(output/inventory_name),"camera_map":cohort.get("camera_map",{})})
            if plan.get("monitoring"):
                plan["monitoring"]["wake_steps"] = [1250,2500,3750,5000]
            # Reseal only new plans against current implementation. Historical pins stay intact.
            for name in plan["implementation_files"]:
                plan["implementation_files"][name] = sha(ROOT / name)
            for name in ("src/rosetta_reality/vla/training/paired_mixture.py","src/rosetta_reality/vla/training/paired_mixture_runtime.py","scripts/prepare_paired_training_bundle.py","scripts/run_paired_training_v2.py","src/rosetta_reality/diagnostics/d_train_native_view.py"):
                plan["implementation_files"][name] = sha(ROOT / name)
            for name in CORE_IMPLEMENTATION:
                plan["implementation_files"][name] = sha(ROOT / name)
            for path in (ROOT / "src/rosetta_reality/vla/training").glob("*.py"):
                plan["implementation_files"][path.relative_to(ROOT).as_posix()] = sha(path)
            validate_plan_structure(plan, known_features=FEATURE_FACTORIES)
            validate_local_implementation(plan, ROOT)
            path = output / f"plan-{arm}-{phase}.yaml"
            with path.open("x") as stream:
                yaml.safe_dump(plan, stream, sort_keys=False)
            saved.append({"arm":arm,"phase":phase,"path":path.name,"sha256":sha(path)})
    save_bundle(output / "bundle.json", {"status":"prepared_unexecuted","N":N,"research_scope":research_scope,"N_strict":0 if research_scope == "fresh_lineage_fixed_development_exploration_v1" else N,"N_exploratory":N if research_scope == "fresh_lineage_fixed_development_exploration_v1" else 0,"legacy_object_scene_nonoverlap":"not_established" if research_scope == "fresh_lineage_fixed_development_exploration_v1" else "verified","normalization":"D_train_only","initialization":"independent_fresh_pinned_base","fixed_final_checkpoint":5000,"native_loss_changed":False,"plans":saved,"unavailable_arms":[a for a in "ABC" if schedules[a]["status"]!="sealed"],"source_original_sha256":sha(args.original),"source_cohort_sha256":sha(args.cohort)})


def main():
    p = argparse.ArgumentParser()
    for name in ("template","original","cohort","output","auxiliary-root"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--identity",default="20260927-001")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
