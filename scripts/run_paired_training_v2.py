"""Explicit native-v2 launcher for an independently accepted self-contained D view.

The legacy launcher and validation reports remain unchanged. Benchmark and
smoke use a fresh base and never become scientific task-success evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def save(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())


def verified(path,expected):
    if sha(path)!=expected:raise ValueError("Registered paired evidence identity changed")
    return json.loads(Path(path).read_text())


def validate_engineering_evidence(record,kind,expected):
    if (record.get("status")!="completed" or record.get("stage")!=kind
            or any(record.get(key)!=value for key,value in expected.items())
            or record.get("native_loss_changed") is not False
            or record.get("hidden_test_loaded") is not False):
        raise ValueError("Actual paired engineering prerequisite is incompatible")
    if kind=="paired_native_optimizer_smoke" and record.get("successful_optimizer_updates")!=2:
        raise ValueError("Two successful native smoke updates required")
    if kind=="paired_native_no_optimizer_benchmark" and (
            record.get("optimizer_updates")!=0 or record.get("gradients_enabled") is not False
            or set(record.get("batch_sizes_measured",[]))!={1,4}):
        raise ValueError("Real batch1/batch4 no-optimizer benchmark required")
    if kind=="paired_native_smoke_reload" and (record.get("optimizer_updates")!=0
            or record.get("serialized_parameters_exact") is not True
            or record.get("gradients_enabled") is not False):
        raise ValueError("Independent exact checkpoint reload required")


def engineering_semantics(plan,plan_path,experiment):
    from rosetta_reality.vla.training.paired_mixture import digest
    feature=next(f for f in plan["features"] if f["name"]=="paired_mixture")
    root=Path(plan_path).parent
    def relative(raw):
        part=Path(raw)
        if part.is_absolute() or ".." in part.parts:raise ValueError("Unsafe engineering identity input")
        return root/part
    schedule=verified(relative(feature["schedule"]),feature["schedule_sha256"])
    verified(relative(feature["auxiliary_inventory"]),feature["auxiliary_inventory_sha256"])
    prefix=schedule["entries"][:8]
    if len(prefix)!=8:raise ValueError("Two full engineering batches required")
    arm=plan.get("arm")
    auxiliary={"A":"D","B":"R","C":"S"}.get(arm)
    if auxiliary is None or any(r["source"]!=(auxiliary if i%4==3 else "D") for i,r in enumerate(prefix)):
        raise ValueError("Registered arm differs from engineering batch mixture")
    if (plan.get("smoke_optimizer_configuration_source")!="training.optimizer_and_scheduler"
            or any(k in plan["optimizer_smoke"] for k in ("optimizer","scheduler"))):
        raise ValueError("Smoke and formal optimizer/scheduler semantics differ")
    return {"arm":arm,"engineering_schedule_prefix_sha256":digest(prefix),
            "auxiliary_inventory_sha256":feature["auxiliary_inventory_sha256"],
            "implementation_identity_sha256":digest(plan["implementation_files"]),
            "optimizer_scheduler_sha256":digest({k:plan["training"][k] for k in ("optimizer","scheduler")}),
            "cuda_budget_sha256":digest(plan["paired_cuda_budget"]),
            "initialization_semantics_sha256":digest({"initialization":plan["initialization"],"seed":experiment["seed"],"policy":plan["training"]["policy"]})}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("benchmark","smoke","reload","train"))
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--view-root",type=Path,required=True)
    p.add_argument("--view-manifest-sha256",required=True)
    p.add_argument("--native-read-sha256",required=True)
    p.add_argument("--normalization-report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    for key in ("benchmark-evidence","smoke-evidence","smoke-reload-evidence"):
        p.add_argument("--"+key,type=Path)
        p.add_argument("--"+key+"-sha256")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--checkpoint-model-sha256")
    p.add_argument("--expected-final-parameters",type=Path)
    p.add_argument("--expected-final-parameters-sha256")
    args=p.parse_args()
    if os.environ.get("HF_HUB_OFFLINE")!="1" or os.environ.get("TRANSFORMERS_OFFLINE")!="1":
        raise RuntimeError("Native paired execution requires the registered offline worker")
    import run_smolvla_v2 as legacy
    import run_smolvla_phase as phases
    from rosetta_reality.vla.training.integrity import validate_local_implementation
    from rosetta_reality.vla.training.launch import build_training_arguments,compose_runtime_experiment
    from rosetta_reality.diagnostics.d_train_native_view import install_native_D_factory,NORMALIZATION_SHA
    plan_path=args.plan.resolve()
    plan,base_path,experiment=legacy._resolve_plan(plan_path)
    validate_local_implementation(plan,ROOT)
    legacy._validate_split(plan,experiment,"train" if args.mode=="train" else "smoke")
    if (plan.get("scope")!="paired_recovery_data_mixture"
            or plan["initialization"].get("source")!="revision_pinned_base_model"
            or plan["initialization"].get("optimizer_state_reused") is not False):
        raise ValueError("Registered fresh-base paired initialization required")
    if args.mode=="train" and plan.get("formal_training_authorized") is not True:
        raise PermissionError("This new plan has no formal training authorization")
    if not {"train_only_statistics","action_boundary_projection","gradient_clip_diagnostics","paired_mixture"} <= {f["name"] for f in plan["features"]}:
        raise ValueError("Native paired statistical/action/numeric safety features are mandatory")
    resources=plan["resources"]
    for env,key in (("ROSETTA_DOCKER_MEMORY_LIMIT","memory_limit"),("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT","memory_swap_limit")):
        if os.environ.get(env)!=resources[key]:raise ValueError("Registered container resource boundary differs")
    device=os.environ.get("ROSETTA_TORCH_DEVICE")
    if device!="cuda":raise ValueError("Paired model execution requires the registered CUDA worker")
    contract=ROOT/experiment["action_contract"]["derived"]
    prerequisites=legacy._validate_prerequisites(plan,experiment,base_path,sha(contract))
    normalization=verified(args.normalization_report,NORMALIZATION_SHA)
    if (plan["normalization"]["report_sha256"]!=NORMALIZATION_SHA
            or normalization["source_split"]!="train" or normalization["hidden_test_loaded"] is not False
            or normalization["validation_episodes_loaded"] is not False
            or normalization["train_episodes"]!=experiment["dataset"]["train_episodes"]):
        raise ValueError("The same original D-only normalization audit is required")
    view=args.view_root.resolve()
    proof=verified(view/"native-read-verification.json",args.native_read_sha256)
    verified(view/"view-manifest.json",args.view_manifest_sha256)
    if proof["view_manifest_sha256"]!=args.view_manifest_sha256:
        raise ValueError("Native reader proof does not bind this view")
    if (proof.get("status")!="passed" or proof.get("native_read_frames")!=20000
            or proof.get("normalization_sha256")!=NORMALIZATION_SHA
            or proof.get("label_parity")!="all_selected_states_and_chunk_actions_exact"
            or proof.get("pixel_parity")!="all_selected_frames_exact"
            or proof.get("source_mixed_payload_opened") is not False):
        raise ValueError("Full D40 native state/action/pixel proof required")
    identity={"view_manifest_sha256":args.view_manifest_sha256,"native_read_sha256":args.native_read_sha256,
              "normalization_sha256":NORMALIZATION_SHA,"model_revision":experiment["model"]["revision"],
              "action_contract_sha256":sha(contract),**engineering_semantics(plan,plan_path,experiment)}
    if args.mode=="train":
        for label,kind in (("benchmark","paired_native_no_optimizer_benchmark"),("smoke","paired_native_optimizer_smoke"),("smoke_reload","paired_native_smoke_reload")):
            path=getattr(args,label+"_evidence");expected=getattr(args,label+"_evidence_sha256")
            if path is None or expected is None:raise ValueError("Training requires bound actual benchmark and smoke evidence")
            validate_engineering_evidence(verified(path,expected),kind,identity)
    args.output.mkdir(parents=True,exist_ok=False)
    run_name=plan["run_name"]+("-"+args.mode if args.mode in {"benchmark","reload"} else "")
    checkpoint_root=phases._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    checkpoint=checkpoint_root/experiment["experiment_id"]/args.mode/run_name
    if checkpoint.exists():raise FileExistsError("Paired checkpoint output must be create-only")
    model_root=phases._model_root(experiment)
    expected_final=None
    if args.mode=="reload":
        if args.checkpoint is None or args.checkpoint_model_sha256 is None or args.expected_final_parameters is None or args.expected_final_parameters_sha256 is None:
            raise ValueError("Reload requires registered serialized checkpoint and actual final tensor evidence")
        if sha(args.checkpoint/"model.safetensors")!=args.checkpoint_model_sha256:
            raise ValueError("Serialized checkpoint model identity changed")
        expected_final=verified(args.expected_final_parameters,args.expected_final_parameters_sha256)
        if expected_final["successful_optimizer_updates"]!=2:
            raise ValueError("Reload source is not the registered two-step smoke")
        model_root=args.checkpoint.resolve()
    # Keep the old normalization report untouched; the context consumes a
    # create-only byte-identical copy under this new source workspace.
    copied=args.output/"D-normalization.json"
    with copied.open("xb") as stream:stream.write(args.normalization_report.read_bytes())
    runtime_plan=json.loads(json.dumps(plan))
    # The original declaration remains historical provenance. The typed new
    # context receives this byte-identical, directly SHA-checked report.
    runtime_plan["normalization_report_actual_sha256"]=sha(copied)
    runtime_plan["native_train_view"]={"root":str(view),**identity,"historical_view_manifest_reused":False}
    import yaml
    runtime_plan_path=args.output/"runtime-plan.yaml"
    with runtime_plan_path.open("x") as stream:yaml.safe_dump(runtime_plan,stream,sort_keys=False)
    runtime=compose_runtime_experiment(experiment,runtime_plan)
    save(args.output/"runtime-experiment.json",runtime)
    active_mode="train" if args.mode=="train" else "smoke"
    arguments=build_training_arguments(plan=runtime_plan,experiment=experiment,mode=active_mode,
        run_name=run_name,model_root=model_root,dataset_root=view,output_dir=checkpoint,device=device)
    save(args.output/"launch.json",{"stage":"paired_native_launch","mode":args.mode,**identity,
        "plan_sha256":sha(plan_path),"runtime_plan_sha256":sha(runtime_plan_path),"argv":arguments,
        "prerequisite_sha256":{k:sha(v) for k,v in prerequisites.items()},
        "legacy_base_validation_reference_only":True,"fixed_final_step":5000,"checkpoint_selection":False,
        "native_loss_changed":False,"hidden_test_loaded":False,"shutdown_requested":False})
    os.environ.update(ROSETTA_VLA_EXPERIMENT_CONFIG=str(base_path),ROSETTA_VLA_RUN_NAME=run_name,
        ROSETTA_VLA_V2_PLAN_PATH=str(runtime_plan_path.resolve()),ROSETTA_VLA_V2_LAUNCHER_VALIDATED="1",
        ROSETTA_VLA_PHASE="formal" if args.mode=="train" else "smoke",
        ROSETTA_VLA_TRAIN_STATS_REPORT=str(copied.resolve()),
        ROSETTA_VLA_NORMALIZATION_SHA256=NORMALIZATION_SHA,
        ROSETTA_VLA_FORMAL_PLAN_SHA256=sha(runtime_plan_path))
    import lerobot.scripts.lerobot_train as native
    restore=install_native_D_factory(native,view,view_manifest_sha256=args.view_manifest_sha256,
        native_read_sha256=args.native_read_sha256,delta_timestamps={"action":[i/50 for i in range(50)]})
    started=time.monotonic()
    try:
        sys.argv=["lerobot-train",*arguments]
        if args.mode in {"benchmark","reload"}:
            benchmark(runtime_plan,plan_path,experiment,base_path,copied,run_name,args.output,identity,
                expected_final=expected_final,checkpoint_model_sha256=args.checkpoint_model_sha256)
        else:
            run_native_training(runtime_plan,plan_path,experiment,base_path,copied,run_name,args.mode)
            updates=json.loads((plan_path.parent/(run_name+"-actual-updates.json")).read_text())
            expected=5000 if args.mode=="train" else 2
            if updates["status"]!="complete" or updates["successful_optimizer_updates"]!=expected:
                raise ValueError("Actual successful optimizer count differs from registration")
            initialized=json.loads((plan_path.parent/(run_name+"-actual-initialization.json")).read_text())
            save(args.output/"completion.json",{"status":"completed","stage":"paired_native_optimizer_smoke" if args.mode=="smoke" else "paired_native_formal_training",**identity,
                "plan_sha256":sha(plan_path),"successful_optimizer_updates":expected,
                "initialization_identity_sha256":initialized["identity_sha256"],
                "optimizer_state_reused":False,"native_loss_changed":False,"hidden_test_loaded":False,
                "checkpoint_root":str(checkpoint),"checkpoint_selection":False,"elapsed_seconds":time.monotonic()-started,"shutdown_requested":False})
    finally:restore()
    return 0


def run_native_training(plan,plan_path,experiment,base_path,normalization,run_name,mode):
    import lerobot.scripts.lerobot_train as native
    from rosetta_reality.vla.training.context import TrainingContext
    from rosetta_reality.vla import load_smolvla_action_space
    from rosetta_reality.vla.training.features import feature_stack_from_plan
    from rosetta_reality.tracking.trackio_lerobot import finish_trackio
    context=TrainingContext(plan,experiment,load_smolvla_action_space(experiment,require_explicit=True),
        plan_path,base_path,ROOT/experiment["action_contract"]["derived"],normalization,
        "formal" if mode=="train" else "smoke","cuda",run_name)
    stack=feature_stack_from_plan(plan)
    stack.install_all(context)
    try:native.main()
    finally:
        try:stack.restore_all(context)
        finally:
            if "trackio_logging" in stack.installed or any(f.name=="trackio_logging" for f in stack.features):
                finish_trackio()


def benchmark(plan,plan_path,experiment,base_path,normalization,run_name,output,identity,*,expected_final=None,checkpoint_model_sha256=None):
    import torch
    import smolvla_forward_check as forward
    import lerobot.scripts.lerobot_train as native
    from rosetta_reality.vla.training.context import TrainingContext
    from rosetta_reality.vla import load_smolvla_action_space
    from rosetta_reality.vla.training.features import feature_stack_from_plan
    context=TrainingContext(plan,experiment,load_smolvla_action_space(experiment,require_explicit=True),
        plan_path,base_path,ROOT/experiment["action_contract"]["derived"],normalization,
        "performance_benchmark","cuda",run_name)
    stack=feature_stack_from_plan(plan)
    stack.install_all(context)
    try:
        cfg=forward._parse_config()
        cfg.validate()
        if cfg.policy is None or cfg.policy.type!="smolvla" or cfg.policy.device!="cuda" or cfg.policy.load_vlm_weights:
            raise ValueError("Validated native SmolVLA CUDA policy must use the combined pinned checkpoint")
        torch.manual_seed(experiment["seed"])
        dataset,_=native.make_train_eval_datasets(cfg)
        policy=native.make_policy(cfg=cfg.policy,ds_meta=dataset.meta,rename_map=cfg.rename_map)
        if expected_final is not None:
            actual=json.loads((plan_path.parent/(run_name+"-actual-initialization.json")).read_text())
            if actual["tensors"]!=expected_final["tensors"] or actual["identity_sha256"]!=expected_final["identity_sha256"]:
                raise ValueError("Independent reload tensors differ from actual saved policy parameters")
        pre,post=forward._make_processors(cfg,policy,dataset,torch.device("cuda"))
        from rosetta_reality.sim import load_action_contract
        from rosetta_reality.vla.processor import ensure_smolvla_action_boundary
        ensure_smolvla_action_boundary(pre,post,load_action_contract(context.contract_path),context.action_space,
            action_contract_sha256=identity["action_contract_sha256"],upstream_revision=experiment["upstream"]["revision"])
        loader=native.torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=0,shuffle=False)
        batch=next(iter(loader));records=[]
        torch.cuda.reset_peak_memory_stats()
        for size in (1,4):
            selected={k:v[:size] if torch.is_tensor(v) or isinstance(v,list) else v for k,v in batch.items()}
            processed=pre(selected)
            for repeat in range(2):
                torch.cuda.synchronize();start=time.monotonic()
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):loss,_=policy.forward(processed)
                torch.cuda.synchronize();elapsed=time.monotonic()-start
                if not torch.isfinite(loss).all():raise FloatingPointError("Nonfinite benchmark loss")
                limits=plan["paired_cuda_budget"]
                if (torch.cuda.max_memory_allocated()>limits["maximum_allocated_bytes"]
                        or torch.cuda.max_memory_reserved()>limits["maximum_reserved_bytes"]):
                    raise MemoryError("No-optimizer CUDA peak exceeded frozen paired budget")
                records.append({"batch_size":size,"repeat":repeat,"forward_seconds":elapsed,"loss":float(loss)})
        save(output/"completion.json",{"status":"completed","stage":"paired_native_smoke_reload" if expected_final is not None else "paired_native_no_optimizer_benchmark",**identity,
            "batch_sizes_measured":[1,4],"records":records,"model_forwards":4,"optimizer_updates":0,"gradients_enabled":False,
            "serialized_parameters_exact":expected_final is not None,"checkpoint_model_sha256":checkpoint_model_sha256,
            "cuda_peak_allocated":torch.cuda.max_memory_allocated(),"cuda_peak_reserved":torch.cuda.max_memory_reserved(),
            "native_loss_changed":False,"hidden_test_loaded":False,"shutdown_requested":False})
    finally:
        # This direct no-optimizer path never constructs the native logging
        # class. Its durable completion records metrics; no Trackio run exists
        # to finish. Native optimizer training retains its logger lifecycle.
        stack.restore_all(context)


if __name__=="__main__":raise SystemExit(main())
