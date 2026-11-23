"""Real Q2 NPZ/native D40 first batches through saved CPU processors, zero weights.

No policy construction, optimizer, CUDA work, dataset download or physics step.
The independent normalization calculation does not call the saved normalizer.
"""
from __future__ import annotations
import argparse
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
VIEW_SHA="019e63cf967c144cc283b40c0cb99f5599c6b2581e5cd54df9f7f37aded656fc"
READ_SHA="39bdb7633bed288796af945050c2e10ff0aad8c6306d0d7866a076346180ec0b"
NORM_SHA="263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988"
CONFIG_SHA="cca7e39c1136a18f3bd93514f446af56f93e3bd5d05b71a9a1cddcfac684348c"
Q2_MANIFEST_SHA="5d44dc89a572f0247b52d24acc3beab997864dbda82d8e3c9195d04b812a5a30"
Q2_AUDIT_SHA="07e92b50a6342737c015c18cd042a74228da02fc84de0a45365f40fa22be5fd4"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def require(condition,reason):
    if not condition:raise ValueError(reason)


def verify_chunk_rows(rows,start):
    require(len(rows)==50,"complete native fifty targets required")
    require([r["step"] for r in rows]==list(range(start,start+50)),"target continuity")
    require(all(r["safe"] is True and r["positive_target"] is True and r["action_source"]=="state_conditioned_teacher" for r in rows),
            "chunk crosses deliberate/unsafe/unverified target")


def arithmetic_normalize(value,stats,mode,eps):
    """Torch tensor arithmetic directly from D audit statistics, no processor calls."""
    import torch
    if mode=="IDENTITY":return value.clone()
    def tensor(key):return torch.as_tensor(stats[key],dtype=value.dtype,device="cpu")
    if mode=="MEAN_STD":return (value-tensor("mean"))/(tensor("std")+eps)
    if mode=="MIN_MAX":return 2*(value-tensor("min"))/(tensor("max")-tensor("min")+eps)-1
    raise ValueError("unregistered normalization mode "+mode)


def run(args):
    if os.environ.get("HF_HUB_OFFLINE")!="1" or os.environ.get("TRANSFORMERS_OFFLINE")!="1":
        raise RuntimeError("offline CPU worker required")
    # Refuse all policy-weight file opens; processor state safetensors are metadata.
    original_open=builtins.open
    def no_weights(file,*a,**kw):
        if isinstance(file,(str,os.PathLike)) and Path(file).name in {"model.safetensors","pytorch_model.bin","model.safetensors.index.json"}:
            raise RuntimeError("policy weights forbidden in CPU input check")
        return original_open(file,*a,**kw)
    builtins.open=no_weights
    started=time.monotonic()
    import numpy as np
    import torch
    import yaml
    import safetensors.torch as safe_torch
    original_tensor_load=safe_torch.load_file
    def processor_state_only(filename,*a,**kw):
        require("processor" in Path(filename).name,"nonprocessor safetensors forbidden")
        return original_tensor_load(filename,*a,**kw)
    safe_torch.load_file=processor_state_only
    torch.load=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError("Torch policy/optimizer deserialization forbidden"))
    from rosetta_reality.diagnostics.d_train_native_view import accepted_training_view,logical_native_dataset,TRAIN
    from rosetta_reality.vla.training.paired_mixture_runtime import PairedMixtureFeature
    from rosetta_reality.vla.processor import (ensure_smolvla_action_boundary,standard_aloha_state_to_pi,
        standard_aloha_action_to_bounded_sine)
    from rosetta_reality.vla.action_space import SmolVLAActionSpace
    from rosetta_reality.sim import load_action_contract
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    import lerobot.scripts.lerobot_train as native
    import smolvla_sim_gate as simulation
    def forbidden(*a,**kw):raise RuntimeError("policy/optimizer construction forbidden")
    native.make_policy=forbidden
    native.make_optimizer_and_scheduler=forbidden
    native.update_policy=forbidden
    require(not torch.cuda.is_initialized(),"CPU run unexpectedly initialized CUDA")
    torch.set_num_threads(2)
    def guard():
        require(time.monotonic()-started<=180,"registered CPU deadline")
        require(sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())<=256*1024**2,"CPU evidence byte cap")
        rss=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
        require(rss<=8*1024**3,"CPU RSS budget")
    require(sha(args.view/"view-manifest.json")==VIEW_SHA and sha(args.view/"native-read-verification.json")==READ_SHA,
            "actual D40 view/read proof identity")
    accepted_training_view(args.view,VIEW_SHA,native_read_sha256=READ_SHA)
    require(sha(args.view/"normalization.json")==NORM_SHA,"original D-only stats identity")
    require(sha(args.artifact/"config.json")==CONFIG_SHA and sha(args.artifact/"normalization.json")==NORM_SHA,
            "base processor metadata differs from canonical D-only identity")
    require(sha(args.collection/"manifest.json")==Q2_MANIFEST_SHA and sha(args.collection_audit)==Q2_AUDIT_SHA,"actual Q2 source identity")
    q2_audit=json.loads(args.collection_audit.read_bytes())
    require(q2_audit["passed"] is True and q2_audit["manifest_sha256"]==Q2_MANIFEST_SHA,"Q2 audit binding")
    # Explicit create-only link plus receipt, not an implicit filesystem assumption.
    auxiliary=args.bundle/"auxiliary"
    if auxiliary.exists() or auxiliary.is_symlink():
        require(auxiliary.is_symlink() and auxiliary.resolve()==args.collection.resolve(),"existing auxiliary input is not the exact audited collection")
    else:auxiliary.symlink_to(args.collection.resolve(),target_is_directory=True)
    receipt={"schema":"rosetta.real_paired_CPU_source.v1","Q2_manifest_sha256":Q2_MANIFEST_SHA,"Q2_audit_sha256":Q2_AUDIT_SHA,
             "D_view_manifest_sha256":VIEW_SHA,"D_native_read_sha256":READ_SHA,"D_normalization_sha256":NORM_SHA,
             "auxiliary_link_resolves_to_audited_collection":True,"policy_weights_loaded":False,"optimizer_updates":0}
    (args.output/"source-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    config=json.loads((args.artifact/"config.json").read_bytes())
    normalization=json.loads((args.artifact/"normalization.json").read_bytes())
    metadata=simulation._ArtifactMetadata(config,normalization)
    pretrained=args.artifact/"pretrained_model"
    cfg=SmolVLAConfig.from_pretrained(pretrained,local_files_only=True)
    cfg.device="cpu"
    pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=pretrained,pretrained_revision=None,dataset_stats=metadata.stats,
        preprocessor_overrides={"device_processor":{"device":"cpu"},"tokenizer_processor":{"tokenizer_name":str(pretrained/"tokenizer")},
            "normalizer_processor":{"features":{**cfg.input_features,**cfg.output_features},"norm_map":cfg.normalization_mapping,"stats":metadata.stats},
            "rename_observations_processor":{"rename_map":config["rename_map"]}},
        postprocessor_overrides={"unnormalizer_processor":{"features":cfg.output_features,"norm_map":cfg.normalization_mapping,"stats":metadata.stats}})
    contract=load_action_contract(ROOT/"configs/sim/aloha_insertion_smolvla.yaml")
    require(sha(ROOT/"configs/sim/aloha_insertion_smolvla.yaml")==config["action_contract_sha256"],"processor physical boundary identity")
    ensure_smolvla_action_boundary(pre,post,contract,SmolVLAActionSpace(**config["action_space"]),
        action_contract_sha256=config["action_contract_sha256"],upstream_revision=config["upstream_revision"])
    normalizers=[s for s in pre.steps if getattr(s.__class__,"_registry_name",None)=="normalizer_processor"]
    require(len(normalizers)==1,"unique saved normalizer required")
    eps=getattr(normalizers[0],"eps",1e-8)
    def mode(name):
        for key,value in cfg.normalization_mapping.items():
            if getattr(key,"value",key)==name:return getattr(value,"value",value)
        raise ValueError("normalization mapping missing "+name)
    dataset=logical_native_dataset(args.view,list(TRAIN),delta_timestamps={"action":[i/50 for i in range(50)]})
    data_stats_before={k:{s:v.detach().cpu().tolist() if torch.is_tensor(v) else v for s,v in stats.items()} for k,stats in dataset.meta.stats.items()}
    outcomes=[]
    for arm,source in (("B","R"),("C","S")):
        guard()
        plan_path=args.bundle/f"plan-{arm}-smoke.yaml"
        plan=yaml.safe_load(plan_path.read_text())
        declaration=next(f for f in plan["features"] if f["name"]=="paired_mixture")
        declaration={k:v for k,v in declaration.items() if k!="name"}
        feature=PairedMixtureFeature(declaration)
        context=types.SimpleNamespace(plan_path=plan_path,phase="performance_benchmark",device="cpu",plan=plan,
            run_name="paired-real-CPU-input-"+arm+"-20260927-001",experiment={"dataset":{"train_episodes":list(TRAIN)}})
        feature.install(context)
        try:
            owner=native if hasattr(native,"DataLoader") else native.torch.utils.data
            loader=owner.DataLoader(dataset,batch_size=4,num_workers=0,shuffle=False)
            raw=next(iter(loader))
            require(loader.dataset.meta.stats is dataset.meta.stats,"D-only metadata was replaced")
            schedule=json.loads((args.bundle/declaration["schedule"]).read_bytes())
            first=schedule["entries"][:4]
            require([e["source"] for e in first]==["D","D","D",source],"real first batch is not sealed 3D+1")
            inventory=json.loads((args.bundle/declaration["auxiliary_inventory"]).read_bytes())
            match=[r for r in inventory["inputs"] if (r["source"],r["episode"],r["frame"])==(source,first[3]["episode"],first[3]["frame"])]
            require(len(match)==1,"real auxiliary source identity")
            item=match[0]
            label_path=args.collection/item["labels"]
            require(sha(label_path)==item["labels_sha256"],"actual auxiliary label identity")
            labels=json.loads(label_path.read_bytes())
            start=item["frame"]
            rows=labels["rows"][start:start+50]
            verify_chunk_rows(rows,start)
            original_action=torch.tensor([r["action"] for r in rows],dtype=raw["action"].dtype)
            original_state=torch.tensor(rows[0]["state"],dtype=raw["observation.state"].dtype)
            require(torch.equal(raw["action"][3],original_action) and torch.equal(raw["observation.state"][3],original_state),"real continuous targets/state differ from saved labels")
            require(tuple(raw["action"].shape)==(4,contract.chunk_length,contract.dimension) and raw["action"].dtype==torch.float32,"native batch action shape/dtype")
            require(not raw["action_is_pad"][3].any(),"auxiliary fifty slots include padding")
            with np.load(args.collection/rows[0]["images"],allow_pickle=False) as image:
                native_pixels=image[declaration["camera_map"]["observation.images.top"]]
            expected_image=torch.from_numpy(native_pixels.copy()).permute(2,0,1).float()/255
            require(torch.equal(raw["observation.images.top"][3],expected_image),"actual auxiliary raw camera differs from original uint8 NPZ")
            # Run the real saved processor once; compute the normalization separately.
            with torch.inference_mode():processed=pre(copy.deepcopy(raw))
            projected=contract.clip(raw["action"])[0]
            encoded_action=standard_aloha_action_to_bounded_sine(projected)
            encoded_state=standard_aloha_state_to_pi(raw["observation.state"])
            expected={"action":arithmetic_normalize(encoded_action,normalization["effective_stats"]["action"],mode("ACTION"),eps),
                      "observation.state":arithmetic_normalize(encoded_state,normalization["effective_stats"]["observation.state"],mode("STATE"),eps)}
            image_key=config["rename_map"]["observation.images.top"]
            expected[image_key]=arithmetic_normalize(raw["observation.images.top"],normalization["visual_statistics"],mode("VISUAL"),eps)
            equalities={}
            for key,value in expected.items():
                require(key in processed,"processed field absent "+key)
                require(processed[key].device.type=="cpu" and torch.isfinite(processed[key]).all(),"nonfinite/nonCPU processed field "+key)
                equalities[key]=torch.equal(processed[key],value)
                require(equalities[key],"independent D-only normalization differs "+key)
            require(torch.equal(processed["action_is_pad"],raw["action_is_pad"]),"saved processor changed target padding")
            arrays={"raw_action":raw["action"].numpy(),"raw_state":raw["observation.state"].numpy(),"action_is_pad":raw["action_is_pad"].numpy()}
            arrays.update({"processed::"+k:v.numpy() for k,v in processed.items() if torch.is_tensor(v)})
            np.savez_compressed(args.output/f"actual-{arm}-batch.npz",**arrays)
            outcomes.append({"arm":arm,"source":source,"actual_inputs":first,"targets_shape":list(raw["action"].shape),
                             "targets_dtype":str(raw["action"].dtype),"continuous_actual_targets_exact":True,
                             "actual_original_camera_exact":True,"independent_normalization_bitwise_exact":equalities,
                             "normalization_modes":{k:mode(k) for k in ("STATE","ACTION","VISUAL")},"normalizer_epsilon":eps,
                             "actual_auxiliary_valid_slots":50,"D_stats_object_identity_preserved":True,
                             "processed_shapes":{k:list(v.shape) for k,v in processed.items() if torch.is_tensor(v)}})
            guard()
        finally:feature.restore(context)
    data_stats_after={k:{s:v.detach().cpu().tolist() if torch.is_tensor(v) else v for s,v in stats.items()} for k,stats in dataset.meta.stats.items()}
    require(data_stats_before==data_stats_after,"auxiliary processor changed D-only dataset stats")
    require(not torch.cuda.is_initialized(),"CPU input check initialized CUDA")
    builtins.open=original_open
    return {"schema":"rosetta.real_paired_CPU_input_check.v1","passed":True,"arms":outcomes,
            "policy_weights_loaded":False,"model_forward_calls":0,"optimizer_updates":0,"physics_steps":0,
            "D_normalization_sha256":NORM_SHA,"D_view_manifest_sha256":VIEW_SHA,"D_native_read_sha256":READ_SHA,
            "Q2_manifest_sha256":Q2_MANIFEST_SHA,"Q2_audit_sha256":Q2_AUDIT_SHA,
            "actual_saved_CPU_processor_used":True,"D_stats_unchanged":True,"elapsed_seconds":time.monotonic()-started,
            "engineering_check_not_task_success":True,"shutdown_requested":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("bundle","collection","collection-audit","view","artifact","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:result=run(args)
    except Exception as error:result={"schema":"rosetta.real_paired_CPU_input_check.v1","passed":False,"error_type":type(error).__name__,"reason":str(error),"optimizer_updates":0,"engineering_check_not_task_success":True}
    result["source_sha256"]=sha(Path(__file__))
    (args.output/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
