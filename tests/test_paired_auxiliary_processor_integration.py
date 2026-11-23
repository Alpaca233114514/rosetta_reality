"""Synthetic NPZ/labels through real saved processors; no weights or optimizer."""
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arm,source",[("B","R"),("C","S")])
def test_synthetic_auxiliary_npz_chunk_matches_native_saved_processor(tmp_path,monkeypatch,arm,source):
    from rosetta_reality.vla.training.paired_mixture import build_schedules,save_bundle,digest
    from rosetta_reality.vla.training.paired_mixture_runtime import PairedMixtureFeature
    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla.action_space import SmolVLAActionSpace
    from rosetta_reality.vla.processor import ensure_smolvla_action_boundary
    import smolvla_sim_gate as simulator
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    metadata_dir=ROOT/"runs/prepost-gate-received-20260924-001/prepost-gate4-20260924-001/artifacts/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/prepost-base-aloha-interface-001"
    pretrained=ROOT/"runs/base-interface-source-20260927-003/pretrained_model"
    if not pretrained.is_dir():pytest.fail("Registered processor-only fixture must be staged")
    config=json.loads((metadata_dir/"config.json").read_text())
    norm=json.loads((metadata_dir/"normalization.json").read_text())
    contract=load_action_contract(ROOT/"configs/sim/aloha_insertion_smolvla.yaml")
    metadata=simulator._ArtifactMetadata(config,norm)
    cfg=SmolVLAConfig.from_pretrained(pretrained,local_files_only=True)
    cfg.device="cpu"
    pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=pretrained,pretrained_revision=None,
        dataset_stats=metadata.stats,preprocessor_overrides={
            "device_processor":{"device":"cpu"},
            "tokenizer_processor":{"tokenizer_name":str(pretrained/"tokenizer")},
            "normalizer_processor":{"features":{**cfg.input_features,**cfg.output_features},"norm_map":cfg.normalization_mapping,"stats":metadata.stats},
            "rename_observations_processor":{"rename_map":config["rename_map"]}},
        postprocessor_overrides={"unnormalizer_processor":{"features":cfg.output_features,"norm_map":cfg.normalization_mapping,"stats":metadata.stats}})
    ensure_smolvla_action_boundary(pre,post,contract,SmolVLAActionSpace(**config["action_space"]),
        action_contract_sha256=config["action_contract_sha256"],upstream_revision=config["upstream_revision"])
    shape=(480,640,3)
    pixels=np.arange(np.prod(shape),dtype=np.uint8).reshape(shape)
    collection=tmp_path/"collection"
    collection.mkdir()
    np.savez_compressed(collection/"frame.npz",top=pixels)
    image_sha=hashlib.sha256((collection/"frame.npz").read_bytes()).hexdigest()
    state=((contract.lower_bounds+contract.upper_bounds)/2).to(torch.float32)
    actions=state.repeat(108,1)
    actions[:,0]+=torch.arange(108)*.0001
    rows=[{"step":i,"positive_target":True,"safe":True,"images":"frame.npz","images_sha256":image_sha,
           "state":state.tolist(),"action":actions[i].tolist(),"task":"Insert the peg into the socket.","action_source":"state_conditioned_teacher"} for i in range(108)]
    native_starts=list(range(59))
    grid=list(range(50))
    labels={"rows":rows,"valid_chunk_starts":native_starts,"native_valid_chunk_starts":native_starts}
    save_bundle(collection/"labels.json",labels)
    sampler={"common_valid_chunk_starts":grid,"common_starts_sha256":digest(grid),
             "S_native_valid_chunk_starts":native_starts,"R_native_valid_chunk_starts":native_starts,
             "sufficient_common_starts":True,"pre_deviation_prefix_exact":True,
             "raw_labels_changed":False,"minimum_common_chunk_starts":50}
    save_bundle(collection/"pair-sampler.json",sampler)
    row=lambda ep,frame:{"episode":ep,"frame":frame,"positive_supervision":True,"chunk_crosses_unqualified_action":False}
    schedule=build_schedules([row(2,0)],[row(100,7)],[row(100,7)],updates=1)[arm]
    save_bundle(tmp_path/"schedule.json",schedule)
    inventory={"hidden_test_loaded":False,"normalization_source":"D_train_only","inputs":[{
        "source":source,"episode":100,"frame":7,"labels":"labels.json","labels_sha256":hashlib.sha256((collection/"labels.json").read_bytes()).hexdigest(),
        "pair_sampler":"pair-sampler.json","pair_sampler_sha256":hashlib.sha256((collection/"pair-sampler.json").read_bytes()).hexdigest(),"common_grid_sha256":digest(grid)}]}
    save_bundle(tmp_path/"inventory.json",inventory)
    native=types.ModuleType("lerobot.scripts.lerobot_train")
    class Loader:
        def __init__(self,dataset,**kwargs):self.dataset,self.kwargs=dataset,kwargs
    native.DataLoader=Loader
    native.make_policy=lambda:None
    native.update_policy=lambda accelerator=None:None
    import lerobot.scripts
    monkeypatch.setattr(lerobot.scripts,"lerobot_train",native,raising=False)
    monkeypatch.setitem(sys.modules,"lerobot.scripts.lerobot_train",native)
    template={"observation.images.top":torch.from_numpy(pixels.copy()).permute(2,0,1).float()/255,
              "observation.state":state.clone(),"action":actions[7:57].clone(),"action_is_pad":torch.zeros(50,dtype=torch.bool),"task":"Insert the peg into the socket."}
    class Data:
        hf_dataset={"episode_index":[2],"frame_index":[0]}
        meta=types.SimpleNamespace(stats=metadata.stats)
        def __getitem__(self,index):return {k:v.clone() if torch.is_tensor(v) else v for k,v in template.items()}
    declaration={"schedule":"schedule.json","schedule_sha256":hashlib.sha256((tmp_path/"schedule.json").read_bytes()).hexdigest(),"auxiliary_root":"collection","auxiliary_inventory":"inventory.json","auxiliary_inventory_sha256":hashlib.sha256((tmp_path/"inventory.json").read_bytes()).hexdigest(),"camera_map":{"observation.images.top":"top"}}
    context=types.SimpleNamespace(plan_path=tmp_path/"plan.yaml",phase="smoke",run_name="synthetic-aux-"+arm,
        plan={"optimizer_smoke":{"steps":1,"batch_size":4}},experiment={"dataset":{"train_episodes":[2]}})
    feature=PairedMixtureFeature(declaration)
    feature.install(context)
    try:
        loader=native.DataLoader(Data(),batch_size=4,num_workers=0,shuffle=False)
        auxiliary=loader.dataset[3]
        assert torch.equal(auxiliary["action"],actions[7:57])
        assert auxiliary["action"].dtype==torch.float32 and auxiliary["action"].shape==(50,14)
        assert torch.equal(auxiliary["observation.images.top"],template["observation.images.top"])
        assert loader.dataset.meta.stats is metadata.stats
        processed_aux=pre(auxiliary)
        processed_D=pre(template)
        for key in ("observation.state","action","action_is_pad","observation.images.camera1"):
            assert torch.equal(processed_aux[key],processed_D[key]),key
            if torch.is_floating_point(processed_aux[key]):assert torch.isfinite(processed_aux[key]).all()
    finally:feature.restore(context)
