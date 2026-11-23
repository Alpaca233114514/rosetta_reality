from rosetta_reality.vla.training.paired_mixture import build_schedules, verify_matched_cohort
import pytest


def rows(episodes):
    return [{"episode": ep, "frame": frame, "positive_supervision": True, "chunk_crosses_unqualified_action": False} for ep in episodes for frame in range(3)]


def test_actual_common_slots_equal_and_no_rng_drift():
    import random
    state = random.getstate()
    plans = build_schedules(rows([2, 3]), rows([100, 101]), rows([200, 201]), updates=7)
    assert state == random.getstate()
    for update in range(7):
        entries = [plans[arm]["entries"][update*4:update*4+3] for arm in "ABC"]
        assert entries[0] == entries[1] == entries[2]
    assert all(plans[arm]["total_exposures"] == 28 for arm in "ABC")
    assert [r["source"] for r in plans["B"]["entries"]].count("R") == 7


def test_missing_auxiliary_is_not_original_substitution():
    plans = build_schedules(rows([2]), updates=2)
    assert plans["A"]["status"] == "sealed"
    assert plans["B"]["status"] == plans["C"]["status"] == "input_unavailable"
    with pytest.raises(ValueError, match="N=0"):
        verify_matched_cohort([])


def test_unknown_action_inside_chunk_rejected():
    d = rows([2])
    d[1]["chunk_crosses_unqualified_action"] = True
    with pytest.raises(ValueError, match="Chunk crosses"):
        build_schedules(d)


def test_exploratory_lineage_does_not_claim_legacy_scene_isolation():
    from rosetta_reality.vla.training.paired_mixture import EXPLORATORY_SCOPE
    pair = {"parent_id":"fresh-parent", "research_scope":EXPLORATORY_SCOPE,
            "fresh_lineage_verified":True, "new_scene_disjoint_verified":True,
            "legacy_object_scene_nonoverlap":"not_established", "N_strict":0,
            "S":{"effective_frames":50,"valid_target_slots":2500,"task_success":True,
                 "labels_independently_verified":True,"ordinary_success_verified":True},
            "R":{"effective_frames":50,"valid_target_slots":2500,"task_success":True,
                 "labels_independently_verified":True,"recovery_chain_verified":True}}
    assert verify_matched_cohort([pair],research_scope=EXPLORATORY_SCOPE)==1
    with pytest.raises(ValueError,match="Scene/split"):
        verify_matched_cohort([pair])
    pair["scene_disjoint_verified"]=True
    with pytest.raises(ValueError,match="cannot claim"):
        verify_matched_cohort([pair],research_scope=EXPLORATORY_SCOPE)
    pair.pop("scene_disjoint_verified")
    pair["fresh_lineage_verified"]=False
    with pytest.raises(ValueError,match="fresh lineage"):
        verify_matched_cohort([pair],research_scope=EXPLORATORY_SCOPE)


@pytest.mark.parametrize("direct_torch_owner",[False,True])
def test_native_loader_consumes_exact_sealed_batches_and_counts_success(tmp_path, monkeypatch,direct_torch_owner):
    import json
    import hashlib
    import sys
    import types
    from rosetta_reality.vla.training.paired_mixture import save_bundle
    from rosetta_reality.vla.training.paired_mixture_runtime import PairedMixtureFeature
    schedule = build_schedules(rows([2]), updates=2)["A"]
    save_bundle(tmp_path / "schedule.json", schedule)
    save_bundle(tmp_path / "aux.json", {"hidden_test_loaded":False,"normalization_source":"D_train_only","inputs":[]})
    native = types.ModuleType("lerobot.scripts.lerobot_train")
    calls=[]
    def native_update(batch=None,policy=None,accelerator=None):
        if policy is not None:policy()
        calls.append("updated")
    native.update_policy = native_update
    original_update = native.update_policy
    class Loader:
        def __init__(self,dataset,**kwargs):
            self.dataset,self.kwargs=dataset,kwargs
    loader_owner=types.SimpleNamespace(DataLoader=Loader) if direct_torch_owner else native
    if direct_torch_owner:
        native.torch=types.SimpleNamespace(utils=types.SimpleNamespace(data=loader_owner))
    else:native.DataLoader=Loader
    import torch
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor(1.))
            self.nonfinite=False
        def forward(self):
            return (self.weight*float("nan") if self.nonfinite else self.weight, {"native_detail":True})
    native.make_policy=Policy
    parent=types.ModuleType("lerobot")
    scripts=types.ModuleType("lerobot.scripts")
    parent.scripts,scripts.lerobot_train=scripts,native
    for name,value in (("lerobot",parent),("lerobot.scripts",scripts),("lerobot.scripts.lerobot_train",native)):
        monkeypatch.setitem(sys.modules,name,value)
    feature=PairedMixtureFeature({"schedule":"schedule.json","schedule_sha256":hashlib.sha256((tmp_path/"schedule.json").read_bytes()).hexdigest(),"auxiliary_root":"collection","auxiliary_inventory":"aux.json","auxiliary_inventory_sha256":hashlib.sha256((tmp_path/"aux.json").read_bytes()).hexdigest(),"camera_map":{}})
    context=types.SimpleNamespace(plan_path=tmp_path/"plan.yaml",phase="smoke",run_name="fake-smoke",plan={"optimizer_smoke":{"steps":2,"batch_size":4}},experiment={"dataset":{"train_episodes":[2]}})
    class Data:
        hf_dataset={"episode_index":[2,2,2],"frame_index":[0,1,2]}
        def __getitem__(self,index): return {"identity":index}
    feature.install(context)
    effective_loader_owner=native.torch.utils.data if direct_torch_owner else native
    if direct_torch_owner:
        assert loader_owner.DataLoader is Loader
        assert native.torch.utils.data.DataLoader is not Loader
    policy=native.make_policy()
    loss, details=policy()
    assert loss is policy.weight and details=={"native_detail":True}
    policy.nonfinite=True
    prepared_forward=policy.forward
    # Represents an upstream AMP wrapper installed after policy construction.
    policy.forward=lambda *a,**kw:prepared_forward(*a,**kw)
    amp_forward=policy.forward
    with pytest.raises(FloatingPointError,match="before backward"):
        native.update_policy(batch={"action_is_pad":torch.zeros((4,50),dtype=torch.bool)},policy=policy)
    assert len(calls)==0
    assert policy.forward is amp_forward
    policy.nonfinite=False
    loader=effective_loader_owner.DataLoader(Data(),batch_size=4,num_workers=0,shuffle=False)
    assert isinstance(loader.kwargs["batch_sampler"],torch.utils.data.BatchSampler)
    assert loader.kwargs["batch_sampler"].batch_size==4
    delivered=[]
    for batch in loader.kwargs["batch_sampler"]:
        delivered.extend(loader.dataset[i]["identity"] for i in batch)
        mask=torch.zeros((4,50),dtype=torch.bool)
        mask[0,-2:]=True
        mask[3,-5:]=True
        native.update_policy(batch={"action_is_pad":mask},policy=policy,accelerator=types.SimpleNamespace(optimizer_step_was_skipped=False))
    assert delivered == [row["frame"] for row in schedule["entries"]]
    assert len(calls)==2
    feature.restore(context)
    assert loader_owner.DataLoader is Loader and native.update_policy is original_update
    report=json.loads((tmp_path/"fake-smoke-actual-updates.json").read_text())
    assert report["successful_optimizer_updates"]==2 and report["sample_exposures"]==8
    assert report["valid_target_slots_actual"]=={"common_D":296,"auxiliary":90}
    assert report["padding_target_slots_actual"]=={"common_D":4,"auxiliary":10}


def test_native_scheduler_horizon_does_not_change_actual_two_update_stop():
    from types import SimpleNamespace
    from rosetta_reality.vla.training.paired_mixture_runtime import scheduler_horizon_config
    cfg=SimpleNamespace(steps=2,optimizer=object(),scheduler=object())
    scheduled=scheduler_horizon_config(cfg,5000)
    assert cfg.steps==2 and scheduled.steps==5000
    assert scheduled.optimizer is cfg.optimizer and scheduled.scheduler is cfg.scheduler
    with pytest.raises(ValueError):scheduler_horizon_config(cfg,2)
