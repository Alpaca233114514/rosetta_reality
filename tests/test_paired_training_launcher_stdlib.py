"""Actual prerequisite semantics, no tensor/model/dataset imports."""
import importlib.util
from pathlib import Path
import pytest


def launcher():
    path=Path(__file__).resolve().parents[1]/"scripts/run_paired_training_v2.py"
    spec=importlib.util.spec_from_file_location("paired_launch_rules",path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def record(stage,**extra):
    return {"status":"completed","stage":stage,"view_manifest_sha256":"a"*64,
            "native_loss_changed":False,"hidden_test_loaded":False,**extra}


def test_benchmark_requires_real_batch1_and4_and_no_optimizer():
    value=launcher();kind="paired_native_no_optimizer_benchmark"
    expected={"view_manifest_sha256":"a"*64}
    r=record(kind,optimizer_updates=0,gradients_enabled=False,batch_sizes_measured=[1,4])
    value.validate_engineering_evidence(r,kind,expected)
    r["batch_sizes_measured"]=[4]
    with pytest.raises(ValueError,match="batch1/batch4"):
        value.validate_engineering_evidence(r,kind,expected)
    r["batch_sizes_measured"]=[1,4];r["optimizer_updates"]=2
    with pytest.raises(ValueError,match="no-optimizer"):
        value.validate_engineering_evidence(r,kind,expected)


@pytest.mark.parametrize("updates",[0,1,3])
def test_smoke_requires_two_actual_updates(updates):
    value=launcher();kind="paired_native_optimizer_smoke"
    with pytest.raises(ValueError,match="Two successful"):
        value.validate_engineering_evidence(record(kind,successful_optimizer_updates=updates),kind,{"view_manifest_sha256":"a"*64})


def test_changed_view_or_native_objective_is_incompatible():
    value=launcher();kind="paired_native_optimizer_smoke"
    r=record(kind,successful_optimizer_updates=2)
    with pytest.raises(ValueError,match="incompatible"):
        value.validate_engineering_evidence(r,kind,{"view_manifest_sha256":"b"*64})
    r["native_loss_changed"]=True
    with pytest.raises(ValueError,match="incompatible"):
        value.validate_engineering_evidence(r,kind,{"view_manifest_sha256":"a"*64})


@pytest.mark.parametrize("key",["arm","engineering_schedule_prefix_sha256","auxiliary_inventory_sha256",
                                 "implementation_identity_sha256","optimizer_scheduler_sha256"])
def test_other_arm_or_training_semantics_cannot_supply_engineering_prerequisite(key):
    value=launcher();kind="paired_native_optimizer_smoke"
    r=record(kind,successful_optimizer_updates=2,**{key:"source-A"})
    with pytest.raises(ValueError,match="incompatible"):
        value.validate_engineering_evidence(r,kind,{"view_manifest_sha256":"a"*64,key:"source-B"})


def test_training_requires_actual_exact_independent_reload():
    value=launcher();kind="paired_native_smoke_reload"
    r=record(kind,optimizer_updates=0,gradients_enabled=False,serialized_parameters_exact=False)
    with pytest.raises(ValueError,match="Independent exact"):
        value.validate_engineering_evidence(r,kind,{"view_manifest_sha256":"a"*64})
    r["serialized_parameters_exact"]=True
    value.validate_engineering_evidence(r,kind,{"view_manifest_sha256":"a"*64})
