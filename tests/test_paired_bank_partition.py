"""Synthetic denominator regressions, not evidence of policy recovery."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("partition",Path(__file__).resolve().parents[1]/"scripts/summarize_paired_recovery_bank.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    registration={"phase":"bank","pairs":[{"pair_id":str(i)} for i in range(10)]}
    source={"bank":[{"pair_id":"0","branch_state":"actual.npz","feasible":True},
                    {"pair_id":"1","branch_state":"actual2.npz","feasible":False}],
            "attempts":[{"attempt_id":"0-R","task_success":True,"recovery":{"step":5},"verified_recovery_chain":True},
                        {"attempt_id":"1-R","task_success":True,"recovery":None,"verified_recovery_chain":False}]}
    policy={"rows":[{"pair_id":"0","teacher_feasible":True,"measured_steps":1,"branch_task_success":False,"recovery_event":None},
                    {"pair_id":"1","teacher_feasible":False,"measured_steps":8,"branch_task_success":True,
                     "recovery_event":{"geometric_recovery":True,"verified_recovery_chain":True}}]}
    return registration,source,policy


def test_exploratory_policy_success_does_not_relabel_teacher_feasible():
    report=module.partition(*fixture())
    assert report["subsets"]["registered"]["denominator"]==10
    assert report["subsets"]["registered"]["policy_unmeasured"]==8
    assert report["subsets"]["actual_branch_created"]["policy_task_success"]==1
    assert report["subsets"]["teacher_feasible"]["denominator"]==1
    assert report["subsets"]["teacher_feasible"]["policy_task_success"]==0


def test_policy_cannot_claim_source_feasibility():
    args=fixture()
    args[2]["rows"][1]["teacher_feasible"]=True
    with pytest.raises(ValueError):module.partition(*args)


def test_feasible_without_real_branch_rejected():
    args=fixture()
    args[1]["bank"][0]["branch_state"]=None
    with pytest.raises(ValueError):module.partition(*args)
