"""Saved-trace recovery audit negatives; no ML or simulation imports."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('independent_bank_trace',Path(__file__).resolve().parents[1]/'scripts/verify_paired_policy_bank_traces.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows(success=6):
    output=[]
    for step in range(1,7):
        output.append({'step':step,'geometry':{'socket_grasp_contact':True,'peg_grasp_contact':True,'peg_socket_contact':True},
            'joint_limit_violations':0,'unexpected_contacts':[],'task_success':step==success,
            'executed_action':[0.0]*14,'state_after':[0.0]*14,'reward':3.0 if step==success else 0.0,
            'recovery_event':{'criterion':'both_grasps','required_consecutive_steps':5,
                'correction_steps_observed':step,'recovery_step':5 if step>=5 else None,
                'geometric_recovery':step>=5,'terminal_success_step':success if step>=success else None,
                'verified_recovery_chain':step>=success and success>5}})
    return output


def test_true_sustained_recovery_followed_by_success():
    result=module.recompute_trace(rows(),'grasp')
    assert result['measured_steps']==6 and result['recovery_event']['verified_recovery_chain']


def test_same_step_success_not_prior_recovery_chain():
    result=module.recompute_trace(rows(5),'grasp')
    assert result['task_success'] and not result['recovery_event']['verified_recovery_chain']


def test_zero_observations_unmeasured():
    assert module.recompute_trace([],'align')['recovery_event'] is None


@pytest.mark.parametrize('change',[ 'gap','nonboolean','nonfinite','align_contact','safety','forged_chain' ])
def test_actual_trace_disagreement_rejected(change):
    data=rows()
    if change=='gap':data[2]['step']=8
    elif change=='nonboolean':data[0]['geometry']['peg_grasp_contact']=1
    elif change=='nonfinite':data[0]['executed_action'][0]=float('nan')
    elif change=='align_contact':
        data[4]['geometry']['peg_socket_contact']=False
        for row in data:row['recovery_event']['criterion']='both_grasps_and_peg_socket_contact'
    elif change=='safety':data[5]['unexpected_contacts']=[['peg','table']]
    elif change=='forged_chain':data[4]['recovery_event']['verified_recovery_chain']=True
    with pytest.raises(ValueError):module.recompute_trace(data,'align' if change=='align_contact' else 'grasp')
