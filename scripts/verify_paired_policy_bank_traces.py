"""Independently recompute saved policy recovery events; no model or simulator imports."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute_trace(rows, trigger):
    if trigger not in ('grasp', 'align', 'approach'):
        raise ValueError('unregistered trigger')
    contacts=['socket_grasp_contact','peg_grasp_contact']
    if trigger == 'align': contacts.append('peg_socket_contact')
    streak=0
    recovery=terminal=None
    unsafe=False
    count=0
    for count,row in enumerate(rows,1):
        if type(row.get('step')) is not int or row['step'] != count:
            raise ValueError('nonconsecutive correction steps')
        if count > 150:
            raise ValueError('correction budget exceeded')
        geometry=row['geometry']
        if any(type(geometry.get(key)) is not bool for key in contacts):
            raise ValueError('contact event not explicit boolean')
        violation=row['joint_limit_violations']
        if type(violation) is not int or violation < 0 or not isinstance(row['unexpected_contacts'],list):
            raise ValueError('invalid safety observations')
        if type(row['task_success']) is not bool:
            raise ValueError('task event not explicit boolean')
        for name in ('executed_action','state_after'):
            if not row[name] or any(type(v) not in (int,float) or not math.isfinite(v) for v in row[name]):
                raise ValueError('nonfinite native action/state')
        if type(row['reward']) not in (int,float) or not math.isfinite(row['reward']):
            raise ValueError('nonfinite physical reward')
        safe=violation == 0 and not row['unexpected_contacts']
        unsafe |= not safe
        streak=streak+1 if safe and all(geometry[key] for key in contacts) else 0
        if recovery is None and streak >= 5: recovery=count
        if terminal is None and row['task_success']: terminal=count
        expected={
            'criterion':'both_grasps_and_peg_socket_contact' if trigger=='align' else 'both_grasps',
            'required_consecutive_steps':5,'correction_steps_observed':count,
            'recovery_step':recovery,'geometric_recovery':recovery is not None,
            'terminal_success_step':terminal,
            'verified_recovery_chain':bool(not unsafe and recovery is not None and terminal is not None and recovery < terminal)}
        if row.get('recovery_event') != expected:
            raise ValueError('saved per-step recovery event differs from independent recomputation')
    return {'measured_steps':count,'task_success':terminal is not None,
            'unsafe_observed':unsafe,'recovery_event':expected if count else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True,type=Path)
    p.add_argument('--source-audit',required=True,type=Path)
    p.add_argument('--measurement',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    read=lambda path:json.loads(Path(path).read_bytes())
    manifest=read(args.bank/'manifest.json')
    source=read(args.bank/'summary.json')
    registration=read(args.bank/'registration.json')
    audit=read(args.source_audit)
    plan=read(args.measurement/'registration.json')
    overall=read(args.measurement/'summary.json')
    if (audit.get('passed') is not True or audit['manifest_sha256'] != sha(args.bank/'manifest.json')
            or audit['summary_sha256'] != sha(args.bank/'summary.json')
            or plan['bank_manifest_sha256'] != audit['manifest_sha256']
            or plan['bank_audit_sha256'] != sha(args.source_audit)
            or overall['plan_sha256'] != sha(args.measurement/'registration.json')):
        raise ValueError('source/plan/measurement identity disagreement')
    if registration['phase'] != 'bank' or registration['recovery_sustain_steps'] != 5:
        raise ValueError('bank protocol drift')
    pairs={r['pair_id']:r for r in registration['pairs']}
    states={r['pair_id']:r for r in source['bank']}
    policy=read(args.measurement/'recovery-bank'/'bank-summary.json')
    if len(pairs) != 10 or len(pairs) != len(registration['pairs']):
        raise ValueError('registered ten-state bank required')
    records={r['pair_id']:r for r in policy['rows']}
    if len(records) != len(policy['rows']) or not records.keys() <= pairs.keys():
        raise ValueError('policy identity duplicates/drift')
    proofs=[]
    for identity,row in records.items():
        state=states.get(identity,{})
        if row['teacher_feasible'] is not (state.get('feasible') is True):
            raise ValueError('teacher feasible label drift')
        trace=args.measurement/'recovery-bank'/(identity+'.jsonl')
        rows=[json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        result=recompute_trace(rows,pairs[identity]['trigger'])
        if result['measured_steps'] != row['measured_steps'] or result['recovery_event'] != row['recovery_event']:
            raise ValueError('policy summary differs from actual trace')
        if rows and (not state.get('branch_state') or row['branch_task_success'] is not result['task_success']):
            raise ValueError('task outcome/branch provenance disagreement')
        if not state.get('branch_state') and (row['status'] != 'unmeasured_branch_not_created' or row['branch_task_success'] is not None):
            raise ValueError('uncreated branch mislabeled as measured outcome')
        proofs.append(dict(pair_id=identity,teacher_feasible=row['teacher_feasible'],
                           actual_branch_created=bool(state.get('branch_state')),
                           trace_sha256=sha(trace) if trace.exists() else None,**result))
    for field,key in [('branch_task_success_count','task_success'),('geometric_recovery_count','geometric_recovery'),('verified_recovery_chain_count','verified_recovery_chain')]:
        actual=sum(r[key] if key=='task_success' else bool((r['recovery_event'] or {}).get(key)) for r in proofs)
        if actual != policy[field]: raise ValueError('aggregate count disagrees with actual trace')
    if policy['registered_states'] != 10 or policy['measured_states'] != sum(r['measured_steps']>0 for r in proofs):
        raise ValueError('registered/measured denominator disagreement')
    result={'schema':'rosetta.independent_policy_bank_trace_check.v1','passed':True,
            'registered':10,'recorded':len(proofs),'unrecorded_parent_ids':sorted(pairs.keys()-records.keys()),
            'rows':proofs,'bank_manifest_sha256':sha(args.bank/'manifest.json'),
            'bank_summary_sha256':sha(args.bank/'summary.json'),'source_audit_sha256':sha(args.source_audit),
            'policy_summary_sha256':sha(args.measurement/'recovery-bank'/'bank-summary.json'),
            'evaluation_plan_sha256':sha(args.measurement/'registration.json'),
            'source_sha256':sha(__file__),'policy_trace_independently_recomputed':True,
            'geometry_is_recomputed_from_saved_contact_observations_not_new_physics':True}
    with args.output.open('x',encoding='utf8') as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(result))
    return 0


if __name__=='__main__':raise SystemExit(main())
