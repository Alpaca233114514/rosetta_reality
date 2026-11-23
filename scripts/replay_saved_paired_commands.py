"""Replay four saved commands in CPU MuJoCo; no policy, pixels or Python restore.

This isolates a physical command interval. Exact teacher integration and student
logical state comparisons delimit what can be concluded from this readout.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def execute(plan, output, job):
    if (plan['schema'] != 'rosetta.saved_command_physics.v1'
            or plan['cases'] != ['teacher', 'A', 'B', 'C']
            or plan['maximum_control_steps'] != 4 or plan['model_forward_calls'] != 0
            or plan['optimizer_updates'] != 0 or plan['hidden_test_loaded'] is not False
            or plan['pair_id'] != 'bank-003b-scene-2445'
            or plan['execution_authorized'] is not True):
        raise ValueError('Registered physical-only scope changed')
    import run_round_experiment_job as supervisor
    registration = supervisor.registration(job)
    supervisor.guard_alive(job, registration)
    for name, expected in plan['source_sha256'].items():
        if sha(ROOT/name) != expected:raise ValueError('Source drift: '+name)
    root = Path(os.environ['ROSETTA_AUTODL_ROOT']).resolve(strict=True)
    for name, expected in plan['input_sha256'].items():
        p = (root/name).resolve(strict=True)
        if not p.is_relative_to(root) or sha(p) != expected:raise ValueError('Input drift: '+name)
    import importlib.metadata as metadata
    for name, expected in plan['packages'].items():
        if metadata.version(name) != expected:raise ValueError('Package version drift: '+name)
    import numpy as np
    import mujoco
    from dm_control import mujoco as dm_mujoco
    from gym_aloha.constants import ASSETS_DIR, DT
    from gym_aloha.tasks.sim import InsertionTask
    import gym_aloha.tasks.sim as installed_task
    if sha(installed_task.__file__) != plan['installed_task_sha256']:
        raise ValueError('Installed task source drift')
    bank = root/plan['bank_root']/plan['pair_id']
    receipt = json.loads(gzip.decompress((bank/'branch-python.json.gz').read_bytes()))
    with np.load(bank/'branch-integration.npz',allow_pickle=False) as arrays:
        initial = arrays['state'].copy()
    with np.load(bank/'branch-observation.npz',allow_pickle=False) as arrays:
        saved_state = arrays['robot_state'].copy()
    source = [json.loads(line) for line in (bank/'R/trace.jsonl').read_text().splitlines()]
    teacher = next(row for row in source if row['step'] == 178)
    if teacher['action_source'] != 'state_conditioned_teacher':raise ValueError('Teacher correction drift')
    output.mkdir(parents=True, exist_ok=False)
    started=time.monotonic()
    result={'status':'incomplete','model_forward_calls':0,'optimizer_updates':0,
            'control_steps':0,'pixel_render_calls':0,'complete_python_restore':False,
            'formal_gate_result':None,'m2_complete':False,'cases':[]}
    def save(name,value):
        with (output/name).open('x') as stream:
            json.dump(value,stream,indent=2,allow_nan=False)
            stream.write('\n')
    def integration(model,data):
        v=np.zeros(mujoco.mj_stateSize(model,receipt['integration_spec']),dtype=np.float64)
        mujoco.mj_getState(model,data,v,receipt['integration_spec'])
        return v
    def snapshot(physics,task):
        model,data=physics.model.ptr,physics.data.ptr
        joints=[]
        for i in range(model.njnt):
            if model.jnt_limited[i]:
                a=int(model.jnt_qposadr[i]);lo,hi=map(float,model.jnt_range[i]);v=float(data.qpos[a])
                joints.append({'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i),
                    'qpos_address':a,'qpos':v,'lower':lo,'upper':hi,'margin':min(v-lo,hi-v),
                    'violation':v<lo-1e-5 or v>hi+1e-5})
        return {'qpos':data.qpos.tolist(),'qvel':data.qvel.tolist(),'ctrl':data.ctrl.tolist(),
                'qacc':data.qacc.tolist(),'qfrc_actuator':data.qfrc_actuator.tolist(),
                'qfrc_constraint':data.qfrc_constraint.tolist(),'joint_limits':joints,
                'logical_state':np.asarray(task.get_qpos(physics),dtype=np.float32).tolist(),
                'contacts':[[physics.model.id2name(int(c.geom1),'geom'),physics.model.id2name(int(c.geom2),'geom')]
                            for c in data.contact[:data.ncon]],'reward':float(task.get_reward(physics))}
    try:
        for case in plan['cases']:
            supervisor.guard_alive(job,registration)
            physics=dm_mujoco.Physics.from_xml_path(str(ASSETS_DIR/'bimanual_viperx_insertion.xml'))
            task=InsertionTask()
            try:
                model,data=physics.model.ptr,physics.data.ptr
                buf=np.zeros(mujoco.mj_sizeModel(model),dtype=np.uint8)
                mujoco.mj_saveModel(model,None,buf)
                model_sha=hashlib.sha256(buf.tobytes()).hexdigest()
                if model_sha!=receipt['model_sha256']:raise ValueError('Compiled model differs from saved bank')
                spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)
                if spec!=receipt['integration_spec']:raise ValueError('Integration specification drift')
                mujoco.mj_setState(model,data,initial,spec)
                mujoco.mj_forward(model,data)
                mujoco.mj_setState(model,data,initial,spec)
                if not np.array_equal(initial,integration(model,data)):raise ValueError('Integration restore not exact')
                before=snapshot(physics,task)
                if not np.array_equal(saved_state,np.array(before['logical_state'],dtype=np.float32)):
                    raise ValueError('Saved observation/state mismatch')
                if any(j['violation'] for j in before['joint_limits']):raise ValueError('Unsafe saved initial state')
                if case=='teacher':
                    command=teacher['action'];expected_state=teacher['after']['geometry']['robot_state']
                    expected_integration=np.array(teacher['after']['integration'],dtype=np.float64)
                    expected_violations=teacher['after']['joint_limits']
                else:
                    rows=[json.loads(line) for line in (root/plan['measurement_traces'][case]).read_text().splitlines()]
                    row=rows[0]
                    if row['step']!=1:raise ValueError('Student first correction missing')
                    command=row['executed_action'];expected_state=row['state_after']
                    expected_integration=None;expected_violations=row['joint_limit_violations']
                action=np.array(command,dtype=np.float32)
                if action.shape!=(14,) or not np.isfinite(action).all():raise ValueError('Saved command invalid')
                interval=float(DT);substeps=int(round(interval/model.opt.timestep))
                if interval!=.02 or substeps!=plan['substeps_per_control'] or abs(substeps*model.opt.timestep-interval)>1e-12:
                    raise ValueError('Physical control interval drift')
                task.before_step(action,physics)
                ctrl=data.ctrl.copy()
                physics.step(substeps)
                task.after_step(physics)
                result['control_steps']+=1
                after=snapshot(physics,task)
                state=np.array(after['logical_state'],dtype=np.float32)
                expected_state=np.array(expected_state,dtype=np.float32)
                exact_state=np.array_equal(state,expected_state)
                exact_integration=None if expected_integration is None else np.array_equal(integration(model,data),expected_integration)
                count=sum(j['violation'] for j in after['joint_limits'])
                record={'case':case,'compiled_model_sha256':model_sha,'initial_integration_exact':True,
                    'initial_logical_state_exact':True,'command':action.tolist(),'translated_ctrl':ctrl.tolist(),
                    'before':before,'after':after,'historical_state_exact':exact_state,
                    'historical_state_max_abs_difference':float(np.max(np.abs(state-expected_state))),
                    'historical_integration_exact':exact_integration,'historical_violation_count':expected_violations,
                    'violation_count':count,'control_seconds':interval,'mujoco_substeps':substeps}
                save(case+'.json',record)
                result['cases'].append({k:record[k] for k in ['case','historical_state_exact','historical_integration_exact','violation_count','historical_violation_count']})
                if not exact_state or count!=expected_violations or case=='teacher' and not exact_integration:
                    raise ValueError('Physical-only replay does not reproduce historical boundary')
            finally:physics.free()
        result['status']='completed'
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        save('result.json',result)
        manifest={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        save('manifest.json',{'files':manifest})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--guard-job',type=Path,required=True)
    args=parser.parse_args()
    execute(read(args.plan),args.output,args.guard_job)
