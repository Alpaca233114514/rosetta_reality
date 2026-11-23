"""Read-only saved-context arithmetic for the paired-003 review; no ML imports."""
import argparse
import collections
import gzip
import importlib.util
import json
from pathlib import Path


def require(condition):
    if not condition:
        raise ValueError('Saved context identity or arithmetic mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--measurement-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo
    if args.output.resolve().is_relative_to((repo / '.work').resolve()):
        parser.error('Context output must be outside historical evidence')
    spec = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('audit_paired_measurement_files.py'))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    read, sha = audit.read, audit.sha
    measurement = read(args.measurement_audit)
    source_checks = {name: {'registered': expected, 'actual': sha(repo/name), 'matches': sha(repo/name)==expected}
                     for name, expected in measurement['arms'][0]['source_sha256'].items()}
    require(all(r['matches'] for r in source_checks.values()))
    require(all(a['source_sha256'] == measurement['arms'][0]['source_sha256'] for a in measurement['arms']))
    bank = repo/'.work/paired-bank-v3-20260927-001-independent-received/extracted/collection'
    manifest = read(bank/'manifest.json')
    bank_plan = read(bank/'registration.json')
    index = {v['path']: v for v in manifest['files']}
    verified = {}
    def checked(name):
        path = audit.safe(bank, name)
        require(sha(path) == index[name]['sha256'])
        verified[name] = sha(path)
        return path
    checked('registration.json')
    checked('summary.json')
    plan = read(repo/'.work/paired-ABC-measurement-20260928-003-received-20260930-001/files/evaluation-plan-A.json')
    require(sha(bank/'manifest.json') == plan['bank_manifest_sha256'])
    require(sha(bank.parent/'collection-audit.json') == plan['bank_audit_sha256'])
    bank_rows=[]
    for registered in bank_plan['pairs']:
        identity=registered['pair_id']
        summary=read(checked(identity+'/R/summary.json'))
        labels=read(checked(identity+'/R/labels.json'))['rows']
        violations=sum(row['after']['joint_limits'] for row in labels)
        contacts=sum(row['after']['unexpected_collisions'] for row in labels)
        require(all(row['safe'] == (not(row['after']['joint_limits'] or row['after']['unexpected_collisions'] or row['after']['info']['clipped_fields'])) for row in labels))
        deviation=summary['deviation']
        sustain=0
        first=None
        for row in labels:
            if summary['correction_start'] is None or row['step'] < summary['correction_start']:
                continue
            g=row['after']['geometry']
            yes=row['safe'] and g['socket_grasp_contact'] and g['peg_grasp_contact']
            if registered['trigger']=='align':yes = yes and g['peg_socket_contact']
            sustain=sustain+1 if yes else 0
            if sustain>=5 and first is None:first=row['step']
        terminals=[row['step'] for row in labels if row['after']['info'].get('is_success')]
        require(first == (summary['recovery']['step'] if summary['recovery'] else None))
        native=None
        if deviation:
            branch=checked(identity+'/branch-python.json.gz')
            receipt=json.loads(gzip.decompress(branch.read_bytes()))
            native={'model_sha256':receipt['model_sha256'],'integration_spec':receipt['integration_spec'],
                    'source_sha256':receipt['source_sha256'],
                    'integration_sha256':sha(checked(identity+'/branch-integration.npz')),
                    'observation_sha256':sha(checked(identity+'/branch-observation.npz')),
                    'source_after_deviation':labels[deviation['step']]['after']['geometry']['robot_state'],
                    'source_initial_safe':labels[deviation['step']]['safe']}
        bank_rows.append({'pair_id':identity,'source_rows':len(labels),'source_joint_counts':violations,
           'source_collision_counts':contacts,'deviation':deviation,'correction_start':summary['correction_start'],
           'recomputed_geometric_recovery_step':first,'terminal_steps':terminals,
           'source_verified_chain':summary['verified_recovery_chain'],'branch':native})
    bundle=repo/'.work/paired-native-ABC-formal-20260927-001-received/source-bundle-20260927-003'
    trainroot=repo/'.work/paired-native-ABC-formal-20260927-001-received/paired-native-ABC-formal-20260927-001'
    training={}
    processors={}
    common=[]
    lr=[]
    for arm in 'ABC':
        schedule=read(bundle/f'schedule-{arm}-train.json')
        actual=read(bundle/f'paired-mixture-{arm.lower()}-train-20260927-008-actual-updates.json')
        completion=read(trainroot/f'training-{arm}/completion.json')
        entries=schedule['entries']
        require([r['successful_update'] for r in actual['update_measurements']] == list(range(1,5001)))
        require(completion['successful_optimizer_updates']==actual['successful_optimizer_updates']==5000)
        require(len(entries)==actual['sample_exposures']==20000)
        require(len(actual['update_measurements'])==actual['target_mask_batches_measured']==5000)
        groups=collections.Counter(e['source'] for e in entries)
        unique=len({(e['source'],e['episode'],e['frame']) for e in entries})
        common.append([e for e in entries if e['slot']<3])
        lr.append([r['learning_rates_after_update'] for r in actual['update_measurements']])
        aux=[e for e in entries if e['slot']==3]
        pretrained = trainroot / 'checkpoints/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/train' / f'paired-mixture-{arm.lower()}-train-20260927-008/checkpoints/005000/pretrained_model'
        artifact = repo / '.work/paired-ABC-measurement-20260928-003-received-20260930-001/files/artifacts' / arm
        processor_manifest = read(artifact / 'manifest.json')
        processors[arm] = {}
        for name, expected in processor_manifest['files'].items():
            if 'processor' in name or name == 'pretrained_model/config.json':
                actual_digest = sha(pretrained / Path(name).name)
                require(actual_digest == expected)
                processors[arm][name] = actual_digest
        processors[arm]['saved_state_shape'] = read(pretrained / 'config.json')['input_features']['observation.state']['shape']
        processors[arm]['dataset_state_shape'] = read(artifact / 'config.json')['dataset_features']['observation.state']['shape']
        processors[arm]['effective_runtime_config_captured'] = False
        training[arm]={'successful_updates':5000,'sample_exposures':len(entries),'source_exposures':dict(groups),
            'unique_inputs':unique,'actual_fields':{k:v for k,v in actual.items() if k!='update_measurements'},
            'schedule_sha256':sha(bundle/f'schedule-{arm}-train.json'),
            'actual_sha256':sha(bundle/f'paired-mixture-{arm.lower()}-train-20260927-008-actual-updates.json'),
            'completion_sha256':sha(trainroot/f'training-{arm}/completion.json'),
            'auxiliary_frame_counts':dict(collections.Counter(e['frame'] for e in aux))}
    require(common[0]==common[1]==common[2])
    require(lr[0]==lr[1]==lr[2])
    context={'source_checks':source_checks,'bank_manifest_sha256':sha(bank/'manifest.json'),
             'bank_audit_sha256':sha(bank.parent/'collection-audit.json'),'bank_verified_metadata':verified,
             'bank':bank_rows,'training':training,'processor_checks':processors,
             'common_D_exact':True,'learning_rates_exact':True,
             'per_batch_target_masks_available_for_recomputation':False,
             'model_or_simulator_executed':False,'recompute_source_sha256':sha(__file__)}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(context,stream,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'source_matches':sum(r['matches'] for r in source_checks.values()),'source_total':len(source_checks),
     'bank':[{k:v for k,v in r.items() if k not in ('branch','deviation')} for r in bank_rows],
     'training':{a:{k:v for k,v in r.items() if k in ('successful_updates','sample_exposures','source_exposures','unique_inputs','actual_fields')} for a,r in training.items()}}))


if __name__ == '__main__':
    main()
