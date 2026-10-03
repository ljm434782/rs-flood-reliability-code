"""Import complete XPU artifacts without changing any trained tensor or threshold."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main():
    config = json.loads((ROOT/'configs/upgrade_20260929.json').read_text())
    original = ROOT/'migration_source'
    source_config = json.loads((original/'configs/upgrade_20260929.json').read_text())
    # Only backend/migration metadata may differ. All study parameters must agree.
    for key in ('seeds','epochs','units_per_epoch','tiles_per_unit_per_epoch','batch_size',
                'optimizer','normalization_chips_per_process','maximum_class_weight','loss',
                'models','manifests','role_grids','auxiliary_metadata','evaluation_protocol','resources'):
        if config[key] != source_config[key]:
            raise RuntimeError('Scientific configuration changed: '+key)
    old_hash = sha(original/'configs/upgrade_20260929.json')
    config_hash = sha(ROOT/'configs/upgrade_20260929.json')
    parent = original/'outputs/upgrade_20260929'
    b1_source = parent/'B1/thresholds.json'
    b1 = json.loads(b1_source.read_text())
    if b1['config_sha256'] != old_hash or b1['fit_manifest_sha256'] != config['manifests']['fit']['sha256']:
        raise RuntimeError('B1 original fit provenance mismatch')
    b1['migration'] = {'parent_thresholds_sha256':sha(b1_source),'parent_config_sha256':old_hash,
                       'thresholds_unchanged':True,'operation':'runtime identity rebinding only'}
    b1['config_sha256'] = config_hash
    b1_target = ROOT/'outputs/upgrade_20260929/B1/thresholds.json'
    if b1_target.exists():
        raise RuntimeError('Refusing to overwrite an imported B1 artifact')
    write(b1_target,b1)
    code = ['scripts/upgrade_train.py','src/flood_reliability/models.py',
            'src/flood_reliability/impactmesh_io.py','src/flood_reliability/availability.py',
            'src/flood_reliability/upgrade_masks.py','src/flood_reliability/smagnet_adapter.py',
            'third_party/smagnet/smagnet.py']
    imported = []
    registry_rows = []
    for source_run in sorted(parent.glob('B2_seed_*')):
        run = json.loads((source_run/'run.json').read_text())
        model_path = source_run/'model_only.pt'
        stats = source_run/'fit_only_statistics.json'
        if (run['status']!='FIT_COMPLETE' or run['completed_epochs']!=24
                or run['hashes']['config']!=old_hash or sha(model_path)!=run['model_only_sha256']
                or sha(stats)!=run['hashes']['statistics']):
            raise RuntimeError('Incomplete or changed original fit artifact')
        payload = torch.load(model_path,map_location='cpu',weights_only=False)
        if payload['hashes']!=run['hashes'] or payload['run_id']!=run['run_id']:
            raise RuntimeError('Original model metadata mismatch')
        target = ROOT/'outputs/upgrade_20260929'/run['run_id']
        if target.exists():
            raise RuntimeError('Refusing to overwrite imported model directory')
        target.mkdir(parents=True)
        provenance = {'parent_model_sha256':sha(model_path),'parent_run_json_sha256':sha(source_run/'run.json'),
                      'parent_hashes':run['hashes'],'original_training_device':run['device'],
                      'operation':'metadata rebinding; all state_dict tensors kept bitwise unchanged'}
        hashes={'config':config_hash,'fit_manifest':config['manifests']['fit']['sha256'],
                'freeze':sha(ROOT/'G3_FREEZE_MANIFEST.json'),
                'independent_review':sha(ROOT/'G3_INDEPENDENT_REVIEW.json'),
                'code':{p:sha(ROOT/p) for p in code},'statistics':sha(stats)}
        payload['hashes']=hashes;payload['migration']=provenance
        destination=target/'model_only.pt'
        torch.save(payload,destination)
        check=torch.load(destination,map_location='cpu',weights_only=False)
        for name,value in payload['model_state'].items():
            if not torch.equal(value,check['model_state'][name]):
                raise RuntimeError('Imported tensor changed')
        (target/'fit_only_statistics.json').write_bytes(stats.read_bytes())
        (target/'epochs.jsonl').write_bytes((source_run/'epochs.jsonl').read_bytes())
        run.update(hashes=hashes,model_only_sha256=sha(destination),migration=provenance)
        write(target/'run.json',run)
        lock_hashes={k:v for k,v in hashes.items() if k!='statistics'}
        write(target/'run.lock.json',{'run_id':run['run_id'],'seed':run['seed'],'hashes':lock_hashes})
        starts=[json.loads(line) for line in (parent/'orchestrator/runs.jsonl').read_text().splitlines()
                if line.strip() and json.loads(line).get('event')=='START'
                and json.loads(line).get('job_id')==run['run_id']]
        elapsed=(datetime.fromisoformat(run['finished_utc'])-datetime.fromisoformat(starts[-1]['started_utc'])).total_seconds()/3600
        registry_rows.append({'event':'FINISH','job_id':run['run_id'],'pid':None,
                              'status':'IMPORTED_COMPLETE_NO_RETRAIN','exit_code':None,
                              'accelerator_task':True,'xpu_task':True,'device':'xpu',
                              'elapsed_accelerator_hours':elapsed,'smagnet_task':False,
                              'source_run_sha256':provenance['parent_run_json_sha256']})
        imported.append({'run_id':run['run_id'],'model_sha256':sha(destination),
                         'source_sha256':provenance['parent_model_sha256'],'device':'xpu',
                         'tensors_bitwise_unchanged':True,'completed_epochs':24})
    registry=ROOT/'outputs/upgrade_20260929/orchestrator/runs.jsonl'
    registry.parent.mkdir(parents=True,exist_ok=True)
    registry.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in registry_rows))
    write(ROOT/'migration_artifact_import.json',{'status':'COMPLETE','imported_models':imported,
          'b1_thresholds_unchanged':True,'source_original_config_sha256':old_hash,
          'cuda_runtime_config_sha256':config_hash,'mixed_training_devices_disclosed':True,
          'remaining_fits_device':'cuda'})
    print(json.dumps({'imported_models':len(imported),'skipped_retraining':True},indent=2))


if __name__=='__main__':
    main()
