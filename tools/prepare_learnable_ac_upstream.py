#!/usr/bin/env python
"""Generate fold-native upstream configs; never starts training or invents evidence.

Unlike the legacy D2 launcher, explicitly bind the selected lock's manifest
instead of retaining its hardcoded pku37_binary_v3 manifest. Existing D1 formal
preflight and teacher evidence gates remain mandatory and unchanged.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sabids.config import load_config,save_config
from sabids.experiments.dose_response import formal_preflight,stable_sha
from sabids.experiments.protocol_lock import load_protocol_lock,validate_checkpoint_config
from tools.ac_vessel_labelsv2.common import read,write,require,sha


def bind_fold_config(cfg,root,lock,manifest,output,stage,pretrained=None):
    """Pure binding tested without launching any optimizer."""
    result=deepcopy(cfg);result.pop('runtime',None)
    result.update(protocol_id=lock['protocol_id'],manifest_root=lock['manifest_root'],seed=42,label_type='binary')
    for key in ('data_plan_sha256','label_inventory_sha256','dataset_inventory_sha256','split_contract_sha256'):
        result[key]=lock[key]
    result['data'].update(manifest=str(manifest),root=str(root),train_split='train',val_split='val',
        train_groups=lock['train_positions'],val_groups=lock['validation_positions'],
        target_size=[512,512],normalization='fixed',max_train_samples=None,max_val_samples=None)
    result['train'].update(stage=stage,output_dir=str(output),pretrained=str(pretrained) if pretrained else None,
        resume=None,batch_size=1,gradient_accumulation_steps=2,amp=False)
    result['model'].update(d2s_enabled=False,s2d_enabled=False,enable_denoise_to_seg=False,enable_seg_to_denoise=False)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default='.')
    p.add_argument('--stage',choices=['d1','teacher','d2'],required=True)
    p.add_argument('--protocol-lock',required=True)
    p.add_argument('--split-contract',required=True)
    p.add_argument('--output',required=True,help='NEW generated YAML path')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--d1-checkpoint');p.add_argument('--d1-initial-inventory');p.add_argument('--d1-checkpoint-binding')
    p.add_argument('--teacher-checkpoint');p.add_argument('--teacher-evidence')
    p.add_argument('--vessel-strata-definition')
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    a=p.parse_args(argv);root=Path(a.project_root).resolve()
    def resolve(value):
        v=Path(value);return v.resolve() if v.is_absolute() else (root/v).resolve()
    lock_path=resolve(a.protocol_lock);split=resolve(a.split_contract);lock=load_protocol_lock(lock_path)
    require(sha(split)==lock['split_contract_sha256'],'Fold split-contract SHA differs')
    output,run=resolve(a.output),resolve(a.run_dir)
    require(not output.exists() and not run.exists(),'Output/config/run already exists; preserved')
    require('learnable_ac_v1' in run.parts and 'current' not in run.parts,'Use isolated learnable_ac_v1 upstream directory')
    require(not (set(lock['train_positions'])|set(lock['validation_positions']))&set(lock['sealed_test_positions']),
            'Outer/development overlap')
    protocol=resolve(lock['manifest_root'])
    import pandas as pd
    manifest=protocol/('train_segment.csv' if a.stage=='teacher' else 'train_denoise.csv')
    table=pd.read_csv(manifest,dtype=str).fillna('')
    require(set(table.split)=={'train','val'},'Upstream manifest must contain train/inner ONLY')
    require(set(table[table.split=='train'].group_id)==set(lock['train_positions']) and
            set(table[table.split=='val'].group_id)==set(lock['validation_positions']),'Upstream manifest groups differ')
    if a.stage!='d1':
        require(a.d1_checkpoint and a.d1_initial_inventory and a.d1_checkpoint_binding,'D1 checkpoint, initial inventory and best binding required')
        gate=formal_preflight(root,str(resolve(a.d1_checkpoint)),str(lock_path),str(split),
            'best_validation_psnr',str(resolve(a.d1_initial_inventory)),str(resolve(a.d1_checkpoint_binding)))
        require(gate['status']=='passed','Existing D1 formal gate BLOCKED: '+str(gate.get('issues')))
    if a.stage=='teacher':
        command=[sys.executable,str(ROOT/'tools/prepare_d2_teacher.py'),'--project-root',str(root),
            '--mode','formal','--template',str(ROOT/'configs/learnable_ac_v1/upstream_teacher_template.yaml'),
            '--protocol-lock',str(lock_path),'--split-contract',str(split),'--d1-checkpoint',str(resolve(a.d1_checkpoint)),
            '--d1-training-asset-inventory',str(resolve(a.d1_initial_inventory)),
            '--d1-checkpoint-binding',str(resolve(a.d1_checkpoint_binding)),
            '--output',str(output),'--run-dir',str(run),'--device',a.device]
        subprocess.run(command,cwd=root,check=True);return 0
    template='configs/next_stage_v3/d1_structure_pku37_v3.yaml' if a.stage=='d1' else 'configs/adaptive_denoising/d2_seed42_template.yaml'
    cfg=bind_fold_config(load_config(ROOT/template),root,lock,manifest,run,'denoise',resolve(a.d1_checkpoint) if a.stage=='d2' else None)
    cfg['device']=a.device
    cfg['training_asset_evidence']=dict(enabled=True,project_root=str(root),protocol_lock=str(lock_path))
    cfg['train'].update(epochs=60 if a.stage=='d1' else 20,fixed_epoch=60 if a.stage=='d1' else 20,
        early_stopping_patience=61,monitor='psnr',strict_pretrained=True,
        checkpoint_selection_rule='best_validation_psnr' if a.stage=='d1' else 'd2_hierarchical_fixed_budget_v1')
    if a.stage=='d2':
        require(a.teacher_checkpoint and a.teacher_evidence and a.vessel_strata_definition,'D25 needs teacher and train-only strata evidence')
        from tools.downstream_seg_benchmark.engine import load
        cp,ev,st=map(resolve,(a.teacher_checkpoint,a.teacher_evidence,a.vessel_strata_definition))
        teacher=load(cp);validate_checkpoint_config(teacher,lock,'fold-native D2 teacher')
        exposure=teacher['config']['data']
        require(Path(exposure['manifest']).resolve()==(protocol/'train_segment.csv').resolve(),'Teacher trained on a different fold manifest')
        evidence=read(ev);strata=read(st);unsigned=dict(strata);claimed=unsigned.pop('definition_sha256',None)
        require(claimed==stable_sha(unsigned) and strata.get('threshold_source')=='development_train_gt_only','Invalid train-only strata')
        require(evidence.get('evidence_type') in ('native_protocol_binding','derived_legacy_binding') and evidence.get('status') in ('passed','completed')
            and (evidence.get('checkpoint_sha256') or evidence.get('best_checkpoint_sha256'))==sha(cp)
            and evidence.get('test_assets_opened')==0,'Teacher evidence missing/mismatched')
        for k in ('selection_rule','training_data','split'):require(evidence.get(k),'Teacher evidence missing '+k)
        from tools.prepare_d2_seed42 import BASE_WEIGHTS,ARMS
        cfg['loss'].update(definition_version='learnable-ac-fold-local-d25-v1',restoration_mode='structure_d2',
            d2=dict(weights={**BASE_WEIGHTS,**ARMS['D25']},boundary_width_pixels=3,epsilon=1e-6,cnr_error_cap=5.,structure_beta=2.))
        cfg['loss']['weights'].update(reconstruction=1.,residual=0.,identity=.05)
        cfg['train']['memory_safe_d2_teacher']=True
        cfg['d2']=dict(enabled=True,template_only=False,arm='D25',run_mode='pilot',scientific_evaluation=True,
            selection=dict(rule='psnr_noninferiority_then_teacher',psnr_noninferiority_db=.2,tie_break='earliest_epoch',fixed_before_training=True),
            vessel_strata_definition=str(st),vessel_strata_definition_sha256=claimed,
            teacher=dict(enabled=True,checkpoint=str(cp),sha256=sha(cp),evidence=str(ev),evidence_sha256=sha(ev),
                **{k:evidence[k] for k in ('selection_rule','training_data','split')}))
    output.parent.mkdir(parents=True,exist_ok=True);save_config(cfg,output)
    write(output.with_suffix('.preparation.json'),dict(status='prepared_NOT_TRAINED',stage=a.stage,
        code_sha256=sha(__file__),protocol_lock_sha256=sha(lock_path),manifest_sha256=sha(manifest),
        no_outer_assets_opened=True,initialization='scratch' if a.stage=='d1' else str(resolve(a.d1_checkpoint)),
        command=[sys.executable,'train.py','--config',str(output)]))
    print(json.dumps(dict(status='prepared_NOT_TRAINED',config=str(output),command=['python','train.py','--config',str(output)]),indent=2));return 0


if __name__=='__main__':sys.exit(main())
