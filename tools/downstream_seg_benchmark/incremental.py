"""Evaluate newly labelled TEST positions using unchanged sealed checkpoints."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import KEY,read_json,require,sha,write_csv,write_json
from .engine import evaluate_rows,load,seal_checkpoints
from .metrics import formal_tables
from .model import Segmenter
from .protocol import verify_plan


def select_new_test(old,new):
    require(not new.duplicated(KEY+['method_id']).any(),'Duplicate incremental keys')
    identity=KEY+['method_id','sha256','label_identity']
    for split in ['train','val']:
        a=old[old.split==split][identity].sort_values(KEY+['method_id']).reset_index(drop=True)
        b=new[new.split==split][identity].sort_values(KEY+['method_id']).reset_index(drop=True)
        require(a.equals(b),'Incremental evaluation cannot change training/validation assets or labels')
    previous=old[old.split=='test']
    overlap=new[new.sample_id.isin(previous.sample_id)&(new.split=='test')]
    a=previous[identity].sort_values(KEY+['method_id']).reset_index(drop=True)
    b=overlap[identity].sort_values(KEY+['method_id']).reset_index(drop=True)
    require(a.equals(b),'Old test cohort changed')
    added=new[(new.split=='test')&~new.position_id.isin(previous.position_id)]
    require(len(added)>0,'No newly labelled test positions')
    require(set(added.method_id)==set(old.method_id),'Incremental methods incomplete')
    canonical=None
    for _,group in added.groupby('method_id'):
        keys=set(map(tuple,group[KEY].to_numpy()))
        require(canonical is None or keys==canonical,'Incremental sample keys differ by method')
        canonical=keys
    return added


def evaluate_incremental(run,audit_run,output,device,resume=False):
    run,audit_run,output=map(Path,[run,audit_run,output])
    lock=verify_plan(run);seal=seal_checkpoints(run)
    require((run/'test_opened.json').exists(),'Initial test evaluation must be completed first')
    require(output.resolve()!=run.resolve(),'Incremental output must be separate')
    output.mkdir(parents=True,exist_ok=True)
    new=pd.read_csv(audit_run/'input_asset_audit.csv',keep_default_na=False)
    old=pd.read_csv(run/'cohort.csv',keep_default_na=False)
    require(new.available.astype(str).str.lower().eq('true').all(),'Incremental assets unavailable')
    added=select_new_test(old,new)
    for r in new.to_dict('records'):require(sha(r['path'])==r['sha256'],'Incremental image hash changed')
    for path,item in read_json(audit_run/'label_asset_inventory.json').items():require(sha(path)==item['sha256'],'Incremental label changed')
    write_csv(output/'incremental_cohort.csv',added)
    signature=dict(plan_sha256=sha(run/'plan_lock.json'),checkpoint_lock_sha256=sha(run/'checkpoint_lock.json'),cohort_sha256=sha(output/'incremental_cohort.csv'))
    marker=output/'incremental_lock.json'
    if marker.exists():require(resume and read_json(marker)==signature,'Incremental lock conflict')
    else:write_json(marker,signature)
    frames=[];config=lock['config']
    for checkpoint in seal['checkpoints']:
        method,seed=checkpoint['method_id'],checkpoint['segmentation_seed']
        path=output/'evaluation'/method/str(seed)/'metrics.csv'
        completed=path.parent/'complete.json'
        if completed.exists():
            require(resume and read_json(completed)['metrics_sha256']==sha(path),'Incremental cached metrics conflict')
        else:
            model=Segmenter(config['channels'],config['depths'],config['decoder_depth']).to(device)
            model.load_state_dict(load(checkpoint['path'])['model'])
            rows=added[added.method_id==method].sort_values('sample_id').to_dict('records')
            frame=evaluate_rows(model,rows,config['evaluation_size'],device,method,seed,True,path.parent/'predictions')
            write_csv(path,frame);write_json(completed,dict(metrics_sha256=sha(path),checkpoint_sha256=checkpoint['sha256']))
        frames.append(pd.read_csv(path,float_precision='round_trip'))
    frame=pd.concat([pd.read_csv(run/'per_frame_metrics.csv',float_precision='round_trip')]+frames,ignore_index=True)
    write_csv(output/'per_frame_metrics.csv',frame);formal_tables(frame,output)
    write_json(output/'incremental_evaluation_status.json',dict(**signature,status='evaluated_without_retraining',new_positions=sorted(added.position_id.unique()),
        total_positions=frame.position_id.nunique(),checkpoint_selection_changed=False))
    return output
