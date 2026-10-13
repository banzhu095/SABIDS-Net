from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tools.downstream_seg_benchmark.common import read_json, require, sha, write_csv, write_json
from tools.downstream_seg_benchmark.metrics import METRICS

ORDER=['noisy_identity','ksvd_self','tv_chambolle','nlm','bm3d_standard','dncnn_paired',
       'tcfl_dncnn','nafnet_paired','sabids_current','clean_oracle']
LABELS=dict(zip(ORDER,['Noisy','K-SVD-self','TV-Chambolle','NLM','BM3D','DnCNN-paired',
    'TCFL-DnCNN-unpaired','NAFNet-paired','SABIDS-current Stage-1/D0','Clean reference']))
SEEDS=[42,123,2026]
POSITIONS=['pku_0006','pku_0017','pku_0031']
MAIN=['layer_dice','vessel_dice','vessel_roi_dice','precision','recall','iou','boundary_dice',
      'vessel_assd','vessel_outside_gt_layer_fraction','vessel_outside_gt_layer_fp',
      'upper_boundary_mae','lower_boundary_mae','thickness_mae']
PAIRS=[('sabids_current',m) for m in ['nafnet_paired','noisy_identity','dncnn_paired','bm3d_standard']]+[('clean_oracle','noisy_identity')]
RATIOS=set(METRICS)-{m for m in METRICS if 'mae' in m or 'assd' in m}


def ordered(df):
    df=df.copy()
    if 'method_id' in df:
        df['display_name']=df.method_id.map(LABELS)
        df['_order']=df.method_id.map({m:i for i,m in enumerate(ORDER)})
        df=df.sort_values(['_order']+[k for k in ['position_id','segmentation_seed'] if k in df]).drop(columns='_order')
    return df


def regroup(frame):
    require(not frame.duplicated(['method_id','segmentation_seed','sample_id']).any(),'Duplicate frame keys')
    ps=frame.groupby(['method_id','segmentation_seed','position_id'],as_index=False)[METRICS].mean()
    pos=ps.groupby(['method_id','position_id'],as_index=False)[METRICS].mean()
    seeds=ps.groupby(['method_id','segmentation_seed'],as_index=False)[METRICS].mean()
    main=seeds.groupby('method_id')[METRICS].agg(['mean','std','min','max'])
    main.columns=['_'.join(c) for c in main.columns]
    return ps,pos,seeds,main.reset_index()


def paired(df,index):
    result=[]
    for a,b in PAIRS:
        x=df[df.method_id==a].set_index(index).sort_index()
        y=df[df.method_id==b].set_index(index).sort_index()
        require(x.index.equals(y.index),'Unpaired keys')
        d=(x[METRICS]-y[METRICS]).reset_index()
        d['comparison']=a+' - '+b
        for m in ['layer_dice','vessel_dice','vessel_roi_dice','precision','recall']:
            d[m+'_direction']=np.where(d[m]>0,'positive',np.where(d[m]<0,'negative','zero'))
        result.append(d)
    return pd.concat(result,ignore_index=True)


def decompose(gains):
    d=gains[gains.comparison.str.startswith('sabids_current')].copy()
    d['delta_full_dice']=d.vessel_dice
    d['delta_roi_dice']=d.vessel_roi_dice
    d['delta_outside_penalty']=d.vessel_roi_dice-d.vessel_dice
    d['outside_penalty_reduction']=-d.delta_outside_penalty
    d['identity_error']=d.delta_full_dice-(d.delta_roi_dice-d.delta_outside_penalty)
    return d


def compare_table(new,old,keys,name):
    require(not old.duplicated(keys).any(),name+' duplicate summary keys')
    a=new.set_index(keys).sort_index();b=old.set_index(keys).sort_index()
    require(a.index.equals(b.index),name+' summary key mismatch')
    result=[]
    for c in a.select_dtypes('number').columns:
        if c not in b:continue
        av,bv=a[c].to_numpy(float),b[c].to_numpy(float)
        same=np.isclose(av,bv,rtol=1e-10,atol=1e-12,equal_nan=True)
        result.append(dict(table=name,metric=c,max_abs_difference=float(np.nanmax(np.abs(av-bv))) if np.isfinite(av-bv).any() else None,
                           mismatches=int((~same).sum()),compared_rows=len(av)))
    return result


def protected_files(run):
    roots=['plan_lock.json','formal_config_lock.json','checkpoint_lock.json','checkpoint_registry.csv',
           'completion_matrix.csv','test_access_log.csv','cohort.csv','input_asset_audit.csv','label_asset_inventory.json',
           'fixed_atlas_samples.csv','resolved_config.yaml','per_frame_metrics.csv','per_position_metrics.csv',
           'per_seed_metrics.csv','per_method_metrics.csv','denoise_segmentation_joined.csv']
    files=[run/n for n in roots]
    files+=list(run.glob('tracks/*/*/*.pth'))+list(run.glob('tracks/*/*/training_curve.csv'))
    files+=list(run.glob('tracks/*/*/complete.json'))+list(run.glob('evaluation/*/*/complete.json'))+list(run.glob('evaluation/*/*/metrics.csv'))
    files+=list(run.glob('plans/*/*'))
    return {str(p.resolve()):sha(p) for p in files if p.is_file()}


def validate_finished_stages(run):
    for name in ['train','lock','evaluate','atlas','report','package']:
        marker=read_json(run/'stages'/(name+'.json'))
        require(marker['status']=='success','Original stage not successful: '+name)
        for path,expected in marker.get('outputs',{}).items():
            require(sha(run/path)==expected,'Original stage output hash mismatch: '+path)


def validate_label_sources(run):
    expected={p:d['sha256'] for p,d in read_json(run/'label_asset_inventory.json').items()}
    for path,value in expected.items():require(sha(path)==value,'Original label hash mismatch: '+path)
    return expected


def audit(run,out,resume=False):
    out.mkdir(parents=True,exist_ok=True)
    require(not out.is_relative_to(run),'Review directory must not be inside original run')
    validate_finished_stages(run)
    label_hashes=validate_label_sources(run)
    write_json(out/'label_source_hash_audit.json',dict(passed=True,labels=len(label_hashes),sealed_label_hashes=label_hashes))
    seal=read_json(run/'checkpoint_lock.json')
    keys={(m,s) for m in ORDER for s in SEEDS}
    require(len(seal['checkpoints'])==30 and {(d['method_id'],d['segmentation_seed']) for d in seal['checkpoints']}==keys,'Checkpoint coverage differs')
    matrix=pd.read_csv(run/'completion_matrix.csv')
    require(len(matrix)==30 and set(zip(matrix.method_id,matrix.segmentation_seed))==keys and matrix.status.eq('complete').all() and matrix.updates.eq(33120).all(),'Completion matrix differs')
    access=pd.read_csv(run/'test_access_log.csv')
    require(access.threshold.eq(.5).all() and access.checkpoint_lock_sha256.eq(sha(run/'checkpoint_lock.json')).all(),'Original test-access binding differs')
    write_json(out/'source_stage_hash_audit.json',dict(passed=True,stages=['train','lock','evaluate','atlas','report','package'],
        checkpoint_keys=30,complete_matrix_keys=30,original_test_threshold=.5,test_access_bound_to_checkpoint_lock=True))
    baseline=out/'immutable_before.json'
    if baseline.exists():
        require(resume,'Review already exists; use --resume')
        before=read_json(baseline)
        require(all(Path(p).is_file() and sha(p)==h for p,h in before.items()),'Original source changed during review')
        return
    lock=read_json(run/'plan_lock.json');c=lock['config']
    require(c['mode']=='formal' and set(c['methods'])==set(ORDER) and c['segmentation_seeds']==SEEDS,'Unexpected formal cohort')
    require(c['threshold']==.5 and c['epochs']==60 and c['target_size']==384 and c['evaluation_size']==640,'Unexpected fixed protocol')
    from tools.downstream_seg_benchmark.protocol import _verify_sealed_assets
    _verify_sealed_assets(run,lock)
    require(sha(run/'plan_lock.json')==read_json(run/'checkpoint_lock.json')['plan_sha256'],'Checkpoint/plan binding mismatch')
    write_json(baseline,protected_files(run))
    sources=out/'source_evidence';sources.mkdir(exist_ok=True)
    for name in ['plan_lock.json','formal_config_lock.json','checkpoint_lock.json','checkpoint_registry.csv','completion_matrix.csv',
                 'test_access_log.csv','fixed_atlas_samples.csv','resolved_config.yaml','bootstrap_confidence_intervals.csv']:
        require((run/name).is_file(),'Missing source '+name);shutil.copyfile(run/name,sources/name)
    write_json(out/'review_scope.json',dict(source_run=str(run),original_plan_sha256=sha(run/'plan_lock.json'),
        original_training_source_sha256=lock['source_sha256'],read_only=True,threshold=.5,no_training=True,no_inference=True,
        order=ORDER,labels=LABELS,independent_test_positions=3,segmentation_seeds=SEEDS,denoiser_primary_seed=42))


def tables(run,out):
    frame=pd.read_csv(run/'per_frame_metrics.csv',float_precision='round_trip')
    require(len(frame)==4500 and set(frame.method_id)==set(ORDER) and set(frame.segmentation_seed)==set(SEEDS),'Incomplete frame coverage')
    require(set(frame.position_id)==set(POSITIONS),'Unexpected test positions')
    cohort=pd.read_csv(run/'cohort.csv',keep_default_na=False)
    canon=cohort[(cohort.split=='test')&(cohort.method_id=='noisy_identity')].set_index('sample_id')
    keys=set(zip(canon.index,canon.position_id))
    for _,g in frame.groupby(['method_id','segmentation_seed']):require(set(zip(g.sample_id,g.position_id))==keys,'Method sample keys differ')
    nan=[]
    for m in METRICS:
        a=frame[m].to_numpy(float)
        require(not np.isinf(a).any(),'Infinite segmentation metric '+m)
        if np.isnan(a).any():
            require(m in ['small_vessel_recall','medium_vessel_recall','large_vessel_recall'],'Unexpected NaN '+m)
            nan.append(dict(metric=m,nan_frames=int(np.isnan(a).sum()),meaning='No GT pixels in size bin; not applicable, never filled with zero'))
    write_csv(out/'metric_missingness_audit.csv',nan,['metric','nan_frames','meaning'])
    ps,pos,seeds,main=regroup(frame)
    diffs=[]
    cached=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in run.glob('evaluation/*/*/metrics.csv')],ignore_index=True)
    diffs+=compare_table(frame,cached,['method_id','segmentation_seed','sample_id'],'per-frame vs persisted per-track evaluation')
    for table,new,ks in [('per_position_seed_metrics.csv',ps,['method_id','segmentation_seed','position_id']),
                         ('per_position_metrics.csv',pos,['method_id','position_id']),('per_seed_metrics.csv',seeds,['method_id','segmentation_seed']),
                         ('per_method_metrics.csv',main,['method_id'])]:
        diffs+=compare_table(new,pd.read_csv(run/table),ks,table)
    write_csv(out/'aggregation_difference_audit.csv',diffs)
    require(all(x['mismatches']==0 for x in diffs),'Reaggregation differs from sealed results')
    write_csv(out/'per_frame_metrics.csv',frame)
    for name,d in [('table2_main_results.csv',main),('table3_position_results.csv',pos),('table4_seed_results.csv',seeds),
                   ('position_seed_results.csv',ps)]:write_csv(out/name,ordered(d))
    gp=paired(pos,['position_id']);gs=paired(seeds,['segmentation_seed']);gps=paired(ps,['segmentation_seed','position_id'])
    write_csv(out/'paired_by_position.csv',gp);write_csv(out/'paired_by_seed.csv',gs);write_csv(out/'paired_position_seed.csv',gps)
    mean=gp.groupby('comparison',as_index=False)[METRICS].mean();write_csv(out/'paired_mean.csv',mean)
    dd=[]
    for scope,data in [('position',gp),('seed',gs),('position_seed',gps),('method',mean)]:
        d=decompose(data);d['scope']=scope;dd.append(d)
    write_csv(out/'dice_gain_decomposition.csv',pd.concat(dd,ignore_index=True))
    write_csv(out/'outside_penalty.csv',ordered(pos.assign(outside_penalty=pos.vessel_roi_dice-pos.vessel_dice)))
    loo=[]
    for comparison,g in gp.groupby('comparison'):
        for omit in POSITIONS:
            vals=g[g.position_id!=omit][METRICS].mean().to_dict()
            loo.append(dict(comparison=comparison,omitted_position=omit,remaining_positions=2,**vals))
    write_csv(out/'leave_one_position_out.csv',loo)
    protocol=[];training=[];missing=[]
    seal=read_json(run/'checkpoint_lock.json')
    for item in seal['checkpoints']:
        m,s=item['method_id'],int(item['segmentation_seed']);track=run/'tracks'/m/str(s)
        done=read_json(track/'complete.json');curve=pd.read_csv(track/'training_curve.csv')
        require(len(curve)==60 and curve.epoch.tolist()==list(range(1,61)) and done['updates']==33120,'Budget incomplete')
        ev=read_json(run/'evaluation'/m/str(s)/'complete.json')
        require(ev['checkpoint_sha256']==item['sha256'] and ev['sha256']==sha(run/'evaluation'/m/str(s)/'metrics.csv'),'Evaluation binding mismatch')
        for sample in canon.index:
            path=run/'evaluation'/m/str(s)/'predictions'/(sample+'.npz')
            if not path.exists():missing.append(dict(method_id=m,segmentation_seed=s,sample_id=sample,reason='NPZ missing; no substitution or inference'))
        p=Path(item['path'])
        if p.exists():
            require(sha(p)==item['sha256'],'Selected checkpoint hash mismatch')
            import torch
            state=torch.load(p,map_location='cpu',weights_only=False)
            ep,updates=int(state['epoch']),int(state['updates']);v,l=map(float,state['validation_score']);status='checkpoint_metadata_verified'
            require(state['plan_sha256']==sha(run/'plan_lock.json') and state['method_id']==m and state['segmentation_seed']==s,'Best metadata binding mismatch')
            del state
        else:
            score=curve.dropna(subset=['val_vessel_roi_dice','val_layer_dice']).iloc[0]
            for row in curve.dropna(subset=['val_vessel_roi_dice']).to_dict('records'):
                if row['val_vessel_roi_dice']>score['val_vessel_roi_dice']+.0001 or (abs(row['val_vessel_roi_dice']-score['val_vessel_roi_dice'])<=.0001 and row['val_layer_dice']>score['val_layer_dice']):score=row
            ep,updates=int(score['epoch']),int(score['updates']);v,l=float(score['val_vessel_roi_dice']),float(score['val_layer_dice']);status='curve_inferred_NOT_independently_verified'
        best=curve[curve.epoch==ep].iloc[0];last=curve.iloc[-1]
        require(int(last['updates'])==33120 and updates<=33120,'Actual updates exceed/differ from uniform budget')
        require(int(best['updates'])==updates and np.isclose(best.val_vessel_roi_dice,v) and np.isclose(best.val_layer_dice,l),'Curve/best metadata mismatch')
        row=dict(method_id=m,segmentation_seed=s,best_epoch=ep,best_checkpoint_updates=updates,total_training_updates=33120,
                 best_validation_roi_dice=v,best_validation_layer_dice=l,last_validation_roi_dice=float(last.val_vessel_roi_dice),
                 last_validation_layer_dice=float(last.val_layer_dice),best_to_last_roi_drop=v-float(last.val_vessel_roi_dice),
                 best_to_last_layer_drop=l-float(last.val_layer_dice),amp_replays=int(curve.amp_window_retries.sum()),
                 completed_epochs=60,uniform_budget_complete=True,checkpoint_metadata_status=status,checkpoint_sha256=item['sha256'],
                 best_in_epoch_5_to_15=5<=ep<=15)
        training.append(row)
        counts={}
        c=cohort[cohort.method_id==m]
        for split in ['train','val','test']:
            counts[split+'_frames']=int((c.split==split).sum());counts[split+'_positions']=int(c[c.split==split].position_id.nunique())
        require([counts[k] for k in ['train_frames','val_frames','test_frames','train_positions','val_positions','test_positions']]==[552,227,150,12,5,3],'Protocol coverage mismatch')
        protocol.append(dict(**row,denoiser_seed=int(done['denoiser_seed']),training_seconds=float(done['seconds']),
            architecture='Independent single-input NAF-style shared encoder; layer/vessel heads; no denoiser or UGBI',
            initialization_sha256=done['initialization_sha256'],data_plan_sha256=done['data_plan_sha256'],
            sampler_plan_sha256=done['sampler_plan_sha256'],augmentation_plan_sha256=done['augmentation_plan_sha256'],
            state='complete',**counts))
        dst=out/'training_curves'/m/str(s);dst.mkdir(parents=True,exist_ok=True);shutil.copyfile(track/'training_curve.csv',dst/'training_curve.csv')
    write_csv(out/'table1_protocol_tracks.csv',ordered(pd.DataFrame(protocol)))
    write_csv(out/'table5_training_selection.csv',ordered(pd.DataFrame(training)))
    write_csv(out/'missing_prediction_assets.csv',missing,['method_id','segmentation_seed','sample_id','reason'])
    for seed in SEEDS:
        group=[d for d in protocol if d['segmentation_seed']==seed]
        for k in ['initialization_sha256','data_plan_sha256','sampler_plan_sha256','augmentation_plan_sha256']:
            require(len({d[k] for d in group})==1,'Common randomness differs across methods')
    write_csv(out/'common_randomness_audit.csv',[dict(segmentation_seed=seed,artifact=k,methods=10,unique_hashes=1,
        sha256=next(d[k] for d in protocol if d['segmentation_seed']==seed),source_seal_verified=True)
        for seed in SEEDS for k in ['initialization_sha256','data_plan_sha256','sampler_plan_sha256','augmentation_plan_sha256']])
    # Source join is already primary-denoiser-seed=42 and restricted to these positions.
    joint=pd.read_csv(run/'denoise_segmentation_joined.csv')
    require(set(joint.position_id)==set(POSITIONS) and not joint.duplicated(['method_id','position_id']).any(),'Denoise join key mismatch')
    joincheck=compare_table(pos,joint,['method_id','position_id'],'denoise/segmentation join')
    require(all(x['mismatches']==0 for x in joincheck),'Join segmentation values differ')
    cols=[c for c in ['psnr','ssim','epi','hf_energy_ratio'] if c in joint]
    joint=pos.merge(joint[['method_id','position_id']+cols],on=['method_id','position_id'],validate='one_to_one')
    joint['denoiser_seed']=joint.method_id.map(lambda m:42 if m in ['sabids_current','nafnet_paired','dncnn_paired','tcfl_dncnn'] else 0)
    write_csv(out/'table6_denoise_segmentation_positions.csv',ordered(joint))
    write_csv(out/'table6_denoise_segmentation_methods.csv',ordered(joint.groupby('method_id',as_index=False)[cols+MAIN].mean()))
    write_csv(out/'missing_test_labels.csv',[dict(position_id=p,status='unlabelled; not evaluated') for p in ['pku_0025','pku_0038','pku_0043']])
    write_json(out/'table_audit.json',dict(tracks=30,test_records=4500,duplicate_records=0,methods=10,seeds=SEEDS,positions=POSITIONS,
        source_reaggregation_matched=True,missing_npz=len(missing),best_metadata_verified=sum(d['checkpoint_metadata_status']=='checkpoint_metadata_verified' for d in training),
        best_epochs_5_to_15=sum(d['best_in_epoch_5_to_15'] for d in training),independent_positions=3,seed_sd_ddof=1,
        aggregation='frame -> position within seed -> seed position mean -> method seed mean/SD'))
    build_workbook_data(out)


def build_workbook_data(out):
    specs=[('主结果','table2_main_results.csv'),('逐位置','table3_position_results.csv'),('逐种子','table4_seed_results.csv'),
           ('配对差值','paired_mean.csv'),('Dice收益拆分','dice_gain_decomposition.csv'),('训练选模','table5_training_selection.csv'),
           ('降噪分割对应','table6_denoise_segmentation_positions.csv'),('协议与资产索引','table1_protocol_tracks.csv')]
    tabs=[]
    for title,name in specs:
        d=pd.read_csv(out/name)
        if title=='主结果':cols=['display_name']+[m+s for m in MAIN for s in ['_mean','_std']]
        elif title in ['逐位置','逐种子']:cols=['display_name']+[c for c in ['position_id','segmentation_seed'] if c in d]+MAIN
        elif title=='配对差值':cols=['comparison']+MAIN
        elif title=='Dice收益拆分':
            d=d[d.scope.isin(['method','position','seed'])];cols=['comparison','scope','position_id','segmentation_seed','delta_full_dice','delta_roi_dice','outside_penalty_reduction','identity_error']
        elif title=='训练选模':cols=['display_name','segmentation_seed','best_epoch','best_checkpoint_updates','total_training_updates','best_validation_roi_dice','best_validation_layer_dice','last_validation_roi_dice','last_validation_layer_dice','best_to_last_roi_drop','best_to_last_layer_drop','amp_replays','checkpoint_metadata_status']
        elif title=='降噪分割对应':cols=['display_name','position_id','denoiser_seed','psnr','ssim','epi','hf_energy_ratio','layer_dice','vessel_dice','vessel_roi_dice','precision','recall']
        else:cols=list(d.columns)
        columns=[]
        for c in cols:
            base=c.removesuffix('_mean').removesuffix('_std')
            fmt='0.00'
            if base in RATIOS or 'validation' in c or 'roi_drop' in c or 'layer_drop' in c:fmt='0.00%'
            if title=='配对差值' and base in RATIOS:fmt='0.00" pp"';d[c]=d[c]*100
            if c in ['delta_full_dice','delta_roi_dice','outside_penalty_reduction']:fmt='0.00" pp"';d[c]=d[c]*100
            if any(t in c for t in ['seed','epoch','updates','frames','positions','amp_replays']):fmt='0'
            label=c.replace('_std',' (种子间SD)').replace('_mean','')+(' (px)' if 'mae' in c or 'assd' in c else '')
            columns.append(dict(key=c,label=label,format=fmt))
        tabs.append(dict(name=title,source=name,columns=columns,rows=json.loads(d[cols].to_json(orient='records'))))
    write_json(out/'workbook_data.json',dict(tabs=tabs,notes='3 independent positions; SD is between segmentation seeds; Clean reference is not a denoising algorithm.'))
