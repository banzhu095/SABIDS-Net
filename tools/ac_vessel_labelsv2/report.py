from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

from .common import *

COMPARISONS=(('A0','N0'),('NA','NN'),('A0','T0'),('NA','N0'),('NA','A0'))


def read_metrics(path):
    # pandas normally interprets the legitimate arm alias "NA" as missing.
    return pd.read_csv(path,keep_default_na=False,na_values=[''],dtype={'sample_id':str,'position_id':str,'group_id':str,'arm':str})


def complete_matrix(snapshot,run,seeds,arms):
    result=[]
    for f in read(Path(snapshot)/'folds.json'):
        for seed in seeds:
            for arm in arms:
                folder=Path(run)/f'fold{f["fold"]}'/f'seed{seed}'/arm
                ok=False;why='missing'
                try:
                    plan=read(folder.parent/'plan.json');done=read(folder/'complete.json');eval_meta=read(folder/'outer_best/complete.json')
                    ok=(done['identity']==plan['identity']==eval_meta['identity'] and done['epochs']==plan['config']['epochs'] and sha(folder/'best.pth')==done['best_sha256']==eval_meta['checkpoint_sha256'] and sha(folder/'outer_best/per_frame.csv')==eval_meta['metrics_sha256'])
                    for p,h in eval_meta['prediction_sha256'].items():ok=ok and sha(folder/'outer_best/predictions'/p)==h
                    why='passed' if ok else 'identity/checkpoint/budget mismatch'
                except (OSError,KeyError,ValueError):pass
                result.append(dict(fold=f['fold'],seed=seed,arm=arm,complete=ok,reason=why))
    return result


def paired_tables(position,metrics,comparison_pairs=COMPARISONS):
    rows=[]
    for a,b in comparison_pairs:
        left=position[position.arm==a].set_index(['seed','position_id','group_id']);right=position[position.arm==b].set_index(['seed','position_id','group_id'])
        require(set(left.index)==set(right.index),'Unpaired positions for '+a+'-'+b)
        right=right.reindex(left.index)
        for key,value in (left[metrics]-right[metrics]).iterrows():
            for metric,d in value.items():
                error=any(t in metric for t in ('error','mae','assd','_missed','_fp','_fn','outside_gt_layer_fraction'))
                unit='px' if 'mae' in metric or 'assd' in metric else 'percentage_points'
                scale=1 if unit=='px' else 100
                rows.append(dict(comparison=a+'-'+b,seed=key[0],position_id=key[1],group_id=key[2],metric=metric,gain=float(d)*(-1 if error else 1)*scale,unit=unit,positive_is_improvement=True))
    return pd.DataFrame(rows)


def intervals(gains):
    rng=np.random.default_rng(20261016);result=[]
    # Seeds are repeated algorithms, not extra subjects. Average them first.
    data=gains.groupby(['comparison','position_id','group_id','metric','unit'],as_index=False).gain.mean()
    for key,g in data.groupby(['comparison','metric','unit']):
        g=g[np.isfinite(g.gain)];values=g.gain.to_numpy();groups=g.group_id.unique();clusters=[g[g.group_id==c].gain.to_numpy() for c in groups]
        if not len(values):continue
        boot=np.array([np.concatenate([clusters[i] for i in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(3000)])
        sums=np.array([a.sum() for a in clusters]);null=((rng.integers(2,size=(10000,len(groups)))*2-1)*sums).sum(1)/len(values)
        raw_p=float((1+(np.abs(null)>=abs(values.mean())).sum())/(len(null)+1))
        result.append(dict(comparison=key[0],metric=key[1],unit=key[2],mean=float(values.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),positions=len(values),clusters=len(groups),
            standardized_paired_effect=float(values.mean()/values.std(ddof=1)) if len(values)>1 and values.std(ddof=1)>0 else None,
            improvement_fraction=float((values>0).mean()),raw_cluster_signflip_p=raw_p,
            cluster_unit='patient when known, otherwise position; seeds averaged before bootstrap',
            ci_scope='descriptive development CV; small clusters and shared training sets limit inference'))
    primary=[r for r in result if r['metric']=='vessel_dice' and r['comparison'] in ('A0-N0','NA-NN')]
    primary.sort(key=lambda r:r['raw_cluster_signflip_p']);previous=0.
    for i,r in enumerate(primary):previous=max(previous,min(1.,(len(primary)-i)*r['raw_cluster_signflip_p']));r['holm_p_two_preregistered_primary_endpoints']=previous
    return result


def atlas(snapshot,cache_root,run,seeds,arms,output,frame_table):
    snapshot=Path(snapshot);output=Path(output);positions=read(snapshot/'positions.json');frames=read(snapshot/'frames.json')
    failures=frame_table.sort_values('vessel_dice').groupby(['seed','arm']).head(2)
    atomic(output/'failure_cases.csv',failures.to_csv(index=False).encode('utf-8-sig'))
    fixed={min(r['sample_id'] for r in frames if r['position_id']==p['position_id']):'fixed_first_frame' for p in positions}
    for sid in failures.sample_id:fixed.setdefault(sid,'failure_not_selected_for_favorable_appearance')
    for seed in seeds:
        for sid,reason in fixed.items():
            item=next(r for r in frames if r['sample_id']==sid);pid=item['position_id']
            fold=next(f['fold'] for f in read(snapshot/'folds.json') if item['group_id'] in f['outer'])
            cache=Path(cache_root)/f'fold{fold}';row=next(r for r in read(cache/'frames.json') if r['sample_id']==sid)
            l=plane(snapshot/'labels'/pid/'layer.png')>0;gt=plane(snapshot/'labels'/pid/'vessel.png')>0;vv=plane(snapshot/'labels'/pid/'vessel_valid.png')>0
            panels=[];dest=output/'atlas'/f'seed{seed}'/sid;dest.mkdir(parents=True)
            def add(name,a):
                a=np.asarray(a,dtype=np.uint8);png(dest/f'{name}.png',a);im=Image.fromarray(a).convert('RGB');im.thumbnail((320,320));tile=Image.new('RGB',(320,344),'#172633');tile.paste(im,(0,24));ImageDraw.Draw(tile).text((5,5),name,fill='white');panels.append(tile)
            for kind in ('noisy','tone','auto')+ (('manual',) if any(a in MANUAL for a in arms) else ()):
                add(kind,np.rint(np.load(cache/row['views'][kind]['path'],allow_pickle=False)*255))
            rgb=np.zeros((*gt.shape,3),np.uint8);rgb[l]=[255,215,40];rgb[gt]=[255,65,70];rgb[~vv]=[140,60,200];add('GT_raw_layer_vessel_validity',rgb)
            for arm in arms:
                pred=np.load(Path(run)/f'fold{fold}'/f'seed{seed}'/arm/'outer_best/predictions'/f'{sid}.npz')
                p=pred['vessel'];v=(p>=.5)&vv;add(arm+'_probability',np.rint(p*255));add(arm+'_P0',v.astype(np.uint8)*255)
                error=np.zeros((*gt.shape,3),np.uint8);error[v&gt]=[0,220,70];error[v&~gt]=[255,50,50];error[~v&gt&vv]=[60,100,255];error[~vv]=[140,60,200];add(arm+'_TP_FP_FN',error)
            sheet=Image.new('RGB',(320*len(panels),344),'#172633')
            for i,panel in enumerate(panels):sheet.paste(panel,(320*i,0))
            stream=__import__('io').BytesIO();sheet.save(stream,format='PNG');atomic(dest/'comparison.png',stream.getvalue())
            write(dest/'identity.json',dict(sample_id=sid,position_id=pid,fold=fold,seed=seed,selection=reason,checkpoint='best',threshold=.5,GT_only_for_evaluation=True))


def summarize(snapshot,cache_root,run,output,seeds=(42,),manual=False):
    snapshot=Path(snapshot);run=Path(run);out=fresh(output);arms=list(MANUAL) if manual else list(ARMS)
    matrix=complete_matrix(snapshot,run,seeds,arms);table(out/'completion_matrix.csv',matrix)
    missing=[r for r in matrix if not r['complete']]
    if missing:
        result=dict(status='blocked',missing=missing,reason='Expected all preregistered arms/folds/seeds, not only existing directories');write(out/'missing_checklist.json',result);return result
    frames=[];configs=[]
    for f in read(snapshot/'folds.json'):
        for seed in seeds:
            configs.append(read(run/f'fold{f["fold"]}'/f'seed{seed}'/'plan.json')['config'])
            for arm in arms:frames.append(read_metrics(run/f'fold{f["fold"]}'/f'seed{seed}'/arm/'outer_best/per_frame.csv'))
    require(all(c==configs[0] for c in configs),'Mixed budgets/model configurations')
    frame=pd.concat(frames,ignore_index=True);expected={r['sample_id'] for r in read(snapshot/'frames.json')}
    for (_,arm),part in frame.groupby(['seed','arm']):require(set(part.sample_id)==expected and not part.sample_id.duplicated().any(),'OOF coverage incomplete/duplicated for '+arm)
    metrics=[c for c in frame.select_dtypes(include=np.number).columns if c not in ('fold','seed')]
    position=frame.groupby(['seed','arm','position_id','group_id'],as_index=False)[metrics].mean()
    per_seed=position.groupby(['seed','arm'],as_index=False)[metrics].mean()
    for name,data in (('per_frame',frame),('per_position',position),('per_seed',per_seed)):
        atomic(out/f'{name}_metrics.csv',data.to_csv(index=False).encode('utf-8-sig'))
        keys=[c for c in data.columns if c not in metrics];atomic(out/f'{name}_long.csv',data.melt(id_vars=keys,value_vars=metrics,var_name='metric',value_name='value').to_csv(index=False).encode('utf-8-sig'))
    gains=paired_tables(position,metrics,(('NM','M0'),) if manual else COMPARISONS);atomic(out/'paired_deltas.csv',gains.to_csv(index=False).encode('utf-8-sig'));ci=intervals(gains);table(out/'paired_confidence_intervals.csv',ci)
    seed_gains=gains.groupby(['comparison','seed','metric','unit'],as_index=False).gain.mean()
    atomic(out/'paired_per_seed.csv',seed_gains.to_csv(index=False).encode('utf-8-sig'))
    spread=seed_gains.groupby(['comparison','metric','unit']).gain.agg(['mean','std']).reset_index();atomic(out/'paired_seed_summary.csv',spread.to_csv(index=False).encode('utf-8-sig'))
    registry=[]
    for f in read(snapshot/'folds.json'):
        fold=f['fold'];cache=Path(cache_root)/f'fold{fold}'
        for seed in seeds:
            seed_dir=run/f'fold{fold}'/f'seed{seed}'
            for arm in arms:
                atomic(out/'histories'/f'fold{fold}_seed{seed}_{arm}.csv',
                    (seed_dir/arm/'history.csv').read_bytes())
            registry.append(dict(fold=fold,seed=seed,split=f,plan=read(seed_dir/'plan.json'),
                data_plan_sha256=sha(seed_dir/'data_plan.json'),
                cache_manifest=read(cache/'cache.json'),
                calibration={p.name:read(p) for p in cache.glob('*.json')
                    if p.name not in ('cache.json','frames.json','fixed_samples.json')},
                arms={arm:dict(initialization=read(seed_dir/arm/'initialization_audit.json'),
                    training=read(seed_dir/arm/'complete.json'),
                    evaluation=read(seed_dir/arm/'outer_best/complete.json')) for arm in arms}))
    write(out/'experiment_registry.json',registry)
    write(out/'snapshot_registry.json',dict(snapshot=read(snapshot/'snapshot.json'),
        positions=read(snapshot/'positions.json'),annotation_states=read(snapshot/'annotation_states.json'),
        historical_access=read(snapshot/'historical_test_access.json'),
        note='Raw arrays/weights excluded. Automatic report does not deserialize per-position manual parameters.'))
    atlas(snapshot,cache_root,run,seeds,arms,out,frame)
    lines=['# AC revised-label development CV','',f'Smoke={configs[0]["smoke"]}; pilot={configs[0]["pilot"]}. Development, previously manually accessed data; not blind test.',
        'P0, threshold0.5, native geometry; best selected only by inner position-macro vessel soft Dice. Last is not mixed in.',
        'Primary Vessel Dice: equal positions after within-position frame averaging. Paired positive values mean improvement; Dice in percentage points.',
        'CI:3000 patient/position-cluster bootstrap draws after seed averaging; shared fold training and small sample count limit inferential claims. Fold SD is NOT a CI.',
        'Primary comparisons A0-N0 and NA-NN:10000 clustered sign flips, Holm family2. Other comparisons/metrics exploratory.',
        'Manual inputs are reference/possibly GT-assisted, never automatic generalization evidence. Historical old-label scores are not direct controls.','']
    for r in ci:
        if r['metric']=='vessel_dice':lines.append(f'{r["comparison"]}: {r["mean"]:.3f} pp, CI[{r["ci_low"]:.3f},{r["ci_high"]:.3f}], improved fraction={r["improvement_fraction"]:.3f}.')
    lines+=['','Interpretation rules: only NA-NN improves -> AC complementary-view effect; only manual improves -> human-assisted potential; old-label-only change -> not AC attribution. Check Recall and small-component trade-offs; retain negative results.']
    atomic(out/'summary.md','\n'.join(lines).encode('utf8'))
    write(out/'missing_checklist.json',dict(status='passed',missing=[],smoke=configs[0]['smoke'],pilot=configs[0]['pilot'],positions=position.position_id.nunique(),seeds=list(seeds),arms=arms))
    return dict(status='passed',report=str(out),positions=position.position_id.nunique(),smoke=configs[0]['smoke'])


def package(report,output):
    report=Path(report);require(read(report/'missing_checklist.json')['status']=='passed','Cannot package an incomplete report as complete')
    output=Path(output);require(not output.exists(),'Existing package preserved');output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(report.rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.csv','.json','.md','.png'):z.write(p,p.relative_to(report).as_posix())
    return dict(path=str(output),sha256=sha(output),excludes='weights, optimizer, caches, npy/npz')
