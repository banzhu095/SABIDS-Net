"""Paired position/patient statistics, fixed native atlas and light packaging."""
from pathlib import Path
import itertools
import json
import zipfile

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion

from .core import *
from .core import COMPARISONS,BASE,ADAPTIVE
from .learning import Frames,build,restore,verified_anchor
from tools.downstream_seg_benchmark.engine import load
from tools.denoise_then_ac_v1.reporting import gray,rgb,sheet,crop_box


def profile(model,x,device):
    """Measured complete segment/control forward; conv MACs only, explicit scope."""
    macs=[0];hooks=[]
    def count(m,args,out):
        if isinstance(m,torch.nn.Conv2d):macs[0]+=out.numel()*(m.in_channels//m.groups)*np.prod(m.kernel_size)
        else:macs[0]+=out.numel()*(m.in_channels//m.groups)*np.prod(m.kernel_size)
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Conv1d)):hooks.append(m.register_forward_hook(count))
    import time
    model.eval()
    if device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    tic=time.monotonic()
    with torch.no_grad():model(x)
    if device.startswith('cuda'):torch.cuda.synchronize()
    seconds=time.monotonic()-tic
    for h in hooks:h.remove()
    return dict(total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                conv_macs=int(macs[0]),mac_scope='Conv1d/2d only; excludes elementwise/NAF matmul/analytic AC and offline D2',
                complete_forward_seconds=seconds,cold_single_sample=True,
                peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None,
                encoder_calls=getattr(model,'last_diagnostics',{}).get('encoder_calls',2 if len(getattr(model,'fusions',{})) else 1),
                offline_d2='frozen cached preprocessing; not included in online profile')


def atlas(snapshot,cache,run,methods,cfg,seed,output,device='cpu',kind='best'):
    meta=verify_cache(cache,snapshot);out=fresh(output);snapshot,cache,run=map(Path,(snapshot,cache,run))
    fixed=set(read(cache/'fixed_samples.json'));rows=[r for r in read(cache/'frames.json') if r['sample_id'] in fixed]
    pending=[];profiles=[]
    for method in methods:
        folder=run/method;model=None
        if (folder/'complete.json').exists():
            done=read(folder/'complete.json');plan=read(folder/'plan.json')
            require(plan['cache_identity']==meta['identity'] and plan['config']==cfg,'Atlas run/cache differs')
            require(sha(folder/(kind+'.pth'))==done[kind+'_sha256'],'Atlas checkpoint differs')
            anc,h=verified_anchor(run,cfg,meta['identity']) if method in ADAPTIVE else (None,None)
            require(h==plan['anchor_sha256'],'Atlas anchor differs')
            model=build(cfg,method,seed,anc['model'] if anc else None).to(device).eval()
            model.load_state_dict(load(folder/(kind+'.pth'))['model'])
        dataset=Frames(snapshot,cache,rows,method,cfg['size'])
        for i,r in enumerate(rows):
            dest=out/r['split']/r['position_id'];dest.mkdir(parents=True,exist_ok=True)
            with np.load(cache/r['analytic'],allow_pickle=False) as data:a={k:data[k].copy() for k in data.files}
            l,v,lv,vv=dataset.native(r);reference=plane(snapshot/'references'/(r['position_id']+'.png')).astype(np.float32)/255
            box=crop_box(l)
            if method==methods[0]:
                for k in ('noisy','m','z0','c','rho','reliability','gain','depth'):
                    npy(dest/(k+'.npy'),a[k].astype(np.float32));png(dest/(k+'.png'),gray(a[k]/6 if k=='gain' else a[k]))
                for k,array in dict(layer=l,vessel=v,layer_valid=lv,vessel_valid=vv,
                    uncertain=plane(snapshot/'labels'/r['position_id']/'uncertain.png')>0,clean_reference=reference).items():
                    npy(dest/(k+'.npy'),array.astype(np.float32));png(dest/(k+'.png'),gray(array))
                panels=[(k,gray(a[k])) for k in ('noisy','m','z0')]+[('clean/reference',gray(reference))]
                title=f'{r["position_id"]} {r["sample_id"]} {r["split"]} native {r["shape"]} a=.25 b=.5 W={meta["ac_metadata"]["config"]["output_window"]:.5g}'
                sheet(dest/'input_stages.png',panels,title);sheet(dest/'input_crop.png',panels,title,box)
                write(dest/'identity.json',dict(sample_id=r['sample_id'],source_sha256=r['sha256'],native_shape=r['shape'],
                    crop_native_xyxy=box,display_window=[0,1],gamma=1,cache_identity=meta['identity'],mode=meta['mode'],
                    provenance_label=cfg.get('provenance_label',meta.get('provenance_label','see upstream audit'))))
            if model is None:
                pending.append(dict(method=method,sample_id=r['sample_id'],status='PENDING TRAINING'));continue
            batch=dataset[i];x=batch[0][None].to(device)
            if i==0:profiles.append(dict(method=method,**profile(model,x,device)))
            with torch.no_grad():pred=model(x)
            lp,vp=[restore(pred[k].sigmoid(),l.shape,cfg['size']) for k in ('layer','vessel')]
            p=dest/method
            for k,array in [('layer_probability',lp),('vessel_probability',vp),('layer_mask',lp>=.5),('vessel_mask',vp>=.5)]:
                npy(p/(k+'.npy'),array.astype(np.float32));png(p/(k+'.png'),gray(array))
            error=np.zeros((*v.shape,3),dtype=np.uint8);pv=(vp>=.5)&vv;gt=v&vv
            error[pv&gt]=[0,220,70];error[pv&~gt]=[255,50,50];error[~pv&gt]=[60,100,255];error[~vv]=[140,60,200]
            overlay=rgb(a['noisy']);overlay[(lp>=.5)^binary_erosion(lp>=.5)]=[255,220,40]
            overlay[pv]=(.6*overlay[pv]+.4*np.array([255,60,70])).astype(np.uint8)
            png(p/'TP_FP_FN.png',error);png(p/'overlay.png',overlay)
            panels=[('overlay',overlay),('TP green FP red FN blue',error)]
            if method in ADAPTIVE:
                diag=model.last_diagnostics
                for name in ('layer_image','vessel_image','effective_beta','delta_beta','rho','coarse_layer','coarse_vessel','coefficients'):
                    if name not in diag:continue
                    tensor=diag[name].detach().cpu();npy(p/(name+'_model_grid.npy'),tensor.numpy())
                    if name=='coefficients':continue
                    for c in range(tensor.shape[1]):
                        native=restore(tensor[:,c:c+1],l.shape,cfg['size'])
                        npy(p/(name+f'_{c}_native_restored.npy'),native.astype(np.float32))
                        # Explicit display for signed coefficient adjustment, not input view.
                        image=gray(.5+native if name=='delta_beta' else native)
                        png(p/(name+f'_{c}.png'),image)
                        if name in ('layer_image','vessel_image','effective_beta','rho'):panels.append((name+str(c),image))
            sheet(p/'comparison.png',panels,f'{method} {r["sample_id"]} {kind} P0 native px | '+cfg.get('provenance_label',meta['mode']))
            sheet(p/'comparison_crop.png',panels,f'{method} common crop',box)
    if profiles:table(out/'profiles.csv',profiles)
    write(out/'atlas_status.json',dict(positions=len(rows),pending=pending,checkpoint=kind,scope=meta['mode'],synthetic=meta['synthetic'],
          actual_input='*_model_grid.npy; native_restored is visualization of the resize/pad canvas, not recomputed native inference'))
    return dict(status='atlas_complete' if not pending else 'incomplete',pending=len(pending),path=str(out))


def paired_statistics(positions,comparisons=COMPARISONS):
    records=[];rng=np.random.default_rng(20261026)
    columns=[c for c in positions.select_dtypes('number').columns if c not in ('fold','seed','synthetic')]
    for a,b in comparisons:
        left=positions[positions.method==a].set_index(['seed','fold','position_id','group_id'])
        right=positions[positions.method==b].set_index(['seed','fold','position_id','group_id'])
        if left.empty or right.empty:continue
        require(set(left.index)==set(right.index),'Incomplete paired locations: '+a+'-'+b)
        for metric in columns:
            reverse=any(s in metric for s in ('error','mae','assd','roi_fp','roi_fn','outside','repeat_probability_std'))
            factor=100 if any(s in metric for s in ('dice','precision','recall','auprc','iou')) else 1
            diff=(left[metric]-right[metric].reindex(left.index))*(-1 if reverse else 1)*factor
            for key,gain in diff.items():records.append(dict(comparison=a+'-'+b,seed=key[0],fold=key[1],position_id=key[2],group_id=key[3],metric=metric,gain=gain,
                unit='percentage_points' if factor==100 else 'native_px' if 'mae' in metric or 'assd' in metric else 'native_ratio',positive_means_improvement=True))
    gains=pd.DataFrame(records);intervals=[]
    if gains.empty:return gains,pd.DataFrame()
    for (comparison,metric),part in gains.groupby(['comparison','metric']):
        # First average seeds and all positions belonging to a known patient.
        values=part.groupby('group_id').gain.mean().dropna().to_numpy()
        n=len(values)
        if not n:continue
        boot=values[rng.integers(n,size=(2000,n))].mean(1)
        if n<=15:
            signs=np.array(list(itertools.product((-1,1),repeat=n)))
            pvalue=float(np.mean(np.abs((signs*values).mean(1))>=abs(values.mean())-1e-12))
        else:
            signs=rng.choice([-1,1],size=(10000,n));pvalue=float((1+np.sum(np.abs((signs*values).mean(1))>=abs(values.mean())))/10001)
        intervals.append(dict(comparison=comparison,metric=metric,mean=values.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),
            clusters=n,improved_clusters=int((values>0).sum()),p_signflip=pvalue,
            cluster_unit='group_id (patient if known, otherwise anatomical position); seeds averaged first',
            limitation='descriptive small-sample bootstrap; one development seed is not stable generalization'))
    ci=pd.DataFrame(intervals)
    ci['p_holm_primary']=np.nan
    fixed='F5-F0' if 'F5-F0' in set(ci.comparison) else 'F5-P3'
    mask=ci.comparison.isin([fixed,'F5-F1','F5-F2']) & ci.metric.eq('vessel_dice')
    ids=ci[mask].sort_values('p_signflip').index
    adjusted=0.
    for rank,index in enumerate(ids):
        adjusted=max(adjusted,min(1.,float(ci.loc[index,'p_signflip'])*(len(ids)-rank)))
        ci.loc[index,'p_holm_primary']=adjusted
    return gains,ci


def summarize(runs,output,methods):
    out=fresh(output);missing=[];scopes={};completed=[];selection=[];snapshots=set();configs=set()
    for run in map(Path,runs):
        for method in methods:
            folder=run/method
            if not (folder/'complete.json').exists():missing.append(str(folder));continue
            plan=read(folder/'plan.json');scopes[str(run)]=plan['mode'];done=read(folder/'complete.json')
            require(plan['sources']==sources(),'Report source changed')
            snapshots.add(plan['snapshot_identity'])
            configs.add(digest({k:v for k,v in plan['config'].items() if k not in ('fold','seed','research_mode')}))
            for kind in ('best','last'):
                p=folder/('outer_'+kind);e=read(p/'complete.json') if (p/'complete.json').exists() else None
                if e is None:missing.append(str(p));continue
                require(sha(folder/(kind+'.pth'))==done[kind+'_sha256']==e['checkpoint_sha256'],'Report checkpoint mismatch')
                require(sha(p/'per_frame.csv')==e['metrics_sha256'],'Report metrics changed')
                for name,h in e['predictions'].items():require(sha(p/'predictions'/name)==h,'Report prediction changed')
                df=pd.read_csv(p/'per_frame.csv');df['checkpoint']=kind;completed.append(df)
                selection.append(dict(run=str(run),method=method,checkpoint=kind,epoch=e['epoch'],sha256=e['checkpoint_sha256'],
                    monitor='inner position-equal vessel soft Dice',tie='strict greater earliest including epoch0',mode=plan['mode']))
    require(len(set(scopes.values()))<=1,'Development and formal cannot share a primary report')
    require(len(snapshots)<=1 and len(configs)<=1,'Primary report cannot mix label snapshots, budgets or sensitivity configs')
    if selection:table(out/'checkpoint_selection.csv',selection)
    if completed:
        combined=pd.concat(completed,ignore_index=True)
        synthetic=bool(combined.synthetic.astype(str).str.lower().eq('true').any())
        for kind in ('best','last'):
            frames=combined[combined.checkpoint==kind].copy()
            if frames.empty:continue
            numeric=[c for c in frames.select_dtypes('number').columns if c not in ('fold','seed','synthetic')]
            pos=frames.groupby(['method','seed','fold','position_id','group_id'],as_index=False)[numeric].mean()
            # No double OOF predictions from two folds for same seed/method/location.
            require(not pos.duplicated(['method','seed','position_id']).any(),'Repeated OOF position across folds')
            groups=pos.groupby(['method','seed','group_id'],as_index=False)[numeric].mean()
            per_seed=groups.groupby(['method','seed'],as_index=False)[numeric].mean()
            gains,ci=paired_statistics(pos)
            for name,df in [('per_frame',frames),('per_position',pos),('per_seed',per_seed),('paired_gains',gains),('cluster_intervals',ci)]:
                df['mode']=next(iter(scopes.values()),'unknown')
                df['provenance_label']='SYNTHETIC ENGINEERING ONLY' if synthetic else 'development / historical upstream exposure' if 'development' in scopes.values() else 'formal / audited upstream'
                df['synthetic']=synthetic
                atomic(out/(kind+'_'+name+'.csv'),df.to_csv(index=False).encode('utf-8-sig'))
            aggregate=per_seed.groupby('method')[numeric].agg(['mean','std'])
            aggregate.columns=['_'.join(c) for c in aggregate.columns]
            atomic(out/(kind+'_seed_mean_std.csv'),aggregate.reset_index().to_csv(index=False).encode('utf-8-sig'))
            atomic(out/(kind+'_failures.csv'),pos.sort_values('vessel_dice').to_csv(index=False).encode('utf-8-sig'))
            # OOF only includes genuinely evaluated rows; never fill absent folds.
            atomic(out/(kind+'_observed_oof.csv'),frames.to_csv(index=False).encode('utf-8-sig'))
        full_cv=bool(not missing and not synthetic and set(combined.seed.unique())=={42,43,44}
                     and combined.groupby(['method','seed']).fold.nunique().eq(4).all())
    else:synthetic=False;full_cv=False
    status=dict(status='incomplete' if missing else 'complete_requested_scope',missing=missing,methods=list(methods),
        mode=next(iter(scopes.values()),'unknown'),synthetic=synthetic,full_cv_complete=full_cv,
        scope='ENGINEERING ONLY' if synthetic else 'development exploratory; upstream exposure retained' if 'development' in scopes.values() else 'formal grouped evaluation',
        primary_comparisons=['F5-F0','F5-F1','F5-F2'],primary_correction='Holm over available primary Vessel Dice comparisons; P3 aliases F0 when F0 absent',
        unrun_mechanism_comparisons='missing methods are not replaced by zero or old models')
    write(out/'report_status.json',status)
    text='# Learnable AC results\n\n'+json.dumps(status,ensure_ascii=False,indent=2)+'\n\n'
    text+='Best primary / last sensitivity are separate. Repeats are correlated. CI clusters by group after averaging seeds; small-sample sign-flip/CI are descriptive.\n'
    text+='No matched benefit => no mechanism claim. F5 must beat P3 AND F1; H2 requires P3>P2 with the same theta. Negative locations including 0001/0004 remain.\n'
    atomic(out/'summary.md',text.encode('utf8'));return status


def package(cache,run,atlas_dir,report,output):
    cache,run,atlas_dir,report=map(Path,(cache,run,atlas_dir,report));output=Path(output)
    require(not output.exists(),'Package exists; choose a new path')
    status=read(report/'report_status.json');manifest=dict(status=status,files={},excluded='weights/optimizers/full native cache/full probability arrays')
    require(atlas_dir.is_dir(),'Fixed atlas missing')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        for base,prefix in [(cache,'provenance'),(run,'training'),(report,'report'),(atlas_dir,'atlas')]:
            for p in sorted(base.rglob('*')):
                if not p.is_file():continue
                rel=p.relative_to(base)
                if prefix=='training' and (not rel.parts or rel.parts[0] not in BASE+ADAPTIVE or p.suffix not in ('.json','.csv','.md')):continue
                if 'native' in rel.parts or 'predictions' in rel.parts:continue
                if p.suffix not in ('.json','.yaml','.csv','.md','.png','.npy'):continue
                if p.suffix=='.npy':
                    representative=any(k in rel.parts for k in ('pku_0001','pku_0004','pku_0005'))
                    input_float=p.name in ('m.npy','z0.npy','c.npy')
                    control_float='pku_0001' in rel.parts and 'F5' in rel.parts and p.name in ('vessel_probability.npy','effective_beta_model_grid.npy','rho_model_grid.npy')
                    if not representative or not (input_float or control_float):continue
                name=prefix+'/'+rel.as_posix();z.write(p,name);manifest['files'][name]=sha(p)
        z.writestr('PACKAGE_STATUS.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(output) as z:require(z.testzip() is None,'Package CRC failure')
    result=dict(status=status['status'],path=str(output),bytes=output.stat().st_size,sha256=sha(output))
    write(output.with_suffix('.json'),result);return result
