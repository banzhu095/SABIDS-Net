"""Shared training budget, epoch-zero anchor, full-state resume and native P0."""
import random
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

from .core import *
from .core import PACK,VIEWS,BASE,ADAPTIVE
from tools.denoise_then_ac_v1.learning import (
    Model as Baseline, canvas,restore,schedule,objective as segmentation_loss,
)
from tools.downstream_seg_benchmark.engine import load,save
from tools.downstream_seg_benchmark.metrics import metrics
from tools.ac_vessel_labelsv2.processing import components
from sabids.models.learnable_ac import AdaptiveSegmenter,control_regularization


class Frames(Dataset):
    def __init__(self,snapshot,cache,rows,method,size,plan=None):
        self.snapshot,self.cache=Path(snapshot),Path(cache)
        self.rows,self.method,self.size,self.plan=rows,method,size,plan

    def __len__(self):return len(self.rows) if self.plan is None else len(self.plan)

    def native(self,row):
        base=self.snapshot/'labels'/row['position_id']
        masks={k:plane(base/(k+'.png'))>0 for k in ('layer','vessel','layer_valid','vessel_valid','coverage','uncertain')}
        known=masks['coverage'] & ~masks['uncertain']
        return masks['layer'],masks['vessel'],masks['layer_valid']&known,masks['vessel_valid']&known

    def __getitem__(self,index):
        item=self.plan[index] if self.plan is not None else dict(index=index,flip_x=False)
        row=self.rows[item['index']]
        with np.load(self.cache/row['analytic'],allow_pickle=False) as arrays:
            if self.method in ADAPTIVE:xx=[arrays[k].copy() for k in PACK]
            else:
                name=VIEWS[self.method]
                a=plane(self.snapshot/'references'/(row['position_id']+'.png')).astype(np.float32)/255 if name=='clean' else arrays[name]
                xx=[a.copy()]
                if self.method=='P5':xx=[arrays['noisy'].copy(),arrays['z0'].copy()]
        x=torch.cat([canvas(a,self.size,not (self.method in ADAPTIVE and j==7)) for j,a in enumerate(xx)])
        ys=[canvas(a,self.size,False) for a in self.native(row)]
        if item['flip_x']:x=x.flip(-1);ys=[a.flip(-1) for a in ys]
        return x,*ys,row['sample_id'],row['position_id']


def model_hash(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build(cfg,method,seed,anchor=None):
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if method in BASE:return Baseline(cfg,method=='P5',seed)
        require(anchor is not None,'Explicit verified P3 best anchor required')
        m=Baseline(cfg,False,seed);m.load_state_dict(anchor)
        return AdaptiveSegmenter(m,method,cfg['delta'],cfg['context_channels'])


def objective(model,output,batch,cfg):
    loss=segmentation_loss(output,*batch[1:5])
    raw={};weighted={}
    if isinstance(model,AdaptiveSegmenter) and model.control is not None:
        raw=control_regularization(model.last_diagnostics)
        weighted={k:v*(cfg['lambda_anchor'] if k=='anchor' else cfg['lambda_smooth']) for k,v in raw.items()}
        loss=loss+sum(weighted.values())
    return loss,dict(segmentation=float((loss-sum(weighted.values())).detach()),
                     **{k+'_raw':float(v.detach()) for k,v in raw.items()},
                     **{k+'_weighted':float(v.detach()) for k,v in weighted.items()})


def average_precision(prob,truth):
    if not len(truth) or not truth.any():return float('nan')
    order=np.argsort(-prob,kind='stable');s=prob[order];y=truth[order]
    last=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[last];precision=tp/(last+1);recall=tp/y.sum()
    return float(np.sum(np.diff(np.r_[0,recall])*precision))


def exterior_vessel_false_positive(prob,layer,vessel,valid):
    # Reviewed Vessel positives outside Layer are genuine positives, not FP.
    negatives=valid & ~layer & ~vessel
    return float(((prob>=.5)&negatives).sum()/max(1,int(negatives.sum())))


def assess(model,snapshot,cache,rows,method,cfg,device,predictions=None):
    model.eval();dataset=Frames(snapshot,cache,rows,method,cfg['size'])
    difficulty=read(Path(cache)/'difficulty.json');records=[];stability={}
    started=time.monotonic()
    with torch.no_grad():
        for i,row in enumerate(rows):
            batch=dataset[i];out=model(batch[0][None].to(device));l,v,lv,vv=dataset.native(row)
            lp,vp=[restore(out[k].sigmoid(),l.shape,cfg['size']) for k in ('layer','vessel')]
            valid_layer=lv&~(v&~l)
            values=metrics(lp,vp,l,v,valid_layer,vv)
            values['vessel_outside_gt_layer_fp']=exterior_vessel_false_positive(vp,l,v,vv)
            # Replace historical hardcoded model-grid component bins by frozen native thresholds.
            for k in ('small_vessel_recall','medium_vessel_recall','large_vessel_recall'):values.pop(k,None)
            gt=v&vv;pred=(vp>=.5)&vv
            values.update(vessel_soft_dice=float((2*(vp*gt).sum()+1e-6)/((vp*vv).sum()+gt.sum()+1e-6)),
                vessel_auprc=average_precision(vp[vv],v[vv]),
                vessel_area_error=float(abs(int(pred.sum())-int(gt.sum()))/max(1,int(gt.sum()))))
            with np.load(Path(cache)/row['analytic'],allow_pickle=False) as a:x=a['noisy']
            ids,cc=components(x,l,v,vv)
            for name,condition in [('small',lambda c:c['area_px']<=difficulty['small_area_px']),
                ('large',lambda c:c['area_px']>difficulty['small_area_px']),
                ('low_contrast',lambda c:c['signed_local_contrast'] is not None and abs(c['signed_local_contrast'])<=difficulty['low_contrast_abs'])]:
                selected=[c for c in cc if c['fully_known'] and condition(c)]
                mask=np.isin(ids,[c['component_id'] for c in selected])&vv
                values[name+'_pixel_recall']=float(pred[mask].mean()) if mask.any() else float('nan')
            yy=np.arange(l.shape[0])[:,None];top=np.argmax(l,axis=0)[None,:]
            bottom=l.shape[0]-1-np.argmax(l[::-1],axis=0)[None,:]
            depth=(yy-top)/np.maximum(bottom-top,1)
            for name,lo,hi in [('shallow',0,1/3),('middle',1/3,2/3),('deep',2/3,1.00001)]:
                mask=gt&l&(depth>=lo)&(depth<hi)
                values[name+'_recall']=float(pred[mask].mean()) if mask.any() else float('nan')
            pid=row['position_id']
            if pid not in stability:stability[pid]=[np.zeros_like(vp,dtype=np.float64),np.zeros_like(vp,dtype=np.float64),0,vv]
            acc=stability[pid];acc[0]+=vp;acc[1]+=vp.astype(np.float64)**2;acc[2]+=1
            records.append(dict(sample_id=row['sample_id'],position_id=pid,group_id=row['group_id'],method=method,**values))
            if predictions:npz(Path(predictions)/(row['sample_id']+'.npz'),layer=lp,vessel=vp)
    for record in records:
        total,total2,n,vv=stability[record['position_id']]
        record['repeat_probability_std']=float(np.sqrt(np.maximum(total2/n-(total/n)**2,0))[vv].mean()) if vv.any() else float('nan')
    return pd.DataFrame(records),time.monotonic()-started


def capture_rng(device):
    return dict(torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])


def restore_rng(s,device):
    torch.set_rng_state(s['torch_rng'].cpu());np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
    if device.startswith('cuda'):torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])


def validate_state(state,plans,epochs):
    required={'model','optimizer','scheduler','epoch','best','history','updates','identity',
              'torch_rng','numpy_rng','python_rng','cuda_rng','anchor_sha256','best_epoch'}
    require(required<=set(state),'Incomplete checkpoint; weights-only is not full resume')
    epoch=state['epoch'];require(0<=epoch<=epochs,'Invalid resume epoch')
    expected=sum((len(p)+1)//2 for p in plans[:epoch])
    require(state['updates']==expected,'Saved update budget differs')
    require(state['scheduler']['last_epoch']==epoch,'Saved scheduler epoch differs')
    if expected:
        steps=[int(s['step']) for s in state['optimizer']['state'].values() if 'step' in s]
        require(steps and max(steps)==expected and min(steps)>0,'Saved AdamW steps differ')
    require(state['history'][-1]['epoch']==epoch,'History/state epoch differs')
    return epoch


def strict_better(score,best):return np.isfinite(score) and score>best


def verified_anchor(run,cfg,cache_identity):
    p=Path(run)/'P3';done=read(p/'complete.json');plan=read(p/'plan.json')
    require(done['epochs']==cfg['base_epochs'] and plan['cache_identity']==cache_identity,'P3 incomplete/cache mismatch')
    require(plan['config']==cfg and done['best_sha256']==sha(p/'best.pth'),'P3 config/checkpoint changed')
    state=load(p/'best.pth')
    require(state['identity']==plan['identity'],'P3 identity differs')
    return state,done['best_sha256']


def train(snapshot,cache,run,methods,cfg,seed=42,device='cpu',resume=False,stop_after=None):
    snapshot,cache,run=Path(snapshot),Path(cache),Path(run)
    meta=verify_cache(cache,snapshot);require(cfg['synthetic']==meta['synthetic'],'Synthetic cache cannot enter scientific run')
    require(seed in (42,43,44),'Unregistered segmentation seed')
    require(not device.startswith('cuda') or torch.cuda.is_available(),'CUDA unavailable')
    run.mkdir(parents=True,exist_ok=True)
    study=dict(config=cfg,cache_identity=meta['identity'],snapshot_identity=meta['snapshot_identity'],seed=seed)
    if (run/'study_plan.json').exists():require(read(run/'study_plan.json')==study,'Study-level config/cache/seed differs; arms cannot silently use different budgets or supervision')
    else:write(run/'study_plan.json',study)
    rows=read(cache/'frames.json');training=[r for r in rows if r['split']=='train'];inner=[r for r in rows if r['split']=='inner']
    all_plans=schedule(training,max(cfg['base_epochs'],cfg['adaptive_epochs']),seed)
    for method in methods:
        adaptive=method in ADAPTIVE
        require(method in BASE+ADAPTIVE,'Unknown method')
        require(adaptive or method in meta['allowed_base'],'Input unavailable: '+method)
        require(not adaptive or not meta['no_mild'],'Adaptive mild path requires legal mild assets')
        anchor_state,anchor_sha=verified_anchor(run,cfg,meta['identity']) if adaptive else (None,None)
        epochs=0 if method=='F0' else cfg['adaptive_epochs'] if adaptive else cfg['base_epochs']
        plans=all_plans[:epochs]
        identity=digest(dict(config=cfg,cache_identity=meta['identity'],method=method,seed=seed,anchor_sha256=anchor_sha,sources=sources()))
        folder=run/method;folder.mkdir(exist_ok=True)
        plan=dict(identity=identity,config=cfg,cache_identity=meta['identity'],method=method,seed=seed,mode=meta['mode'],
                  snapshot_identity=meta['snapshot_identity'],anchor_sha256=anchor_sha,epochs=epochs,sources=sources(),data_plan_sha256=digest(plans))
        if (folder/'plan.json').exists():require(read(folder/'plan.json')==plan,'Run plan differs; preserve and use new directory')
        else:write(folder/'plan.json',plan)
        model=build(cfg,method,seed,anchor_state['model'] if adaptive else None).to(device)
        trainable=[p for p in model.parameters() if p.requires_grad]
        frozen_before=model_hash(model.anchor) if adaptive else None
        if (folder/'complete.json').exists():
            done=read(folder/'complete.json')
            require(resume and done['identity']==identity and done['best_sha256']==sha(folder/'best.pth') and done['last_sha256']==sha(folder/'last.pth'),'Completed run changed or resume not requested')
            state=load(folder/'last.pth')
            if method!='F0':validate_state(state,plans,epochs)
            require(state['epoch']==epochs,'Complete record disagrees with checkpoint')
            continue
        random.seed(seed+773);np.random.seed(seed+773);torch.manual_seed(seed+773)
        optimizer=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) if trainable else None
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,epochs),eta_min=cfg['min_lr']) if optimizer else None
        start=0;updates=0;best=-float('inf');best_epoch=0;history=[]
        if (folder/'last.pth').exists():
            require(resume,'Partial run exists; use resume')
            state=load(folder/'last.pth');require(state['identity']==identity,'Resume identity differs');start=validate_state(state,plans,epochs)
            selected=load(folder/'best.pth');require(selected['epoch']==state['best_epoch'] and selected['score']==state['best'],'Best/last transaction differs; preserve evidence')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state,device)
            updates,best,best_epoch,history=state['updates'],state['best'],state['best_epoch'],state['history']
        else:
            require(not any((folder/n).exists() for n in ('epoch0.pth','best.pth')),
                    'Existing checkpoint without full last state; preserve this run and inspect, never overwrite best')
            validation,_=assess(model,snapshot,cache,inner,method,cfg,device)
            best=float(validation.groupby('position_id').vessel_soft_dice.mean().mean());require(np.isfinite(best),'Epoch0 validation nonfinite')
            save(folder/'epoch0.pth',dict(model=model.state_dict(),epoch=0,score=best,identity=identity,anchor_sha256=anchor_sha))
            save(folder/'best.pth',load(folder/'epoch0.pth'))
            history=[dict(epoch=0,inner_position_vessel_soft_dice=best,best=best,best_epoch=0,updates=0,seconds=0)]
            # First-epoch interruptions also resume from a COMPLETE boundary.
            # A best-ahead-of-last interruption still stops at transaction check.
            if epochs:
                save(folder/'last.pth',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    epoch=0,best=best,best_epoch=0,history=history,updates=0,identity=identity,
                    anchor_sha256=anchor_sha,**capture_rng(device)))
        write(folder/'initialization.json',dict(anchor_sha256=anchor_sha,frozen_anchor_sha256=frozen_before,
              total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in trainable),
              data_plan_sha256=digest(plans),initial_model_sha256=model_hash(build(cfg,method,seed,anchor_state['model'] if adaptive else None)),
              boundary_auxiliary=False,encoder_calls=3 if adaptive and method!='F0' else 2 if method=='P5' else 1))
        for epoch in range(start,epochs):
            if stop_after is not None and epoch>=stop_after:break
            tic=time.monotonic();model.train();dataset=Frames(snapshot,cache,training,method,cfg['size'],plans[epoch])
            loader=DataLoader(dataset,batch_size=1,num_workers=0,shuffle=False,generator=torch.Generator().manual_seed(seed+epoch+1927))
            if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True);sum_loss=0;raw_totals={};window=min(2,len(loader));grad_norms=[]
            controller_before=[p.detach().clone() for p in trainable] if adaptive else None
            for j,batch in enumerate(loader):
                data=[t.to(device) for t in batch[:5]];out=model(data[0]);loss,parts=objective(model,out,data,cfg)
                require(torch.isfinite(loss).item(),'Nonfinite loss: '+batch[5][0])
                (loss/window).backward();sum_loss+=float(loss.detach())
                for k,v in parts.items():raw_totals[k]=raw_totals.get(k,0)+v
                if (j+1)%2==0 or j+1==len(loader):
                    norm=torch.nn.utils.clip_grad_norm_(trainable,cfg['clip']);require(torch.isfinite(norm).item(),'Nonfinite gradient')
                    grad_norms.append(float(norm))
                    probe=next((p for p in trainable if p.grad is not None and p.grad.abs().max()>0),None) if updates==0 else None
                    probe_before=probe.detach().clone() if probe is not None else None
                    optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                    if updates==1:
                        changed=probe is not None and not torch.equal(probe_before,probe.detach())
                        require(changed,'First optimizer step did not update an active parameter')
                        write(folder/'first_update.json',dict(pid=os.getpid(),method=method,epoch=epoch+1,updates=updates,
                            loss=float(loss.detach()),finite=True,parameter_changed=True,gradient_norm=float(norm),
                            config_sha256=digest(cfg),code_sha256=digest(sources()),mode=meta['mode'],
                            provenance_label=cfg.get('provenance_label',meta.get('provenance_label','see audit'))))
                    window=min(2,len(loader)-j-1)
            validation,_=assess(model,snapshot,cache,inner,method,cfg,device)
            score=float(validation.groupby('position_id').vessel_soft_dice.mean().mean());require(np.isfinite(score),'Nonfinite inner score')
            if strict_better(score,best):
                best,best_epoch=score,epoch+1
                save(folder/'best.pth',dict(model=model.state_dict(),epoch=best_epoch,score=best,identity=identity,anchor_sha256=anchor_sha))
            scheduler.step()
            history.append(dict(epoch=epoch+1,loss=sum_loss/len(loader),inner_position_vessel_soft_dice=score,
                mode=meta['mode'],provenance_label=cfg.get('provenance_label',meta.get('provenance_label','see audit')),
                best=best,best_epoch=best_epoch,updates=updates,seconds=time.monotonic()-tic,
                peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0,
                gradient_norm_mean=float(np.mean(grad_norms)),
                controller_update_l2=float(sum((p.detach()-q).square().sum() for p,q in zip(trainable,controller_before)).sqrt()) if adaptive else 0,
                **{k:v/len(loader) for k,v in raw_totals.items()}))
            if adaptive:require(model_hash(model.anchor)==frozen_before,'Frozen anchor changed')
            save(folder/'last.pth',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                epoch=epoch+1,best=best,best_epoch=best_epoch,history=history,updates=updates,identity=identity,
                anchor_sha256=anchor_sha,**capture_rng(device)))
            table(folder/'history.csv',history);write(folder/'status.json',dict(epoch=epoch+1,target=epochs,updates=updates,best_epoch=best_epoch))
            print(dict(method=method,**history[-1]),flush=True)
        if method=='F0':
            save(folder/'last.pth',dict(model=model.state_dict(),epoch=0,identity=identity,anchor_sha256=anchor_sha));table(folder/'history.csv',history)
        if method=='F0' or history[-1]['epoch']==epochs:
            write(folder/'complete.json',dict(identity=identity,epochs=epochs,updates=updates,best_epoch=best_epoch,
                best_sha256=sha(folder/'best.pth'),last_sha256=sha(folder/'last.pth'),frozen_anchor_sha256=frozen_before))
    return dict(status='training_dispatch_complete',methods=list(methods),scope=meta['mode'],synthetic=meta['synthetic'])


def evaluate(snapshot,cache,run,methods,cfg,seed,device,kind):
    meta=verify_cache(cache,snapshot);rows=[r for r in read(Path(cache)/'frames.json') if r['split']=='outer']
    for method in methods:
        folder=Path(run)/method;plan=read(folder/'plan.json');done=read(folder/'complete.json')
        require(plan['config']==cfg and plan['cache_identity']==meta['identity'] and plan['seed']==seed,'Evaluation plan differs')
        require(done['identity']==plan['identity'] and sha(folder/(kind+'.pth'))==done[kind+'_sha256'],'Incomplete/changed checkpoint')
        out=folder/('outer_'+kind)
        if (out/'complete.json').exists():
            e=read(out/'complete.json');require(e['checkpoint_sha256']==done[kind+'_sha256'] and e['metrics_sha256']==sha(out/'per_frame.csv'),'Evaluation changed')
            for name,h in e['predictions'].items():require(sha(out/'predictions'/name)==h,'Prediction changed')
            continue
        out.mkdir(parents=True,exist_ok=True)
        anc,anc_sha=verified_anchor(run,cfg,meta['identity']) if method in ADAPTIVE else (None,None)
        require(anc_sha==plan['anchor_sha256'],'Anchor changed')
        model=build(cfg,method,seed,anc['model'] if anc else None).to(device)
        state=load(folder/(kind+'.pth'));model.load_state_dict(state['model'])
        results,seconds=assess(model,snapshot,cache,rows,method,cfg,device,out/'predictions')
        results['fold']=meta['fold'];results['seed']=seed;results['mode']=meta['mode'];results['synthetic']=meta['synthetic']
        results['provenance_label']=cfg.get('provenance_label',meta.get('provenance_label','see upstream audit'))
        atomic(out/'per_frame.csv',results.to_csv(index=False).encode('utf-8-sig'))
        write(out/'complete.json',dict(checkpoint_sha256=done[kind+'_sha256'],metrics_sha256=sha(out/'per_frame.csv'),
            epoch=state['epoch'],frames=len(rows),positions=len(set(r['position_id'] for r in rows)),
            checkpoint=kind,P0=True,threshold=.5,coordinate_domain='native px',seconds=seconds,
            predictions={p.name:sha(p) for p in (out/'predictions').glob('*.npz')}))
    return dict(status='evaluated',checkpoint=kind,methods=list(methods))
