import hashlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from scipy.ndimage import binary_dilation,binary_erosion

from tools.downstream_seg_benchmark.model import Segmenter
from tools.downstream_seg_benchmark.engine import optimizer_window,save,load
from tools.downstream_seg_benchmark.metrics import metrics,dice
from sabids.models.dual_view_segmenter import DualViewFusion
from sabids.losses.common import masked_bce_dice_loss,masked_negative_bce_loss
from .common import *
from .processing import verify_cache,components


class Model(Segmenter):
    def __init__(self,cfg,dual=False,seed=42):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            super().__init__(cfg['channels'],cfg['depths'],cfg['decoder_depth'])
            self.fusions=nn.ModuleDict({str(i):DualViewFusion(cfg['channels'][i],0.) for i in range(1,len(cfg['channels']))}) if dual else nn.ModuleDict()

    def forward(self,packed):
        f=self.encode(packed[:,:1])
        if len(self.fusions):
            aux=self.encode(packed[:,1:2])
            for i in range(1,len(f)):f[i]=self.fusions[str(i)](f[i],aux[i])[0]
        return self.decode(f)


def objective(output,layer,vessel,lv,vv):
    conflict=vessel*(1-layer)
    outside,_=masked_negative_bce_loss(output['vessel'],vv*(1-layer)*(1-vessel))
    return (masked_bce_dice_loss(output['layer'],layer,lv*(1-conflict))+
        masked_bce_dice_loss(output['vessel'],vessel,vv*torch.maximum(layer,vessel))+.5*outside)


def geometry(shape,size):
    h,w=shape;scale=min(size/h,size/w);rh,rw=max(1,round(h*scale)),max(1,round(w*scale))
    top,left=(size-rh)//2,(size-rw)//2
    return rh,rw,top,left


def canvas(array,size,is_image):
    a=torch.from_numpy(np.array(array,dtype=np.float32,copy=True))[None,None]
    rh,rw,t,l=geometry(array.shape,size)
    a=F.interpolate(a,size=(rh,rw),mode='bilinear' if is_image else 'nearest',**({'align_corners':False} if is_image else {}))
    return F.pad(a,(l,size-rw-l,t,size-rh-t))[0]


def restore(prob,shape,size):
    rh,rw,t,l=geometry(shape,size)
    return F.interpolate(prob[...,t:t+rh,l:l+rw],size=shape,mode='bilinear',align_corners=False)[0,0].cpu().numpy()


class Frames(Dataset):
    def __init__(self,snapshot,cache,rows,arm,size,plan=None):
        self.snapshot=Path(snapshot);self.cache=Path(cache);self.rows=rows;self.arm=arm;self.size=size;self.plan=plan;self.labels={}
    def __len__(self):return len(self.plan) if self.plan is not None else len(self.rows)
    def native(self,row):
        pid=row['position_id']
        if pid not in self.labels:self.labels[pid]=tuple(plane(self.snapshot/'labels'/pid/f'{n}.png')>0 for n in ('layer','vessel','layer_valid','vessel_valid'))
        return self.labels[pid]
    def __getitem__(self,i):
        choice=self.plan[i] if self.plan is not None else dict(index=i,flip_x=False)
        row=self.rows[choice['index']];primary={'T0':'tone','A0':'auto','M0':'manual'}.get(self.arm,'noisy')
        aux={'NA':'auto','NM':'manual'}.get(self.arm,primary)
        arrays=[np.load(self.cache/row['views'][name]['path'],allow_pickle=False) for name in (primary,aux)]
        x=torch.cat([canvas(a,self.size,True) for a in arrays]);ys=[canvas(a,self.size,False) for a in self.native(row)]
        if choice['flip_x']:x=x.flip(-1);ys=[t.flip(-1) for t in ys]
        return x,*ys,row['sample_id'],row['position_id']


def schedule(rows,epochs,seed):
    rng=np.random.default_rng(seed+10991);groups=sorted({r['position_id'] for r in rows});plans=[]
    pool={g:np.array([i for i,r in enumerate(rows) if r['position_id']==g]) for g in groups}
    count=max(map(len,pool.values()))
    for epoch in range(epochs):
        indices=np.concatenate([np.resize(rng.permutation(pool[g]),count) for g in groups]);rng.shuffle(indices)
        plans.append([dict(index=int(i),flip_x=bool(rng.integers(2))) for i in indices])
    return plans


def base_hash(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        if not n.startswith('fusions.'):h.update(n.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def assess(model,snapshot,cache,rows,arm,cfg,device,predictions=None):
    model.eval();dataset=Frames(snapshot,cache,rows,arm,cfg['size']);records=[]
    thresholds=read(Path(cache)/'difficulty.json')
    with torch.no_grad():
        for i,row in enumerate(rows):
            batch=dataset[i];out=model(batch[0][None].to(device));layer,vessel,lv,vv=dataset.native(row)
            lp=restore(torch.sigmoid(out['layer']),layer.shape,cfg['size']);vp=restore(torch.sigmoid(out['vessel']),layer.shape,cfg['size'])
            layer_eval_valid=lv & ~(vessel&~layer)
            values=metrics(lp,vp,layer,vessel,layer_eval_valid,vv)
            for k in ('small_vessel_recall','medium_vessel_recall','large_vessel_recall'):values.pop(k,None)
            v=(vp>=.5)&vv;gt=vessel&vv
            values['vessel_soft_dice']=float((2*(vp*gt).sum()+1e-6)/((vp*vv).sum()+gt.sum()+1e-6)) if vv.any() else float('nan')
            values['vessel_area_error']=float(abs(int(v.sum())-int(gt.sum()))/max(1,int(gt.sum())))
            values['vessel_outside_gt_layer_fp']=float((v&~layer&~vessel).sum()/max(1,int((~layer&~vessel&vv).sum())))
            values['vessel_outside_gt_layer_fraction']=float((v&~layer&~vessel).sum()/max(1,int(v.sum())))
            band=binary_dilation(vessel ^ binary_erosion(vessel),iterations=2)&vv
            values['boundary_band_dice']=dice(v&band,gt&band)
            x=np.load(Path(cache)/row['views']['noisy']['path'],allow_pickle=False)
            ids,cc=components(x,layer,vessel,vv)
            for name,predicate in (('small',lambda c:c['area_px']<=thresholds['small_area_px']),('low_contrast',lambda c:c['signed_local_contrast'] is not None and abs(c['signed_local_contrast'])<=thresholds['low_contrast_abs'])):
                selected=[c for c in cc if c['fully_known'] and predicate(c)]
                covered=[float(v[ids==c['component_id']].mean()) for c in selected]
                values[name+'_component_coverage']=float(np.mean(covered)) if covered else float('nan')
                values[name+'_component_missed']=float(np.mean(np.array(covered)==0)) if covered else float('nan')
            records.append(dict(sample_id=row['sample_id'],position_id=row['position_id'],group_id=row['group_id'],arm=arm,**values))
            if predictions:
                path=Path(predictions)/f'{row["sample_id"]}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,layer=lp,vessel=vp)
    return pd.DataFrame(records)


def _train(snapshot,cache,run,arms,seed,cfg,device='cpu',resume=False,stop_after=None):
    snapshot=Path(snapshot);cache=Path(cache);run=Path(run);meta=verify_cache(cache,snapshot)
    require(not device.startswith('cuda') or torch.cuda.is_available(),'CUDA requested but unavailable')
    require(set(arms)<=set(ARMS+MANUAL),'Unknown arms');require(not set(arms)&set(MANUAL) or meta['manual'],'Manual cache absent')
    rows=read(cache/'frames.json');training=[r for r in rows if r['split']=='train'];inner=[r for r in rows if r['split']=='inner']
    identity=digest(dict(config=cfg,cache_identity=meta['identity'],seed=seed,sources=source_identity()))
    seed_dir=run/f'seed{seed}';seed_dir.mkdir(parents=True,exist_ok=True)
    if (seed_dir/'plan.json').exists():require(read(seed_dir/'plan.json')['identity']==identity,'Budget/config/cache/source changed; cannot resume or extend cosine plan')
    else:write(seed_dir/'plan.json',dict(identity=identity,config=cfg,cache_identity=meta['identity'],seed=seed,sources=source_identity()))
    plans=schedule(training,cfg['epochs'],seed+meta['fold']*100)
    plan_path=seed_dir/'data_plan.json'
    if plan_path.exists():require(read(plan_path)==plans,'Sampling/augmentation plan changed')
    else:write(plan_path,plans)
    torch.use_deterministic_algorithms(True)
    for arm in arms:
        folder=seed_dir/arm;folder.mkdir(exist_ok=True);dual=arm in ('NN','NA','NM')
        model=Model(cfg,dual,seed);init_kind='dual' if dual else 'single';init_file=seed_dir/f'{init_kind}_initialization.pth'
        if init_file.exists():model.load_state_dict(load(init_file))
        else:save(init_file,model.state_dict())
        model=model.to(device);base=base_hash(model)
        if (folder/'complete.json').exists():
            done=read(folder/'complete.json');require(resume and done['identity']==identity and done['best_sha256']==sha(folder/'best.pth') and done['last_sha256']==sha(folder/'last.pth'),'Completed run differs or --resume absent');continue
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=cfg['min_lr'])
        scaler=torch.cuda.amp.GradScaler(enabled=False)
        random.seed(seed+773);np.random.seed(seed+773);torch.manual_seed(seed+773)
        begin=0;best=-float('inf');history=[];updates=0
        if (folder/'last.pth').exists():
            require(resume,'Partial run exists; use --resume');state=load(folder/'last.pth');require(state['identity']==identity,'Resume identity changed')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
            begin=state['epoch'];best=state['best'];history=state['history'];updates=state['updates']
            torch.set_rng_state(state['torch_rng'].cpu());np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
            if device.startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        write(folder/'resolved_config.json',cfg)
        write(folder/'initialization_audit.json',dict(common_parameter_sha256=base,initialization_sha256=sha(init_file),data_plan_sha256=sha(plan_path),trainable=sum(p.numel() for p in model.parameters()),encoder_calls=2 if dual else 1,UGBI=False,restoration=False,RMAC=False,pseudo=False,teacher_checkpoint=None))
        for epoch in range(begin,cfg['epochs']):
            if stop_after is not None and epoch>=stop_after:break
            start=time.monotonic();model.train();loader=DataLoader(Frames(snapshot,cache,training,arm,cfg['size'],plans[epoch]),batch_size=1,num_workers=0,shuffle=False,generator=torch.Generator().manual_seed(seed+epoch+1927))
            pending=[];loss_sum=0.
            if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            for j,batch in enumerate(loader):
                pending.append(batch)
                if len(pending)==cfg['accumulation'] or j+1==len(loader):
                    value,_=optimizer_window(model,optimizer,scaler,pending,device,False,objective_fn=objective);loss_sum+=value;updates+=1;pending=[]
            validation=assess(model,snapshot,cache,inner,arm,cfg,device)
            score=float(validation.groupby('position_id').vessel_soft_dice.mean().mean());require(np.isfinite(score),'Invalid inner soft Dice')
            if score>best:
                best=score;save(folder/'best.pth',dict(model=model.state_dict(),epoch=epoch+1,score=score,identity=identity,arm=arm))
            scheduler.step();history.append(dict(epoch=epoch+1,loss=loss_sum/len(loader),inner_position_vessel_soft_dice=score,best=best,updates=updates,seconds=time.monotonic()-start,learning_rate=optimizer.param_groups[0]['lr'],gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0))
            save(folder/'last.pth',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch+1,best=best,history=history,updates=updates,identity=identity,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []))
            table(folder/'history.csv',history);print(dict(arm=arm,fold=meta['fold'],seed=seed,**history[-1]),flush=True)
            write(folder/'status.json',dict(identity=identity,epoch=epoch+1,expected_epochs=cfg['epochs'],complete=epoch+1==cfg['epochs']))
        if history and history[-1]['epoch']==cfg['epochs']:
            expected=sum((len(p)+cfg['accumulation']-1)//cfg['accumulation'] for p in plans)
            require(updates==expected,'Optimizer update budget mismatch')
            write(folder/'complete.json',dict(identity=identity,epochs=cfg['epochs'],updates=updates,best_sha256=sha(folder/'best.pth'),last_sha256=sha(folder/'last.pth'),base_initialization=base,data_plan_sha256=sha(plan_path)))
    return dict(run=str(run),seed=seed,arms=list(arms),identity=identity)


def train(*args,**kwargs):
    before=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:return _train(*args,**kwargs)
    finally:torch.use_deterministic_algorithms(before,warn_only=warn)


def evaluate(snapshot,cache,run,arms,seed,device='cpu',checkpoint='best',resume=False):
    require(checkpoint in ('best','last'),'Checkpoint must be best or last');meta=verify_cache(cache,snapshot)
    run=Path(run);seed_dir=run/f'seed{seed}';plan=read(seed_dir/'plan.json');cfg=plan['config']
    require(meta['identity']==plan['cache_identity'] and plan['sources']==source_identity(),'Run/cache/source changed')
    rows=[r for r in read(Path(cache)/'frames.json') if r['split']=='outer']
    for arm in arms:
        done=read(seed_dir/arm/'complete.json');require(done['identity']==plan['identity'] and done['epochs']==cfg['epochs'],'Incomplete or wrong-budget run')
        require(sha(seed_dir/arm/f'{checkpoint}.pth')==done[checkpoint+'_sha256'],'Checkpoint changed')
    for arm in arms:
        folder=seed_dir/arm;out=folder/f'outer_{checkpoint}'
        if (out/'complete.json').exists():
            value=read(out/'complete.json');require(resume and value['checkpoint_sha256']==sha(folder/f'{checkpoint}.pth') and value['metrics_sha256']==sha(out/'per_frame.csv'),'Evaluation exists or changed')
            for p,h in value['prediction_sha256'].items():require(sha(out/'predictions'/p)==h,'Prediction changed')
            continue
        out.mkdir(parents=True,exist_ok=True);model=Model(cfg,arm in ('NN','NA','NM'),seed).to(device)
        weights=load(folder/f'{checkpoint}.pth');require(weights.get('identity',plan['identity'])==plan['identity'],'Checkpoint plan mismatch');model.load_state_dict(weights['model'])
        started=time.monotonic();result=assess(model,snapshot,cache,rows,arm,cfg,device,out/'predictions');result['fold']=meta['fold'];result['seed']=seed
        atomic(out/'per_frame.csv',result.to_csv(index=False).encode('utf-8-sig'))
        write(out/'complete.json',dict(identity=plan['identity'],checkpoint_sha256=sha(folder/f'{checkpoint}.pth'),metrics_sha256=sha(out/'per_frame.csv'),threshold=.5,P0=True,checkpoint=checkpoint,epoch=weights.get('epoch'),seconds=time.monotonic()-started,frames=len(rows),prediction_sha256={p.name:sha(p) for p in (out/'predictions').glob('*.npz')}))
    return dict(status='evaluated_development_outer',fold=meta['fold'],seed=seed,checkpoint=checkpoint,arms=list(arms))
