"""Two-update + exact full-state replay; never a scientific result."""
from pathlib import Path
import copy
import random
import time

import numpy as np
import torch

from .core import *
from .learning import build,Frames,objective,model_hash,capture_rng,restore_rng
from .reporting import profile
from tools.downstream_seg_benchmark.engine import load,save


def _equal_state(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(_equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(_equal_state(x,y) for x,y in zip(a,b))
    return a==b


def two_update_check(snapshot,cache,output,cfg,device='cpu',run=None):
    """Paired F1/F5 check. Optional verified P3 best, otherwise random probe.

    F1 has no image-Jacobian training path by design; parameter equality is
    NOT a claim of equal backward cost. Both use accumulation2 and clip1.
    """
    meta=verify_cache(cache,snapshot);out=fresh(output)
    require(device=='cpu' or torch.cuda.is_available(),'CUDA requested but unavailable')
    rows=[r for r in read(Path(cache)/'frames.json') if r['split']=='train'][:2]
    require(len(rows)==2,'Two real training frames required')
    from .learning import verified_anchor
    if run:
        state,anchor_sha=verified_anchor(run,cfg,meta['identity']);anchor=state['model']
    else:anchor=build(cfg,'P3',42).state_dict();anchor_sha=None
    dataset=Frames(snapshot,cache,rows,'F5',cfg['size'])
    batches=[[x[None].to(device) for x in dataset[i][:5]] for i in range(2)]
    def step(m,o,s,b):
        m.train();o.zero_grad(set_to_none=True);total=0.;input_norm=0.
        if device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        tic=time.monotonic()
        for sample in b:
            pred=m(sample[0]);loss,_=objective(m,pred,sample,cfg)
            require(torch.isfinite(loss).item(),'Nonfinite check loss')
            target=m.last_diagnostics['adjustment'] if m.method=='F1' else m.last_diagnostics['vessel_image']
            g=torch.autograd.grad(loss,target,retain_graph=True,allow_unused=True)[0]
            require(g is not None and torch.isfinite(g).all() and g.abs().sum()>0,'Refinement input direction unreachable')
            input_norm+=float(g.norm());(loss/len(b)).backward();total+=float(loss.detach())/len(b)
        require(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.control.parameters()),'No controller gradient')
        require(all(p.grad is None for p in m.anchor.parameters()),'Frozen anchor received parameter gradient')
        body_grad=sum(float(p.grad.norm()) for n,p in m.control.named_parameters() if n.startswith('context.') and p.grad is not None)
        norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],cfg['clip'])
        require(torch.isfinite(norm),'Nonfinite check gradient');o.step();s.step()
        if device.startswith('cuda'):torch.cuda.synchronize()
        return dict(loss=total,input_gradient_norm=input_norm,context_gradient_norm=body_grad,
                    step_seconds=time.monotonic()-tic,peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None)
    records={};tic=time.monotonic()
    for method in ('F1','F5'):
        model=build(cfg,method,42,anchor).to(device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,2)
        frozen_before=model_hash(model.anchor);before=model_hash(model.control)
        random.seed(718);np.random.seed(718);torch.manual_seed(718)
        first=step(model,optimizer,scheduler,batches)
        state=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
            scheduler=scheduler.state_dict(),epoch=0,step=1,**capture_rng(device))
        folder=out/method;folder.mkdir();save(folder/'step1.pth',state)
        second=step(model,optimizer,scheduler,batches)
        require(second['context_gradient_norm']>0,'Zero head permanently blocked context gradients')
        expected=copy.deepcopy(model.state_dict());expected_optimizer=copy.deepcopy(optimizer.state_dict());expected_rng=capture_rng(device)
        # Release the first GPU model before constructing the replay copy.
        with torch.no_grad():
            off=model(batches[0][0],adaptive_off=True);reference=model.anchor(batches[0][0][:,1:2])
        require(all(torch.equal(off[k],reference[k]) for k in off),'Adaptive-off not exact anchor')
        require(model_hash(model.anchor)==frozen_before and model_hash(model.control)!=before,'Freeze/update failure')
        measured=profile(model,batches[0][0],device)
        del model,optimizer,scheduler,state,off,reference
        if device.startswith('cuda'):torch.cuda.empty_cache()
        resumed=build(cfg,method,42,anchor).to(device)
        opt=torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2)
        checkpoint=load(folder/'step1.pth');resumed.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer']);sch.load_state_dict(checkpoint['scheduler']);restore_rng(checkpoint,device)
        replay=step(resumed,opt,sch,batches)
        require(all(torch.equal(expected[k],resumed.state_dict()[k]) for k in expected),'Resume parameters differ')
        require(_equal_state(expected_optimizer,opt.state_dict()),'Resume AdamW state differs')
        require(second['loss']==replay['loss'] and sch.last_epoch==2,'Resume loss/scheduler differs')
        require(_equal_state(expected_rng,capture_rng(device)),'Resume Python/NumPy/Torch/CUDA RNG differs')
        records[method]=dict(steps=[first,second],resume_bit_exact=True,profile=measured,
                            input_gradient='logit residual' if method=='F1' else 'frozen S0 image Jacobian')
        del resumed,opt,sch,checkpoint,expected,expected_optimizer
        if device.startswith('cuda'):torch.cuda.empty_cache()
    result=dict(status='passed',scope='ENGINEERING ONLY; verified S0' if run else 'ENGINEERING ONLY; random segmentation anchor; no efficacy result',
        synthetic=meta['synthetic'],size=cfg['size'],device=device,updates=2,losses=[first,second],
        resume_bit_exact=True,anchor_unchanged=True,D2='offline frozen cache, absent from model/optimizer',
        anchor_sha256=anchor_sha,accumulation=2,methods=records,compute_matched=False,
        seconds=time.monotonic()-tic,profile=records['F5']['profile'])
    write(out/'check.json',result);return result


def cpu_smoke(output):
    from tools.ac_vessel_labelsv2.smoke import synthetic_project
    from tools.ac_vessel_labelsv2.snapshot import freeze
    from .learning import train,evaluate
    from .reporting import atlas,summarize,package
    from sabids.experiments.attenuation_aux import fit_train_statistics,config_from_statistics
    from sabids.models.learnable_ac import NativeStableAC
    import shutil
    out,root,workspace,evidence=synthetic_project(output)
    snapshot=out/'snapshot';freeze(root,workspace,root/'Data/PKU37_OCT_Denoising/noisy',snapshot,evidence)
    cache=out/'cache';cache.mkdir();split=read(snapshot/'folds.json')[0];rows=[]
    for r in read(snapshot/'frames.json'):
        stage=next(s for s in ('train','inner','outer') if r['group_id'] in split[s]);x=noisy(root/r['path']).astype(np.float32)
        d=torch.nn.functional.avg_pool2d(torch.from_numpy(x)[None,None],3,1,1)[0,0].numpy()
        rows.append(dict(**r,split=stage,shape=list(x.shape),_x=x,_m=.75*x+.25*d))
    fixed=[min((r for r in rows if r['position_id']==pid),key=lambda r:r['sample_id']) for pid in sorted({r['position_id'] for r in rows})]
    chosen=[r for r in fixed if r['split']=='train'];stats=fit_train_statistics([r['_m'] for r in chosen],source_records=[dict(split='train',sample_id=r['sample_id']) for r in chosen]);ac_cfg=config_from_statistics(stats)
    ac=NativeStableAC(ac_cfg)
    for r in rows:
        x,m=r.pop('_x'),r.pop('_m');data={k:v[0,0].numpy() for k,v in ac(torch.from_numpy(m)[None,None]).items()}
        data.update(noisy=x,direct_ac=ac(torch.from_numpy(x)[None,None])['z0'][0,0].numpy(),tone=np.sqrt(m))
        p=Path('native')/(r['sample_id']+'.npz');npz(cache/p,**data);r['analytic']=p.as_posix();r['sha256_native']=sha(cache/p)
    write(cache/'frames.json',rows);write(cache/'fixed_samples.json',[r['sample_id'] for r in fixed]);write(cache/'split.json',split)
    write(cache/'ac_statistics.json',stats);write(cache/'ac_config.json',asdict(ac_cfg))
    write(cache/'difficulty.json',difficulty_fit(snapshot,root,[r for r in rows if r['split']=='train']))
    meta=dict(version=VERSION,mode='development',fold=0,no_mild=False,synthetic=True,allowed_base=list(BASE),
        snapshot_identity=verify_lock(snapshot)['identity'],full_cache_complete=True,frames=len(rows),
        ac_metadata=ac.metadata(),sources=sources(),files=[dict(path=p.relative_to(cache).as_posix(),sha256=sha(p)) for p in sorted(cache.rglob('*')) if p.is_file()])
    meta['identity']=digest(meta);write(cache/'cache.json',meta)
    cfg=config(True);run=out/'training'
    two_update_check(snapshot,cache,out/'two_updates',cfg)
    train(snapshot,cache,run,list(BASE),cfg)
    train(snapshot,cache,run,list(ADAPTIVE),cfg,stop_after=1)
    train(snapshot,cache,run,list(ADAPTIVE),cfg,resume=True)
    # Independently continuous same-anchor control, compare every tensor/Adam step.
    continuous=out/'continuous';continuous.mkdir();shutil.copytree(run/'P3',continuous/'P3')
    train(snapshot,cache,continuous,['F5'],cfg)
    left,right=load(run/'F5/last.pth'),load(continuous/'F5/last.pth')
    require(all(torch.equal(v,right['model'][k]) for k,v in left['model'].items()),'Epoch resume not bit-exact')
    require(left['updates']==right['updates'] and left['scheduler']==right['scheduler'],'Epoch resume counters differ')
    for kind in ('best','last'):evaluate(snapshot,cache,run,list(BASE+ADAPTIVE),cfg,42,'cpu',kind)
    summarize([run],out/'report',list(BASE+ADAPTIVE));atlas(snapshot,cache,run,list(BASE+ADAPTIVE),cfg,42,out/'atlas')
    result=package(cache,run,out/'atlas',out/'report',out/'synthetic_light.zip')
    result.update(status='passed',epoch_resume_bit_exact=True,scope='SYNTHETIC CPU engineering only; not a real performance result')
    write(out/'smoke_result.json',result);return result
