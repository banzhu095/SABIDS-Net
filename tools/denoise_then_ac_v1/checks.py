"""Short optimizer/full-state recovery checks, never a formal training launcher."""
import copy
import random
from pathlib import Path
import numpy as np
import torch
from tools.downstream_seg_benchmark.engine import save,load,optimizer_window
from .core import read,write,sha,require,config,verify_cache,load_d2,native_d2,model_fingerprint
from .learning import Model,Frames,objective


def _update_check(snapshot,cache,output,device='cpu',smoke=False):
    # The same deterministic kernels are used by the full training loop.
    torch.use_deterministic_algorithms(True)
    out=Path(output);require(not out.exists(),'Check exists; choose new directory')
    require(not device.startswith('cuda') or torch.cuda.is_available(),'CUDA unavailable')
    verify_cache(cache,snapshot,full=False)
    rows=[r for r in read(Path(cache)/'frames.json') if r['split']=='train']
    require(rows,'No train frames');out.mkdir(parents=True)
    cfg=config(smoke);records=[];d2_check=dict(scope='synthetic smoke; no formal D2 checkpoint')
    if not smoke:
        identity=read(Path(cache)/'denoiser_identity.json');d2=load_d2(identity,device)
        before_d2=model_fingerprint(d2)
        native=np.load(Path(cache)/rows[0]['views']['noisy']['path'],allow_pickle=False)
        restored=native_d2(native,d2,device)
        require(model_fingerprint(d2)==before_d2,'D2 changed in CUDA native check')
        d2_check=dict(sha256=identity['sha256'],native_shape=list(native.shape),output_shape=list(restored.shape),
                      parameters_unchanged=True,requires_grad_parameters=sum(p.requires_grad for p in d2.parameters()))
        del d2
        if device.startswith('cuda'):torch.cuda.empty_cache()
    for arm in ('N0','NMA'):
        torch.manual_seed(478);model=Model(cfg,arm=='NMA',42).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=cfg['min_lr'])
        scaler=torch.cuda.amp.GradScaler(enabled=False)
        dataset=Frames(snapshot,cache,rows[:1],arm,cfg['size'])
        item=dataset[0];batch=tuple(t[None] for t in item[:5])+tuple(item[5:])
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        loss,_=optimizer_window(model,opt,scaler,[batch,batch],device,False,objective_fn=objective)
        scheduler.step()
        path=out/f'{arm}_short_check.pth'
        save(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),epoch=1,updates=1,torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []))
        second=Model(cfg,arm=='NMA',42).to(device);state=load(path);second.load_state_dict(state['model'])
        opt2=torch.optim.AdamW(second.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        opt2.load_state_dict(state['optimizer'])
        scheduler2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=60,eta_min=cfg['min_lr'])
        scheduler2.load_state_dict(state['scheduler']);scaler2=torch.cuda.amp.GradScaler(enabled=False);scaler2.load_state_dict(state['scaler'])
        crng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []
        optimizer_window(model,opt,scaler,[batch,batch],device,False,objective_fn=objective);scheduler.step()
        expected_rng=torch.get_rng_state().clone()
        torch.set_rng_state(state['torch_rng'].cpu())
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        if crng:torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        optimizer_window(second,opt2,scaler2,[batch,batch],device,False,objective_fn=objective);scheduler2.step()
        equal=all(torch.equal(v,second.state_dict()[k]) for k,v in model.state_dict().items())
        changed=sum(not torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        opt_equal=all(torch.equal(v,opt2.state_dict()['state'][k][n]) if torch.is_tensor(v)
            else v==opt2.state_dict()['state'][k][n]
            for k,s in opt.state_dict()['state'].items() for n,v in s.items())
        rng_equal=torch.equal(expected_rng,torch.get_rng_state())
        require(equal and changed>0 and opt_equal and rng_equal and scheduler.state_dict()==scheduler2.state_dict(),
                'Recovery equivalence/parameter update failed')
        records.append(dict(arm=arm,size=cfg['size'],loss=loss,changed_tensors=changed,
            resumed_parameters_exact=equal,scheduler_exact=scheduler.state_dict()==scheduler2.state_dict(),
            optimizer_state_exact=opt_equal,torch_rng_exact=rng_equal,updates=2,
            checkpoint_sha256=sha(path),peak_mb=torch.cuda.max_memory_allocated()/2**20 if crng else 0))
        del model,second,opt,opt2
        if device.startswith('cuda'):torch.cuda.empty_cache()
    write(out/'check.json',dict(status='passed',scope='short engineering check; NOT 60-epoch result',d2=d2_check,records=records))
    return read(out/'check.json')


def update_check(*args,**kwargs):
    before=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:return _update_check(*args,**kwargs)
    finally:torch.use_deterministic_algorithms(before,warn_only=warn)
