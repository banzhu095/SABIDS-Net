from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import KEY, TRAIN_ORDER, digest, read_json, require, sha, write_csv, write_json
from .data import Inputs
from .metrics import METRICS, aggregate, classify, metrics, formal_tables
from .model import Segmenter, objective
from .protocol import verify_plan


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def save(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate_rows(model, rows, size, device, method, seed, allow_test=False, prediction_dir=None):
    dataset = Inputs(rows, size, allow_test=allow_test)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    records = []
    with torch.no_grad():
        for x, l, v, valid, vv, sid, position in loader:
            output = model(x.to(device))
            lp, vp = [torch.sigmoid(output[k]).cpu().numpy()[0, 0] for k in ['layer', 'vessel']]
            values = metrics(lp, vp, l.numpy()[0,0] > .5, v.numpy()[0,0] > .5,
                             valid.numpy()[0,0] > .5, vv.numpy()[0,0] > .5)
            records.append(dict(sample_id=sid[0], position_id=position[0], method_id=method, segmentation_seed=seed, **values))
            if prediction_dir:
                prediction_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(prediction_dir / (sid[0]+'.npz'), layer=lp, vessel=vp)
    return pd.DataFrame(records)


def runtime_probe(run, config, device):
    """Training-data-only timing/OOM probe; no choice depends on model quality."""
    data = pd.read_csv(Path(run)/'input_asset_audit.csv', keep_default_na=False)
    rows = data[(data.method_id == 'noisy_identity') & (data.split == 'train')].sort_values('sample_id').to_dict('records')
    require(rows and all(r['available'] for r in rows), 'No usable noisy train probe')
    if device.startswith('cuda'):
        require(torch.cuda.is_available(), 'CUDA requested but unavailable')
    batch = 4
    timings = []
    while batch:
        try:
            torch.manual_seed(42)
            model = Segmenter(config['channels'], config['depths'], config['decoder_depth']).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
            loader = DataLoader(Inputs(rows[:max(4,batch)], config['target_size']), batch_size=batch)
            samples = next(iter(loader))
            for i in range(3):
                if device.startswith('cuda'): torch.cuda.synchronize()
                start = time.monotonic()
                x, l, v, valid, vv = [t.to(device) for t in samples[:5]]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', enabled=device.startswith('cuda') and config['amp']):
                    output = model(x)
                loss = objective(output, l, v, valid, vv)
                loss.backward(); optimizer.step()
                if device.startswith('cuda'): torch.cuda.synchronize()
                if i: timings.append(time.monotonic()-start)
            break
        except torch.cuda.OutOfMemoryError:
            batch //= 2
            del model
            torch.cuda.empty_cache()
    require(batch > 0, 'OOM at batch 1; cannot train this protocol')
    config = dict(config, batch_size=batch)
    seconds_per_update = max(timings)
    val_count = int(((data.method_id == 'noisy_identity') & (data.split == 'val')).sum())
    test_count = int(((data.method_id == 'noisy_identity') & (data.split == 'test')).sum())
    proposals = [(config['methods'], 15, 0, 'none'),
                 (config['methods'][:3], 15, 0, 'drop_clean'),
                 (config['methods'][:3], 10, 0, 'drop_clean_10_epochs'),
                 (config['methods'][:3], 10, 20, 'drop_clean_10_epochs_20_frames_per_position')]
    for methods, epochs, maximum, degradation in proposals:
        n = len(rows) if not maximum else sum(min(maximum, len(g)) for _,g in pd.DataFrame(rows).groupby('position_id'))
        estimate = len(methods)*epochs*(math.ceil(n/batch)*seconds_per_update + val_count*seconds_per_update/batch*.8) + len(methods)*test_count*seconds_per_update/batch + 600
        config.update(methods=methods, epochs=epochs, max_train_frames_per_position=maximum, degradation=degradation)
        if estimate <= config['wall_budget_seconds']*.9:
            break
    evidence = dict(seconds_per_update=seconds_per_update, estimated_seconds=estimate, physical_batch=batch,
                    device=device, degraded=config['degradation'], feasible=estimate<=config['wall_budget_seconds']*.9)
    write_json(Path(run)/'runtime_preflight.json', evidence)
    require(evidence['feasible'], 'Four-hour pilot infeasible on this device after prescribed reductions')
    optional=data[data.method_id=='dncnn_paired']
    # Freeze the optional arm before any test access, using timing only.
    if config['degradation']=='none' and estimate<10800 and estimate*1.25<config['wall_budget_seconds']*.9 and len(optional) and optional.available.astype(str).str.lower().eq('true').all():
        config['methods']=config['methods']+['dncnn_paired']
        config['optional_dncnn_decision']='included_before_test_from_conservative_runtime_projection'
    else:
        config['optional_dncnn_decision']='not_included_time_or_asset_budget'
    return config


def verify_assets(run):
    frame = pd.read_csv(Path(run)/'cohort.csv', keep_default_na=False)
    for row in frame.to_dict('records'):
        require(sha(row['path']) == row['sha256'], 'Input bytes changed: '+row['sample_id'])
    for path, meta in read_json(Path(run)/'label_asset_inventory.json').items():
        require(sha(path) == meta['sha256'], 'Label bytes changed')
    return frame


def train(run, device, resume=False, methods=None, seeds=None):
    run = Path(run)
    lock = verify_plan(run); config = lock['config']
    require(not (run/'test_opened.json').exists(), 'Test already opened; training is sealed')
    data = verify_assets(run)
    selected_ids = set(pd.read_csv(run/'training_samples.csv').sample_id)
    torch.use_deterministic_algorithms(True)
    start_all = time.monotonic()
    chosen_methods=methods or [m for m in TRAIN_ORDER if m in config['methods']]
    chosen_seeds=seeds or config['segmentation_seeds']
    require(set(chosen_methods)<=set(config['methods']) and set(chosen_seeds)<=set(config['segmentation_seeds']),'Selected task absent from sealed plan')
    for seed in chosen_seeds:
        folder = run/'plans'/str(seed)
        plans = read_json(folder/'data_plan.json')
        for method in chosen_methods:
            dest = run/'tracks'/method/str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            if (dest/'complete.json').exists():
                require(resume, 'Training exists; use --resume')
                done = read_json(dest/'complete.json')
                require(done['plan_sha256'] == sha(run/'plan_lock.json') and done['best_sha256']==sha(dest/'best.pth'), 'Completed track differs')
                continue
            arm = data[data.method_id == method]
            rows = arm[(arm.split == 'train') & arm.sample_id.isin(selected_ids)].sort_values('sample_id').to_dict('records')
            val = arm[arm.split == 'val'].sort_values('sample_id').to_dict('records')
            torch.manual_seed(seed)
            model = Segmenter(config['channels'], config['depths'], config['decoder_depth']).to(device)
            model.load_state_dict(load(folder/'initialization.pth'))
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],weight_decay=config.get('weight_decay',.01))
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
            scaler = torch.cuda.amp.GradScaler(enabled=config['amp'] and device.startswith('cuda'))
            begin, updates, history, best = 0, 0, [], (-float('inf'),-float('inf'))
            if (dest/'last.pth').exists():
                require(resume, 'Partial training exists; use --resume')
                state = load(dest/'last.pth')
                require(state['plan_sha256'] == sha(run/'plan_lock.json'), 'Resume plan changed')
                model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
                if 'scheduler' in state:scheduler.load_state_dict(state['scheduler'])
                begin, updates, history, best = state['epoch'], state['updates'], state['history'], tuple(state['best'])
                torch.set_rng_state(state['rng'].cpu())
                if device.startswith('cuda') and state.get('cuda_rng'):
                    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
            metadata = dict(method_id=method, display_name='SABIDS-current Stage-1/D0 output' if method=='sabids_current' else method,
                            segmentation_seed=seed, denoiser_seed=42 if method in ['sabids_current','nafnet_paired','dncnn_paired','tcfl_dncnn'] else 0,
                            initialization_sha256=sha(folder/'initialization.pth'), data_plan_sha256=sha(folder/'data_plan.json'),
                            sampler_plan_sha256=sha(folder/'sampler_plan.json'), model_source_sha256=lock['source_sha256'], plan_sha256=sha(run/'plan_lock.json'), device=device)
            metadata['augmentation_plan_sha256']=sha(folder/'augmentation_plan.json')
            write_json(dest/'run_metadata.json', metadata)
            (dest/'resolved_config.yaml').write_bytes((run/'resolved_config.yaml').read_bytes())
            (dest/'label_asset_inventory.json').write_bytes((run/'label_asset_inventory.json').read_bytes())
            write_json(dest/'parameter_audit.json', dict(trainable=sum(p.numel() for p in model.parameters()),
                       denoising_parameters=0, interactions=0, input_channels=1, heads=['layer','vessel']))
            for epoch in range(begin, config['epochs']):
                model.train()
                generator = torch.Generator(); generator.set_state(load(folder/'loader_generator.pth'))
                loader = DataLoader(Inputs(rows, config['target_size'], plans[epoch]), batch_size=config['batch_size'],
                                    shuffle=False, num_workers=config['num_workers'], generator=generator)
                epoch_start, loss_sum, last_log = time.monotonic(), 0., time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                accumulation=config.get('gradient_accumulation',1)
                for batch_index,batch in enumerate(loader):
                    x,l,v,valid,vv = [t.to(device) for t in batch[:5]]
                    with torch.autocast(device_type='cuda', enabled=config['amp'] and device.startswith('cuda')):
                        output = model(x)
                    loss = objective(output,l,v,valid,vv)
                    require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                    window=min(accumulation,len(loader)-(batch_index//accumulation)*accumulation)
                    scaler.scale(loss/window).backward()
                    if (batch_index+1)%accumulation==0 or batch_index+1==len(loader):
                        scaler.unscale_(optimizer)
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        require(bool(torch.isfinite(norm)),'Nonfinite gradient; optimizer update not counted')
                        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);updates+=1
                    loss_sum += float(loss.detach())
                    if time.monotonic()-last_log >= 600:
                        print(dict(method=method, epoch=epoch+1, updates=updates, validation='pending',
                                   eta_seconds=(time.monotonic()-epoch_start)/max(1,updates-epoch*len(loader))*(lock['updates_per_method']-updates),
                                   gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0,
                                   last_validation=history[-1].get('val_vessel_roi_dice') if history else None), flush=True)
                        write_json(run/'progress.json',dict(method=method,seed=seed,epoch=epoch+1,updates=updates,
                            remaining_updates=lock['updates_per_method']-updates,gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0))
                        last_log = time.monotonic()
                score=(None,None)
                if (epoch+1)%config.get('validation_every',5)==0 or epoch+1==config['epochs']:
                    validation = evaluate_rows(model,val,config['evaluation_size'],device,method,seed)
                    scores = validation.groupby('position_id')[['vessel_roi_dice','layer_dice']].mean().mean()
                    score = (float(scores.vessel_roi_dice),float(scores.layer_dice))
                    better=score[0]>best[0]+.0001 or (abs(score[0]-best[0])<=.0001 and score[1]>best[1])
                    if better:
                        best = score
                        save(dest/'best.pth',dict(model=model.state_dict(),epoch=epoch+1,updates=updates,validation_score=score,**metadata))
                scheduler.step()
                history.append(dict(epoch=epoch+1,updates=updates,loss=loss_sum/len(loader),val_vessel_roi_dice=score[0],
                                    val_layer_dice=score[1],seconds=time.monotonic()-epoch_start,learning_rate=optimizer.param_groups[0]['lr']))
                save(dest/'last.pth',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                     scheduler=scheduler.state_dict(),epoch=epoch+1,updates=updates,history=history,best=best,rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],plan_sha256=sha(run/'plan_lock.json')))
                write_csv(dest/'training_curve.csv',history)
                print(dict(method=method,seed=seed,**history[-1]),flush=True)
            require(updates == lock['updates_per_method'], 'Optimizer update mismatch')
            write_json(dest/'complete.json',dict(**metadata,updates=updates,best_sha256=sha(dest/'best.pth'),seconds=sum(h['seconds'] for h in history)))
    if all((run/'tracks'/m/str(s)/'complete.json').is_file() for m in config['methods'] for s in config['segmentation_seeds']):seal_checkpoints(run)


def seal_checkpoints(run):
    run=Path(run); lock=verify_plan(run); inventory=[]
    for seed in lock['config']['segmentation_seeds']:
        for method in lock['config']['methods']:
            folder=run/'tracks'/method/str(seed)
            require((folder/'complete.json').is_file(), 'All methods must finish before test lock')
            done=read_json(folder/'complete.json')
            require(done['updates']==lock['updates_per_method'] and done['best_sha256']==sha(folder/'best.pth'), 'Completion mismatch')
            require(done['plan_sha256']==sha(run/'plan_lock.json') and done['model_source_sha256']==lock['source_sha256'], 'Track source/plan mismatch')
            for key,file in [('initialization_sha256','initialization.pth'),('data_plan_sha256','data_plan.json'),('sampler_plan_sha256','sampler_plan.json'),('augmentation_plan_sha256','augmentation_plan.json')]:
                require(done[key]==sha(run/'plans'/str(seed)/file),'Common-randomness audit mismatch')
            inventory.append(dict(method_id=method,segmentation_seed=seed,path=str(folder/'best.pth'),sha256=done['best_sha256'],updates=done['updates']))
    seal=dict(plan_sha256=sha(run/'plan_lock.json'),checkpoints=inventory)
    if (run/'checkpoint_lock.json').exists():
        require(read_json(run/'checkpoint_lock.json')==seal,'Checkpoint lock changed')
    else:
        write_json(run/'checkpoint_lock.json',seal)
    write_csv(run/'checkpoint_inventory.csv',inventory)
    write_csv(run/'checkpoint_registry.csv',inventory)
    formal=run/'formal_config_lock.json'
    if formal.exists():
        value=read_json(formal);value.update(status='all_checkpoints_locked',checkpoint_lock_sha256=sha(run/'checkpoint_lock.json'))
        write_json(formal,value)
    return seal


def evaluate(run, device, resume=False):
    run=Path(run); lock=verify_plan(run); config=lock['config']
    require((run/'checkpoint_lock.json').is_file(),'Test access requires checkpoint lock')
    seal=seal_checkpoints(run)
    data=verify_assets(run)
    marker=run/'test_opened.json'
    if marker.exists():
        require(resume and read_json(marker)['checkpoint_lock_sha256']==sha(run/'checkpoint_lock.json'),'Test access already recorded; immutable resume required')
    else:
        write_json(marker,dict(checkpoint_lock_sha256=sha(run/'checkpoint_lock.json'),threshold=.5))
    if not (run/'test_access_log.csv').exists():
        from datetime import datetime,timezone
        write_csv(run/'test_access_log.csv',[dict(opened_at_utc=datetime.now(timezone.utc).isoformat(),checkpoint_lock_sha256=sha(run/'checkpoint_lock.json'),threshold=.5,reason='All participating method/seed checkpoints locked')])
    outputs=[]
    for item in seal['checkpoints']:
        method,seed=item['method_id'],item['segmentation_seed']
        dest=run/'evaluation'/method/str(seed); dest.mkdir(parents=True,exist_ok=True)
        path=dest/'metrics.csv'
        if path.exists() and (dest/'complete.json').exists():
            cached=read_json(dest/'complete.json')
            require(cached['sha256']==sha(path) and cached['checkpoint_sha256']==item['sha256'],'Cached evaluation changed')
            for name,expected in cached['prediction_hashes'].items():require(sha(dest/'predictions'/name)==expected,'Cached prediction changed')
            result=pd.read_csv(path)
        else:
            model=Segmenter(config['channels'],config['depths'],config['decoder_depth']).to(device)
            model.load_state_dict(load(item['path'])['model'])
            rows=data[(data.method_id==method)&(data.split=='test')].sort_values('sample_id').to_dict('records')
            result=evaluate_rows(model,rows,config['evaluation_size'],device,method,seed,True,dest/'predictions')
            write_csv(path,result);write_json(dest/'complete.json',dict(sha256=sha(path),checkpoint_sha256=item['sha256'],
                prediction_hashes={p.name:sha(p) for p in (dest/'predictions').glob('*.npz')}))
        # Parse the persisted representation on first and resumed evaluation alike.
        outputs.append(pd.read_csv(path,float_precision='round_trip'))
    frame=pd.concat(outputs,ignore_index=True)
    require(not frame.duplicated(['method_id','segmentation_seed','sample_id']).any(),'Duplicate evaluation rows')
    position,summary,gains=aggregate(frame)
    write_csv(run/'per_frame_metrics.csv',frame);write_csv(run/'per_position_metrics.csv',position)
    write_csv(run/'per_method_summary.csv',summary);write_csv(run/'method_comparison_table.csv',summary)
    write_csv(run/'paired_position_gains.csv',gains)
    delta=gains.groupby('comparison')[METRICS].agg(['mean','min','max']);delta.columns=['_'.join(c) for c in delta.columns]
    write_csv(run/'paired_method_differences.csv',delta.reset_index())
    if config['mode']=='formal':
        summary=formal_tables(frame,run)
        write_json(run/'evaluation_status.json',dict(status='complete_limited_3_position_test' if position.position_id.nunique()<6 else 'complete_six_position_test',
                checkpoints=len(seal['checkpoints']),frames=len(frame),threshold=.5))
        return summary
    selected=gains[gains.comparison=='sabids_current - nafnet_paired']
    write_csv(run/'pilot_success_criteria.csv',[dict(status=classify(selected),criterion=config['criterion_version'],test_positions=position.position_id.nunique())])
    return summary
