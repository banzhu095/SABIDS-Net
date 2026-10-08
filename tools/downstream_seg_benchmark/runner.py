"""Hash-checked atomic stage runner; partial arms never unlock test."""
from __future__ import annotations

import os
import time
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd
import torch

from .common import FORMAL,digest,read_json,require,sha,source_hash,write_csv,write_json

STAGES=['preflight','materialize','audit','prepare','train','lock','evaluate','atlas','report','package']


def completion_matrix(run):
    rows=[]
    for method in FORMAL:
        for seed in [42,123,2026]:
            track=Path(run)/'tracks'/method/str(seed)
            complete=track/'complete.json'
            d=read_json(complete) if complete.exists() else {}
            rows.append(dict(method_id=method,segmentation_seed=seed,status='complete' if d else ('partial' if (track/'last.pth').exists() else 'pending'),
                             updates=d.get('updates',0),seconds=d.get('seconds',0)))
    write_csv(Path(run)/'completion_matrix.csv',rows)
    return rows


def stage(run,name,signature,action,resume=False):
    folder=Path(run)/'stages';folder.mkdir(exist_ok=True)
    marker=folder/(name+'.json')
    if marker.exists():
        prior=read_json(marker)
        require(prior['input_signature']==signature,'Stage configuration changed: '+name)
        if prior['status']=='success':
            require(resume,'Stage already completed; use --resume')
            for path,expected in prior.get('outputs',{}).items():require(sha(Path(run)/path)==expected,'Stage output changed: '+path)
            print('[stage skipped] '+name,flush=True);return
    started=time.monotonic()
    write_json(marker,dict(stage=name,status='running',input_signature=signature,started_at_utc=datetime.now(timezone.utc).isoformat()))
    try:
        print('[stage start] '+name,flush=True)
        outputs=action() or []
        hashes={str(Path(p).relative_to(run)):sha(p) for p in outputs if Path(p).is_file()}
        write_json(marker,dict(stage=name,status='success',input_signature=signature,outputs=hashes,
                   seconds=time.monotonic()-started,completed_at_utc=datetime.now(timezone.utc).isoformat()))
        print('[stage complete] '+name,flush=True)
    except Exception as exc:
        write_json(marker,dict(stage=name,status='failed',input_signature=signature,error=type(exc).__name__,reason=str(exc)))
        write_csv(Path(run)/'failures.csv',[dict(stage=name,reason=str(exc),at_utc=datetime.now(timezone.utc).isoformat())])
        raise
    finally:completion_matrix(run)


def run_formal(args):
    from .assets import audit
    from .materialize import materialize,find_sources
    from .protocol import build_plan,default_config
    from .engine import train,seal_checkpoints,evaluate
    from .report import atlas,build_report,workbook,package
    root,run=args.project_root.resolve(),args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    source=find_sources(root,args.source_denoise_run)
    config=default_config('formal');config.update(num_workers=args.num_workers,batch_size=args.batch_size,target_size=args.patch_size)
    request=dict(config=config,source=str(source),source_hash=source_hash(),manifest=sha(args.manifest) if args.manifest else None)
    signature=digest(request)
    # Arm selectors are recovery controls, not scientific protocol changes.
    manifest=args.manifest or run/'materialized_primary_inputs.csv'
    def preflight():
        require(args.device.startswith('cuda') and torch.cuda.is_available(),'Formal runner requires CUDA; CPU is engineering smoke only')
        require(args.patch_size in [256,384,512,640],'Unsupported native patch size')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        write_json(run/'preflight.json',dict(request=request,device=torch.cuda.get_device_name(),torch=torch.__version__,
            budget_reason='Existing roi_outside_no_d2s uses 60 epochs and LR 5e-5; fixed two-head comparison disables early stopping and uses a common cosine scheduler',free_disk_bytes=__import__('shutil').disk_usage(run).free))
        return [run/'preflight.json']
    def materialization():
        if args.manifest:return [args.manifest] if args.manifest.is_relative_to(run) else []
        return [materialize(root,run,source,resume=args.resume,reinfer=args.reinfer,device=args.device)]
    def auditing():
        require(not (run/'plan_lock.json').exists(),'Cannot rerun audit after training plan is sealed')
        audit(root,run,manifest);return [run/'input_asset_audit.csv',run/'label_asset_inventory.json']
    def preparing():build_plan(run,config,args.resume);return [run/'plan_lock.json',run/'resolved_config.yaml']
    def training():
        train(run,args.device,args.resume,args.methods,args.seg_seeds)
        matrix=completion_matrix(run)
        require(all(r['status']=='complete' for r in matrix),'Selected arms finished; remaining arms must train before lock/evaluate')
        return [run/'checkpoint_registry.csv']
    def locking():seal_checkpoints(run);return [run/'checkpoint_lock.json']
    def evaluating():evaluate(run,args.device,args.resume);return [run/'per_frame_metrics.csv']
    def atlasing():atlas(run);return list((run/'atlas').glob('*.png'))
    def reporting():build_report(run);workbook(run);return [run/'downstream_segmentation_report.md',run/'benchmark_summary.xlsx']
    def packaging():package(run);return [run/'package_verification.json']
    actions=[preflight,materialization,auditing,preparing,training,locking,evaluating,atlasing,reporting,packaging]
    start=STAGES.index(args.from_stage)
    for name,action in zip(STAGES[start:],actions[start:]):
        stage(run,name,signature,action,args.resume)
