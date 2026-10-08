from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from .assets import audit
from .common import FORMAL, PILOT, read_json, require, write_csv, write_json
from .engine import evaluate, runtime_probe, train, seal_checkpoints
from .protocol import build_plan, default_config
from .report import atlas, build_report, package, workbook


def main(argv=None):
    parser=argparse.ArgumentParser(description='Unified denoised-only segmentation; no Joint training')
    parser.add_argument('command',choices=['audit','build-plan','train','lock','evaluate','build-atlas','build-report','package-light','run-pilot','run-formal',
        'materialize-inputs','verify-materialized-inputs','export-input-assets','evaluate-incremental','smoke','status'])
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--source-denoise-run',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--incremental-audit-dir',type=Path)
    parser.add_argument('--reinfer',action='store_true',help='Exact sealed denoiser replay only; historical SHA must match')
    parser.add_argument('--methods',type=lambda v:v.split(','))
    parser.add_argument('--seg-seeds',type=lambda v:[int(x) for x in v.split(',')])
    parser.add_argument('--from-stage',choices=['preflight','materialize','audit','prepare','train','lock','evaluate','atlas','report','package'],default='preflight')
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--patch-size',type=int,default=384)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode',choices=['pilot','formal'],default='pilot')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--num-workers',type=int,default=4)
    args=parser.parse_args(argv)
    root=args.project_root.resolve()
    mode='formal' if args.command=='run-formal' else args.mode
    run=(args.run_dir or root/'runs'/('downstream_seg_'+mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    try:
        if args.command=='run-formal':
            from .runner import run_formal
            args.run_dir=run;run_formal(args);return
        if args.command=='status':
            from .runner import completion_matrix
            print(pd.DataFrame(completion_matrix(run)).to_string(index=False));return
        if args.command=='evaluate-incremental':
            from .incremental import evaluate_incremental
            require(args.incremental_audit_dir and args.output_dir,'Provide separate --incremental-audit-dir and --output-dir')
            print(evaluate_incremental(run,args.incremental_audit_dir,args.output_dir,args.device,args.resume));return
        if args.command in ['materialize-inputs','verify-materialized-inputs','export-input-assets']:
            from .materialize import materialize,verify,export
            if args.command=='materialize-inputs':print(materialize(root,run,args.source_denoise_run,args.methods or FORMAL,args.resume,args.reinfer,args.device))
            elif args.command=='verify-materialized-inputs':print(len(verify(run)))
            else:print(export(run,args.output_dir or run/'exports'))
            return
        if args.command=='smoke':
            from .smoke import real_train_smoke
            print(real_train_smoke(run,args.output_dir or run/'real_train_smoke'));return
        if args.command=='audit':
            require(not (run/'plan_lock.json').exists(),'Cannot replace audit of a sealed run')
            print(json.dumps(audit(root,run,args.manifest),indent=2));return
        if args.command in ['run-pilot','run-formal','build-plan']:
            if not (run/'audit.json').exists():audit(root,run,args.manifest)
            config=default_config(mode);config['num_workers']=args.num_workers
            if (run/'plan_lock.json').exists():
                require(args.resume,'Run already planned; use --resume')
                config=read_json(run/'plan_lock.json')['config']
            else:
                failures=pd.read_csv(run/'missing_or_ambiguous_assets.csv',keep_default_na=False)
                require(failures[failures.method_id.isin(config['methods']+['*'])].empty,'Missing/ambiguous inputs or labels; see audit CSV')
                if mode=='pilot':config=runtime_probe(run,config,args.device)
            build_plan(run,config,args.resume)
            if args.command=='build-plan':return
            train(run,args.device,args.resume)
            evaluate(run,args.device,args.resume)
            atlas(run);build_report(run);workbook(run);print(package(run));return
        if args.command=='train':
            if read_json(run/'plan_lock.json')['config']['mode']=='formal':require(args.device.startswith('cuda') and torch.cuda.is_available(),'Formal training requires CUDA')
            train(run,args.device,args.resume,args.methods,args.seg_seeds)
        elif args.command=='lock':seal_checkpoints(run)
        elif args.command=='evaluate':evaluate(run,args.device,args.resume)
        elif args.command=='build-atlas':atlas(run)
        elif args.command=='build-report':build_report(run);workbook(run)
        elif args.command=='package-light':print(package(run))
    except Exception as exc:
        write_json(run/'failure.json',dict(stage=args.command,error=type(exc).__name__,reason=str(exc)))
        write_csv(run/'failures.csv',[dict(stage=args.command,reason=str(exc))])
        if (run/'audit.json').exists():build_report(run)
        raise


if __name__=='__main__':main()
