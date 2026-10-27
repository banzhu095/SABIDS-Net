#!/usr/bin/env python
"""Isolated learnable AC CLI. No mode implicitly starts long CUDA training."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
import uuid
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.learnable_ac_v1.core import (
    audit,prepare,config,read,write,require,BASE,ADAPTIVE,sha,fresh,atomic,
)


def processes():
    """Bounded read-only inspection. GPU CUDA writers additionally fail closed."""
    result=[]
    if os.name=='posix':
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name)==os.getpid():continue
            try:
                cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if ('python' in cmd and any(x in cmd for x in ('train.py','run_learnable_ac_v1.py','dispatch_denoise_then_ac','run_denoise_then_ac_v1.py'))):
                    result.append(dict(pid=int(p.name),command=cmd,start_ticks=(p/'stat').read_text().split(') ',1)[1].split()[19]))
            except (OSError,IndexError):pass
    return result


def gpu_idle():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    require(p.returncode==0,'Cannot inspect GPU occupancy: '+p.stderr)
    require(not p.stdout.strip(),'GPU has active compute processes; no task was stopped: '+p.stdout)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['status','audit','prepare','preview','cuda-check','cpu-smoke',
        'train-base','train-adaptive','resume','evaluate-best','evaluate-last','summarize','atlas','package','generate-configs','inspect-local-assets','reference-check'])
    p.add_argument('--project-root',default='.')
    p.add_argument('--research-mode',choices=['development','formal'],default='development')
    p.add_argument('--snapshot',default='runs/ac_vessel_labelsv2/snapshot_20261016_v1')
    p.add_argument('--input-cache',help='Verified full native denoise_then_ac cache; never a20-frame preview')
    p.add_argument('--upstream',help='Explicit checkpoint/config/SHA/binding records for D1,teacher,D2')
    p.add_argument('--review-bundle',help='Copied real preview/provenance bundle, NOT a full training cache')
    p.add_argument('--cache',default='runs/learnable_ac_v1/fold0/cache')
    p.add_argument('--run',default='runs/learnable_ac_v1/fold0/seed42')
    p.add_argument('--runs',nargs='+',help='Explicit run folders for grouped OOF summary')
    p.add_argument('--atlas',default='runs/learnable_ac_v1/fold0/atlas')
    p.add_argument('--report',default='runs/learnable_ac_v1/fold0/report')
    p.add_argument('--output')
    p.add_argument('--fold',type=int,choices=range(4),default=0)
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--methods',nargs='+',choices=BASE+ADAPTIVE)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--no-mild',action='store_true',help='Independent P0/P2 path when no legal D2 exists')
    p.add_argument('--config',help='Generated resolved config; budgets/range fixed before training')
    p.add_argument('--queue',action='store_true',help='After each explicitly selected method, evaluate both endpoints serially')
    p.add_argument('--check-anchor',action='store_true',help='CUDA check requires verified P3 best instead of a random engineering anchor')
    a=p.parse_args(argv)
    root=Path(a.project_root).resolve()
    def path(value):
        q=Path(value);return q.resolve() if q.is_absolute() else (root/q).resolve()
    snapshot,cache,run,atlas_dir,report=[path(getattr(a,k)) for k in ('snapshot','cache','run','atlas','report')]
    output=path(a.output) if a.output else None
    running=processes()
    if a.mode=='status':
        print(json.dumps(dict(processes=running,run=str(run),methods={m:read(run/m/'status.json') if (run/m/'status.json').is_file() else
            dict(complete=read(run/m/'complete.json')) if (run/m/'complete.json').is_file() else dict(status='not_started') for m in BASE+ADAPTIVE}),indent=2));return 0
    import torch
    torch.set_num_threads(a.threads)
    if a.device=='cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
    cfg=read(path(a.config)) if a.config else config()
    require(cfg['amp'] is False and cfg['boundary_loss']==0 and not cfg['joint_finetune'],'Unsupported public protocol')
    require(cfg.get('fold',a.fold)==a.fold and cfg.get('seed',a.seed)==a.seed,'Generated config fold/seed differs from CLI')
    if a.mode=='train-base':methods=a.methods or ['P0','P1','P2','P3','P4']
    elif a.mode=='train-adaptive':methods=a.methods or ['F0','F1','F2','F5']
    else:methods=a.methods or ['P0','P1','P2','P3','P4','F0','F1','F2','F5']
    mutating=a.mode not in ('audit',)
    if a.device=='cuda' and a.mode in ('prepare','train-base','train-adaptive','resume','evaluate-best','evaluate-last','atlas','cuda-check'):gpu_idle()
    if a.mode in ('train-base','train-adaptive','resume'):
        require(not any('run_learnable_ac_v1.py' in q['command'] for q in running),'Another learnable AC process is active; inspect before dispatch')
    if a.mode not in ('audit','prepare','generate-configs','cpu-smoke','inspect-local-assets'):
        from tools.learnable_ac_v1.core import verify_cache
        meta=verify_cache(cache,snapshot)
        require(meta['fold']==a.fold and meta['mode']==a.research_mode,'Cache fold/research-mode differs from CLI')
    run.parent.mkdir(parents=True,exist_ok=True)
    lock=run.parent/'.learnable_ac_writer.lock'
    token=uuid.uuid4().hex;owner=dict(token=token,pid=os.getpid(),host=socket.gethostname(),mode=a.mode)
    if mutating:
        with lock.open('x',encoding='utf8') as f:json.dump(owner,f)
    log=run.parent/'logs'/('execution_'+token+'.json')
    activity=dict(**owner,argv=sys.argv,started_at=datetime.now(timezone.utc).isoformat(),status='running')
    write(log,activity)
    try:
        if a.mode=='inspect-local-assets':
            from tools.learnable_ac_v1.asset_review import inspect_local
            require(a.review_bundle and output,'--review-bundle and new --output required')
            result=inspect_local(root,snapshot,path(a.review_bundle),output)
        elif a.mode=='generate-configs':
            require(output is not None,'--output new config directory required');fresh(output)
            for mode in ('development','formal'):
                for fold in range(4):
                    for seed in (42,43,44):
                        value=dict(cfg,fold=fold,seed=seed,research_mode=mode)
                        write(output/f'{mode}_fold{fold}_seed{seed}.json',value)
            write(output/'upstream_TEMPLATE.json',dict(
                **{role:dict(checkpoint='REQUIRED_ACTUAL_PATH',sha256='REQUIRED_SHA256',resolved_config='REQUIRED_CONFIG_PATH',config_sha256='REQUIRED_CONFIG_SHA256',
                    **({'binding':'REQUIRED_BINDING'} if role=='D2' else dict(initial_inventory='REQUIRED_PRE_OPTIMIZER_INVENTORY',initial_inventory_sha256='REQUIRED_INITIAL_SHA256'))) for role in ('D1','teacher','D2')}))
            result=dict(status='generated',configs=24,note='Formal requires verified upstream.json; template is NOT approval')
        elif a.mode=='audit':
            result=audit(root,snapshot,path(a.input_cache) if a.input_cache else None,
                read(path(a.upstream)) if a.upstream else None,a.research_mode,a.fold,a.no_mild)
            if output:require(not output.exists(),'Audit exists');write(output,result)
        elif a.mode=='prepare':
            result=prepare(root,snapshot,output or cache,path(a.input_cache) if a.input_cache else None,
                read(path(a.upstream)) if a.upstream else None,a.research_mode,a.fold,a.no_mild,a.device)
            result=dict(status='prepared',identity=result['identity'],frames=result['frames'])
        elif a.mode=='cpu-smoke':
            from tools.learnable_ac_v1.checks import cpu_smoke
            require(output is not None,'--output new persistent smoke directory required');result=cpu_smoke(output)
        elif a.mode=='cuda-check':
            from tools.learnable_ac_v1.checks import two_update_check
            require(a.device=='cuda' and output is not None,'cuda-check requires cuda and --output')
            result=two_update_check(snapshot,cache,output,cfg,'cuda',run if a.check_anchor else None)
        elif a.mode=='reference-check':
            from tools.learnable_ac_v1.native_reference import compare
            require(output is not None,'--output new native reference directory required')
            result=compare(cache,snapshot,output)
        elif a.mode in ('train-base','train-adaptive','resume'):
            from tools.learnable_ac_v1.learning import train,evaluate
            if a.mode=='train-base':require(set(methods)<=set(BASE),'train-base cannot train refiners')
            if a.mode=='train-adaptive':require(set(methods)<=set(ADAPTIVE),'train-adaptive cannot train bases')
            for m in methods:
                result=train(snapshot,cache,run,[m],cfg,a.seed,a.device,resume=a.mode=='resume')
                if a.queue:
                    for kind in ('best','last'):evaluate(snapshot,cache,run,[m],cfg,a.seed,a.device,kind)
        elif a.mode.startswith('evaluate-'):
            from tools.learnable_ac_v1.learning import evaluate
            result=evaluate(snapshot,cache,run,methods,cfg,a.seed,a.device,a.mode.split('-')[1])
        elif a.mode in ('preview','atlas'):
            from tools.learnable_ac_v1.reporting import atlas
            result=atlas(snapshot,cache,run,methods,cfg,a.seed,output or atlas_dir,a.device)
        elif a.mode=='summarize':
            from tools.learnable_ac_v1.reporting import summarize
            result=summarize([path(q) for q in a.runs] if a.runs else [run],output or report,methods)
        else:
            from tools.learnable_ac_v1.reporting import package
            require(output is not None and output.suffix=='.zip','--output new ZIP required')
            result=package(cache,run,atlas_dir,report,output)
        print(json.dumps(result,ensure_ascii=False,indent=2));activity['status']=result.get('status','finished')
        return 2 if result.get('status') in ('blocked','incomplete') else 0
    except BaseException:
        activity.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        activity['finished_at']=datetime.now(timezone.utc).isoformat();write(log,activity)
        if mutating and read(lock).get('token')==token:lock.unlink()


if __name__=='__main__':sys.exit(main())
