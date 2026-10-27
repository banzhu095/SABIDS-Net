#!/usr/bin/env python
"""Persistent minimal DEVELOPMENT queue; never broadens to all folds/seeds."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.learnable_ac_v1.core import config,read,write,sha,digest,require,sources

CORE=('P3','F0','F1','F2','F5')
LABEL='development / historical upstream exposure'


def now():return datetime.now(timezone.utc).isoformat()


def proc(pid):
    try:
        p=Path('/proc')/str(pid)
        return dict(pid=int(pid),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
                    start_ticks=(p/'stat').read_text().split(') ',1)[1].split()[19])
    except (OSError,IndexError):return None


def validate_completed(run,cache,method,cfg):
    """Payload/step/budget/selection verification, not history/file-existence."""
    from tools.downstream_seg_benchmark.engine import load
    from tools.learnable_ac_v1.learning import validate_state
    from tools.denoise_then_ac_v1.learning import schedule
    folder=Path(run)/method;done=read(folder/'complete.json');plan=read(folder/'plan.json')
    meta=read(Path(cache)/'cache.json')
    require(plan['config']==cfg and plan['cache_identity']==meta['identity'] and done['identity']==plan['identity'],'Completion identity differs')
    best,last=load(folder/'best.pth'),load(folder/'last.pth')
    require(best['identity']==last['identity']==plan['identity'],'Checkpoint payload identity differs')
    for kind in ('best','last'):require(sha(folder/(kind+'.pth'))==done[kind+'_sha256'],'Checkpoint SHA differs')
    if method=='F0':require(done['epochs']==0 and last['epoch']==0 and done['updates']==0,'F0 is not the untrained anchor')
    else:
        epochs=cfg['base_epochs'] if method in ('P3','P2') else cfg['adaptive_epochs']
        rows=[r for r in read(Path(cache)/'frames.json') if r['split']=='train']
        plans=schedule(rows,epochs,42);validate_state(last,plans,epochs)
        require(last['epoch']==epochs==done['epochs'],'Training has not reached fixed endpoint')
        expected=max(last['history'],key=lambda r:r['inner_position_vessel_soft_dice'])
        require(best['epoch']==last['best_epoch']==expected['epoch'] and best['score']==last['best'],
                'Inner-best/earliest tie binding differs')
    result=dict(identity=plan['identity'],code_sha256=digest(sources()),config_sha256=digest(cfg),
        cache_identity=meta['identity'],snapshot_identity=meta['snapshot_identity'],mode='development',provenance_label=LABEL,
        synthetic=meta['synthetic'],
        method=method,completed_epochs=last['epoch'],updates=done['updates'],best_epoch=best['epoch'],
        best_sha256=done['best_sha256'],last_sha256=done['last_sha256'],
        anchor_sha256=plan['anchor_sha256'],monitor='inner position-equal Vessel soft Dice',tie='strict greater; earliest',
        threshold=.5,postprocessing='P0',two_tasks_same_checkpoint=True)
    target=folder/'checkpoint_selection_binding.json'
    if target.exists():require(read(target)==result,'Existing selection binding differs')
    else:write(target,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True)
    p.add_argument('--mode',choices=['run','status'],required=True)
    p.add_argument('--input-cache')
    p.add_argument('--snapshot',default='runs/ac_vessel_labelsv2/snapshot_20261016_v1')
    p.add_argument('--with-p2',action='store_true',help='Supplement AFTER complete core report/atlas/ZIP')
    a=p.parse_args();base=Path(a.base).resolve()
    if a.mode=='status':
        state=read(base/'dispatch/status.json') if (base/'dispatch/status.json').exists() else {'status':'not_started'}
        owner=read(base/'dispatch/owner.json') if (base/'dispatch/owner.json').exists() else None
        actual=proc(owner['pid']) if owner else None
        print(json.dumps(dict(state=state,owner=owner,actual_process=actual,
            owner_matches=bool(actual and actual['start_ticks']==owner.get('start_ticks'))),indent=2));return
    require(os.name=='posix','Use Linux cloud terminal for persistent scheduling')
    require(a.input_cache,'--input-cache verified historical FULL native cache required')
    import fcntl
    base.mkdir(parents=True,exist_ok=True);dispatch=base/'dispatch';dispatch.mkdir(exist_ok=True)
    stream=(dispatch/'scheduler.lock').open('a+')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Existing scheduler owns this exact run; no duplicate launched')
    owner=dict(**proc(os.getpid()),started_at=now(),code_sha256=sha(__file__),base=str(base),mode='development',provenance_label=LABEL)
    write(dispatch/'owner.json',owner)
    if (base/'deployment.json').exists():deployment=read(base/'deployment.json')
    else:deployment=dict(note='No deployment receipt supplied; actual physical source fingerprints below')
    cfg=config();cfg.update(fold=0,seed=42,research_mode='development',provenance_label=LABEL,execution_scope='minimal_seed42_v1')
    cf=base/'resolved_config.json'
    if cf.exists():require(read(cf)==cfg,'Existing frozen config differs')
    else:write(cf,cfg)
    cache,run=base/'cache',base/'training';snapshot=Path(a.snapshot)
    if not snapshot.is_absolute():snapshot=ROOT/snapshot
    state={}
    def record(status,**fields):
        state.clear();state.update(updated_at=now(),driver_pid=os.getpid(),status=status,mode='development',provenance_label=LABEL,**fields)
        write(dispatch/'status.json',state)
        with (dispatch/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
    def child(mode,methods=None,output=None,anchor=False):
        command=[sys.executable,str(ROOT/'tools/run_learnable_ac_v1.py'),'--project-root',str(ROOT),
            '--mode',mode,'--research-mode','development','--fold','0','--seed','42',
            '--snapshot',str(snapshot),'--cache',str(cache),'--run',str(run),'--config',str(cf),'--threads','4']
        if methods:command+=['--methods',*methods]
        if output:command+=['--output',str(output)]
        if mode=='prepare':command+=['--input-cache',str(Path(a.input_cache).resolve())]
        if mode in ('resume','cuda-check','evaluate-best','evaluate-last','atlas'):command+=['--device','cuda']
        if anchor:command+=['--check-anchor']
        log=dispatch/(mode+'_'+('_'.join(methods or []))+'_'+uuid.uuid4().hex[:8]+'.log')
        env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with log.open('x',encoding='utf8') as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
            info=proc(process.pid)
            record('running',stage=mode,methods=methods,child=info,command=command,log=str(log),config_sha256=sha(cf))
            code=process.wait()
        require(code==0,'Child failed code='+str(code)+'; preserved log='+str(log))
        record('step_complete',stage=mode,methods=methods,log=str(log),exit_code=code)
    def checked(mode,path,anchor_sha=None):
        if (path/'check.json').exists():
            check=read(path/'check.json')
            require(check['status']=='passed' and check.get('cache_identity',read(cache/'cache.json')['identity'])==read(cache/'cache.json')['identity'],'Previous check mismatch')
            if anchor_sha:require(check['anchor_sha256']==anchor_sha,'Previous CUDA anchor changed')
            return
        child(mode,output=path,anchor=bool(anchor_sha))
    def evaluate_one(method):
        validate_completed(run,cache,method,cfg)
        for kind in ('best','last'):child('evaluate-'+kind,[method])
    def finish(methods,tag):
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        report=base/('report_'+tag+'_'+stamp);atlas=base/('atlas_'+tag+'_'+stamp)
        child('summarize',methods,report);child('atlas',methods,atlas)
        target=base/('learnable_ac_'+tag+'_seed42_'+stamp+'.zip')
        cmd=[sys.executable,str(ROOT/'tools/run_learnable_ac_v1.py'),'--mode','package','--snapshot',str(snapshot),
            '--cache',str(cache),'--run',str(run),'--report',str(report),'--atlas',str(atlas),'--output',str(target)]
        subprocess.run(cmd,cwd=ROOT,check=True)
        receipt=read(target.with_suffix('.json'));record(tag+'_complete',report=str(report),atlas=str(atlas),package=receipt,full_cv_complete=False)
        write(base/(tag+'_result.json'),dict(state))
    try:
        record('preparing',source_fingerprints=sources(),deployment=deployment)
        write(base/'reuse_audit.json',dict(P3=dict(reuse=False,
            reason='Historical MA0 lacks matching learnable cache/candidate/coverage-mask/source/study binding; new60epoch P3 required'),
            P0='Historical numbers not protocol-proven; excluded, rerun required if requested',
            P2='New matched theta from this train-mild cache; no historical result reuse'))
        if not (cache/'cache.json').exists():child('prepare',output=cache)
        checked('reference-check',base/'native_reference')
        checked('cuda-check',base/'cuda_before_P3')
        child('resume',['P3']);evaluate_one('P3')
        anchor=read(run/'P3/complete.json')['best_sha256']
        checked('cuda-check',base/'cuda_verified_anchor',anchor)
        for method in CORE[1:]:
            child('resume',[method]);evaluate_one(method)
        finish(CORE,'core')
        if a.with_p2:
            # Core output is persisted and complete BEFORE extra GPU use.
            child('resume',['P2']);evaluate_one('P2');finish((*CORE,'P2'),'core_plus_P2')
        else:write(base/'P2_pending.json',dict(status='configured_not_started',reason='Optional budget-sensitive supplement; core was prioritized',command='rerun dispatcher --with-p2; verified complete methods skip'))
    except BaseException:
        failure=dict(previous_state=dict(state),traceback=traceback.format_exc(),at=now())
        write(dispatch/('failure_'+uuid.uuid4().hex[:8]+'.json'),failure)
        record('failed',failure=failure);raise
    finally:fcntl.flock(stream,fcntl.LOCK_UN);stream.close()


if __name__=='__main__':main()
