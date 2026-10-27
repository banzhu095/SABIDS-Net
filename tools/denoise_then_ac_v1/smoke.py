"""Deliberately synthetic CPU engineering smoke; never accepted for real training."""
from pathlib import Path
import numpy as np
import torch
from .core import *
from .learning import train,evaluate
from .reporting import preview,summarize,package
from .checks import update_check


class SyntheticRestorer(torch.nn.Module):
    channels=(4,8,16)
    def forward_denoise_only(self,x):
        return {'denoised':torch.nn.functional.avg_pool2d(x,3,1,1)}


def run(output):
    from tools.ac_vessel_labelsv2.smoke import synthetic_project
    from tools.ac_vessel_labelsv2.snapshot import freeze,registration_audit,approve
    out,project,workspace,evidence=synthetic_project(output);snapshot=out/'snapshot'
    freeze(project,workspace,project/'Data/PKU37_OCT_Denoising/noisy',snapshot,evidence)
    registration_audit(project,snapshot,out/'registration')
    approve(snapshot,out/'registration',out/'approval.json','synthetic generator','Known generated identities; NOT real clinical registration')
    cache=out/'cache';cache.mkdir();split=read(snapshot/'folds.json')[0];rows=[];model=SyntheticRestorer()
    for r in read(snapshot/'frames.json'):
        part=next(k for k in ('train','inner','outer') if r['group_id'] in split[k]);x=noisy(project/r['path']).astype(np.float32)
        d=native_d2(x,model,'cpu');m=mild_input(x,d)
        rows.append(dict(**r,split=part,shape=list(x.shape),_arrays=dict(noisy=x,denoised=d,mild=m)))
    chosen=[min((r for r in rows if r['position_id']==pid),key=lambda r:r['sample_id']) for pid in split['train']]
    stats=fit_mild(chosen,[r['_arrays']['mild'] for r in chosen]);cfg=config_from_statistics(stats)
    for r in rows:
        arrays=r.pop('_arrays');arrays['auto']=stable_ac_v1(arrays['mild'],cfg)[0];views={}
        for key,a in arrays.items():
            p=Path('arrays')/f'{r["sample_id"]}_{key}.npy';npy(cache/p,a);views[key]=dict(path=p.as_posix(),sha256=sha(cache/p))
        r['views']=views
    write(cache/'frames.json',rows);write(cache/'auto_config.json',asdict(cfg));write(cache/'ac_statistics.json',stats)
    write(cache/'difficulty.json',difficulty_fit(snapshot,project,[r for r in rows if r['split']=='train']))
    write(cache/'fixed_samples.json',[min(r['sample_id'] for r in rows if r['position_id']==pid) for pid in sorted({r['position_id'] for r in rows})])
    write(cache/'split.json',split)
    meta=dict(version=VERSION,fold=0,synthetic=True,manual=False,full_cache_complete=True,frames=len(rows),
        snapshot_identity=read(snapshot/'snapshot.json')['identity'],denoiser_sha256='SYNTHETIC_NOT_D2',
        ac_config=asdict(cfg),sources=source_identity(),files=[dict(path=p.relative_to(cache).as_posix(),sha256=sha(p)) for p in sorted(cache.rglob('*')) if p.is_file()])
    meta['identity']=digest(meta);write(cache/'cache.json',meta)
    update_check(snapshot,cache,out/'short_check','cpu',smoke=True)
    cfg=config(smoke=True);cfg['epochs']=2
    train(snapshot,cache,out/'training',ARMS,42,cfg,stop_after=1)
    train(snapshot,cache,out/'training',ARMS,42,cfg,resume=True)
    evaluate(snapshot,cache,out/'training',ARMS,42,checkpoint='best')
    evaluate(snapshot,cache,out/'training',ARMS,42,checkpoint='last')
    preview(snapshot,cache,out/'preview',out/'training');summarize(out/'training',out/'report')
    result=package(cache,out/'preview',out/'report',out/'synthetic_gpt.zip')
    result.update(status='passed',scope='SYNTHETIC CPU smoke; no real D2/AC efficacy result')
    write(out/'smoke_result.json',result);return result
