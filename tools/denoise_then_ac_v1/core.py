"""Frozen provenance, native inference and train-only calibration.

No old AC cache or manual review parameters are accepted as an input view.
"""
from dataclasses import asdict
from pathlib import Path
import json
import io

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from tools.ac_vessel_labelsv2.common import (
    ROOT, atomic, digest, read, write, table, sha, noisy, npy, png, plane,
    require, verify_lock, source_identity as inherited_sources, config as inherited_config,
)
from tools.ac_vessel_labelsv2.snapshot import preflight as snapshot_preflight
from tools.ac_vessel_labelsv2.processing import difficulty_fit
from sabids.experiments.attenuation_aux import (
    CACHE_VERSION, fit_train_statistics, config_from_statistics, stable_ac_v1,
    StableACConfig, array_sha256,
)

VERSION = 'denoise_then_ac_v1.native_r1'
GEOMETRY = 'native dtype normalization; right-bottom stride pad; crop native HxW; full-depth AC; then common aspect resize/pad'
SNAPSHOT_ID = '8bb6d39bda627d7204d023f7b2db0b7a0a8a5cb5be247f5e7b8628a00203e5b7'
ARMS = ('N0', 'M0', 'MA0', 'NN', 'NM', 'NMA')
OPTIONAL = ('A0', 'NA')
VIEWS = dict(N0=('noisy','noisy'), M0=('mild','mild'), MA0=('auto','auto'),
             NN=('noisy','noisy'), NM=('noisy','mild'), NMA=('noisy','auto'),
             A0=('direct_auto','direct_auto'), NA=('noisy','direct_auto'))
DUAL = ('NN', 'NM', 'NMA', 'NA')
COMPARISONS = (('M0','N0'),('MA0','M0'),('NM','NN'),('NMA','NM'),
               ('NMA','NN'),('NMA','MA0'),('NMA','N0'),('NMA','M0'))


def source_identity():
    result = inherited_sources()
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT/'tools/run_denoise_then_ac_v1.py',
            ROOT/'configs/denoise_then_ac_v1/protocol.yaml', ROOT/'sabids/models/sabids_net.py']
    result.update({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    return result


def config(smoke=False):
    import yaml
    cfg = yaml.safe_load((ROOT/'configs/denoise_then_ac_v1/protocol.yaml').read_text(encoding='utf8'))
    base = inherited_config(smoke)
    for k in base:
        if k not in ('version','smoke','pilot'):
            require(cfg[k] == inherited_config()[k], 'Frozen training contract differs: '+k)
    if smoke:
        for k in ('channels','depths','decoder_depth','size','epochs'): cfg[k] = base[k]
    cfg.update(smoke=smoke, pilot=False)
    return cfg


def resolve(root, value):
    p = Path(value)
    return p if p.is_absolute() else Path(root)/p


def discover(root):
    """Candidates only: never silently select the newest checkpoint."""
    return [str(p.relative_to(root)) for p in sorted(Path(root).glob(
        'runs/**/checkpoint_binding_best_task_preserving.json'))]


def exposure(root, cfg, split):
    """Metadata only. Never open the image paths in a historical manifest."""
    data = cfg.get('data', {})
    path = resolve(root, data.get('manifest','__missing_manifest__'))
    result = dict(manifest=str(path), train=[], selection=[], unknown=[])
    if not path.is_file():
        result['unknown'].append('manifest_missing'); return result
    records = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'group_id' not in records or 'split' not in records:
        result['unknown'].append('manifest_group_or_split_missing'); return result
    for role, key in (('train','train'),('selection','val')):
        part = records[records.split == data.get(key+'_split', 'train' if key=='train' else 'val')]
        if data.get(key+'_datasets') and 'dataset' in part:
            part = part[part.dataset.isin(data[key+'_datasets'])]
        if data.get(key+'_groups'): part = part[part.group_id.isin(data[key+'_groups'])]
        result[role] = sorted(set(part.group_id))
    result['manifest_sha256'] = sha(path)
    for key in ('inner','outer'):
        result[key+'_train_overlap'] = sorted(set(split[key]) & set(result['train']))
        result[key+'_selection_overlap'] = sorted(set(split[key]) & set(result['selection']))
    return result


def audit_d2(root, checkpoint, binding_path, split):
    """Validate the existing binding, not the old formal downstream split gate."""
    from sabids.experiments.d2 import audit_d2_checkpoint_binding
    from sabids.config import load_config
    from tools.downstream_seg_benchmark.engine import load
    cp, bp = resolve(root, checkpoint), resolve(root, binding_path)
    require(cp.name == 'best_task_preserving.pth', 'D2-task best checkpoint required')
    binding = audit_d2_checkpoint_binding(bp, cp, 'best_task_preserving')
    raw = load(cp); cfg = raw['config']; resolved = cp.parent/'resolved_config.yaml'
    other = load_config(resolved)
    require(cfg.get('d2',{}).get('enabled') is True, 'Not a D2 checkpoint')
    require(cfg['d2'].get('run_mode') in ('pilot','full'), 'Only completed pilot/full D2 is eligible; smoke/overfit/template prohibited')
    require(cfg.get('data',{}).get('normalization') == 'fixed', 'D2 was not trained with the required fixed normalization')
    require(cfg.get('train',{}).get('stage') == 'denoise', 'D2 stage is not denoise')
    require(cfg.get('loss',{}).get('restoration_mode') == 'structure_d2', 'Not structure_d2')
    for key in ('model','data','loss','train','d2','seed'):
        require(cfg.get(key) == other.get(key), 'D2 resolved/checkpoint mismatch: '+key)
    d2_exposure = exposure(root,cfg,split)
    teacher = dict(unknown=['teacher checkpoint/config not yet identified'])
    # Search configuration values, never guess a historical teacher run.
    def walk(value, prefix=''):
        if isinstance(value,dict):
            for k,v in value.items(): yield from walk(v,prefix+'.'+k)
        else: yield prefix,value
    paths = [(k,v) for k,v in walk(cfg) if 'teacher' in k.lower() and isinstance(v,str) and v.endswith(('.pth','.pt'))]
    if len(set(v for _,v in paths)) == 1:
        tp = resolve(root,paths[0][1])
        if tp.is_file():
            expected_teacher_sha=cfg.get('d2',{}).get('teacher',{}).get('sha256')
            require(expected_teacher_sha and sha(tp)==expected_teacher_sha,
                    'Bound D2 teacher SHA missing/mismatched; exposure cannot be attributed')
            traw = load(tp)
            teacher = exposure(root,traw.get('config',{}),split)
            teacher.update(checkpoint=str(tp),sha256=sha(tp))
        else: teacher = dict(checkpoint=str(tp),unknown=['bound teacher checkpoint unavailable'])
    initial = read(binding['initial_inventory_path'])
    require(initial.get('recorded_before_optimizer_step') is True,'Missing training-start inventory')
    require(d2_exposure.get('manifest_sha256') == initial.get('manifest_sha256'),
            'D2 exposure manifest is missing/changed relative to training-start inventory')
    return dict(checkpoint=str(cp),sha256=sha(cp),binding_sha256=sha(bp),binding=binding,
        config_sha256=sha(resolved),config=cfg, selection=binding['selection'],
        d2_exposure=d2_exposure,teacher_exposure=teacher,
        assessment='development exploration; historical denoiser/teacher exposure reported, NOT strict independent generalization',
        output_semantics='forward_denoise_only[denoised] is clipped restoration, NOT residual',
        alpha=.25,beta=.5,power=2.,gamma=1.,ac_version=CACHE_VERSION)


def audit(root,snapshot,approval,checkpoint=None,binding=None):
    gate = snapshot_preflight(root,snapshot,approval)
    split = read(Path(snapshot)/'folds.json')[0]
    frames = read(Path(snapshot)/'frames.json')
    counts = {s:dict(positions=len(split[s]),frames=sum(r['group_id'] in split[s] for r in frames))
              for s in ('train','inner','outer')}
    problems = list(gate['problems']); d2 = None
    if gate['snapshot_identity'] != SNAPSHOT_ID: problems.append('not_the_frozen_labelsv2_snapshot')
    if counts != dict(train=dict(positions=12,frames=548),inner=dict(positions=3,frames=141),outer=dict(positions=5,frames=240)):
        problems.append('frozen_fold0_counts_differ')
    if not checkpoint or not binding: problems.append('explicit_D2_checkpoint_and_binding_required')
    else:
        try: d2 = audit_d2(root,checkpoint,binding,split)
        except Exception as exc: problems.append(type(exc).__name__+': '+str(exc))
    return dict(version=VERSION,status='blocked' if problems else 'passed',problems=problems,
                snapshot=gate,split=split,counts=counts,denoiser=d2,
                candidates=discover(root),new_sealed_test_assets_opened=0)


def load_d2(identity,device):
    from sabids.engine.trainer import build_model
    from tools.downstream_seg_benchmark.engine import load
    require(sha(identity['checkpoint']) == identity['sha256'], 'D2 changed after audit')
    raw = load(identity['checkpoint']); model = build_model(identity['config'])
    model.load_state_dict(raw['model'],strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def native_d2(image,model,device):
    """Right/bottom reflect pad to encoder stride; crop BEFORE full-depth AC."""
    x = np.asarray(image,dtype=np.float32)
    require(x.ndim==2 and np.isfinite(x).all() and x.min()>=0 and x.max()<=1,'Invalid native gray')
    model.eval().requires_grad_(False)
    stride = 2**(len(model.channels)-1)
    h,w = x.shape; ph,pw = (-h)%stride,(-w)%stride
    t = torch.from_numpy(x.copy())[None,None].to(device)
    if ph or pw: t=F.pad(t,(0,pw,0,ph),mode='reflect' if h>ph and w>pw else 'replicate')
    d = model.forward_denoise_only(t)['denoised'][0,0,:h,:w].float().cpu().numpy().copy()
    require(d.shape==x.shape and np.isfinite(d).all() and d.min()>=0 and d.max()<=1,
            'Nonfinite/out-of-range D2 or geometry change')
    return d


def mild_input(x,d,alpha=.25):
    require(np.shape(x)==np.shape(d), 'D2/native geometry mismatch')
    require(0<=alpha<=1 and np.isfinite(d).all(), 'Invalid mild input')
    return np.clip(x-alpha*(x-d),0,1).astype(np.float32)


def fit_mild(chosen, arrays):
    require(len(chosen)==len(arrays),'Calibration source/array length mismatch')
    require(chosen and all(r['split']=='train' for r in chosen),'Calibration must be train only')
    require(len({r['position_id'] for r in chosen})==len(chosen),'One fixed frame per train position')
    return fit_train_statistics(arrays, power=2., source_records=[dict(
        position_id=r['position_id'],sample_id=r['sample_id'],sha256=r['sha256'],
        mild_array_sha256=array_sha256(a),split='train') for r,a in zip(chosen,arrays)])


def model_fingerprint(model):
    return digest({k:array_sha256(v.detach().cpu().numpy()) for k,v in model.state_dict().items()})


def verify_cache(cache,snapshot,full=True):
    cache=Path(cache);meta=read(cache/'cache.json');unsigned=dict(meta);claimed=unsigned.pop('identity')
    require(digest(unsigned)==claimed,'Cache identity changed')
    require(meta['snapshot_identity']==verify_lock(snapshot)['identity'],'Snapshot identity changed')
    require(meta['sources']==source_identity(),'Code changed; new cache identity required')
    require(not full or meta['full_cache_complete'],'Preview-only cache cannot train/evaluate')
    for r in meta['files']: require(sha(cache/r['path'])==r['sha256'],'Cache file changed: '+r['path'])
    if full and not meta.get('synthetic',False):
        actual=read(cache/'frames.json');expected=read(Path(snapshot)/'frames.json')
        require({r['sample_id']:r['sha256'] for r in actual}=={r['sample_id']:r['sha256'] for r in expected}
                and len(actual)==len(expected),'Full cache does not contain the complete frozen frame pool')
        require(meta.get('geometry_policy')==GEOMETRY,'Cache geometry policy differs')
        require(meta.get('ac_config_sha256')==digest(read(cache/'auto_config.json')),'AC configuration SHA changed')
    return meta


def prepare(root,snapshot,approval,checkpoint,binding,output,device='cpu',full=False,optional=False):
    root=Path(root);snapshot=Path(snapshot);out=Path(output)
    evidence=audit(root,snapshot,approval,checkpoint,binding)
    require(evidence['status']=='passed','Preflight blocked: '+str(evidence['problems']))
    require(not out.exists(),'Existing output preserved; select a new cache directory')
    out.mkdir(parents=True);write(out/'audit.json',evidence)
    split=evidence['split'];rows=[]
    for r in read(snapshot/'frames.json'):
        part=next(s for s in ('train','inner','outer') if r['group_id'] in split[s])
        rows.append(dict(**r,split=part))
    fixed=[min((r for r in rows if r['position_id']==p),key=lambda r:r['sample_id'])
           for p in sorted({r['position_id'] for r in rows})]
    chosen=[r for r in fixed if r['split']=='train']
    model=load_d2(evidence['denoiser'],device);memo={};weights_before=model_fingerprint(model)
    for r in chosen:
        x=noisy(root/r['path']).astype(np.float32);d=native_d2(x,model,device)
        memo[r['sample_id']]=(x,d,mild_input(x,d))
    statistics=fit_mild(chosen,[memo[r['sample_id']][2] for r in chosen])
    cfg=config_from_statistics(statistics,beta=.5,gain_cap=6.,reliability_floor=.2)
    write(out/'ac_statistics.json',statistics);write(out/'auto_config.json',asdict(cfg))
    write(out/'difficulty.json',difficulty_fit(snapshot,root,[r for r in rows if r['split']=='train']))
    write(out/'split.json',split);write(out/'fixed_samples.json',[r['sample_id'] for r in fixed])
    records=[];diagnostics=[]
    for r in (rows if full else fixed):
        sid=r['sample_id'];pid=r['position_id']
        require(sha(root/r['path'])==r['sha256'],'Source image changed: '+sid)
        if sid in memo: x,d,m=memo.pop(sid)
        else:
            x=noisy(root/r['path']).astype(np.float32);d=native_d2(x,model,device);m=mild_input(x,d)
        a,maps,stats=stable_ac_v1(m,cfg)  # r2 already beta-mixes exactly ONCE
        arrays=dict(noisy=x,denoised=d,mild=m,auto=a)
        if optional: arrays['direct_auto']=stable_ac_v1(x,cfg)[0]
        paths={}
        for key,val in arrays.items():
            require(val.shape==x.shape and np.isfinite(val).all(),'Native view mismatch')
            p=Path('arrays')/f'{sid}_{key}.npy';npy(out/p,val.astype(np.float32))
            paths[key]=dict(path=p.as_posix(),sha256=sha(out/p))
        records.append(dict(**r,shape=list(x.shape),views=paths))
        diagnostics.append(dict(sample_id=sid,position_id=pid,split=r['split'],
            input_q01=float(np.quantile(m,.01)),input_q50=float(np.quantile(m,.5)),input_q99=float(np.quantile(m,.99)),
            output_q01=float(np.quantile(a,.01)),output_q50=float(np.quantile(a,.5)),output_q99=float(np.quantile(a,.99)),
            bottom_high_clip=float(maps['output_high_clip'][int(.9*x.shape[0]):].mean()),
            bottom_saturation=float((a[int(.9*x.shape[0]):]>=1-1e-6).mean()),
            **{k:v for k,v in stats.items() if k.endswith('_fraction')}))
        print(dict(event='native_input_cached',sample=sid,split=r['split']),flush=True)
    weights_after=model_fingerprint(model)
    require(weights_before==weights_after and not model.training and not any(p.requires_grad for p in model.parameters()),
            'Frozen D2 changed during preparation')
    write(out/'frozen_d2_audit.json',dict(status='passed',before=weights_before,after=weights_after,
        eval=True,requires_grad_parameters=0,native_inference=True,geometry_policy=GEOMETRY))
    del model
    write(out/'frames.json',records);table(out/'input_diagnostics.csv',diagnostics)
    df=pd.DataFrame(diagnostics);table(out/'input_diagnostics_by_position.csv',
        df.groupby(['position_id','split'],as_index=False).mean(numeric_only=True).to_dict('records'))
    write(out/'denoiser_identity.json',evidence['denoiser']);write(out/'source_confirmation.json',read(approval))
    for name in ('snapshot.json','positions.json','annotation_states.json','historical_test_access.json'):
        if (snapshot/name).is_file(): write(out/('frozen_'+name),read(snapshot/name))
    write(out/'reuse_audit.json',dict(N0='not reused: full cross-run identity equivalence not established',
        NN='not reused: full cross-run identity equivalence not established',
        historical_A0_NA='reference only; different train-noisy theta',optional_A0_NA_same_theta=optional))
    write(out/'access_record.json',dict(positions=sorted({r['position_id'] for r in records}),
        purpose='user-authorized existing development positions; NOT blind test',
        historical_access_preserved=True,new_sealed_test_assets_opened=0))
    files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    meta=dict(version=VERSION,fold=0,manual=False,full_cache_complete=full,frames=len(records),
        snapshot_identity=read(snapshot/'snapshot.json')['identity'],denoiser_sha256=evidence['denoiser']['sha256'],
        alpha=.25,beta=.5,gamma=1,power=2,synthetic=False,ac_version=CACHE_VERSION,ac_config=asdict(cfg),
        ac_config_sha256=digest(asdict(cfg)),geometry_policy=GEOMETRY,
        source_frame_identity_sha256=digest([dict(sample_id=r['sample_id'],sha256=r['sha256']) for r in records]),
        statistics_sha256=statistics['statistics_sha256'],sources=source_identity(),files=files)
    meta['identity']=digest(meta);write(out/'cache.json',meta)
    return dict(status='prepared',path=str(out),identity=meta['identity'],full_cache_complete=full)
