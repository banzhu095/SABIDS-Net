"""Provenance and immutable native analytic cache. No historical gate changes."""
from dataclasses import asdict
from pathlib import Path
import io
import subprocess

import numpy as np
import torch
import yaml

from tools.ac_vessel_labelsv2.common import (
    ROOT, atomic, digest, read, write, table, sha, noisy, npy, png, plane,
    require, verify_lock, fresh,
)
from tools.denoise_then_ac_v1.core import exposure, audit_d2
from tools.ac_vessel_labelsv2.processing import difficulty_fit
from sabids.experiments.attenuation_aux import (
    StableACConfig, fit_train_statistics, config_from_statistics, stable_ac_v1,
)
from sabids.models.learnable_ac import NativeStableAC

VERSION='learnable_ac_v1.r1'
BASE=('P0','P1','P2','P3','P4','P5','P6')
ADAPTIVE=('F0','F1','F2','F3','F4','F5','F6')
VIEWS=dict(P0='noisy',P1='m',P2='direct_ac',P3='z0',P4='tone',P5='z0',P6='clean')
COMPARISONS=(('F5','F0'),('F5','P3'),('F5','F1'),('P3','P1'),('P3','P2'),('P3','P4'),
             ('F5','F3'),('F5','F6'),('F5','F4'),('F5','F2'))
PACK=('m','z0','c','depth','reliability','gain','rho','valid')


def config(smoke=False):
    value=yaml.safe_load((ROOT/'configs/learnable_ac_v1/protocol.yaml').read_text(encoding='utf8'))
    require(value['amp'] is False and value['boundary_loss']==0 and not value['joint_finetune'],
            'Unregistered training contract')
    value['synthetic']=smoke
    if smoke:value.update(channels=[4,8,16],depths=[1,1,1],decoder_depth=1,
                          context_channels=[4,8,16],size=32,base_epochs=2,adaptive_epochs=2)
    return value


def sources():
    files=list((ROOT/'tools/learnable_ac_v1').glob('*.py'))+[
        ROOT/'sabids/models/learnable_ac.py',ROOT/'tools/run_learnable_ac_v1.py',
        ROOT/'configs/learnable_ac_v1/protocol.yaml',ROOT/'sabids/experiments/attenuation_aux.py',
        ROOT/'tools/downstream_seg_benchmark/model.py',ROOT/'tools/denoise_then_ac_v1/learning.py',
        ROOT/'tools/downstream_seg_benchmark/metrics.py',ROOT/'tools/downstream_seg_benchmark/engine.py',
        ROOT/'tools/ac_vessel_labelsv2/common.py',ROOT/'tools/ac_vessel_labelsv2/processing.py',
        ROOT/'tools/denoise_result_review/image_io.py',ROOT/'sabids/models/dual_view_segmenter.py',
        ROOT/'sabids/models/blocks.py',ROOT/'sabids/losses/common.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}


def npz(path,**arrays):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);atomic(path,buffer.getvalue())


def verify_historical_cache(cache,snapshot):
    """Validate recorded bytes, not pretend current source == deployed source."""
    cache=Path(cache);meta=read(cache/'cache.json');unsigned=dict(meta);claimed=unsigned.pop('identity')
    require(digest(unsigned)==claimed,'Historical cache identity invalid')
    require(meta['snapshot_identity']==verify_lock(snapshot)['identity'],'Snapshot differs')
    require(meta.get('full_cache_complete') and not meta.get('synthetic',False),'Real full native cache required')
    require(meta.get('alpha')==.25 and meta.get('beta')==.5 and 'full-depth' in meta.get('geometry_policy',''),
            'Historical preprocessing differs')
    for r in meta['files']:require(sha(cache/r['path'])==r['sha256'],'Historical cache changed: '+r['path'])
    denoiser=read(cache/'denoiser_identity.json')
    require(denoiser['sha256']==meta['denoiser_sha256'] and denoiser['ac_version']=='attenuation_aux_v1.ac_r2',
            'Historical denoiser/cache binding differs')
    require(denoiser['selection']['completed_epochs']>=20 and denoiser['selection']['test_assets_opened']==0,
            'Smoke/incomplete denoiser cannot prepare scientific cache')
    return meta


def upstream_audit(root, split, evidence, mode, require_mild=True):
    """Actual checkpoint/config/manifest evidence, never self-declared safe=true.

    D1/teacher/D2 dependencies must be explicitly supplied in formal mode.
    Bind their exact paths in upstream.json; no discovery by newest timestamp.
    Unknown exposure is a formal blocker. Development keeps all disclosures.
    """
    from tools.downstream_seg_benchmark.engine import load
    result={};problems=[]
    for role in ('D1','teacher','D2'):
        item=(evidence or {}).get(role)
        if not item:
            result[role]=dict(status='missing',unknown=['no bound checkpoint/config'])
            if require_mild:problems.append(role+': provenance missing')
            continue
        try:
            cp=Path(item['checkpoint']);cp=cp if cp.is_absolute() else Path(root)/cp
            require(sha(cp)==item['sha256'],role+': checkpoint SHA differs')
            raw=load(cp);cfg=raw['config']
            from sabids.config import load_config
            rp=Path(item['resolved_config']);rp=rp if rp.is_absolute() else Path(root)/rp
            require(sha(rp)==item['config_sha256'],role+': resolved config SHA differs')
            other=load_config(rp)
            for key in ('model','data','train','loss','d2'):
                require(cfg.get(key)==other.get(key),role+': embedded/resolved config differs')
            e=exposure(root,cfg,split)
            require(not e['unknown'] and bool(e['train']) and bool(e['selection']),role+': exposure unknown')
            recorded_manifest=cfg.get('runtime',{}).get('manifest_sha256')
            require(recorded_manifest==e.get('manifest_sha256'),role+': manifest no longer bound to training-time SHA')
            result[role]=dict(status='verified',checkpoint=str(cp),sha256=item['sha256'],
                              config_sha256=item['config_sha256'],epoch=raw.get('epoch'),exposure=e)
            if set(e['train']+e['selection']) & set(split['outer']):problems.append(role+': outer exposure')
            if role in ('D1','teacher'):
                inv_path=item.get('initial_inventory') or cfg.get('runtime',{}).get('training_asset_inventory_initial')
                require(inv_path and item.get('initial_inventory_sha256'),role+': immutable initial inventory/SHA missing')
                inv_path=Path(inv_path);inv_path=inv_path if inv_path.is_absolute() else Path(root)/inv_path
                require(sha(inv_path)==item['initial_inventory_sha256'],role+': initial inventory SHA differs')
                inv=read(inv_path)
                require(inv.get('recorded_at_training') is True and inv.get('recorded_before_optimizer_step') is True
                    and inv.get('manifest_sha256')==e['manifest_sha256'],role+': initial inventory not bound to actual cohort')
            if role=='D2':
                bound=audit_d2(root,cp,item['binding'],split)
                result[role]['binding']=bound
            if role=='D1':
                initialization=cfg.get('runtime',{}).get('initialization_checkpoint') or cfg.get('train',{}).get('pretrained')
                # This first formal path intentionally starts D1 from scratch;
                # inherited D1 initialization requires a separate complete
                # upstream lineage review, never implicitly treated as safe.
                require(not initialization,'D1 has an additional unreviewed initialization source')
        except Exception as exc:
            result[role]=dict(status='invalid',error=type(exc).__name__+': '+str(exc))
            if require_mild:problems.append(role+': invalid evidence')
    if require_mild and all(result[r].get('status')=='verified' for r in ('D1','teacher','D2')):
        # Trace both the D2 initialization and its teacher back to supplied files.
        try:
            cfg=load(result['D2']['checkpoint'])['config']
            teacher_cfg=cfg['d2']['teacher']
            require(teacher_cfg['sha256']==result['teacher']['sha256'],'D2 teacher lineage differs')
            from sabids.experiments.d2 import audit_d2_checkpoint_binding
            bound=result['D2']['binding']['binding']
            # This field is part of the audited binding, derived from the
            # checkpoint's embedded runtime by write_d2_checkpoint_binding.
            expected=bound.get('initialization_checkpoint_sha256')
            require(expected==result['D1']['sha256'],'D1 initialization lineage unknown/mismatched')
            teacher_raw=load(result['teacher']['checkpoint'])
            require(teacher_raw['config'].get('runtime',{}).get('initialization_checkpoint_sha256')==result['D1']['sha256'],
                    'Teacher D1 initialization lineage unknown/mismatched')
        except Exception as exc:problems.append('lineage: '+str(exc))
    return dict(mode=mode,roles=result,problems=problems,
                formal_allowed=not problems,development_exploratory=mode=='development',
                note='No independent generalization claim; observed outer remains development' if mode=='development' else 'Outer excluded from all upstream fitting/selection',
                new_sealed_test_assets_opened=0)


def audit(root,snapshot,input_cache=None,upstream=None,mode='development',fold=0,no_mild=False):
    snapshot=Path(snapshot);problems=[];meta=None
    try:
        lock=verify_lock(snapshot);split=read(snapshot/'folds.json')[fold]
        rows=read(snapshot/'frames.json');states=read(snapshot/'annotation_states.json')
        accepted={r['position_id'] for r in states if r['status']=='confirmed'}
        require(all(r['position_id'] in accepted for r in rows),'Unconfirmed supervision')
        groups={s:set(split[s]) for s in ('train','inner','outer')}
        require(not any(groups[a]&groups[b] for a,b in (('train','inner'),('train','outer'),('inner','outer'))),'Split overlap')
        for r in rows:
            require(sum(r['group_id'] in groups[s] for s in groups)==1,'Unassigned/repeated split')
        verified_positions={};checked=0
        for r in rows:
            src=Path(root)/r['path']
            require(sha(src)==r['sha256'],'Actual noisy SHA differs: '+r['sample_id'])
            x=noisy(src);pid=r['position_id']
            require(x.ndim==2 and np.isfinite(x).all(),'Invalid native source: '+r['sample_id'])
            if pid not in verified_positions:
                masks={k:plane(snapshot/'labels'/pid/(k+'.png')) for k in ('layer','vessel','layer_valid','vessel_valid','coverage','uncertain')}
                require(all(a.shape==x.shape and np.isin(a,(0,255)).all() for a in masks.values()),'Label geometry/encoding differs: '+pid)
                require(not np.any((masks['vessel_valid']>0)&((masks['coverage']==0)|(masks['uncertain']>0))),
                        'Unknown/uncertain mistakenly marked Vessel-valid: '+pid)
                verified_positions[pid]=dict(shape=list(x.shape),outside_vessel_pixels=int(((masks['vessel']>0)&(masks['layer']==0)).sum()))
            require(list(x.shape)==verified_positions[pid]['shape'],'Noisy/native label dimensions differ: '+r['sample_id'])
            checked+=1
        if input_cache:
            meta=verify_historical_cache(input_cache,snapshot)
            require(meta['fold']==fold,'Inherited cache fold differs')
        provenance=upstream_audit(root,split,upstream or {},mode,not no_mild)
        if not input_cache and not no_mild and provenance['roles']['D2'].get('status')!='verified':
            problems.append('Neither a verified native full mild cache nor an actual audited D2 is available')
        if input_cache:
            inherited=read(Path(input_cache)/'denoiser_identity.json')
            provenance['inherited_development_evidence']=dict(
                sha256=inherited['sha256'],checkpoint=inherited['checkpoint'],selection=inherited['selection'],
                D2_exposure=inherited['d2_exposure'],teacher_exposure=inherited['teacher_exposure'],
                evidence_type='SHA-verified historical cache metadata; NOT a fresh checkpoint inspection')
            if mode=='formal':
                require(provenance['roles']['D2'].get('sha256')==meta['denoiser_sha256'],
                        'Formal cache denoiser is not the audited fold-local D2')
        if mode=='formal':problems.extend(provenance['problems'])
        counts={s:dict(positions=len(groups[s]),frames=sum(r['group_id'] in groups[s] for r in rows)) for s in groups}
        return dict(status='blocked' if problems else 'passed',version=VERSION,mode=mode,problems=problems,
                    counts=counts,split=split,snapshot_identity=lock['identity'],upstream=provenance,
                    verified_frames=checked,positions=verified_positions,
                    historical_cache_identity=meta['identity'] if meta else None,
                    allowed_base=['P0','P2'] if no_mild else list(BASE),
                    registration='existing user_confirmation; no new registration proof required',
                    new_sealed_test_assets_opened=0)
    except Exception as exc:
        return dict(status='blocked',mode=mode,problems=[type(exc).__name__+': '+str(exc)],new_sealed_test_assets_opened=0)


def prepare(root,snapshot,output,input_cache=None,upstream=None,mode='development',fold=0,no_mild=False,device='cpu'):
    root,snapshot=Path(root),Path(snapshot)
    gate=audit(root,snapshot,input_cache,upstream,mode,fold,no_mild)
    require(gate['status']=='passed','Asset audit blocked: '+str(gate['problems']))
    out=fresh(output);write(out/'audit.json',gate)
    split=gate['split'];native=[];old={};denoiser=None;memo={};frozen_before=None
    if input_cache:old={r['sample_id']:r for r in read(Path(input_cache)/'frames.json')}
    elif not no_mild:
        from tools.denoise_then_ac_v1.core import load_d2,model_fingerprint
        denoiser=load_d2(gate['upstream']['roles']['D2']['binding'],device)
        frozen_before=model_fingerprint(denoiser)
    for r in read(snapshot/'frames.json'):
        require(sha(root/r['path'])==r['sha256'],'Actual noisy identity differs: '+r['sample_id'])
        stage=next(s for s in ('train','inner','outer') if r['group_id'] in split[s])
        native.append(dict(**r,split=stage))
    fixed=[min((r for r in native if r['position_id']==p),key=lambda r:r['sample_id'])
           for p in sorted({r['position_id'] for r in native})]
    chosen=[r for r in fixed if r['split']=='train']
    def inputs(r):
        if r['sample_id'] in memo:return memo[r['sample_id']]
        x=noisy(root/r['path']).astype(np.float32)
        if no_mild:return x,x
        if denoiser is not None:
            from tools.denoise_then_ac_v1.core import native_d2,mild_input
            d=native_d2(x,denoiser,device)
            return x,mild_input(x,d,.25)
        q=old[r['sample_id']];require(q['sha256']==r['sha256'],'Mild/source mismatch')
        v=q['views']['mild'];m=np.load(Path(input_cache)/v['path'],allow_pickle=False)
        require(sha(Path(input_cache)/v['path'])==v['sha256'] and x.shape==m.shape,'Mild SHA/geometry differs')
        d=np.load(Path(input_cache)/q['views']['denoised']['path'],allow_pickle=False)
        require(d.shape==x.shape and np.isfinite(m).all() and np.isfinite(d).all() and m.min()>=0 and m.max()<=1,
                'Invalid native mild/restoration')
        expected=np.clip(x-.25*(x-d),0,1).astype(np.float32)
        require(np.array_equal(m,expected),'Mild is not fixed alpha=.25 restoration mix')
        return x,m
    for r in chosen:memo[r['sample_id']]=inputs(r)
    calibration=[memo[r['sample_id']][1] for r in chosen]
    # Reproduce the same train-mild calibration rule, never fit on labels/clean.
    stats=fit_train_statistics(calibration,source_records=[dict(split='train',sample_id=r['sample_id'],
             position_id=r['position_id'],sha256=r['sha256']) for r in chosen])
    cfg=config_from_statistics(stats,beta=.5)
    write(out/'ac_statistics.json',stats);write(out/'ac_config.json',asdict(cfg))
    # Global quantile mapping matches AC marginal gray distribution on train.
    before=np.concatenate([a.ravel()[::max(1,a.size//20000)] for a in calibration])
    after=np.concatenate([stable_ac_v1(a,cfg)[0].ravel()[::max(1,a.size//20000)] for a in calibration])
    q=np.linspace(0,1,257);xp,ix=np.unique(np.quantile(before,q),return_index=True)
    yp=np.maximum.accumulate(np.quantile(after,q)[ix])
    write(out/'tone.json',dict(input=xp.tolist(),output=yp.tolist(),sources=[r['sample_id'] for r in chosen],
        fit_split='train',rule='pooled equal-position quantile mapping mild to fixed AC; no depth, no GT'))
    write(out/'difficulty.json',difficulty_fit(snapshot,root,[r for r in native if r['split']=='train']))
    ac=NativeStableAC(cfg);records=[];diagnostics=[]
    for r in native:
        x,m=inputs(r)
        a=ac(torch.from_numpy(m)[None,None])
        arrays={k:v[0,0].detach().numpy() for k,v in a.items()}
        reference=stable_ac_v1(m,cfg)[0]
        require(np.max(np.abs(arrays['z0']-reference))<=1e-7,'r2 compatibility failure')
        arrays.update(noisy=x,direct_ac=stable_ac_v1(x,cfg)[0],tone=np.interp(m,xp,yp).astype(np.float32))
        for v in arrays.values():require(v.shape==x.shape and np.isfinite(v).all(),'Native analytic finite/shape error')
        p=Path('native')/(r['sample_id']+'.npz');npz(out/p,**arrays)
        records.append(dict(**r,shape=list(x.shape),analytic=p.as_posix(),sha256_native=sha(out/p)))
        diagnostics.append(dict(sample_id=r['sample_id'],position_id=r['position_id'],split=r['split'],
            **{k:float(arrays[k].mean()) for k in ('denominator_active','gain_cap_active','high_clip','low_clip','fallback')},
            rho_mean=float(arrays['rho'].mean()),r2_max_abs=float(np.max(np.abs(arrays['z0']-reference)))))
    write(out/'frames.json',records);write(out/'fixed_samples.json',[r['sample_id'] for r in fixed]);write(out/'split.json',split)
    table(out/'input_diagnostics.csv',diagnostics)
    if denoiser is not None:
        from tools.denoise_then_ac_v1.core import model_fingerprint
        frozen_after=model_fingerprint(denoiser)
        require(frozen_before==frozen_after and not denoiser.training and not any(p.requires_grad for p in denoiser.parameters()),'D2 changed')
        write(out/'frozen_d2.json',dict(before=frozen_before,after=frozen_after,eval=True,requires_grad_parameters=0,
                                      native_before_ac=True,device=device))
    result=dict(version=VERSION,mode=mode,fold=fold,no_mild=no_mild,synthetic=False,
        provenance_label='development / historical upstream exposure' if mode=='development' else 'formal / audited fold-local upstream',
        snapshot_identity=gate['snapshot_identity'],upstream=gate['upstream'],
        allowed_base=gate['allowed_base'],full_cache_complete=True,frames=len(records),
        alpha=0 if no_mild else .25,beta=.5,ac_metadata=ac.metadata(),sources=sources(),
        source_cache=gate['historical_cache_identity'],geometry='native full valid depth AC BEFORE common aspect resize/pad',
        files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()])
    result['identity']=digest(result);write(out/'cache.json',result)
    return result


def verify_cache(cache,snapshot):
    cache=Path(cache);meta=read(cache/'cache.json');unsigned=dict(meta);identity=unsigned.pop('identity')
    require(meta['version']==VERSION and digest(unsigned)==identity,'Cache identity differs')
    require(meta['snapshot_identity']==verify_lock(snapshot)['identity'],'Labels changed')
    require(meta['sources']==sources(),'Task code changed; new namespace/cache required')
    require(meta['full_cache_complete'],'Incomplete/preview-only cache')
    for row in meta['files']:require(sha(cache/row['path'])==row['sha256'],'Cache file changed: '+row['path'])
    if meta['mode']=='formal':require(meta['upstream']['formal_allowed'],'Formal provenance blocked')
    return meta
