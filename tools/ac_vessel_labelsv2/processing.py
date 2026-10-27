from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation,label

from sabids.experiments.attenuation_aux import fit_train_statistics,config_from_statistics,stable_ac_v1,StableACConfig
from tools.pku37_label_review.display import compute_intermediates,render,resolve_parameters
from .common import *
from .snapshot import preflight


def components(x,layer,vessel,valid):
    ids,n=label(vessel,np.ones((3,3)));result=[]
    for i in range(1,n+1):
        mask=ids==i;known=bool(valid[mask].all());ring=binary_dilation(mask,iterations=3)&layer&~vessel&valid
        contrast=None
        if known and ring.any():
            values=x[ring];iqr=float(np.quantile(values,.75)-np.quantile(values,.25))
            contrast=float((np.median(values)-np.median(x[mask]))/max(iqr,1/255))
        result.append(dict(component_id=i,area_px=int(mask.sum()),fully_known=known,signed_local_contrast=contrast,
            ring_pixels=int(ring.sum())))
    return ids,result


def difficulty_fit(snapshot,project,training):
    snapshot=Path(snapshot);frames=read(snapshot/'frames.json');rows=[];sources=[]
    for position in sorted({r['position_id'] for r in training}):
        r=min((r for r in training if r['position_id']==position),key=lambda r:r['sample_id']);sources.append(r['sample_id'])
        x=noisy(Path(project)/r['path']);l=plane(snapshot/'labels'/position/'layer.png')>0;v=plane(snapshot/'labels'/position/'vessel.png')>0;vv=plane(snapshot/'labels'/position/'vessel_valid.png')>0
        _,records=components(x,l,v,vv);rows.extend(records)
    areas=[r['area_px'] for r in rows if r['fully_known']];contrasts=[abs(r['signed_local_contrast']) for r in rows if r['signed_local_contrast'] is not None]
    require(areas and contrasts,'Train subset cannot define component thresholds; no historical threshold substitution')
    return dict(small_area_px=float(np.quantile(areas,.25)),low_contrast_abs=float(np.quantile(contrasts,.25)),
        sources=sources,fit_split='train',definition='8-connected original native GT; fully known components only; local3px stroma ring; signed(median_stroma-median_lumen)/max(stromaIQR,1/255); low=abs(signed)<=trainQ25; small=area<=trainQ25',
        units='native pixels; no fabricated physical spacing',not_image_or_label_resampling=True)


def prepare(project,snapshot,fold,output,approval,manual=False):
    project=Path(project);snapshot=Path(snapshot);gate=preflight(project,snapshot,approval)
    if manual:verify_lock(snapshot,include_manual=True)
    require(gate['status']=='passed','Preflight blocked: '+str(gate['problems']))
    out=fresh(output);lock=read(snapshot/'snapshot.json');split=read(snapshot/'folds.json')[fold]
    require(split['fold']==fold,'Fold indexing mismatch');positions=read(snapshot/'positions.json');frames=read(snapshot/'frames.json')
    train=[r for r in frames if r['group_id'] in split['train']]
    chosen=[min((r for r in train if r['position_id']==p),key=lambda r:r['sample_id']) for p in sorted({r['position_id'] for r in train})]
    # Exactly one deterministic noisy frame/position: equal contribution even if
    # position frame counts differ. No clean, GT or evaluation manual parameters.
    statistics=fit_train_statistics([noisy(project/r['path']) for r in chosen],source_records=[dict(position_id=r['position_id'],sample_id=r['sample_id'],sha256=r['sha256'],split='train') for r in chosen])
    auto=config_from_statistics(statistics,beta=.5,gain_cap=6.,reliability_floor=.2)
    before=[];after=[]
    for r in chosen:
        x=noisy(project/r['path']);a=stable_ac_v1(x,auto)[0]
        ids=np.linspace(0,x.size-1,min(20000,x.size),dtype=int);before.append(x.ravel()[ids]);after.append(a.ravel()[ids])
    q=np.linspace(0,1,257);xp=np.quantile(np.concatenate(before),q);yp=np.quantile(np.concatenate(after),q)
    xp,unique=np.unique(xp,return_index=True);yp=np.maximum.accumulate(yp[unique]);grid=np.linspace(0,1,4097)
    lut=np.interp(grid,xp,yp)
    tone=dict(input=grid.tolist(),output=lut.tolist(),sources=[r['sample_id'] for r in chosen],
        rule='train position-equal pooled fixed quantile monotone LUT, no depth/GT/clean, not independent per-image normalization')
    difficulty=difficulty_fit(snapshot,project,train)
    write(out/'ac_statistics.json',statistics);write(out/'auto_config.json',asdict(auto));write(out/'tone_lut.json',tone);write(out/'difficulty.json',difficulty)
    write(out/'split.json',split);component_rows=[];cache_rows=[]
    # Manual config is never opened by the automatic path.
    manual_config=StableACConfig(**read(snapshot/'manual_algorithm.json')) if manual else None
    for r in frames:
        x=noisy(project/r['path']);pid=r['position_id'];sid=r['sample_id']
        stage='train' if r['group_id'] in split['train'] else 'inner' if r['group_id'] in split['inner'] else 'outer'
        a=stable_ac_v1(x,auto)[0];t=np.interp(x,grid,lut)
        arrays=dict(noisy=x.astype(np.float32),tone=t.astype(np.float32),auto=a.astype(np.float32))
        if manual:
            profile=read(snapshot/'manual'/f'{pid}.json');params=resolve_parameters(profile['parameters'],dict(beta=manual_config.beta,W=manual_config.output_window,gamma=1.,opacity=.35))
            arrays['manual']=render(compute_intermediates(x,manual_config),params)[1].astype(np.float32)
        paths={}
        for name,value in arrays.items():
            require(np.isfinite(value).all() and value.min()>=0 and value.max()<=1,'Non-finite or invalid view')
            path=Path('arrays')/f'{sid}_{name}.npy';npy(out/path,value);paths[name]=dict(path=path.as_posix(),sha256=sha(out/path))
        l=plane(snapshot/'labels'/pid/'layer.png')>0;v=plane(snapshot/'labels'/pid/'vessel.png')>0;vv=plane(snapshot/'labels'/pid/'vessel_valid.png')>0
        require(l.shape==v.shape==vv.shape==x.shape,'Native geometry mismatch after registration gate')
        _,records=components(x,l,v,vv)
        for c in records:
            c.update(sample_id=sid,position_id=pid,small=c['area_px']<=difficulty['small_area_px'],
                low_contrast=c['signed_local_contrast'] is not None and abs(c['signed_local_contrast'])<=difficulty['low_contrast_abs'])
        component_rows.extend(records);cache_rows.append(dict(**r,split=stage,shape=list(x.shape),views=paths))
    write(out/'frames.json',cache_rows);table(out/'components.csv',component_rows)
    write(out/'fixed_samples.json',[min((r['sample_id'] for r in frames if r['position_id']==p['position_id'])) for p in positions])
    identities=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    result=dict(version=VERSION,snapshot_identity=lock['identity'],fold=fold,manual=manual,files=identities,
        calibration_positions=[r['position_id'] for r in chosen],registration_approval_sha256=sha(approval),
        sources=source_identity(),native_ac_before_geometry=True,normalization='fixed dtype range; shared geometry; no independent contrast normalization',frames=len(frames))
    result['identity']=digest(result);write(out/'cache.json',result);return dict(path=str(out),frames=len(frames),identity=result['identity'])


def verify_cache(cache,snapshot):
    cache=Path(cache);meta=read(cache/'cache.json');unsigned=dict(meta);claimed=unsigned.pop('identity')
    require(digest(unsigned)==claimed,'Cache identity changed')
    require(meta['snapshot_identity']==verify_lock(snapshot,include_manual=meta['manual'])['identity'],'Cache/snapshot mismatch')
    require(meta['sources']==source_identity(),'Code changed since cache preparation; do not silently mix versions')
    for r in meta['files']:require(sha(cache/r['path'])==r['sha256'],'Cache changed: '+r['path'])
    return meta
