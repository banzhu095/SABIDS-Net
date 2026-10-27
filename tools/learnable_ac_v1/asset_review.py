"""Read-only real local asset audit, with explicitly historical D2 evidence."""
from pathlib import Path
import numpy as np
import torch

from .core import *
from sabids.models.learnable_ac import NativeStableAC


def inspect_local(root,snapshot,bundle,output):
    root,snapshot,bundle=map(Path,(root,snapshot,bundle));out=fresh(output)
    geometry=audit(root,snapshot,no_mild=True)
    provenance=bundle/'provenance';d2=read(provenance/'denoiser_identity.json')
    before={str(p):sha(p) for p in [provenance/'denoiser_identity.json',provenance/'auto_config.json',snapshot/'snapshot.json']}
    cfg=StableACConfig(**read(provenance/'auto_config.json'));ac=NativeStableAC(cfg)
    fixed=read(provenance/'fixed_samples.json');rows=[];missing=[]
    for identity in sorted((bundle/'preview').glob('*/*/identity.json')):
        folder=identity.parent;meta=read(identity);pid=meta['position_id'];sid=meta['sample_id']
        if sid not in fixed:continue
        files={k:folder/(k+'.npy') for k in ('noisy','mild','auto','denoised')}
        if not all(p.is_file() for p in files.values()):
            # The intentionally LIGHT historical bundle includes floats only
            # for representative positions. Never reconstruct a numerical
            # reference from quantized PNG and call it float verification.
            for k in ('noisy','mild','auto'):
                pp=folder/(k+'.png');before[str(pp)]=sha(pp)
                require(tuple(plane(pp).shape)==tuple(geometry.get('positions',{}).get(pid,{}).get('shape',())),
                        'Real PNG/native shape differs: '+pid)
            missing.append(dict(position_id=pid,sample_id=sid,status='PNG present; native floating reference absent in light bundle'))
            continue
        a={k:np.load(p,allow_pickle=False) for k,p in files.items()}
        for p in files.values():before[str(p)]=sha(p)
        result=ac(torch.from_numpy(a['mild'])[None,None]);z=result['z0'][0,0].numpy()
        numerical=float(np.max(np.abs(z-a['auto'])))
        require(numerical<=1e-7,'Real r2 Torch/numpy mismatch: '+pid)
        formula=np.clip(a['noisy']-.25*(a['noisy']-a['denoised']),0,1).astype(np.float32)
        require(np.array_equal(a['mild'],formula),'Historical mild formula differs: '+pid)
        for k in ('noisy','mild','auto'):
            png_path=folder/(k+'.png');before[str(png_path)]=sha(png_path)
            require(np.array_equal(plane(png_path),np.rint(a[k]*255).astype(np.uint8)),'PNG/float differs: '+pid+'/'+k)
        rows.append(dict(position_id=pid,sample_id=sid,split=meta['split'],shape=str(a['mild'].shape),
            r2_max_abs=numerical,finite=True,base_reliability_once=True,
            effective_base_beta=.5,fallback=float(result['fallback'].mean()),
            gain_cap=float(result['gain_cap_active'].mean()),high_clip=float(result['high_clip'].mean()),
            rho_mean=float(result['rho'].mean()),checkpoint_available_locally=Path(d2['checkpoint']).is_file()))
    require(len(rows)+len(missing)==len(fixed),'Real fixed preview pool incomplete')
    require(all(sha(p)==h for p,h in before.items()),'Source assets changed while reading')
    table(out/'real_native_compatibility.csv',rows)
    result=dict(status='local_geometry_and_real_r2_passed' if geometry['status']=='passed' else 'blocked',
        geometry=geometry,real_fixed_previews=len(rows)+len(missing),float_reference_positions=len(rows),
        missing_float_references=missing,r2_max_abs=max((r['r2_max_abs'] for r in rows),default=None),
        preserved_asset_hashes=before,historical_d2=d2,
        current_checkpoint_verified=False,
        formal_allowed=False,formal_blockers=['D2/teacher outer exposure','Fresh D1 initialization lineage incomplete','Actual upstream checkpoint files not available locally'],
        new_training_started=False,new_sealed_test_assets_opened=0,
        GPU='user confirms no current rented instance; no more SSH attempted')
    write(out/'asset_audit.json',result)
    return dict(status=result['status'],positions=result['real_fixed_previews'],float_reference_positions=len(rows),
                frames=geometry.get('verified_frames'),r2_max_abs=result['r2_max_abs'],formal_allowed=False,path=str(out))
