"""Independent NumPy r2 versus torch, on a common fixed native sample pool."""
from pathlib import Path
import numpy as np
import torch
from .core import read,write,table,npy,fresh,sha,require,verify_cache
from sabids.experiments.attenuation_aux import StableACConfig,stable_ac_v1
from sabids.models.learnable_ac import NativeStableAC


def compare(cache,snapshot,output):
    meta=verify_cache(cache,snapshot);out=fresh(output);cache=Path(cache)
    config=StableACConfig(**read(cache/'ac_config.json'));model=NativeStableAC(config)
    fixed=set(read(cache/'fixed_samples.json'));results=[]
    for row in read(cache/'frames.json'):
        if row['sample_id'] not in fixed:continue
        with np.load(cache/row['analytic'],allow_pickle=False) as arrays:
            for view,key in [('mild','m'),('noisy','noisy')]:
                x=arrays[key].astype(np.float32)
                # Independent production implementations: numpy function never
                # invokes NativeStableAC or consumes its cached output.
                expected,maps,_=stable_ac_v1(x,config)
                actual=model(torch.from_numpy(x)[None,None])
                dest=out/row['position_id']/view
                npy(dest/'input_native.npy',x);npy(dest/'numpy_r2.npy',expected)
                npy(dest/'torch_r2.npy',actual['z0'][0,0].numpy())
                for name,np_array,torch_key in [('mixed',expected,'z0'),('protected_candidate',maps['safe_output'],'c'),('integral',maps['denominator'],'energy')]:
                    observed=actual[torch_key][0,0].numpy()
                    # Integrals exported float32; compare in that same output
                    # precision, while both implementations integrate float64.
                    delta=observed-np_array.astype(np.float32)
                    error=float(np.abs(delta).max())
                    require(error<=1e-7 if name!='integral' else error<=1e-4,'Native independent reference mismatch: '+row['sample_id'])
                    results.append(dict(sample_id=row['sample_id'],position_id=row['position_id'],split=row['split'],
                        view=view,quantity=name,max_abs=error,rmse=float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
                        native_shape=str(x.shape),source_sha256=row['sha256'],mode=meta['mode'],
                        provenance_label=meta.get('provenance_label','see upstream audit')))
    table(out/'numpy_torch_native_comparison.csv',results)
    result=dict(status='passed',positions=len(fixed),comparisons=len(results),cache_identity=meta['identity'],
        independent_numpy_source_sha256=sha(Path(__file__).resolve().parents[2]/'sabids/experiments/attenuation_aux.py'),
        torch_source_sha256=sha(Path(__file__).resolve().parents[2]/'sabids/models/learnable_ac.py'),
        arithmetic='float64 integration; compare exported float32 arrays; no PNG reconstruction',
        mode=meta['mode'],provenance_label=meta.get('provenance_label','see upstream audit'),
        max_abs=max(r['max_abs'] for r in results))
    write(out/'check.json',result);return result
