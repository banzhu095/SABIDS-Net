"""Versioned annotation-only display; the production ac_r2 formula is unchanged."""
from __future__ import annotations

import math
from dataclasses import asdict, replace

import numpy as np

from sabids.experiments.attenuation_aux import CACHE_VERSION, stable_ac_v1

LEGACY = 'legacy_r2_v1'
WINDOW = 'protected_window_v2'
DISPLAY_CACHE_VERSION = 'annotation_display_v2.protected_preclip_r2'


def resolve_parameters(value, defaults):
    """Unversioned four-field records retain exactly their previous semantics."""
    result={key:float(value.get(key,defaults[key])) for key in ('beta','W','gamma','opacity')}
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('Display parameters must be finite')
    mode=value.get('display_mode',LEGACY)
    if 'display_mode' not in value and any(k in value for k in ('black','white','auto_window')):raise ValueError('New window fields require an explicit display mode/version')
    if value.get('algorithm_version',CACHE_VERSION)!=CACHE_VERSION:raise ValueError('Display algorithm version mismatch')
    if mode not in (LEGACY,WINDOW):raise ValueError('Unknown display mode')
    expected=1 if mode==LEGACY else 2
    if int(value.get('display_version',expected))!=expected:raise ValueError('Display version/mode mismatch')
    gamma_bounds=(.1,5.) if mode==LEGACY else (.2,3.)
    if not (0<=result['beta']<=1 and 0<=result['opacity']<=1 and 0<result['W']<=1e12 and gamma_bounds[0]<=result['gamma']<=gamma_bounds[1]):
        raise ValueError(f'beta/opacity0..1; positive W; gamma{gamma_bounds}')
    if value.get('input_source','reference')!='reference':raise ValueError('Only the bound reference is an AC input; noisy registration is unverified')
    if mode==LEGACY and 'display_mode' not in value:
        return result
    result.update(display_mode=mode,display_version=expected,input_source='reference',algorithm_version=CACHE_VERSION)
    if 'observation_roi' in value:result['observation_roi']=value['observation_roi']
    if mode==WINDOW:
        black=float(value.get('black',0.));white=float(value.get('white',black+result['W']))
        if not math.isfinite(black) or not math.isfinite(white) or white<=black or white-black<=np.finfo(float).eps*max(1.,abs(black),abs(white)):
            raise ValueError('Finite white must be greater than black; non-degenerate span required')
        if not math.isclose(white-black,result['W'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('W must equal white-black in protected_window_v2')
        result.update(black=black,white=white,auto_window=value.get('auto_window'),observation_roi=value.get('observation_roi'))
        if result['auto_window'] is not None and not isinstance(result['auto_window'],dict):raise ValueError('auto_window must be metadata or null')
    if result.get('observation_roi') is not None:
        rectangle=np.asarray(result['observation_roi'],float)
        if rectangle.shape!=(4,) or not np.isfinite(rectangle).all() or not np.equal(rectangle,np.floor(rectangle)).all():raise ValueError('Observation ROI must use four integral native coordinates')
    return result


def compute_intermediates(image,config):
    """One production forward per image/config; no GT enters the AC algorithm."""
    x=np.asarray(image,dtype=np.float64)
    baseline,maps,_=stable_ac_v1(x,replace(config,beta=1.))
    return dict(source=x,regularized_raw=maps['regularized_raw'],weight=maps['mix_weight'],
        reliability=maps['reliability'],valid=np.isfinite(x),baseline_ac=baseline,
        reference_W=float(config.output_window),gain_cap=float(config.gain_cap))


def render(intermediate,params):
    x=intermediate['source'];raw=intermediate['regularized_raw'];weight=intermediate['weight']
    reference_W=intermediate['reference_W'];cap=intermediate['gain_cap'];mode=params.get('display_mode',LEGACY)
    # The independent raw units are a coefficient-like display domain, NOT a
    # calibrated physical attenuation coefficient. Preserve the gain bound and
    # reliability interpolation, defer only the display clamp in the new mode.
    protected_raw=weight*np.minimum(raw,x*cap*reference_W)+(1-weight)*x*reference_W
    if mode==LEGACY:
        mapped=np.clip(raw/params['W'],0,1)
        safe=weight*np.minimum(mapped,x*cap)+(1-weight)*x
        ac=np.clip((1-params['beta'])*x+params['beta']*safe,0,1).astype(np.float32)
        if params['beta']==0:ac=x.copy()
        A=ac
        u=np.clip(ac,0,1)
        low=raw<=0;high=raw>=params['W']
    else:
        A=(1-params['beta'])*reference_W*x+params['beta']*protected_raw
        u=np.clip((A-params['black'])/(params['white']-params['black']),0,1)
        # ac.npy is always the original, default-window r2 numerical result,
        # independent of the new manual display window. raw values are separate.
        ac=np.clip((1-params['beta'])*x+params['beta']*(weight*np.minimum(np.clip(raw/reference_W,0,1),x*cap)+(1-weight)*x),0,1).astype(np.float32)
        if params['beta']==0:ac=x.copy()
        low=A<=params['black'];high=A>=params['white']
    display=u**params['gamma']
    maps=dict(reliability=intermediate['reliability'],low_reliability=weight<=0,
        output_low_clip=low,output_high_clip=high,A=A,protected_raw=protected_raw,
        regularized_raw=raw,pre_gamma=u,valid=intermediate['valid'])
    return ac,display,maps


def auto_window(intermediate,params,roi,valid):
    """Explicit, one-shot P1/P99 fit to a frozen Layer or manual rectangle."""
    params=dict(params,display_mode=WINDOW,display_version=2,black=0.,white=params['W'])
    _,_,maps=render(intermediate,params)
    selected=np.asarray(roi,bool)&np.asarray(valid,bool)&np.isfinite(maps['A'])
    values=maps['A'][selected]
    if not values.size:raise ValueError('Empty/invalid observation ROI; draw a valid rectangle or use an original Layer')
    black,white=np.quantile(values,[.01,.99]);degenerate=bool(white-black<=1e-10*max(1.,abs(black),abs(white)))
    if degenerate:
        span=max(1e-6,abs(float(black))*.01)
        black-=span/2;white+=span/2
    params.update(black=float(black),white=float(white),W=float(white-black),gamma=min(3.,max(.2,params['gamma'])),
        auto_window=dict(percentiles=[1,99],finite_valid_pixels=int(values.size),quantiles=[float(np.quantile(values,.01)),float(np.quantile(values,.99))],degenerate=degenerate,
            excludes='source nonfinite/padding and original unknown; never current Vessel',continuous_refit=False,fit_beta=params['beta']))
    return params
