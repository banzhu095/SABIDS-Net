"""Numerical/gradient/provenance tests; synthetic is never scientific evidence."""
import copy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from sabids.experiments.attenuation_aux import StableACConfig,stable_ac_v1
from sabids.models.learnable_ac import NativeStableAC,AdaptiveSegmenter,ACController
from tools.learnable_ac_v1.core import config,upstream_audit,read,write
from tools.learnable_ac_v1.learning import build,objective,model_hash,strict_better,average_precision


@pytest.fixture(autouse=True)
def one_thread():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.mark.parametrize('shape',[(31,47),(48,16),(1,9),(13,1)])
def test_native_r2_exact_and_single_mix(shape):
    x=np.random.default_rng(17).uniform(0,1,shape).astype(np.float32)
    cfg=StableACConfig(output_window=31.04,denominator_floor=.0857)
    valid=np.ones(shape,bool)
    if shape[0]>8:valid[-3:]=False;valid[3:5,0]=False
    z,maps,_=stable_ac_v1(x,cfg,valid=valid)
    actual=NativeStableAC(cfg)(torch.from_numpy(x)[None,None],torch.from_numpy(valid)[None,None])
    assert np.max(np.abs(actual['z0'][0,0].numpy()-z))<=1e-7
    assert np.max(np.abs(actual['energy'][0,0].numpy()-maps['denominator']))<1e-5
    torch.testing.assert_close(actual['z0'],.5*actual['m']+.5*actual['c'],atol=1e-7,rtol=0)
    assert np.array_equal(actual['z0'][0,0].numpy()[~valid],x[~valid])


@pytest.mark.parametrize('level',[0.,.001,.25,1.])
def test_dark_constant_last_row_finite(level):
    ac=NativeStableAC(StableACConfig(output_window=.01))
    a=ac(torch.full((1,1,17,23),level))
    assert all(torch.isfinite(v).all() for v in a.values())
    assert a['energy'][0,0,-1].eq(0).all()
    assert a['rho'][0,0,-1].eq(0).all()
    assert a['z0'].min()>=0 and a['z0'].max()<=1


def analytic_pack():
    x=torch.rand(1,1,32,32)*.7+.05
    a=NativeStableAC(StableACConfig())(x)
    return torch.cat([a[k] for k in ('m','z0','c','depth','reliability','gain','rho','valid')],1)


@pytest.mark.parametrize('method',['F0','F1','F2','F3','F4','F5','F6'])
def test_zero_and_off_anchor_equivalence(method):
    cfg=config(True);base=build(cfg,'P3',42);m=build(cfg,method,42,base.state_dict())
    x=analytic_pack()
    with torch.no_grad():
        expected=base.eval()(x[:,1:2]);out=m(x);off=m(x,adaptive_off=True)
    for task in ('layer','vessel'):
        torch.testing.assert_close(out[task],expected[task],rtol=0,atol=0)
        assert torch.equal(off[task],expected[task])


def test_controller_input_gradient_and_frozen_state():
    cfg=config(True);base=build(cfg,'P3',42);m=build(cfg,'F5',42,base.state_dict()).train()
    x=analytic_pack();frozen=model_hash(m.anchor);opt=torch.optim.AdamW(m.control.parameters(),lr=.001)
    grads=[]
    for step in range(3):
        opt.zero_grad();out=m(x)
        # Nonzero segmentation gradient through frozen input, not a regularizer shortcut.
        loss=out['layer'].square().mean()+out['vessel'].square().mean();loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.control.head.parameters())
        grads.append(sum(float(p.grad.abs().sum()) for p in m.control.context.parameters() if p.grad is not None))
        assert all(p.grad is None for p in m.anchor.parameters());opt.step()
    assert grads[0]==0 and grads[-1]>0
    assert model_hash(m.anchor)==frozen and not m.anchor.training
    assert m.last_diagnostics['effective_beta'].min()>=.25
    assert m.last_diagnostics['effective_beta'].max()<=.75


def test_refiner_capacity_context_and_guidance():
    f1,f5=[ACController(k,channels=(4,8,16)) for k in ('F1','F5')]
    assert sum(p.numel() for p in f1.parameters())==sum(p.numel() for p in f5.parameters())
    cfg=config(True);base=build(cfg,'P3',42)
    a,b=[build(cfg,k,42,base.state_dict()) for k in ('F5','F6')]
    # Common names have exactly the same seeded initial values.
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
    assert list(inspect.signature(a.forward).parameters)==['packed','adaptive_off']


def test_global_beta_really_global_not_spatial_reliability():
    ctrl=ACController('F2',channels=(4,8,16))
    with torch.no_grad():ctrl.global_raw.copy_(torch.tensor([.4,-.3]))
    result=ctrl(torch.rand(1,10,12,15),torch.rand(1,1,12,15),torch.rand(1,1,12,15))
    for task in range(2):assert torch.unique(result['effective_beta'][0,task]).numel()==1


def test_validity_unknown_uncertain_excluded_from_loss():
    from tools.denoise_then_ac_v1.learning import objective
    torch.manual_seed(1)
    output={k:torch.randn(1,1,8,8,requires_grad=True) for k in ('layer','vessel')}
    layer=torch.ones(1,1,8,8);vessel=layer.clone();valid=layer.clone();valid[:,:,:3]=0
    a=objective(output,layer,vessel,valid,valid)
    other={k:v.clone() for k,v in output.items()}
    for v in other.values():v[:,:,:3]=100
    b=objective(other,layer,vessel,valid,valid)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.backward()
    assert all(v.grad[:,:,:3].eq(0).all() for v in output.values())


def test_outer_exposure_fail_closed(tmp_path):
    split=dict(train=['a'],inner=['b'],outer=['c'])
    a=upstream_audit(tmp_path,split,{},'formal',True)
    assert not a['formal_allowed'] and len(a['problems'])==3
    assert upstream_audit(tmp_path,split,{},'development',True)['development_exploratory']
    # Independent noisy-only AC does not need D2 or teacher.
    assert upstream_audit(tmp_path,split,{},'formal',False)['formal_allowed']


def test_actual_manifest_exposure_is_not_self_declared_safe(tmp_path):
    from sabids.config import load_config
    from tools.learnable_ac_v1.core import sha
    manifest=tmp_path/'manifest.csv';manifest.write_text('group_id,split\nc,train\nb,val\n',encoding='utf8')
    cfg_file=tmp_path/'cfg.yaml'
    cfg_file.write_text('data:\n  manifest: '+manifest.as_posix()+'\n  train_split: train\n  val_split: val\nruntime:\n  manifest_sha256: '+sha(manifest)+'\n',encoding='utf8')
    cfg=load_config(cfg_file);cp=tmp_path/'cp.pth';torch.save(dict(config=cfg,epoch=2,model={'test':torch.ones(1)}),cp)
    item=dict(checkpoint=str(cp),sha256=sha(cp),resolved_config=str(cfg_file),config_sha256=sha(cfg_file),safe=True)
    result=upstream_audit(tmp_path,dict(train=['a'],inner=['b'],outer=['c']),{'D1':item},'formal',True)
    assert not result['formal_allowed']
    assert 'D1: outer exposure' in result['problems']


def test_tie_epoch0_and_auprc():
    assert not strict_better(.5,.5) and strict_better(.50001,.5)
    assert not strict_better(float('nan'),.5)
    assert average_precision(np.array([.9,.7,.4]),np.array([1,1,0],bool))==1
    assert average_precision(np.ones(4),np.array([1,1,0,0],bool))==.5


def test_weights_only_not_full_resume():
    from tools.learnable_ac_v1.learning import validate_state
    with pytest.raises(ValueError,match='Incomplete checkpoint'):
        validate_state({'model':{}},[],20)


def test_prepare_can_compute_native_from_an_audited_d2_without_old_fold0_cache(tmp_path,monkeypatch):
    # Computational branch test only. Provenance is explicitly mocked on a
    # generated project, not declared to be an actual clinical D2 audit.
    from tools.ac_vessel_labelsv2.smoke import synthetic_project
    from tools.ac_vessel_labelsv2.snapshot import freeze
    from tools.learnable_ac_v1 import core
    from tools.denoise_then_ac_v1 import core as inherited
    from tools.denoise_then_ac_v1.smoke import SyntheticRestorer
    out,root,workspace,evidence=synthetic_project(tmp_path/'generated')
    snapshot=out/'snapshot';freeze(root,workspace,root/'Data/PKU37_OCT_Denoising/noisy',snapshot,evidence)
    split=core.read(snapshot/'folds.json')[0]
    monkeypatch.setattr(core,'audit',lambda *a,**k:dict(status='passed',problems=[],split=split,
        snapshot_identity=core.verify_lock(snapshot)['identity'],allowed_base=list(core.BASE),historical_cache_identity=None,
        upstream={'roles':{'D2':{'binding':{'synthetic_test':True}}}}))
    monkeypatch.setattr(inherited,'load_d2',lambda identity,device:SyntheticRestorer().eval().requires_grad_(False))
    meta=core.prepare(root,snapshot,out/'cache',device='cpu')
    assert meta['frames']==16 and meta['source_cache'] is None
    assert core.read(out/'cache/frozen_d2.json')['requires_grad_parameters']==0
    row=core.read(out/'cache/frames.json')[0]
    with np.load(out/'cache'/row['analytic']) as a:
        assert np.max(np.abs(a['m']-a['noisy']))>0
        assert a['m'].shape==a['z0'].shape==(32,32)


def test_paired_cluster_units_and_error_sign():
    import pandas as pd
    from tools.learnable_ac_v1.reporting import paired_statistics
    rows=[]
    for seed in (42,43,44):
        for i in range(3):
            for method,score in [('P3',.5),('F1',.5),('F5',.6)]:
                rows.append(dict(method=method,seed=seed,fold=0,position_id=str(i),group_id=str(i),vessel_dice=score,upper_boundary_mae=2-score))
    gains,ci=paired_statistics(pd.DataFrame(rows))
    assert np.allclose(gains[gains.metric=='vessel_dice'].gain,10)
    assert np.allclose(gains[gains.metric=='upper_boundary_mae'].gain,.1)
    assert (ci.clusters==3).all() # Not nine seeds/positions, not repeated frames.
    assert ci[ci.metric=='vessel_dice'].p_holm_primary.notna().all()


def test_complete_cpu_closure(tmp_path):
    from tools.learnable_ac_v1.checks import cpu_smoke
    result=cpu_smoke(tmp_path/'smoke')
    assert result['status']=='passed' and result['epoch_resume_bit_exact']
    status=read(tmp_path/'smoke/report/report_status.json')
    assert status['synthetic'] and not status['full_cv_complete']
    a=read(tmp_path/'smoke/two_updates/check.json')
    assert a['updates']==2 and a['resume_bit_exact'] and a['anchor_unchanged']
    assert set(a['methods'])=={'F1','F5'} and not a['compute_matched']
    assert all(v['steps'][1]['context_gradient_norm']>0 for v in a['methods'].values())
    from tools.learnable_ac_v1.native_reference import compare
    comparison=compare(tmp_path/'smoke/cache',tmp_path/'smoke/snapshot',tmp_path/'native_reference')
    assert comparison['status']=='passed' and comparison['positions']==8
    from tools.dispatch_learnable_ac_v1 import validate_completed,CORE
    assert CORE==('P3','F0','F1','F2','F5')
    binding=validate_completed(tmp_path/'smoke/training',tmp_path/'smoke/cache','F5',config(True))
    assert binding['synthetic'] and binding['completed_epochs']==2
    for p in (tmp_path/'smoke/atlas').glob('outer/*/F5/vessel_probability.npy'):
        prob=np.load(p,allow_pickle=False);mask=np.load(p.with_name('vessel_mask.npy'),allow_pickle=False)
        assert np.array_equal(mask,prob>=.5)
        sid=read(p.parent.parent/'identity.json')['sample_id']
        with np.load(tmp_path/f'smoke/training/F5/outer_best/predictions/{sid}.npz') as pred:
            assert np.array_equal(prob,pred['vessel'])
    from tools.learnable_ac_v1.learning import train
    cfg=config(True);cfg['learning_rate']*=2
    with pytest.raises(ValueError,match='Study-level'):
        train(tmp_path/'smoke/snapshot',tmp_path/'smoke/cache',tmp_path/'smoke/training',['P0'],cfg,resume=True)
