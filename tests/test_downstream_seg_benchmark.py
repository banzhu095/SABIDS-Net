from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from tools.downstream_seg_benchmark.common import KEY, FORMAL, sha, write_csv, write_json, read_json
from tools.downstream_seg_benchmark.data import Inputs
from tools.downstream_seg_benchmark.engine import train, evaluate
from tools.downstream_seg_benchmark.metrics import METRICS, aggregate, classify
from tools.downstream_seg_benchmark.model import Segmenter
from tools.downstream_seg_benchmark.protocol import build_plan, default_config


def fixture_run(tmp_path,formal=False):
    rows=[]
    image=tmp_path/'image.png';layer=tmp_path/'layer.png';vessel=tmp_path/'vessel.png'
    Image.fromarray(np.full((16,16),127,np.uint8)).save(image)
    mask=np.zeros((16,16),np.uint8);mask[5:13]=255;Image.fromarray(mask).save(layer)
    mask[:]=0;mask[7:10,4:12]=255;Image.fromarray(mask).save(vessel)
    for method in FORMAL if formal else ['noisy_identity','nafnet_paired','sabids_current','clean_oracle']:
        for split in ['train','val','test']:
            rows.append(dict(dataset='PKU37',split=split,position_id=split,frame_id=1,sample_id=split+'_1',
                method_id=method,path=str(image),layer_mask_path=str(layer),vessel_mask_path=str(vessel),
                multiclass_label_path='',label_valid_mask_path='',vessel_valid_mask_path='',label_identity='same',
                available=True,height=16,width=16,sha256=sha(image)))
    write_csv(tmp_path/'input_asset_audit.csv',rows)
    write_csv(tmp_path/'missing_or_ambiguous_assets.csv',[],KEY+['method_id','reason'])
    write_json(tmp_path/'label_asset_inventory.json',{str(p):{'sha256':sha(p)} for p in [layer,vessel]})
    config=default_config('formal' if formal else 'pilot');config.update(epochs=1,channels=[4,8],depths=[1,1],decoder_depth=1,
        target_size=16,evaluation_size=16,num_workers=0,batch_size=1,amp=False)
    return rows,config


def test_shared_initialization_plans_no_method_input_and_resume(tmp_path):
    torch.set_num_threads(1)
    rows,config=fixture_run(tmp_path)
    build_plan(tmp_path,config)
    model=Segmenter([4,8],[1,1],1)
    assert not any('denoise' in name or 'interaction' in name for name,_ in model.named_parameters())
    with pytest.raises(TypeError):model(torch.zeros(1,1,16,16),method_id='sabids_current')
    with pytest.raises(ValueError,match='checkpoint lock'):evaluate(tmp_path,'cpu')
    train(tmp_path,'cpu')
    from tools.downstream_seg_benchmark.common import read_json
    metadata=[read_json(p) for p in (tmp_path/'tracks').glob('*/*/run_metadata.json')]
    for key in ['initialization_sha256','data_plan_sha256','sampler_plan_sha256']:
        assert len({m[key] for m in metadata})==1
    evaluate(tmp_path,'cpu');first=sha(tmp_path/'per_frame_metrics.csv')
    evaluate(tmp_path,'cpu',True)
    assert sha(tmp_path/'per_frame_metrics.csv')==first
    with pytest.raises(ValueError,match='sealed'):train(tmp_path,'cpu',True)


def test_missing_and_unpaired_inputs_fail_closed(tmp_path):
    rows,config=fixture_run(tmp_path)
    rows[0]['available']=False;write_csv(tmp_path/'input_asset_audit.csv',rows)
    with pytest.raises(ValueError,match='Missing input'):build_plan(tmp_path,config)
    rows[0]['available']=True;rows[0]['sample_id']='wrong';write_csv(tmp_path/'input_asset_audit.csv',rows)
    with pytest.raises(ValueError,match='keys differ'):build_plan(tmp_path,config)


def test_test_dataset_seal():
    with pytest.raises(ValueError,match='checkpoint lock'):Inputs([{'split':'test'}],256)


def test_position_macro_and_preset_rules():
    frame=pd.DataFrame([dict(method_id=m,segmentation_seed=42,position_id=p,**{k:v for k in METRICS})
                       for m in ['sabids_current','nafnet_paired'] for p,v in [('a',1),('a',1),('b',0)]])
    _,summary,_=aggregate(frame)
    assert summary.layer_dice_mean.tolist()==[.5,.5]
    gains=pd.DataFrame([{k:0. for k in METRICS} for _ in range(3)])
    gains['vessel_roi_dice']=[.01,.01,.002];gains['recall']=.01
    assert classify(gains)=='supportive'
    gains['precision']=-.02
    assert classify(gains)=='not_supportive'
    gains['precision']=0.;gains['vessel_roi_dice']=[.001,-.001,.001]
    assert classify(gains)=='inconclusive'
    gains['vessel_roi_dice']=.002
    assert classify(gains)=='weak_support'


def test_ten_methods_three_seeds_equal_budget_and_checkpoint_conflict(tmp_path):
    torch.set_num_threads(1)
    rows,config=fixture_run(tmp_path,True)
    assert len(rows)==30
    build_plan(tmp_path,config)
    train(tmp_path,'cpu',methods=['noisy_identity'],seeds=[42])
    with pytest.raises(ValueError,match='checkpoint lock'):evaluate(tmp_path,'cpu')
    train(tmp_path,'cpu',True)
    records=[read_json(p) for p in (tmp_path/'tracks').glob('*/*/complete.json')]
    assert len(records)==30 and {r['updates'] for r in records}=={1}
    for seed in [42,123,2026]:
        for key in ['initialization_sha256','data_plan_sha256','sampler_plan_sha256','augmentation_plan_sha256']:
            assert len({r[key] for r in records if r['segmentation_seed']==seed})==1
    evaluate(tmp_path,'cpu');first=sha(tmp_path/'per_frame_metrics.csv')
    evaluate(tmp_path,'cpu',True);assert first==sha(tmp_path/'per_frame_metrics.csv')
    assert read_json(tmp_path/'evaluation_status.json')['status']=='complete_limited_3_position_test'
    assert pd.read_csv(tmp_path/'per_position_seed_metrics.csv').shape[0]==30
    path=tmp_path/'tracks/noisy_identity/42/best.pth'
    with path.open('ab') as stream:stream.write(b'conflict')
    with pytest.raises(ValueError,match='mismatch'):evaluate(tmp_path,'cpu',True)


def test_ignore_padding_no_gradient_and_canonical_subset():
    from tools.downstream_seg_benchmark.model import objective
    layer=torch.zeros(1,1,8,8);layer[:,:,2:6]=1
    vessel=layer.clone();vessel[:,:,:,:4]=0
    assert not bool(((vessel>0)&~(layer>0)).any())
    valid=torch.ones_like(layer);valid[:,:,:,0]=0;valid[:,:,-1]=0
    output={k:torch.randn_like(layer,requires_grad=True) for k in ['layer','vessel']}
    loss=objective(output,layer,vessel,valid,valid);loss.backward()
    for prediction in output.values():
        assert not bool(prediction.grad[:,:,:,0].any())
        assert not bool(prediction.grad[:,:,-1].any())


def test_atlas_registration_only_noisy_gt(tmp_path):
    from tools.downstream_seg_benchmark.selection import fixed_samples
    rows,_=fixture_run(tmp_path)
    selected=fixed_samples(pd.DataFrame(rows))
    assert {r['role'] for r in selected}=={'middle','small_vessel_rich','weak_boundary'}
    assert not list(tmp_path.glob('*metrics.csv'))
    other=pd.DataFrame(rows);other.loc[other.method_id!='noisy_identity','path']='unreadable_other_method'
    assert fixed_samples(other)==selected


def test_seed_position_macro_and_descriptive_bootstrap(tmp_path):
    from tools.downstream_seg_benchmark.metrics import formal_tables
    rows=[]
    for method,offset in [('sabids_current',.1),('nafnet_paired',0.)]:
        for seed in [42,123,2026]:
            for p,values in [('a',[1.,1.]),('b',[0.])]:
                for v in values:rows.append(dict(method_id=method,segmentation_seed=seed,position_id=p,**{k:v+offset for k in METRICS}))
    formal_tables(pd.DataFrame(rows),tmp_path)
    position=pd.read_csv(tmp_path/'per_position_metrics.csv')
    assert len(position)==4
    ci=pd.read_csv(tmp_path/'bootstrap_confidence_intervals.csv')
    assert np.allclose(ci['mean'],.1) and np.allclose(ci.ci_low,.1) and np.allclose(ci.ci_high,.1)


def test_incremental_changes_only_new_test_positions(tmp_path):
    from tools.downstream_seg_benchmark.incremental import select_new_test
    rows,_=fixture_run(tmp_path,True);old=pd.DataFrame(rows)
    added=old[old.split=='test'].copy();added['sample_id']='new_1';added['position_id']='new'
    updated=pd.concat([old,added],ignore_index=True)
    assert len(select_new_test(old,updated))==10
    updated.loc[updated.split=='train','sha256']='changed'
    with pytest.raises(ValueError,match='training/validation'):select_new_test(old,updated)


def test_atomic_stage_resume_checks_output_hash(tmp_path):
    from tools.downstream_seg_benchmark.runner import stage
    output=tmp_path/'result.csv';calls=[]
    def action():
        calls.append(1);write_csv(output,[{'value':1}]);return [output]
    stage(tmp_path,'fixture','same',action)
    stage(tmp_path,'fixture','same',action,True);assert len(calls)==1
    write_csv(output,[{'value':2}])
    with pytest.raises(ValueError,match='output changed'):stage(tmp_path,'fixture','same',action,True)
