import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tools.downstream_seg_benchmark.metrics import METRICS
from tools.downstream_seg_review.tables import regroup,paired,decompose,compare_table
from tools.downstream_seg_review.images import probability_assets,png,basic_metrics
from tools.downstream_seg_review.reporting import zip_package


def test_reaggregation_is_position_macro_then_seed_not_frame_weighted():
    rows=[]
    for seed in [42,123,2026]:
        for pos,vals in [('a',[1.,1.,1.]),('b',[0.])]:
            for j,value in enumerate(vals):rows.append(dict(method_id='m',segmentation_seed=seed,position_id=pos,sample_id=pos+str(j),**{k:value for k in METRICS}))
    ps,pos,seeds,main=regroup(pd.DataFrame(rows))
    assert main.vessel_dice_mean.iloc[0]==.5 and main.vessel_dice_std.iloc[0]==0
    assert len(ps)==6 and len(seeds)==3 and len(pos)==2


def test_duplicate_frame_keys_rejected():
    row=dict(method_id='m',segmentation_seed=42,position_id='p',sample_id='x',**{k:0 for k in METRICS})
    with pytest.raises(ValueError,match='Duplicate'):regroup(pd.DataFrame([row,row]))


def test_decomposition_identity_signs():
    gains=pd.DataFrame([dict(comparison='sabids_current - nafnet_paired',vessel_dice=.0083,vessel_roi_dice=-.004)])
    d=decompose(gains).iloc[0]
    assert np.isclose(d.outside_penalty_reduction,.0123) and abs(d.identity_error)<1e-15


def test_masks_are_raw_threshold_ignore_only_affects_error_display(tmp_path):
    lp=np.full((8,8),.49,np.float32);vp=np.zeros_like(lp);vp[1,1]=.5;vp[2,2]=.9
    gl=np.zeros_like(lp,dtype=bool);gl[1,1]=True;gv=gl.copy()
    valid=np.ones_like(gl);valid[2,2]=False
    a=probability_assets(lp,vp,gl,gv,valid,valid,np.ones_like(lp)*.2)
    assert a['vessel_mask.png'][1,1]==255 and a['vessel_mask.png'][2,2]==255
    assert tuple(a['vessel_tp_fp_fn.png'][1,1])==(0,255,0)
    assert tuple(a['vessel_tp_fp_fn.png'][2,2])==(128,128,128)
    assert a['vessel_probability_u16.png'].dtype==np.uint16
    row=png(tmp_path,tmp_path/'prob.png',a['vessel_probability_u16.png'],'m',42,'p','s')
    assert row['sha256']==hashlib.sha256((tmp_path/'prob.png').read_bytes()).hexdigest()
    decoded=np.asarray(Image.open(tmp_path/'prob.png'))/65535
    assert np.abs(decoded-vp).max()<=.5/65535+1e-7


def test_native_shapes_probabilities_fail_closed():
    x=np.zeros((8,8));mask=x.astype(bool)
    with pytest.raises(ValueError,match='Geometry'):probability_assets(x[:2],x,mask,mask,~mask,~mask,x)
    x[0,0]=np.nan
    with pytest.raises(ValueError,match='probability'):probability_assets(x,x,mask,mask,~mask,~mask,np.zeros((8,8)))


def test_summary_nan_preserved_and_difference_detected():
    a=pd.DataFrame([dict(method_id='x',small_vessel_recall=np.nan,vessel_dice=.5)])
    assert all(x['mismatches']==0 for x in compare_table(a,a,['method_id'],'test'))
    b=a.copy();b.vessel_dice=.6
    assert sum(x['mismatches'] for x in compare_table(a,b,['method_id'],'test'))==1


def test_zip_has_member_hash_size_manifest_and_crc(tmp_path):
    p=tmp_path/'table.csv';p.write_text('x,y\n1,2\n')
    result=zip_package(tmp_path,'fixture.zip',[p])
    assert result['crc_passed'] and result['member_sha256_verified']
    with zipfile.ZipFile(tmp_path/'fixture.zip') as z:
        assert z.testzip() is None
        row=pd.read_csv(io.BytesIO(z.read('PACKAGE_MANIFEST.csv'))).iloc[0]
        assert row.sha256==hashlib.sha256(z.read('table.csv')).hexdigest()
    with pytest.raises(ValueError,match='Package exists'):zip_package(tmp_path,'fixture.zip',[p])
