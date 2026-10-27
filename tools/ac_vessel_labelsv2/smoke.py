from dataclasses import asdict
import io
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from sabids.experiments.attenuation_aux import StableACConfig,CACHE_VERSION
from tools.pku37_label_review.workspace import initialize
from .common import *
from .snapshot import freeze,registration_audit,approve
from .processing import prepare
from .learning import train,evaluate
from .report import summarize,package


def synthetic_project(output):
    out=fresh(output);project=out/'project';atlas=project/'atlas';atlas.mkdir(parents=True)
    (project/'Manifests').mkdir();rows=[]
    for i in range(1,9):
        rng=np.random.default_rng(i);x=cv2.GaussianBlur(rng.uniform(40,200,(32,32)).astype(np.float32),(3,3),0)
        y,xx=np.mgrid[:32,:32];gt=np.zeros((32,32),np.uint8);gt[7:29]=1;v=((y-17)**2+(xx-12)**2<=16)|((y-22)**2+(xx-24)**2<=9);gt[v]=2
        gt[:2]=255;x[v]*=.3
        ref=project/'Data/PKU37_OCT_Denoising/clean'/f'{i:04d}.png';png(ref,np.rint(x));lab=project/'Label'/f'{i:04d}.png';png(lab,gt)
        for f in (1,2):png(project/'Data/PKU37_OCT_Denoising/noisy'/f'{i:04d}{f:02d}.png',np.clip(np.rint(x+rng.normal(0,.5,x.shape)),0,255))
        rows.append(dict(position_id=f'pku_{i:04d}',position=f'{i:04d}',split='synthetic_development',reference_path=str(ref),representative_noisy_path='',repeat_frame_count=2,label_reference_path='',multiclass_label_path=str(lab),layer_mask_path='',vessel_mask_path='',annotation_validity='synthetic',label_coordinate_claim='known synthetic identity'))
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);atomic(atlas/'selection_manifest.csv',stream.getvalue().encode())
    atomic(atlas/'resolved_ac_config.yaml',yaml.safe_dump(dict(version=CACHE_VERSION,stable_ac_v1=asdict(StableACConfig(output_window=25.)))).encode())
    write(atlas/'source_file_inventory.json',dict(synthetic=True));write(atlas/'asset_inventory.json',dict(synthetic=True,test_positions=[]))
    workspace=out/'workspace';review=initialize(project,atlas,workspace,expected_positions=8)
    for pid in review.index['positions']:
        _,state,m=review.load(pid);m['vessel'][3,4]=True
        review.save(pid,0,m,review.default,'confirmed',notes='SYNTHETIC engineering test; not human review')
    evidence=out/'synthetic_service_evidence.json';write(evidence,dict(workspace=str(workspace),index_sha256=sha(workspace/'index.json'),basis='synthetic generator, no real active service'))
    return out,project,workspace,evidence


def run(output):
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        out,project,workspace,evidence=synthetic_project(output);snapshot=out/'snapshot'
        freeze(project,workspace,project/'Data/PKU37_OCT_Denoising/noisy',snapshot,evidence)
        registration_audit(project,snapshot,out/'registration');approval=out/'approval.json'
        approve(snapshot,out/'registration',approval,'synthetic generator','Known synthetic identity transforms under generator; NOT evidence for real data')
        for fold in range(4):
            cache=out/'cache'/f'fold{fold}';prepare(project,snapshot,fold,cache,approval,manual=True)
            cfg=config(smoke=True);train(snapshot,cache,out/'experiment'/f'fold{fold}',ARMS,42,cfg)
            evaluate(snapshot,cache,out/'experiment'/f'fold{fold}',ARMS,42)
        result=summarize(snapshot,out/'cache',out/'experiment',out/'report')
        package(out/'report',out/'gpt_light_smoke.zip')
        # Exercise optional human-assisted paths separately, never mix their report
        # with the five-arm automatic-input primary report.
        for fold in range(4):
            cache=out/'cache'/f'fold{fold}'
            train(snapshot,cache,out/'experiment'/f'fold{fold}',MANUAL,42,config(smoke=True))
            evaluate(snapshot,cache,out/'experiment'/f'fold{fold}',MANUAL,42)
        manual_result=summarize(snapshot,out/'cache',out/'experiment',out/'manual_report',manual=True)
        require(manual_result['status']=='passed','Optional manual smoke report failed')
        result.update(scope='SYNTHETIC CPU engineering smoke; NOT real scientific AC result',output=str(out));write(out/'smoke_result.json',result);return result
    finally:torch.set_num_threads(old_threads)
