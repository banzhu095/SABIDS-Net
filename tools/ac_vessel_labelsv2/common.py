import hashlib
import io
import json
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
ARMS=('N0','T0','A0','NN','NA')
MANUAL=('M0','NM')
VERSION='ac_vessel_labelsv2.v1'


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    import uuid
    temp=path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
    temp.write_bytes(data);os.replace(temp,path)


def write(path,value):atomic(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf8'))


def table(path,rows):
    rows=list(rows);require(bool(rows),'Cannot write empty table: '+str(path))
    stream=io.StringIO();keys=list(dict.fromkeys(k for r in rows for k in r))
    w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)
    atomic(path,stream.getvalue().encode('utf-8-sig'))


def png(path,array):
    stream=io.BytesIO();Image.fromarray(np.asarray(array,dtype=np.uint8)).save(stream,format='PNG');atomic(path,stream.getvalue())


def plane(path):
    with Image.open(path) as im:return np.asarray(im).copy()


def noisy(path):
    from tools.denoise_result_review.image_io import decode_lossless
    a=decode_lossless(path)
    require(a.dtype in (np.dtype('uint8'),np.dtype('uint16')),'No inferred float range: '+str(path))
    return a.astype(np.float64)/np.iinfo(a.dtype).max


def npy(path,array):
    s=io.BytesIO();np.save(s,np.asarray(array),allow_pickle=False);atomic(path,s.getvalue())


def fresh(path):
    path=Path(path).resolve();require(not path.exists(),'Existing artifacts preserved; choose new directory: '+str(path));path.mkdir(parents=True);return path


def verify_lock(folder,include_manual=False):
    folder=Path(folder);lock=read(folder/'snapshot.json')
    unsigned=dict(lock);claimed=unsigned.pop('identity');require(digest(unsigned)==claimed,'Snapshot identity altered')
    for r in lock['files']:
        if not include_manual and (r['path'].startswith(('manual/','revisions/')) or r['path']=='manual_algorithm.json'):continue
        require(sha(folder/r['path'])==r['sha256'],'Frozen snapshot changed: '+r['path'])
    return lock


def source_identity():
    files=list((ROOT/'tools/ac_vessel_labelsv2').glob('*.py'))+[ROOT/'configs/ac_vessel_labelsv2/protocol.yaml',ROOT/'tools/run_ac_vessel_labelsv2.py',ROOT/'tools/downstream_seg_benchmark/model.py',ROOT/'tools/downstream_seg_benchmark/engine.py',ROOT/'tools/downstream_seg_benchmark/metrics.py',ROOT/'sabids/models/blocks.py',ROOT/'sabids/losses/common.py',ROOT/'sabids/experiments/attenuation_aux.py',ROOT/'tools/pku37_label_review/display.py',ROOT/'tools/denoise_result_review/image_io.py']
    import inspect
    from sabids.models.dual_view_segmenter import DualViewFusion
    result={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result['symbol:DualViewFusion']=hashlib.sha256(inspect.getsource(DualViewFusion).encode()).hexdigest()
    return result


def config(smoke=False,pilot=False):
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/ac_vessel_labelsv2/protocol.yaml').read_text(encoding='utf8'))
    require(cfg['version']==VERSION and cfg['amp'] is False and cfg['batch_size']==1 and cfg['accumulation']==2 and cfg['clip']==1.,'Unsupported common optimizer contract')
    cfg.update(smoke=smoke,pilot=pilot)
    if smoke:cfg.update(channels=[4,8,16],depths=[1,1,1],decoder_depth=1,size=32,epochs=1)
    elif pilot:cfg['epochs']=2
    return cfg
