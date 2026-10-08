"""Real train-only engineering check. Never generates pilot performance claims."""
from pathlib import Path
import time

import pandas as pd
import torch

from .common import require, sha, write_json
from .data import Inputs
from .model import Segmenter, objective


def real_train_smoke(audit_run, output):
    output=Path(output)
    require(not output.exists(),'Smoke output already exists')
    output.mkdir(parents=True)
    data=pd.read_csv(Path(audit_run)/'input_asset_audit.csv',keep_default_na=False)
    rows=data[(data.split=='train') & (data.method_id=='sabids_current') & (data.available==True)].sort_values('sample_id').head(2).to_dict('records')
    require(len(rows)==2,'Two real train images required')
    torch.set_num_threads(4);torch.manual_seed(42)
    model=Segmenter();optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5)
    dataset=Inputs(rows,64)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    losses=[];start=time.monotonic()
    for index in range(2):
        x,l,v,valid,vv,*_=dataset[index]
        optimizer.zero_grad(set_to_none=True)
        loss=objective(model(x[None]),l[None],v[None],valid[None],vv[None])
        require(bool(torch.isfinite(loss)),'Nonfinite smoke loss')
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    require(any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()),'No parameter update')
    checkpoint=output/'smoke.pth';torch.save(model.state_dict(),checkpoint)
    restored=Segmenter();restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False))
    model.eval();restored.eval()
    with torch.no_grad():
        a,b=model(x[None]),restored(x[None])
    require(all(torch.equal(a[k],b[k]) for k in a),'Checkpoint restore changed predictions')
    result=dict(status='passed_engineering_only',dataset='real_PKU37_train',samples=[r['sample_id'] for r in rows],
                updates=2,size=64,device='cpu',losses=losses,seconds=time.monotonic()-start,
                checkpoint_sha256=sha(checkpoint),restore_verified=True,test_opened=False,pilot_result=False)
    write_json(output/'smoke_result.json',result)
    return result
