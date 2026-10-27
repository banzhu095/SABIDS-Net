from pathlib import Path
from tools.prepare_learnable_ac_upstream import bind_fold_config


def test_upstream_config_never_retains_legacy_manifest_or_initialization():
    cfg=dict(data=dict(manifest='OLD/train.csv'),train=dict(pretrained='OLD/weights.pth'),model={},runtime={'old':True})
    lock=dict(protocol_id='new_fold',manifest_root='new',train_positions=['a'],validation_positions=['b'],
        data_plan_sha256='data',label_inventory_sha256='label',dataset_inventory_sha256='inventory',split_contract_sha256='split')
    new=bind_fold_config(cfg,Path('/project'),lock,Path('/new/train.csv'),Path('/new/run'),'denoise')
    assert Path(new['data']['manifest'])==Path('/new/train.csv')
    assert new['train']['pretrained'] is None and 'runtime' not in new
    assert new['data']['train_groups']==['a'] and new['data']['val_groups']==['b']
    assert new['train']['batch_size']==1 and new['train']['gradient_accumulation_steps']==2
    assert cfg['data']['manifest']=='OLD/train.csv' # no in-place alteration
