from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .common import FORMAL, KEY, PILOT, digest, read_json, require, sha, source_hash, write_csv, write_json
from .model import Segmenter


def default_config(mode='pilot'):
    return dict(version='downstream-denoised-only-v1', mode=mode,
                methods=PILOT.copy() if mode == 'pilot' else FORMAL.copy(),
                segmentation_seeds=[42] if mode == 'pilot' else [42, 123, 2026],
                denoiser_primary_seed=42, epochs=15 if mode == 'pilot' else 60,
                target_size=256 if mode == 'pilot' else 384, evaluation_size=256 if mode == 'pilot' else 640,
                batch_size=4, learning_rate=5e-5, amp=True, num_workers=4,
                gradient_accumulation=1,weight_decay=.0001,scheduler='cosine_epoch',validation_every=5,
                layer_patch_fraction=.75,component_small_area=64,component_medium_area=256,
                max_train_frames_per_position=0, threshold=.5, wall_budget_seconds=14400,
                channels=[32,64,128,256], depths=[2,2,4,6], decoder_depth=2,
                loss='layer masked BCE+Dice; vessel inside-layer masked BCE+Dice; outside negative BCE 0.5',
                selection='max validation position-macro vessel ROI Dice; tie max layer Dice; earliest tie',
                degradation='none', criterion_version='v1: harmful precision<-0.01 or outside-FP>0.01; negative, supportive, mixed-inconclusive, weak precedence')


def build_plan(run, config, resume=False):
    run = Path(run)
    path = run / 'plan_lock.json'
    if path.exists():
        require(resume, 'Plan exists; use --resume')
        lock = read_json(path)
        require(lock['config'] == config and lock['source_sha256'] == source_hash(), 'Plan config/source changed')
        verify_plan(run)
        return lock
    data = pd.read_csv(run / 'input_asset_audit.csv', keep_default_na=False)
    failures = pd.read_csv(run/'missing_or_ambiguous_assets.csv', keep_default_na=False)
    require(failures[failures.method_id.isin(config['methods']+['*'])].empty, 'Asset/label audit failed for selected methods')
    selected = data[data.method_id.isin(config['methods'])]
    require(set(selected.method_id) == set(config['methods']), 'Missing method')
    require(selected.available.astype(str).str.lower().eq('true').all(), 'Missing input assets: training fails closed')
    require((selected.height == selected.width).all(), 'Non-square geometry requires explicit aspect-preserving protocol')
    canonical = selected[selected.method_id == config['methods'][0]].sort_values(KEY)
    for method in config['methods']:
        arm = selected[selected.method_id == method].sort_values(KEY)
        require(arm[KEY].reset_index(drop=True).equals(canonical[KEY].reset_index(drop=True)), 'Method sample keys differ')
        require(arm.label_identity.tolist() == canonical.label_identity.tolist(), 'Method labels/validity differ')
    require(canonical[canonical.split == 'test'].position_id.nunique() >= 1,'No labelled test position')
    train = canonical[canonical.split == 'train'].sort_values('sample_id')
    if config['max_train_frames_per_position']:
        pieces = []
        for _, group in train.groupby('position_id'):
            group = group.sort_values('frame_id')
            ids = np.linspace(0, len(group)-1, min(len(group), config['max_train_frames_per_position']), dtype=int)
            pieces.append(group.iloc[ids])
        train = pd.concat(pieces).sort_values('sample_id')
    require(len(train) and (canonical.split == 'val').any() and (canonical.split == 'test').any(), 'Empty split')
    from .selection import fixed_samples
    fixed = fixed_samples(selected)
    write_csv(run / 'fixed_atlas_samples.csv', fixed)
    write_csv(run / 'training_samples.csv', train[KEY])
    write_csv(run / 'cohort.csv', selected)
    seeds = {}
    for seed in config['segmentation_seeds']:
        folder = run / 'plans' / str(seed)
        folder.mkdir(parents=True, exist_ok=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = Segmenter(config['channels'], config['depths'], config['decoder_depth'])
        torch.save(model.state_dict(), folder / 'initialization.pth')
        shared=run/'shared_initializations';shared.mkdir(exist_ok=True)
        import shutil
        shutil.copyfile(folder/'initialization.pth',shared/f'seed_{seed}.pth')
        rng = np.random.default_rng(seed)
        epochs, sampler = [], []
        from PIL import Image
        layer_points={row.sample_id:np.nonzero(np.asarray(Image.open(row.layer_mask_path))>0) for _,row in train.iterrows()}
        for epoch in range(config['epochs']):
            order = rng.permutation(len(train)).tolist()
            sampler.append(order)
            plan = []
            for index in order:
                row = train.iloc[index]
                crop = config['target_size'] if config['mode'] == 'formal' else 0
                require(not crop or (row.height >= crop and row.width >= crop), 'Formal crop exceeds source geometry')
                x=int(rng.integers(0,int(row.width)-crop+1)) if crop else 0
                y=int(rng.integers(0,int(row.height)-crop+1)) if crop else 0
                layer_patch=bool(crop and (len(plan)%4!=3))
                if layer_patch:
                    yy,xx=layer_points[row.sample_id]
                    require(len(yy)>0,'Layer-centred patch requires nonempty train layer')
                    point=int(rng.integers(len(yy)))
                    x=int(np.clip(xx[point]-crop//2,0,int(row.width)-crop))
                    y=int(np.clip(yy[point]-crop//2,0,int(row.height)-crop))
                plan.append(dict(index=index, sample_id=row.sample_id, crop_size=crop,
                                 x=x,y=y,layer_patch=layer_patch,
                                 flip_x=bool(rng.integers(2)), flip_y=False))
            epochs.append(plan)
        write_json(folder / 'data_plan.json', epochs)
        write_json(folder / 'sampler_plan.json', sampler)
        augmentation=[[{k:v for k,v in item.items() if k in ['sample_id','flip_x','flip_y']} for item in e] for e in epochs]
        write_json(folder/'augmentation_plan.json',augmentation)
        plan_dir=run/'data_plans';plan_dir.mkdir(exist_ok=True)
        write_csv(plan_dir/f'seed_{seed}.csv',[dict(epoch=i+1,order=j,**item) for i,e in enumerate(epochs) for j,item in enumerate(e)])
        torch.save(torch.Generator().manual_seed(seed).get_state(), folder / 'loader_generator.pth')
        seeds[str(seed)] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
    lock = dict(config=config, source_sha256=source_hash(), seeds=seeds,
                updates_per_method=config['epochs'] * math.ceil(math.ceil(len(train)/config['batch_size'])/config['gradient_accumulation']),
                assets_sha256=sha(run / 'input_asset_audit.csv'), cohort_sha256=sha(run / 'cohort.csv'),
                training_samples_sha256=sha(run / 'training_samples.csv'), fixed_atlas_sha256=sha(run / 'fixed_atlas_samples.csv'))
    (run / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    write_json(path, lock)
    write_json(run/'formal_config_lock.json',dict(status='training_protocol_locked_test_closed',plan_sha256=sha(path),threshold=.5,
        test_scope='limited_3_position_test' if canonical[canonical.split=='test'].position_id.nunique()<6 else 'six_position_test'))
    return lock


def verify_plan(run):
    run = Path(run)
    lock = read_json(run / 'plan_lock.json')
    require(lock['source_sha256'] == source_hash(), 'Runtime source differs from sealed plan')
    _verify_sealed_assets(run, lock)
    return lock


def _verify_sealed_assets(run, lock):
    for key, file in [('cohort_sha256', 'cohort.csv'), ('training_samples_sha256', 'training_samples.csv'),
                      ('fixed_atlas_sha256', 'fixed_atlas_samples.csv'), ('assets_sha256', 'input_asset_audit.csv')]:
        require(sha(run / file) == lock[key], 'Sealed file changed: ' + file)
    for seed, files in lock['seeds'].items():
        for name, expected in files.items():
            require(sha(run / 'plans' / seed / name) == expected, 'Shared randomness artifact changed')


def recover_unstarted_plan(source, destination):
    """Copy immutable randomness/assets into a new seal; never migrate weights."""
    import shutil
    from datetime import datetime, timezone

    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(not destination.exists(), 'Recovery destination already exists')
    old = read_json(source/'plan_lock.json')
    _verify_sealed_assets(source, old)
    require(not any((source/name).exists() for name in
                    ['test_opened.json', 'checkpoint_lock.json', 'checkpoint_registry.csv', 'per_frame_metrics.csv']),
            'Recovery forbidden after checkpoint seal/test access')
    require(not list((source/'tracks').glob('*/*/*.pth')) and
            not list((source/'tracks').glob('*/*/complete.json')) and
            not list((source/'tracks').glob('*/*/training_curve.csv')),
            'Recovery forbidden after any saved training checkpoint/history')
    status = read_json(source/'formal_config_lock.json')
    require(status['status'] == 'training_protocol_locked_test_closed' and
            status['plan_sha256'] == sha(source/'plan_lock.json'), 'Old formal lock mismatch')
    for seed in old['seeds']:
        require(sha(source/'shared_initializations'/f'seed_{seed}.pth') ==
                old['seeds'][seed]['initialization.pth'], 'Shared initialization mismatch')
    destination.mkdir(parents=True)
    provenance = destination/'recovery_source_audit'
    provenance.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.suffix in ['.csv', '.json', '.yaml', '.md']:
            shutil.copy2(path, provenance/path.name)
            if path.name not in ['plan_lock.json', 'formal_config_lock.json', 'failure.json', 'failures.csv', 'completion_matrix.csv']:
                shutil.copy2(path, destination/path.name)
    for name in ['plans', 'shared_initializations', 'data_plans']:
        shutil.copytree(source/name, destination/name)
    for name in ['stages', 'logs', 'a30_resume_20261009']:
        if (source/name).is_dir():
            shutil.copytree(source/name, provenance/name)
    recovery = dict(reason='uniform AMP overflow window replay before any saved epoch',
                    source_run=str(source), source_plan_sha256=sha(source/'plan_lock.json'),
                    previous_source_sha256=old['source_sha256'], new_source_sha256=source_hash(),
                    copied_randomness_byte_identical=True, original_results_modified=False,
                    created_at_utc=datetime.now(timezone.utc).isoformat())
    lock = dict(old, source_sha256=source_hash(), recovery=recovery)
    write_json(destination/'plan_lock.json', lock)
    write_json(destination/'formal_config_lock.json', dict(status, plan_sha256=sha(destination/'plan_lock.json')))
    write_json(destination/'recovery_provenance.json', recovery)
    verify_plan(destination)
    return recovery
