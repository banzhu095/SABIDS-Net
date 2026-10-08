from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tools.denoise_result_review.image_io import decode_lossless

from .common import DEEP, FORMAL, KEY, digest, require, sha, write_csv, write_json


def discover(root, manifest=None):
    if manifest:
        path = Path(manifest).resolve()
        require(path.is_file(), f'Manifest absent: {path}')
        return path
    candidates = []
    for folder in [root / 'runs', root / 'Manifests', root.parent / 'ROI_Denoise_Analysis']:
        if folder.is_dir():
            candidates.extend(folder.rglob('segmentation_primary_inputs.csv'))
    if not candidates:
        for folder in [root / 'runs', root.parent]:
            candidates.extend(folder.rglob('downstream_segmentation_inputs.csv'))
    require(bool(candidates), 'No primary segmentation manifest; provide --manifest')
    # Copies with identical bytes are harmless; independent protocols require explicit selection.
    identities = {}
    for path in candidates:
        identities.setdefault(sha(path), path)
    require(len(identities) == 1, 'Ambiguous manifests; use --manifest: ' + ', '.join(map(str, candidates)))
    return next(iter(identities.values())).resolve()


def resolve(root, value):
    if not value or str(value).lower() == 'nan':
        return None
    normalized = str(value).replace('\\', '/')
    candidates = [Path(normalized), root / normalized]
    if '/SABIDS-Net/' in normalized:
        candidates.append(root / normalized.split('/SABIDS-Net/', 1)[1])
    for old, new in [('Label/layer_binary/', 'Label/binary_choroid_layer/'),
                     ('Label/vessel_binary/', 'Label/binary_choroid_vessel/')]:
        if old in normalized:
            candidates.append(root / (new + normalized.split(old)[1]))
            suffix = '_choroid_layer' if 'layer' in old else '_choroid_vessel'
            name = Path(normalized.split(old)[1])
            candidates.append(root / new / (name.stem + suffix + name.suffix))
    existing = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(existing) > 1:
        require(len({sha(p) for p in existing}) == 1, f'Ambiguous path: {value}')
    return existing[0] if existing else None


@lru_cache(maxsize=30000)
def inspect_file(path):
    path = Path(path)
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = decode_lossless(path)
    require(a.ndim == 2 and np.isfinite(a).all(), f'Invalid grayscale image: {path}')
    return dict(sha256=sha(path), height=a.shape[0], width=a.shape[1],
                bytes=path.stat().st_size, min=float(a.min()), max=float(a.max()), dtype=str(a.dtype))


def audit(root, output, manifest=None, methods=FORMAL):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = discover(root, manifest)
    frame = pd.read_csv(source, keep_default_na=False)
    require(set(KEY + ['method_id']).issubset(frame), 'Manifest lacks explicit logical keys')
    frame = frame[(frame.dataset == 'PKU37') & frame.split.isin(['train', 'val', 'test'])].copy()
    seed_col = 'denoiser_seed' if 'denoiser_seed' in frame else 'seed'
    require(seed_col in frame, 'Explicit denoiser seed required')
    frame = frame[(~frame.method_id.isin(DEEP)) | (pd.to_numeric(frame[seed_col]) == 42)]
    require(not frame.duplicated(KEY + ['method_id']).any(), 'Duplicate method/sample logical keys')
    require(frame.groupby('position_id').split.nunique().max() == 1, 'Position crosses splits')
    canonical = frame[KEY].drop_duplicates().sort_values(KEY)
    require(not canonical.sample_id.duplicated().any(), 'Ambiguous sample identity')
    raw_path = root / 'Manifests/manifest_all.csv'
    raw = pd.read_csv(raw_path, keep_default_na=False) if raw_path.is_file() else pd.DataFrame()
    raw_lookup = {r['sample_id']: r for r in raw.to_dict('records')}
    failures, audited, coverage, repairs = [], [], [], []
    lookup = {(r['method_id'], r['sample_id']): r for r in frame.to_dict('records')}
    for identity in canonical.to_dict('records'):
        sid = identity['sample_id']
        available_rows = [lookup[(m, sid)] for m in FORMAL if (m, sid) in lookup]
        source_row = available_rows[0]
        labels = {}
        for field in ['layer_mask_path', 'vessel_mask_path', 'multiclass_label_path',
                      'label_valid_mask_path', 'vessel_valid_mask_path']:
            values = {str(r.get(field, '')) for r in available_rows if r.get(field, '')}
            values.add(str(raw_lookup.get(sid, {}).get(field, '')))
            paths = [resolve(root, v) for v in values if v]
            paths = [p for p in paths if p]
            if len({sha(p) for p in paths}) > 1:
                failures.append({**identity, 'method_id': '*', 'reason': 'label disagreement: ' + field})
            labels[field] = str(paths[0]) if paths else ''
        labelled = bool(labels['layer_mask_path'] and labels['vessel_mask_path'])
        if labelled:
            # Public naming is an explicit position-key convention, never sorted pairing.
            voc = root/'Label/voc_seg'/(identity['position_id'].removeprefix('pku_')+'.png')
            if not labels['multiclass_label_path'] and voc.is_file():
                labels['multiclass_label_path'] = str(voc)
            if labels['multiclass_label_path']:
                multi = np.asarray(Image.open(labels['multiclass_label_path']))
                require(set(np.unique(multi)).issubset({0,1,2,255}), 'Unexpected multiclass values')
                derived = output/'derived_labels'/identity['position_id']
                derived.mkdir(parents=True,exist_ok=True)
                for field, binary in [('layer_mask_path',np.isin(multi,[1,2])),('vessel_mask_path',multi==2),
                                      ('label_valid_mask_path',multi!=255)]:
                    target=derived/(field+'.png')
                    if not target.exists():Image.fromarray(binary.astype(np.uint8)*255).save(target)
                    require(np.array_equal(np.asarray(Image.open(target))>0,binary),'Existing canonical cache conflicts with multiclass source')
                    if labels[field] and not np.array_equal(np.asarray(Image.open(labels[field]))>0,binary):
                        repairs.append(dict(position_id=identity['position_id'],field=field,original_path=labels[field],
                                            multiclass_source=labels['multiclass_label_path'],derived_path=str(target),reason='Canonical class 1|2 layer, class 2 vessel, class 255 ignored'))
                    labels[field]=str(target)
        declared = any(str(r.get('has_vessel_label', '')).lower() == 'true' for r in available_rows)
        if declared and not labelled:
            failures.append({**identity, 'method_id': '*', 'reason': 'Declared label missing'})
        coverage.append({**identity, 'labelled': labelled, **labels})
        if not labelled:
            continue
        label_meta = {k: inspect_file(v) for k, v in labels.items() if v}
        shapes = {(m['height'], m['width']) for m in label_meta.values()}
        require(len(shapes) == 1, f'Label geometry mismatch: {sid}')
        layer = np.asarray(Image.open(labels['layer_mask_path'])) > 0
        vessel = np.asarray(Image.open(labels['vessel_mask_path'])) > 0
        if (vessel & ~layer).any():
            failures.append({**identity, 'method_id': '*', 'reason': f'Vessel outside layer label: {int((vessel & ~layer).sum())} pixels'})
        if labels['multiclass_label_path']:
            multi = np.asarray(Image.open(labels['multiclass_label_path']))
            require(set(np.unique(multi)).issubset({0, 1, 2, 255}), f'Unknown multiclass values: {sid}')
            known = multi != 255
            if not (np.array_equal(layer[known], np.isin(multi[known], [1, 2]))
                    and np.array_equal(vessel[known], multi[known] == 2)):
                failures.append({**identity, 'method_id': '*', 'reason': 'Binary/multiclass label mismatch'})
        for method in methods:
            row = lookup.get((method, sid), {})
            candidates = [resolve(root, row.get(k, '')) for k in ['local_image_path', 'packaged_path', 'path', 'image_path']]
            if method in ['noisy_identity', 'clean_oracle']:
                candidates.append(resolve(root, raw_lookup.get(sid, {}).get('image_path' if method == 'noisy_identity' else 'clean_path', '')))
            candidates = list(dict.fromkeys(p for p in candidates if p))
            result = {**identity, 'method_id': method, 'denoiser_seed': 42 if method in DEEP else 0,
                      **labels, 'label_identity': digest(label_meta), 'path': '', 'available': False, 'decode_status': 'missing'}
            result.update(config_sha256=row.get('config_sha256',''),checkpoint_sha256=row.get('checkpoint_sha256',''),
                          asset_origin=row.get('asset_origin','local_original_or_downloaded_formal'),is_primary=True)
            try:
                require(bool(row), 'No matching method manifest row')
                require(bool(candidates), 'Missing input image')
                if len({sha(p) for p in candidates}) > 1:
                    decoded=[decode_lossless(p) for p in candidates]
                    require(all(np.array_equal(decoded[0],a) for a in decoded[1:]), 'Conflicting image candidates')
                path = candidates[0]
                meta = inspect_file(str(path))
                require((meta['height'], meta['width']) in shapes, 'Image/label shape mismatch')
                expected = row.get('image_sha256') or row.get('sha256') or row.get('output_sha256')
                if expected and method not in ['noisy_identity','clean_oracle']:
                    require(meta['sha256'] == expected, 'Image SHA256 differs from source manifest')
                result['formal_output_sha256']=expected or ''
                result['source_kind']='original_or_lossless_copy' if method in ['noisy_identity','clean_oracle'] else 'formal_denoised_output'
                result.update(path=str(path), available=True, decode_status='ok', **meta)
            except (ValueError, OSError) as exc:
                failures.append({**identity, 'method_id': method, 'reason': str(exc)})
            audited.append(result)
    data = pd.DataFrame(audited)
    require(not data.empty, 'No labelled images')
    balance = data.groupby(['method_id', 'split']).agg(expected=('sample_id', 'size'), available=('available', 'sum'), positions=('position_id', 'nunique')).reset_index()
    counts = pd.DataFrame(coverage).query('labelled').groupby('split').agg(frames=('sample_id', 'size'), positions=('position_id', 'nunique')).reset_index()
    write_csv(output / 'input_asset_audit.csv', data)
    write_csv(output / 'asset_inventory.csv', data)
    formal=data.copy()
    formal['input_group']=formal.method_id
    formal['image_path']=formal.path;formal['image_sha256']=formal.get('sha256','')
    formal['layer_label_path']=formal.layer_mask_path;formal['vessel_label_path']=formal.vessel_mask_path
    formal['label_sha256']=formal.label_identity
    formal['has_layer_label']=True;formal['has_vessel_label']=True
    formal['materialization_status']=np.where(formal.available,'verified','missing')
    write_csv(output/'formal_segmentation_inputs.csv',formal)
    write_csv(output/'asset_provenance.csv',formal[KEY+['method_id','denoiser_seed','path','image_sha256','config_sha256','checkpoint_sha256','asset_origin','materialization_status']])
    write_csv(output / 'input_balance.csv', balance)
    write_csv(output / 'label_coverage.csv', coverage)
    write_csv(output/'label_derivation_audit.csv',pd.DataFrame(repairs).drop_duplicates() if repairs else [],
              ['position_id','field','original_path','multiclass_source','derived_path','reason'])
    write_csv(output / 'missing_or_ambiguous_assets.csv', failures, KEY + ['method_id', 'reason'])
    inventory = {p: inspect_file(p) for p in sorted({str(r[k]) for r in audited for k in labels if r[k]})}
    write_json(output / 'label_asset_inventory.json', inventory)
    status = dict(source_manifest=str(source), source_sha256=sha(source), passed=not failures,
                  methods=methods, labelled_counts=counts.to_dict('records'), failures=len(failures),
                  audit_sha256=sha(output / 'input_asset_audit.csv'))
    write_json(output / 'audit.json', status)
    evidence=[]
    for folder in [source.parent/'source_formal_light',root/'runs']:
        if folder.is_dir():
            for name in ['per_image_metrics.csv','denoised_dataset_manifest_primary.csv']:
                for p in folder.rglob(name):
                    evidence.append(dict(kind=name,path=str(p.resolve()),sha256=sha(p)))
    write_csv(output/'formal_source_inventory.csv',evidence,['kind','path','sha256'])
    test_positions = int(counts.loc[counts.split == 'test', 'positions'].sum())
    canonical_exports(output,data,repairs)
    formal_status = 'limited_3_position_test' if test_positions < 6 else 'six_position_test'
    (output / 'protocol_audit.md').write_text(
        '# Input protocol audit\n\nSABIDS-current Stage-1/D0 output; denoised-only information recoverability.\n\n'
        + balance.to_markdown(index=False) + '\n\n' + counts.to_markdown(index=False)
        + f'\n\nAsset failures: {len(failures)}. Formal: {formal_status}. '
        + 'Audit decodes files and labels for integrity only; prediction/test scoring remains sealed until all checkpoints lock. '
        + 'Complete missing annotations for pku_0025, pku_0038, pku_0043 when applicable.\n', encoding='utf-8')
    return status


def canonical_exports(output,data,repairs):
    unique=data.drop_duplicates('position_id')
    columns=['position_id','split','layer_mask_path','vessel_mask_path','multiclass_label_path','label_valid_mask_path','label_identity']
    write_csv(output/'canonical_label_manifest.csv',unique[columns])
    rows=[]
    for row in pd.DataFrame(repairs).drop_duplicates().to_dict('records') if repairs else []:
        old=np.asarray(Image.open(row['original_path']))>0
        new=np.asarray(Image.open(row['derived_path']))>0
        rows.append({**row,'different_pixels':int((old!=new).sum()),'total_pixels':old.size,'difference_fraction':float((old!=new).mean())})
    columns=['position_id','field','original_path','multiclass_source','derived_path','reason','different_pixels','total_pixels','difference_fraction']
    write_csv(output/'legacy_vs_canonical_label_audit.csv',rows,columns)
    summary=pd.DataFrame(rows,columns=columns)
    write_csv(output/'label_difference_by_position.csv',summary[['position_id','field','different_pixels','total_pixels','difference_fraction']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    preview=output/'canonical_label_preview';preview.mkdir(exist_ok=True)
    for row in unique.sort_values('position_id').to_dict('records'):
        layer=np.asarray(Image.open(row['layer_mask_path']))>0
        vessel=np.asarray(Image.open(row['vessel_mask_path']))>0
        require(not (vessel&~layer).any(),'Canonical vessel is not a layer subset')
        fig,axes=plt.subplots(1,3,figsize=(9,3))
        valid=np.asarray(Image.open(row['label_valid_mask_path']))>0 if row['label_valid_mask_path'] else np.ones_like(layer)
        for ax,a,title in zip(axes,[layer,vessel,valid],['Canonical layer: 1 or 2','Canonical vessel: 2','Valid: excludes 255']):
            ax.imshow(a,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=9);ax.axis('off')
        fig.suptitle(row['position_id']);fig.tight_layout();fig.savefig(preview/(row['position_id']+'.png'),dpi=100);plt.close(fig)
    (output/'canonical_label_policy.md').write_text('Class semantics confirmed by AGENTS.md, docs/DATASET_PROTOCOL.md and tools/prepare_current_data.py. Layer=(1|2), vessel=2, 255 ignored. Generated masks remain inside this run. Padding is absent from native square PKU37 images and out-of-image crops are prohibited. Integrity auditing may decode test labels, but training and checkpoint selection never use them.\n',encoding='utf-8')
