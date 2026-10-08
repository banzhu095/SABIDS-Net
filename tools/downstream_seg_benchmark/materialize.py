"""Reuse sealed denoising outputs, or replay the exact sealed adapter/config."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from .assets import discover, resolve
from .common import DEEP, FORMAL, KEY, digest, read_json, require, sha, write_csv, write_json


def find_sources(root, source=None):
    root=Path(root).resolve()
    if source:
        selected=Path(source).resolve();require(selected.is_dir(),'Source denoise run absent')
        return selected
    paths=[]
    for base in [root/'runs', root.parent/'ROI_Denoise_Analysis']:
        if base.is_dir():
            paths.extend(p.parent.parent for p in base.rglob('denoised_dataset_manifest_primary.csv') if p.parent.name=='manifests')
    # Prefer a merged light snapshot with all ten methods; never choose by timestamps.
    complete=[]
    for path in paths:
        table=pd.read_csv(path/'manifests/denoised_dataset_manifest_primary.csv',keep_default_na=False)
        if set(FORMAL)-{'clean_oracle'}<=set(table.method_id):complete.append(path)
    require(len(complete)==1,'Provide --source-denoise-run; no unique complete formal source')
    return complete[0]


def source_manifest(root, source, run):
    source=Path(source);run=Path(run)
    for name in ['segmentation_primary_inputs.csv','downstream_segmentation_inputs.csv']:
        for folder in [source, source/'manifests']:
            path=folder/name
            if path.is_file():return path
    primary=source/'manifests/denoised_dataset_manifest_primary.csv'
    if not primary.is_file():primary=source/'manifests/denoised_dataset_manifest.csv'
    require(primary.is_file(),'Formal primary denoising manifest missing')
    outputs=pd.read_csv(primary,keep_default_na=False)
    from tools.oct_denoise_benchmark.data import load_protocol_manifest
    protocol=load_protocol_manifest(Path(root)).fillna('')
    require(set(KEY)<=set(protocol),'Denoising protocol lacks sample keys')
    if 'sample_id' not in outputs:
        metrics=pd.read_csv(source/'metrics/per_image_metrics.csv',keep_default_na=False)
        join=['dataset','split','position_id','frame_id','method_id','seed','denoised_path']
        require(not metrics.duplicated(join).any(),'Ambiguous legacy logical join')
        outputs=outputs.merge(metrics[join+['sample_id']],on=join,how='left',validate='many_to_one')
    require(set(KEY)<=set(outputs),'Formal output keys incomplete')
    rows=[]
    raw=protocol.set_index('sample_id')
    for record in outputs.to_dict('records'):
        if record['dataset']!='PKU37' or record['split'] not in ['train','val','test']:continue
        method=record['method_id'];seed=int(record.get('seed',0))
        if method in DEEP and seed!=42:continue
        require(record['sample_id'] in raw.index,'Output absent from original protocol')
        identity=raw.loc[record['sample_id']]
        require(all(str(record[k])==str(identity[k]) for k in KEY),'Protocol/output split or key conflict')
        row={**record,'path':record.get('denoised_path',''),'image_path':record.get('denoised_path',''),
             'image_sha256':record.get('output_sha256',''),'denoiser_seed':seed,'is_primary':True}
        for key in ['layer_mask_path','vessel_mask_path','multiclass_label_path','label_valid_mask_path','vessel_valid_mask_path']:
            row[key]=identity.get(key,'')
        rows.append(row)
    for identity in protocol[protocol.dataset=='PKU37'].to_dict('records'):
        rows.append({**identity,'method_id':'clean_oracle','denoiser_seed':0,'path':identity['clean_path'],
                     'image_path':identity['clean_path'],'image_sha256':'','is_primary':True,
                     'config_sha256':digest({'method_id':'clean_oracle'}),'checkpoint_sha256':''})
    path=run/'source_primary_inputs.csv';write_csv(path,rows)
    return path


def zip_asset(root,run,row):
    """Extract only an explicitly manifest-keyed image; never pair sorted filenames."""
    import io,hashlib
    matches=[]
    folders=[Path(root)/'Downloaded_Denoise_Review',Path(root).parent/'SABIDS_PKU37_denoise_benchmark_GPT_light_PLUS_SABIDS_TCFL_20260918_121447']
    for folder in folders:
        if not folder.exists():continue
        for archive in folder.rglob('*.zip'):
            with zipfile.ZipFile(archive) as z:
                names=[n for n in z.namelist() if Path(n).name in ['IMAGE_MANIFEST.csv','image_manifest.csv','package_manifest.csv']]
                for name in names:
                    table=pd.read_csv(io.BytesIO(z.read(name)),keep_default_na=False)
                    if not set(KEY+['method_id'])<=set(table):continue
                    mask=table.method_id.eq(row['method_id'])
                    for key in KEY:mask=mask & table[key].astype(str).eq(str(row[key]))
                    for record in table[mask].to_dict('records'):
                        entry=record.get('archive_image_path') or record.get('archive_path') or record.get('relative_path')
                        if not entry:continue
                        entry=str(entry).replace('\\','/')
                        require(entry in z.namelist(),'ZIP manifest names absent image')
                        payload=z.read(entry);actual=hashlib.sha256(payload).hexdigest()
                        expected=row.get('image_sha256') or row.get('output_sha256')
                        require(not expected or expected==actual,'ZIP image historical SHA conflict')
                        matches.append((payload,actual,Path(entry).suffix))
    if not matches:return None
    require(len({x[1] for x in matches})==1,'Conflicting manifest-keyed ZIP images')
    payload,actual,suffix=matches[0]
    path=Path(run)/'materialized_images'/row['method_id']/row['split']/(row['sample_id']+suffix)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(sha(path)==actual,'Existing extracted ZIP image conflict')
    else:path.write_bytes(payload)
    return path


def locked_entry(root, source, method, record):
    from tools.oct_denoise_benchmark.registry import stable_sha256
    roots={Path(source)}
    for value in [record.get('path',''),record.get('image_path','')]:
        normalized=str(value).replace('\\','/')
        if '/runs/' in normalized and '/images/' in normalized:
            relative=normalized.split('/runs/',1)[1].split('/images/',1)[0]
            roots.add(Path(root)/'runs'/relative)
    found=[]
    for folder in roots:
        registry=folder/'configs/inference_registry.yaml'
        if not registry.is_file():continue
        entries=yaml.safe_load(registry.read_text(encoding='utf-8')).get('methods',{})
        if method not in entries:continue
        entry=entries[method];config=dict(entry.get('config',entry));config['method_id']=method
        if record.get('config_sha256') and stable_sha256(config)!=record['config_sha256']:continue
        lockfile=folder/'audit/extension_config_lock.json'
        if not lockfile.is_file():lockfile=folder/'audit/config_lock.json'
        require(lockfile.is_file(),'Unsealed denoising source')
        lock=read_json(lockfile);require(lock.get('status')=='locked','Denoising source is not locked')
        require(lock.get('config_sha256',{}).get('inference_registry.yaml')==sha(registry),'Registry bytes differ from lock')
        checkpoint=None
        if method in DEEP:
            require(int(entry.get('seed',0))==42,'Locked primary denoiser is not seed 42')
            checkpoint=resolve(Path(root),entry.get('checkpoint',''))
            require(checkpoint and sha(checkpoint)==record.get('checkpoint_sha256'),'Checkpoint absent/hash conflict')
            allowed=lock.get('checkpoint_sha256',{}).get(method,[])
            require(any(int(x.get('seed',-1))==42 and x.get('sha256')==sha(checkpoint) for x in allowed),'Primary checkpoint not sealed')
        if config.get('config_path'):
            original=config['config_path'];resolved=resolve(Path(root),original)
            require(resolved,'SABIDS locked resolved config missing')
            # Path relocation changes no model/config content, and is explicitly logged.
            config['config_path']=str(resolved)
        found.append((config,checkpoint,str(folder)))
    require(len(found)==1,'No unique config/checkpoint matching historical SHA for '+method)
    return found[0]


def materialize(root, run, source=None, methods=FORMAL, resume=False, reinfer=False, device='cuda'):
    root,run=Path(root).resolve(),Path(run).resolve();run.mkdir(parents=True,exist_ok=True)
    require(not (run/'plan_lock.json').exists(),'Training plan is sealed; materialize into a separate asset directory')
    source=find_sources(root,source)
    base=source_manifest(root,source,run)
    frame=pd.read_csv(base,keep_default_na=False)
    require(set(KEY+['method_id'])<=set(frame),'Incomplete source keys')
    seed='denoiser_seed' if 'denoiser_seed' in frame else 'seed'
    frame=frame[(frame.dataset=='PKU37') & frame.method_id.isin(methods)]
    frame=frame[(~frame.method_id.isin(DEEP)) | (pd.to_numeric(frame[seed])==42)]
    require(not frame.duplicated(KEY+['method_id']).any(),'Ambiguous primary rows')
    old={}
    target=run/'materialized_primary_inputs.csv'
    if target.exists():
        require(resume,'Materialization exists; use --resume')
        old={(r['method_id'],r['sample_id']):r for r in pd.read_csv(target,keep_default_na=False).to_dict('records')}
    rows=[]
    for index,row in enumerate(frame.to_dict('records')):
        # Only labelled positions enter the segmentation assets; test labels are not decoded here.
        declared=bool(row.get('layer_mask_path') and row.get('vessel_mask_path'))
        if not declared:continue
        key=(row['method_id'],row['sample_id']);previous=old.get(key)
        expected=row.get('image_sha256') or row.get('output_sha256','')
        candidates=[resolve(root,row.get(k,'')) for k in ['local_image_path','packaged_path','path','image_path','denoised_path']]
        candidates=[p for p in candidates if p]
        if row['method_id'] in ['clean_oracle','noisy_identity']:
            from tools.oct_denoise_benchmark.data import load_protocol_manifest
            raw=load_protocol_manifest(root);match=raw[raw.sample_id==row['sample_id']]
            require(len(match)==1,'Original image key ambiguous')
            name='clean_path' if row['method_id']=='clean_oracle' else 'image_path'
            candidate=resolve(root,match.iloc[0][name])
            if candidate:candidates.append(candidate)
        origin='formal_existing_output'
        if previous:
            p=Path(previous['local_image_path']);require(p.is_file() and sha(p)==previous['materialized_sha256'],'Resume asset hash conflict')
            require(not expected or previous['historical_output_sha256']==expected,'Historical manifest changed')
            rows.append(previous);continue
        if candidates:
            p=candidates[0]
        else:
            zipped=zip_asset(root,run,row)
            if zipped:
                actual=sha(zipped)
                rows.append({**row,'local_image_path':str(zipped),'materialized_sha256':actual,
                    'historical_output_sha256':expected,'asset_origin':'manifest_keyed_downloaded_zip','materialization_status':'verified'})
                write_csv(target,rows);continue
            require(reinfer,'Missing cloud output: '+str(key)+'; use --reinfer only with exact sealed checkpoint/config')
            require(row['method_id'] not in {'clean_oracle','noisy_identity'},'Original/reference input must exist')
            config,checkpoint,config_source=locked_entry(root,source,row['method_id'],row)
            noisy=resolve(root,row.get('noisy_path',''))
            if not noisy:
                from tools.oct_denoise_benchmark.data import load_protocol_manifest
                raw=load_protocol_manifest(root);matches=raw[raw.sample_id==row['sample_id']]
                require(len(matches)==1,'No unique original noisy input')
                noisy=Path(matches.iloc[0].image_path)
            from tools.oct_denoise_benchmark.io import read_image,save_image
            from tools.oct_denoise_benchmark.methods import AdapterContext,denoise
            image,meta=read_image(noisy)
            prediction=denoise(image,config,AdapterContext(device=device,checkpoint=checkpoint,seed=42))
            p=run/'materialized_images'/row['method_id']/row['split']/(row['sample_id']+'.tif')
            require(not p.exists(),'Orphan materialized output exists; verify explicitly before reuse')
            save_image(p,prediction,meta,True);origin='deterministic_reinference_locked_primary'
        actual=sha(p)
        require(not expected or actual==expected,'Historical output SHA conflict for '+str(key))
        rows.append({**row,'local_image_path':str(p),'materialized_sha256':actual,
                     'historical_output_sha256':expected,'asset_origin':origin,'materialization_status':'verified'})
        write_csv(target,rows)
        if index%20==0:print(f'materialize {index+1}/{len(frame)} {key}',flush=True)
    verify(run)
    return target


def verify(run):
    path=Path(run)/'materialized_primary_inputs.csv'
    data=pd.read_csv(path,keep_default_na=False)
    require(not data.duplicated(KEY+['method_id']).any(),'Duplicate materialized keys')
    for row in data.to_dict('records'):
        require(sha(row['local_image_path'])==row['materialized_sha256'],'Materialized image changed')
        require(not row['historical_output_sha256'] or row['historical_output_sha256']==row['materialized_sha256'],'Historical SHA conflict')
    present=set(data.method_id)
    keys=None
    for method,group in data.groupby('method_id'):
        current=set(map(tuple,group[KEY].to_numpy()))
        require(keys is None or current==keys,'Materialized method sample keys differ')
        keys=current
    write_json(Path(run)/'materialization_verification.json',dict(rows=len(data),manifest_sha256=sha(path),passed=True))
    return data


def export(run, output):
    data=verify(run);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    archives=[]
    for method,group in data.groupby('method_id'):
        path=output/(method+'_labelled_primary.zip');require(not path.exists(),'Asset ZIP already exists')
        inventory=[]
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for row in group.to_dict('records'):
                entry='images/'+row['split']+'/'+row['sample_id']+Path(row['local_image_path']).suffix
                z.write(row['local_image_path'],entry);inventory.append({**row,'archive_image_path':entry})
            z.writestr('IMAGE_MANIFEST.csv',pd.DataFrame(inventory).to_csv(index=False))
        with zipfile.ZipFile(path) as z:
            require(z.testzip() is None,'Asset ZIP CRC failure')
            import hashlib
            for row in inventory:require(hashlib.sha256(z.read(row['archive_image_path'])).hexdigest()==row['materialized_sha256'],'Asset ZIP SHA failure')
        archives.append(dict(method_id=method,path=str(path),sha256=sha(path),images=len(group)))
    write_csv(output/'export_inventory.csv',archives)
    return archives
