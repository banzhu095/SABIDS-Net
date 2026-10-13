from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion

from tools.downstream_seg_benchmark.common import read_json, require, sha, write_csv, write_json
from tools.downstream_seg_benchmark.data import image_array,mask_array
from .tables import ORDER,LABELS,SEEDS,POSITIONS


def labels(row):
    layer=mask_array(row['layer_mask_path'])>0;vessel=mask_array(row['vessel_mask_path'])>0
    valid=np.ones_like(layer)
    if row.get('multiclass_label_path'):valid &= mask_array(row['multiclass_label_path'])!=255
    if row.get('label_valid_mask_path'):valid &= mask_array(row['label_valid_mask_path'])>0
    vv=valid.copy()
    if row.get('vessel_valid_mask_path'):vv &= mask_array(row['vessel_valid_mask_path'])>0
    return layer,vessel,valid,vv


def probability_assets(layer,vessel,gt_layer,gt_vessel,valid,vv,image):
    require(layer.shape==vessel.shape==gt_layer.shape==gt_vessel.shape==valid.shape==vv.shape==image.shape,'Geometry mismatch')
    for a in [layer,vessel]:require(np.isfinite(a).all() and a.min()>=0 and a.max()<=1,'Invalid probability')
    lm,vm=layer>=.5,vessel>=.5
    errors=np.zeros((*vm.shape,3),np.uint8)
    errors[vm&gt_vessel&vv]=(0,255,0);errors[vm&~gt_vessel&vv]=(255,0,0);errors[~vm&gt_vessel&vv]=(0,0,255)
    errors[~vv]=(128,128,128)
    gray=np.rint(np.clip(image,0,1)*255).astype(np.uint8)
    def overlay(pred,gt,mask):
        a=np.repeat(gray[...,None],3,axis=2)
        a[(gt^binary_erosion(gt))&mask]=(0,255,255)
        a[(pred^binary_erosion(pred))&mask]=(255,255,0)
        a[~mask]=(128,128,128)
        return a
    from matplotlib import colormaps
    return {'layer_mask.png':lm.astype(np.uint8)*255,'vessel_mask.png':vm.astype(np.uint8)*255,
        'layer_probability_u16.png':np.rint(layer*65535).astype(np.uint16),
        'vessel_probability_u16.png':np.rint(vessel*65535).astype(np.uint16),
        'vessel_tp_fp_fn.png':errors,'layer_overlay.png':overlay(lm,gt_layer,valid),
        'vessel_overlay.png':overlay(vm,gt_vessel,vv),
        'layer_probability_preview.png':np.rint(colormaps['viridis'](layer)[...,:3]*255).astype(np.uint8),
        'vessel_probability_preview.png':np.rint(colormaps['viridis'](vessel)[...,:3]*255).astype(np.uint8)}


def asset_record(out,path,blob,method,seed,position,sample,kind,shape,dtype,checkpoint=''):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    return dict(method_id=method,display_name=LABELS.get(method,method),segmentation_seed=seed,position_id=position,sample_id=sample,
        asset_type=kind,path=str(path.resolve()),relative_path=path.relative_to(out).as_posix(),shape=shape,dtype=dtype,
        threshold=.5 if kind in ['layer_mask','vessel_mask','vessel_tp_fp_fn','layer_overlay','vessel_overlay'] else None,
        sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),checkpoint_sha256=checkpoint)


def png(out,path,a,method,seed,position,sample,checkpoint=''):
    stream=io.BytesIO();Image.fromarray(a).save(stream,format='PNG',compress_level=3)
    blob=stream.getvalue()
    decoded=np.asarray(Image.open(io.BytesIO(blob)))
    require(np.array_equal(decoded,a),'PNG round-trip changed values')
    return asset_record(out,path,blob,method,seed,position,sample,path.stem,'x'.join(map(str,a.shape)),str(a.dtype),checkpoint)


def basic_metrics(lp,vp,layer,vessel,valid,vv):
    l=(lp>=.5)&valid;v=(vp>=.5)&vv;gl=layer&valid;gv=vessel&vv
    def dice(a,b):return 2*(a&b).sum()/(a.sum()+b.sum()) if a.sum()+b.sum() else 1.
    tp=(v&gv).sum();fp=(v&~gv).sum();fn=(~v&gv).sum()
    return dict(layer_dice=dice(l,gl),vessel_dice=dice(v,gv),vessel_roi_dice=dice(v&gl,gv&gl),
        precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        vessel_outside_gt_layer_fraction=(v&~gl).sum()/max(v.sum(),1),
        vessel_outside_gt_layer_fp=(v&~gl&vv).sum()/max((~gl&vv).sum(),1))


def export(run,out,resume=False,workers=4):
    cohort=pd.read_csv(run/'cohort.csv',keep_default_na=False);cohort=cohort[cohort.split=='test']
    canon=cohort[cohort.method_id=='noisy_identity'].set_index('sample_id')
    gt={pos:labels(canon[canon.position_id==pos].iloc[0].to_dict()) for pos in POSITIONS}
    records=[];public=[];missing=[]
    shared=out/'exports/common/positions'
    for pos in POSITIONS:
        gl,gv,valid,vv=gt[pos]
        clean=cohort[(cohort.position_id==pos)&(cohort.method_id=='clean_oracle')].iloc[0]
        for name,a in [('layer_gt',gl.astype(np.uint8)*255),('vessel_gt',gv.astype(np.uint8)*255),
                       ('layer_valid',valid.astype(np.uint8)*255),('vessel_valid',vv.astype(np.uint8)*255),
                       ('reference',np.rint(image_array(clean.path)*65535).astype(np.uint16))]:
            public.append(png(out,shared/pos/('inputs' if name=='reference' else 'ground_truth')/(name+'.png'),a,'__shared',0,pos,'__position'))
    inputs={}
    for method in ORDER:
        for row in cohort[cohort.method_id==method].to_dict('records'):
            pos,sid=row['position_id'],row['sample_id']
            require(sha(Path(row['path']))==row['sha256'],'Method input asset changed')
            if method=='clean_oracle':
                path=shared/pos/'inputs/reference.png'
                reference=np.asarray(Image.open(path))
                require(np.array_equal(np.rint(image_array(row['path'])*65535).astype(np.uint16),reference),'Clean reference differs within position')
            else:
                path=out/'exports/primary_seed42/positions'/pos/sid/'inputs'/(method+'.png')
                x=image_array(row['path']);require(x.shape==(640,640),'Input geometry not native 640')
                public.append(png(out,path,np.rint(x*65535).astype(np.uint16),method,0,pos,sid))
            inputs[method,sid]=path
    frame=pd.read_csv(run/'per_frame_metrics.csv').set_index(['method_id','segmentation_seed','sample_id'])
    checkpoints={(x['method_id'],int(x['segmentation_seed'])):x['sha256'] for x in read_json(run/'checkpoint_lock.json')['checkpoints']}
    counts=[]
    for method in ORDER:
        rows=cohort[cohort.method_id==method].sort_values('sample_id').to_dict('records')
        for seed in SEEDS:
            cached=out/'exports/export_records'/(method+'_'+str(seed)+'.csv')
            if resume and cached.exists():
                old=pd.read_csv(cached,keep_default_na=False)
                require(all(sha(out/p)==h for p,h in zip(old.relative_path,old.sha256)),'Cached export changed')
                records+=old.to_dict('records');counts.append(dict(method_id=method,segmentation_seed=seed,frames=old.sample_id.nunique(),status='verified_cached'))
                continue
            ev=read_json(run/'evaluation'/method/str(seed)/'complete.json')
            ck=checkpoints[method,seed]
            def one(row):
                pos,sid=row['position_id'],row['sample_id']
                source=run/'evaluation'/method/str(seed)/'predictions'/(sid+'.npz')
                if not source.is_file():return [],dict(method_id=method,segmentation_seed=seed,position_id=pos,sample_id=sid,reason='NPZ missing; no replacement')
                blob=source.read_bytes();h=hashlib.sha256(blob).hexdigest()
                require(h==ev['prediction_hashes'][source.name],'Sealed probability changed')
                with np.load(io.BytesIO(blob),allow_pickle=False) as data:lp,vp=data['layer'].copy(),data['vessel'].copy()
                require(lp.shape==vp.shape==(640,640),'NPZ not native 640x640')
                gl,gv,valid,vv=gt[pos]
                vals=basic_metrics(lp,vp,gl,gv,valid,vv);original=frame.loc[method,seed,sid]
                require(all(np.isclose(value,original[name],rtol=1e-10,atol=1e-12) for name,value in vals.items()),'Probability/record metric mismatch '+sid)
                image=np.asarray(Image.open(inputs[method,sid])).astype(np.float32)/65535
                arrays=probability_assets(lp,vp,gl,gv,valid,vv,image)
                require(np.array_equal(arrays['layer_mask.png']>0,lp>=.5) and np.array_equal(arrays['vessel_mask.png']>0,vp>=.5),'Raw threshold mismatch')
                require(max(np.abs(arrays['vessel_probability_u16.png']/65535-vp).max(),np.abs(arrays['layer_probability_u16.png']/65535-lp).max())<=.5/65535+1e-7,'Probability quantization mismatch')
                folder=(out/'exports/primary_seed42/positions'/pos/sid/method if seed==42 else out/'exports/all_seeds'/method/('seed_'+str(seed))/pos/sid)
                rs=[png(out,folder/name,a,method,seed,pos,sid,ck) for name,a in arrays.items()]
                rs.append(asset_record(out,folder/'probabilities.npz',blob,method,seed,pos,sid,'probabilities','layer:640x640;vessel:640x640',str(lp.dtype),ck))
                # Link common/method input rather than copying across segmentation seeds.
                path=inputs[method,sid]
                rs.append(dict(method_id=method,display_name=LABELS[method],segmentation_seed=seed,position_id=pos,sample_id=sid,
                    asset_type='input_link',path=str(path.resolve()),relative_path=path.relative_to(out).as_posix(),shape='640x640',dtype='uint16',
                    threshold=None,sha256=sha(path),bytes=path.stat().st_size,checkpoint_sha256=ck))
                return rs,None
            track_records=[]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i,(rs,fail) in enumerate(pool.map(one,rows),1):
                    track_records+=rs
                    if fail:missing.append(fail)
                    if i%50==0:print('export',method,seed,i,'/150',flush=True)
            write_csv(cached,track_records);records+=track_records
            if seed==42:write_csv(out/'exports/all_seeds'/method/'seed_42/index.csv',track_records)
            counts.append(dict(method_id=method,segmentation_seed=seed,frames=len({r['sample_id'] for r in track_records}),status='exported_from_sealed_npz'))
    write_csv(out/'prediction_export_manifest.csv',public+records)
    write_csv(out/'public_asset_manifest.csv',public)
    write_csv(out/'prediction_export_coverage.csv',counts)
    write_csv(out/'prediction_export_missing.csv',missing,['method_id','segmentation_seed','position_id','sample_id','reason'])
    write_json(out/'prediction_export_audit.json',dict(frames=sum(c['frames'] for c in counts),assets=len(public)+len(records),missing=len(missing),
        threshold=.5,native_shape=[640,640],raw_masks_verified=True,npz_hashes_verified=True,basic_metrics_reproduced=True,
        no_postprocessing=True,probability_u16_mapping='round(p * 65535)',ignore_error_color=[128,128,128],
        all_seed42_layout='seed42 index refers to primary_seed42; no duplicate probability assets',
        public_assets='GT and reference once per position; inputs once per method/sample; linked across segmentation seeds'))


def atlas(run,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fixed=pd.read_csv(run/'fixed_atlas_samples.csv');cohort=pd.read_csv(run/'cohort.csv',keep_default_na=False)
    cohort=cohort[cohort.split=='test'].set_index(['method_id','sample_id'])
    folder=out/'qualitative_atlas';folder.mkdir(exist_ok=True)
    write_csv(folder/'atlas_selection.csv',fixed)
    write_json(folder/'display_protocol.json',dict(gray_range=[0,1],gamma=1.,probability_range=[0,1],probability_cmap='viridis',
        resize='nearest',crop='locked fixed_atlas_samples.csv coordinates',threshold=.5,gt_contour='cyan',prediction_contour='yellow',
        error_colors={'TP':'green','FP':'red','FN':'blue','ignore':'gray'},primary_segmentation_seed=42,no_postprocessing=True,
        role_warning='small_vessel_rich with count=0 is preregistered but has no small components'))
    def panel(item,seed,methods,filename):
        sid=item['sample_id'];gl,gv,valid,vv=labels(cohort.loc['noisy_identity',sid].to_dict())
        y,x,n=int(item.crop_y),int(item.crop_x),int(item.crop_size);crop=np.s_[y:y+n,x:x+n]
        fig,axes=plt.subplots(len(methods),8,figsize=(20,len(methods)*2.5),squeeze=False)
        titles=['Method input','GT/Pred layer contour','GT vessel','Vessel probability [0,1]','Raw vessel p>=0.5','TP green / FP red / FN blue','Layer crop: GT cyan / pred yellow','Vessel crop: GT cyan / pred yellow']
        for j,method in enumerate(methods):
            row=cohort.loc[method,sid];image=image_array(row.path)
            source=run/'evaluation'/method/str(seed)/'predictions'/(sid+'.npz')
            if not source.exists():
                for ax in axes[j]:ax.text(.5,.5,'MISSING NPZ',ha='center');ax.axis('off')
                continue
            with np.load(source,allow_pickle=False) as p:lp,vp=p['layer'],p['vessel']
            assets=probability_assets(lp,vp,gl,gv,valid,vv,image)
            pictures=[image,assets['layer_overlay.png'],gv.astype(float),vp,assets['vessel_mask.png']/255,
                      assets['vessel_tp_fp_fn.png'],assets['layer_overlay.png'][crop],assets['vessel_overlay.png'][crop]]
            for k,(ax,pic) in enumerate(zip(axes[j],pictures)):
                ax.imshow(pic,cmap='viridis' if k==3 else 'gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
                if j==0:ax.set_title(titles[k],fontsize=8)
                if k==0:ax.text(-.04,.5,LABELS[method],transform=ax.transAxes,ha='right',va='center',fontsize=8)
        warning='; preregistered role, NO small components' if item.role=='small_vessel_rich' and item.small_component_count==0 else ''
        fig.suptitle(f'{item.position_id} / {sid} / {item.role}; segmentation seed {seed}{warning}',fontsize=12)
        fig.subplots_adjust(left=.15,right=.995,top=.955,bottom=.005,wspace=.04,hspace=.06)
        fig.savefig(folder/filename,dpi=100);plt.close(fig)
    for seed in SEEDS:
        for _,item in fixed.iterrows():panel(item,seed,ORDER,f'all_methods_{item.position_id}_{item.role}_seed{seed}.png')
        for pos in POSITIONS:
            item=fixed[(fixed.position_id==pos)&(fixed.role=='weak_boundary')].iloc[0]
            panel(item,seed,['noisy_identity','clean_oracle','dncnn_paired','nafnet_paired','sabids_current'],f'core_{pos}_seed{seed}.png')
    thumbs=out/'atlas_thumbnails';thumbs.mkdir(exist_ok=True)
    for p in folder.glob('*.png'):
        with Image.open(p) as im:
            im.thumbnail((1200,1600),Image.Resampling.LANCZOS);im.save(thumbs/p.name,optimize=True)
    (thumbs/'DISPLAY_ONLY.md').write_text('Display-only downsampled atlas thumbnails. Quantitative masks/probabilities remain 640x640 in download packages. Full atlas uses identical registered crops, fixed [0,1] gray/probability range, gamma 1, nearest panel sampling.\n',encoding='utf-8')
