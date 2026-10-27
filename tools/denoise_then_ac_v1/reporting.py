"""Fixed native previews; development-only paired results and bounded light ZIP."""
from pathlib import Path
import io
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .core import (read,write,table,atomic,png,npy,plane,sha,require,verify_cache,
                   ARMS,COMPARISONS)
from tools.ac_vessel_labelsv2.report import read_metrics,paired_tables


def gray(a):
    return np.rint(np.clip(a,0,1)*255).astype(np.uint8)


def save_image(path,a):
    png(path,a)


def rgb(a):
    return np.repeat(gray(a)[...,None],3,axis=2)


def contour(mask):
    return mask ^ binary_erosion(mask)


def crop_box(layer):
    h,w=layer.shape
    yy,xx=np.where(layer)
    # Shared read-only Layer crop, fixed horizontal middle third. No model/Vessel selection.
    return (w//3,max(0,int(yy.min())-16) if len(yy) else h//3,
            2*w//3,min(h,int(yy.max())+17) if len(yy) else 2*h//3)


def sheet(path,panels,title,box=None):
    tiles=[]
    for name,a in panels:
        im=Image.fromarray(a).convert('RGB')
        if box: im=im.crop(box)
        im.thumbnail((320,320))
        tile=Image.new('RGB',(320,344),'#152533');tile.paste(im,(0,24))
        ImageDraw.Draw(tile).text((4,4),name,fill='white');tiles.append(tile)
    canvas=Image.new('RGB',(320*max(1,len(tiles)),370),'#152533')
    ImageDraw.Draw(canvas).text((4,3),title,fill='white')
    for i,t in enumerate(tiles): canvas.paste(t,(i*320,26))
    stream=io.BytesIO();canvas.save(stream,format='PNG');atomic(path,stream.getvalue())


def preview(snapshot,cache,output,run=None):
    snapshot=Path(snapshot);cache=Path(cache);out=Path(output)
    meta=verify_cache(cache,snapshot,full=False)
    require(not out.exists(),'Preview exists; choose a new directory')
    out.mkdir(parents=True)
    rows=read(cache/'frames.json');fixed=set(read(cache/'fixed_samples.json'));metrics=[];missing=[]
    # Fixed train/inner pictures are qualitative only, not outer evaluation metrics.
    # Generate their best predictions too, using the same restored native grid.
    extra={}
    if run:
        from .learning import Model,Frames,restore
        from tools.downstream_seg_benchmark.engine import load
        import torch
        seed=Path(run)/'seed42';plan=read(seed/'plan.json') if (seed/'plan.json').is_file() else None
        if plan:
            require(plan['cache_identity']==meta['identity'],'Atlas/run cache mismatch')
            require(plan['sources']==meta['sources'],'Atlas/run source identity mismatch')
            selected=[r for r in rows if r['sample_id'] in fixed]
            for arm in ARMS:
                folder=seed/arm
                if not (folder/'complete.json').is_file(): continue
                done=read(folder/'complete.json');require(sha(folder/'best.pth')==done['best_sha256'],'Atlas checkpoint changed')
                from .core import DUAL
                model=Model(plan['config'],arm in DUAL,42).eval()
                model.load_state_dict(load(folder/'best.pth')['model']);dataset=Frames(snapshot,cache,selected,arm,plan['config']['size'])
                with torch.inference_mode():
                    for i,r in enumerate(selected):
                        prediction=model(dataset[i][0][None])
                        extra[(arm,r['sample_id'])]={k:restore(torch.sigmoid(prediction[k]),tuple(r['shape']),plan['config']['size']) for k in ('layer','vessel')}
    for r in rows:
        if r['sample_id'] not in fixed: continue
        sid,pid=r['sample_id'],r['position_id'];dest=out/r['split']/pid
        arrays={k:np.load(cache/v['path'],allow_pickle=False) for k,v in r['views'].items()}
        ref=plane(snapshot/'references'/f'{pid}.png').astype(np.float32)/255.
        require(ref.shape==arrays['noisy'].shape,'Reference/native mismatch')
        arrays['clean_reference']=ref
        for name in ('layer','vessel','layer_valid','vessel_valid','coverage','uncertain'):
            arrays[name]= (plane(snapshot/'labels'/pid/f'{name}.png')>0).astype(np.float32)
        arrays['noisy_minus_mild']=arrays['noisy']-arrays['mild']
        arrays['auto_minus_mild']=arrays['auto']-arrays['mild']
        for name,a in arrays.items():
            npy(dest/f'{name}.npy',a.astype(np.float32))
            # Signed diagnostics have a fixed, explicitly marked [-.5,.5] window.
            image=gray(a+.5) if name in ('noisy_minus_mild','auto_minus_mild') else gray(a)
            png(dest/f'{name}.png',image)
            require(np.array_equal(plane(dest/f'{name}.png'),image),'PNG/array mismatch')
        x,m,a=arrays['noisy'],arrays['mild'],arrays['auto']
        layer=arrays['layer']>0;v=arrays['vessel']>0;box=crop_box(layer)
        overlay=rgb(x);overlay[contour(layer)]=[255,220,40];overlay[contour(v)]=[255,60,70]
        png(dest/'GT_contours.png',overlay)
        title=f'{pid} {sid} {r["split"]} native{x.shape} alpha=.25 beta=.5 W={meta["ac_config"]["output_window"]:.5g} D2={meta["denoiser_sha256"][:12]}'
        panels=[(n,gray(arrays[k])) for n,k in [('noisy','noisy'),('mild a=.25','mild'),('AC(mild) b=.5','auto'),('clean/reference','clean_reference')]]
        sheet(dest/'input_stages.png',panels,title)
        sheet(dest/'input_stages_crop.png',panels,title,box)
        segmentation=[('GT contours',overlay)]
        for arm in ARMS:
            p=Path(run)/'seed42'/arm/'outer_best/predictions'/f'{sid}.npz' if run else None
            if (arm,sid) in extra or (p and p.is_file() and r['split']=='outer'):
                pred=extra[(arm,sid)] if (arm,sid) in extra else np.load(p,allow_pickle=False)
                lp,vp=pred['layer'],pred['vessel']
                vv=arrays['vessel_valid']>0;pv=(vp>=.5)&vv;gt=v&vv
                error=np.zeros((*pv.shape,3),np.uint8)
                error[pv&gt]=[0,220,70];error[pv&~gt]=[255,50,50];error[~pv&gt]=[60,100,255];error[~vv]=[140,60,200]
                over=rgb(x);over[contour(lp>=.5)]=[255,220,40];over[pv]=(.6*over[pv]+.4*np.array([255,60,70])).astype(np.uint8)
                for name,value in (('layer_probability',lp),('vessel_probability',vp)):
                    npy(dest/arm/f'{name}.npy',value.astype(np.float32));png(dest/arm/f'{name}.png',gray(value))
                for name,value in (('layer_mask',lp>=.5),('vessel_mask',vp>=.5)):
                    png(dest/arm/f'{name}.png',value.astype(np.uint8)*255)
                png(dest/arm/'overlay.png',over);png(dest/arm/'TP_FP_FN.png',error)
                segmentation.extend([(arm+' overlay',over),(arm+' TP/FP/FN',error)])
            else:
                tile=Image.new('RGB',(x.shape[1],x.shape[0]),'#152533')
                label='PENDING TRAINING' if not run or not (Path(run)/'seed42'/arm/'complete.json').is_file() else 'NOT OUTER EVALUATION / PENDING EVALUATION'
                ImageDraw.Draw(tile).text((10,20),arm+' '+label,fill='white')
                segmentation.append((arm,np.asarray(tile)));missing.append(dict(sample_id=sid,arm=arm,status=label))
        sheet(dest/'segmentation_comparison.png',segmentation,title)
        sheet(dest/'segmentation_comparison_crop.png',segmentation,title,box)
        for name in ('noisy','denoised','mild','auto'):
            y=arrays[name];rmse=float(np.sqrt(np.mean((y-ref)**2)))
            edge=float(np.mean(np.abs(sobel(y,axis=0)-sobel(ref,axis=0))+np.abs(sobel(y,axis=1)-sobel(ref,axis=1))))
            metrics.append(dict(sample_id=sid,position_id=pid,split=r['split'],view=name,
                psnr=float(min(120,peak_signal_noise_ratio(ref,y,data_range=1))),
                ssim=float(structural_similarity(ref,y,data_range=1)),rmse=rmse,
                sobel_l1_error=edge,interpretation='AC gray mapping: PSNR decline alone is not segmentation failure'))
        write(dest/'identity.json',dict(sample_id=sid,position_id=pid,split=r['split'],
            source_sha256=r['sha256'],reference_sha256=sha(snapshot/'references'/f'{pid}.png'),
            cache_identity=meta['identity'],d2_sha256=meta['denoiser_sha256'],alpha=.25,beta=.5,gamma=1,
            ac_config=meta['ac_config'],geometry='native full depth; no resize before AC',
            main_window=[0,1],signed_diagnostic_window=[-.5,.5],crop_native_xyxy=list(box)))
    table(out/'restoration_preview_metrics.csv',metrics);write(out/'pending_segmentation.json',missing)
    write(out/'preview_manifest.json',dict(positions=len(fixed),fixed_samples=sorted(fixed),
        scope='fixed 20 development positions; no favorable selection; not a model efficacy result',
        cache_identity=meta['identity'],new_sealed_test_assets_opened=0))
    return dict(status='preview_generated',positions=len(fixed),path=str(out),segmentation_pending=bool(missing))


def summarize(run,output):
    run=Path(run);out=Path(output);require(not out.exists(),'Report exists; choose new output')
    out.mkdir(parents=True);missing=[];completion=[]
    for arm in ARMS:
        folder=run/'seed42'/arm
        for kind in ('best','last'):
            p=folder/f'outer_{kind}/complete.json'
            ok=p.is_file() and (folder/'complete.json').is_file()
            if ok:
                done=read(folder/'complete.json');ev=read(p)
                ok=(sha(folder/f'{kind}.pth')==done[kind+'_sha256']==ev['checkpoint_sha256']
                    and sha(p.parent/'per_frame.csv')==ev['metrics_sha256'])
                ok=ok and all(sha(p.parent/'predictions'/f)==h for f,h in ev['prediction_sha256'].items())
            completion.append(dict(arm=arm,checkpoint=kind,complete=ok))
            if not ok: missing.append(arm+'/'+kind)
    table(out/'completion_matrix.csv',completion)
    result=dict(status='incomplete' if missing else 'complete_fold0_seed42',missing=missing,
        scope='fold0 seed42 development; outer not used for segmentation checkpoint selection; D2/teacher exposure audit still applies',
        full_cv_complete=False,independent_unit='position; patient identity unavailable',
        repeated_frames_are_not_independent=True)
    if (run/'seed42/plan.json').is_file():
        plan=read(run/'seed42/plan.json')
        write(out/'resolved_config.json',plan['config'])
    if not missing:
        markdown=[];selection=[]
        plan=read(run/'seed42/plan.json')
        write(out/'training_plan.json',plan)
        from tools.downstream_seg_benchmark.engine import load
        for arm in ARMS:
            folder=run/'seed42'/arm
            for kind in ('best','last'):
                state=load(folder/f'{kind}.pth');ev=read(folder/f'outer_{kind}/complete.json')
                selection.append(dict(arm=arm,checkpoint=kind,epoch=state['epoch'],
                    score=state.get('score',state.get('best')),sha256=sha(folder/f'{kind}.pth'),
                    threshold=.5,frames=ev['frames'],monitor='inner position-equal vessel soft Dice',
                    tie='earliest; strictly greater updates',boundary_units='native pixels',
                    last_score_note='last best field is best-so-far, NOT last score'))
        table(out/'checkpoint_selection.csv',selection)
        for kind in ('best','last'):
            frames=pd.concat([read_metrics(run/'seed42'/a/f'outer_{kind}/per_frame.csv') for a in ARMS],ignore_index=True)
            keys=['fold','seed','arm','position_id','group_id']
            metrics=[c for c in frames.select_dtypes('number').columns if c not in ('fold','seed')]
            positions=frames.groupby(keys,as_index=False)[metrics].mean()
            summary=positions.groupby(['seed','arm'],as_index=False)[metrics].mean()
            gains=paired_tables(positions,[m for m in metrics if m!='pred_layer_vessel_dice'],COMPARISONS)
            gain_summary=gains.groupby(['comparison','metric','unit'],as_index=False).agg(
                mean=('gain','mean'),improved_positions=('gain',lambda x:int((x>0).sum())),positions=('gain','size'))
            for name,df in [('per_frame',frames),('per_position',positions),('position_equal',summary),('paired_gains',gains),('gain_summary',gain_summary)]:
                atomic(out/f'{kind}_{name}.csv',df.to_csv(index=False).encode('utf-8-sig'))
            failure=positions.sort_values('vessel_dice').groupby('arm').head(2)
            atomic(out/f'{kind}_failure_positions.csv',failure.to_csv(index=False).encode('utf-8-sig'))
            markdown.append('\n### '+kind+'\n\n```text\n'+summary[['arm','vessel_dice','precision','recall','layer_dice']].to_string(index=False)+'\n```\n')
        text='## Fold0 seed42 development results\n\nBest is primary; last is sensitivity. No blind-test or multi-seed generalization claim.\n\n'
        text+=''.join(markdown)
    else: text='## INCOMPLETE — no efficacy claim\n\nMissing training/evaluations: '+', '.join(missing)+'\n'
    text+='\nM0>N0 supports mild input; MA0>M0 or NMA>NM is required for incremental AC benefit. NMA>N0 alone is insufficient. No matched brightness control in the six arms.\n'
    atomic(out/'summary.md',text.encode('utf8'));write(out/'report_status.json',result)
    return result


def package(cache,preview_dir,report,output,run=None):
    cache=Path(cache);preview_dir=Path(preview_dir);report=Path(report);target=Path(output)
    require(not target.exists(),'ZIP exists; choose new filename')
    status=read(report/'report_status.json') if (report/'report_status.json').is_file() else dict(status='incomplete',missing=['quantitative_report'])
    require(preview_dir.is_dir(),'Real preview absent: cannot package a preview claim')
    target.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(scope='fold0 seed42',status=status,full_cv_complete=False,
                  excludes='weights/optimizer/full repeated-frame cache',files={})
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        sources=[(cache,'provenance'),(preview_dir,'preview'),(report,'report')]
        if run: sources.append((Path(run),'training_records'))
        for base,prefix in sources:
            if not base.is_dir(): continue
            for p in sorted(base.rglob('*')):
                if not p.is_file() or p.suffix not in ('.json','.yaml','.csv','.md','.png','.npy'): continue
                if prefix=='training_records' and ('predictions' in p.parts or p.suffix=='.npy'): continue
                if base==cache and (p.suffix=='.npy' or 'arrays' in p.relative_to(base).parts): continue
                if p.suffix=='.npy' and not any(s in p.parts for s in ('pku_0001','pku_0004','pku_0005')): continue
                name=prefix+'/'+p.relative_to(base).as_posix();z.write(p,name);manifest['files'][name]=sha(p)
        z.writestr('PACKAGE_STATUS.json',__import__('json').dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as z: require(z.testzip() is None,'ZIP CRC failed')
    result=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),status=status['status'])
    write(target.with_suffix('.json'),result);return result
