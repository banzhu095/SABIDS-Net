from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from tools.downstream_seg_benchmark.common import read_json,require,sha,write_csv,write_json
from .tables import ORDER,LABELS,MAIN,PAIRS,POSITIONS,RATIOS,SEEDS


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    folder=out/'figures';folder.mkdir(exist_ok=True)
    m=pd.read_csv(out/'table2_main_results.csv').set_index('method_id').loc[ORDER]
    names=[LABELS[x].replace(' Stage-1/D0','\nStage-1/D0') for x in ORDER]
    x=np.arange(10)
    def save(fig,name):fig.tight_layout();fig.savefig(folder/name,dpi=160);plt.close(fig)
    def bars(cols,name,title,unit='%'):
        fig,ax=plt.subplots(figsize=(15,5))
        for i,c in enumerate(cols):
            scale=100 if c in RATIOS else 1
            ax.bar(x+(i-(len(cols)-1)/2)*.34,m[c+'_mean']*scale,width=.32,
                   yerr=m[c+'_std']*scale,capsize=3,label=c+' (between-seed SD)')
        ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylabel(unit);ax.set_title(title+'; n=3 independent positions, 3 segmentation seeds')
        ax.legend();save(fig,name)
    bars(['vessel_dice','vessel_roi_dice'],'full_vs_roi_dice.png','Full-image vs GT-layer ROI Vessel Dice')
    bars(['layer_dice'],'layer_dice.png','Layer Dice')
    bars(['precision','recall'],'precision_recall.png','Precision and Recall')
    bars(['vessel_outside_gt_layer_fraction'],'outside_vessel_fraction.png','Outside predicted vessels / all predicted vessels')
    gp=pd.read_csv(out/'paired_by_position.csv');gs=pd.read_csv(out/'paired_by_seed.csv')
    for d,index,file,title in [(gp,'position_id','sabids_nafnet_position_differences.png','Position-paired SABIDS - NAFNet'),
                                (gs,'segmentation_seed','three_seed_paired_differences.png','Seed-paired SABIDS - NAFNet')]:
        d=d[d.comparison=='sabids_current - nafnet_paired'];fig,ax=plt.subplots(figsize=(9,5));xx=np.arange(len(d))
        for i,c in enumerate(['layer_dice','vessel_dice','vessel_roi_dice','precision','recall']):
            ax.bar(xx+(i-2)*.15,d[c]*100,.14,label=c)
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(xx,d[index].astype(str));ax.set_ylabel('percentage points (pp)');ax.set_title(title);ax.legend(fontsize=8);save(fig,file)
    t=pd.read_csv(out/'table5_training_selection.csv')
    d=t.groupby('method_id')[['best_validation_roi_dice','last_validation_roi_dice','best_validation_layer_dice','last_validation_layer_dice']].mean().loc[ORDER]
    fig,axes=plt.subplots(1,2,figsize=(17,5))
    for ax,part in zip(axes,['roi','layer']):
        for i,time in enumerate(['best','last']):ax.bar(x+(i-.5)*.35,d[f'{time}_validation_{part}_dice']*100,.34,label=time)
        ax.set_xticks(x,names,rotation=35,ha='right');ax.set_ylabel('%');ax.set_title('Validation '+part+' Dice: selected epoch vs epoch 60');ax.legend()
    save(fig,'best_vs_epoch60.png')
    joint=pd.read_csv(out/'table6_denoise_segmentation_methods.csv').dropna(subset=['psnr'])
    fig,ax=plt.subplots(figsize=(9,6))
    for row in joint.to_dict('records'):
        ax.scatter(row['psnr'],row['vessel_dice']*100);ax.annotate(LABELS[row['method_id']],(row['psnr'],row['vessel_dice']*100),fontsize=8)
    ax.set_xlabel('Denoising PSNR (dB), primary denoiser seed on SAME 3 positions');ax.set_ylabel('Vessel Dice (%)');ax.set_title('Descriptive relationship, no causal or significance claim');save(fig,'psnr_vs_vessel_dice.png')
    for seed in SEEDS:
        fig,axes=plt.subplots(5,2,figsize=(13,15))
        for method,ax in zip(ORDER,axes.ravel()):
            curve=pd.read_csv(out/'training_curves'/method/str(seed)/'training_curve.csv')
            ax.plot(curve.epoch,curve.val_vessel_roi_dice*100,'o-',label='validation ROI Dice')
            ax.plot(curve.epoch,curve.val_layer_dice*100,'s-',label='validation Layer Dice')
            best=t[(t.method_id==method)&(t.segmentation_seed==seed)].iloc[0]
            ax.axvline(best.best_epoch,color='gray',ls='--');ax.set_title(LABELS[method]);ax.set_xlabel('epoch');ax.set_ylabel('%');ax.legend(fontsize=7)
        save(fig,f'training_dynamics_seed{seed}.png')


def markdown_table(frame,columns,percent=True):
    d=frame[columns].copy()
    for c in d.select_dtypes('number'):
        base=c.removesuffix('_mean').removesuffix('_std')
        d[c]=d[c].map(lambda v:'不适用' if pd.isna(v) else f'{v*100:.2f}%' if percent and base in RATIOS else f'{v:.3f}')
    return d.to_markdown(index=False)


def report(out):
    m=pd.read_csv(out/'table2_main_results.csv').set_index('method_id');p=pd.read_csv(out/'table3_position_results.csv')
    g=pd.read_csv(out/'paired_by_position.csv');mean=pd.read_csv(out/'paired_mean.csv').set_index('comparison')
    tr=pd.read_csv(out/'table5_training_selection.csv');audit=read_json(out/'table_audit.json')
    a=mean.loc['sabids_current - nafnet_paired'];b=mean.loc['sabids_current - noisy_identity']
    sd=pd.read_csv(out/'paired_by_seed.csv');sn=sd[sd.comparison=='sabids_current - nafnet_paired']
    decomposition=pd.read_csv(out/'dice_gain_decomposition.csv')
    dm=decomposition[(decomposition.scope=='method')&(decomposition.comparison=='sabids_current - nafnet_paired')].iloc[0]
    write_json(out/'headline_results.json',dict(sabids_minus_nafnet_full_pp=float(a.vessel_dice*100),
        sabids_minus_nafnet_roi_pp=float(a.vessel_roi_dice*100),outside_penalty_reduction_pp=float(dm.outside_penalty_reduction*100),
        sabids_minus_noisy_full_pp=float(b.vessel_dice*100),sabids_minus_noisy_roi_pp=float(b.vessel_roi_dice*100),
        positive_full_seed_count=int((sn.vessel_dice>0).sum()),positive_roi_seed_count=int((sn.vessel_roi_dice>0).sum())))
    text='# 统一下游分割实验阶段总结\n\n'
    text+='## 1. 目的与范围\n\n比较各降噪图作为单输入的下游分割表现。10组输入使用相同分割架构，按42、123、2026分别训练。SABIDS-current仅指Stage-1/D0输出，不评价Joint、D→S、S→D或UGBI。TCFL-DnCNN明确为unpaired；Clean reference是参考上限，不参与普通降噪算法排名。\n\n'
    text+='## 2. 数据与公平性协议\n\ntrain为12位置/552帧，val为5位置/227帧，test为3位置/150帧。共30/30训练组、4500条逐帧测试记录。每组60轮/33120次总更新，384×384原生训练裁剪、640×640测试，固定阈值0.5，不追加后处理。同一分割seed跨方法共享初始化、采样、增强与数据计划哈希。深度降噪仅用primary seed42。\n\n主聚合为frame → position within segmentation seed → seed average → method，病例数是3而不是150或9。均值、种子间SD和范围保留独立定义。原汇总已与逐帧复算逐项核对。\n\n'
    text+='## 3. 全方法主结果\n\n均值按百分数显示，SD为三分割种子的SD。完整13项主指标及SD见CSV/Excel，空间误差单位px。非Clean最优与Clean参考单独标记。\n\n'
    text+=markdown_table(m.loc[ORDER].reset_index(),['display_name','layer_dice_mean','vessel_dice_mean','vessel_roi_dice_mean','precision_mean','recall_mean'])+'\n\n'
    nonclean=m.drop(index='clean_oracle')
    text+=f"非Clean全图Vessel Dice最高为{LABELS[nonclean.vessel_dice_mean.idxmax()]}；Layer Dice最高为{LABELS[nonclean.layer_dice_mean.idxmax()]}；GT层内ROI Dice最高为{LABELS[nonclean.vessel_roi_dice_mean.idxmax()]}。\n\n"
    text+='## 4. SABIDS与NAFNet配对结果\n\n'
    text+=f'SABIDS−NAFNet：全图Vessel Dice **{a.vessel_dice*100:+.3f} pp**，GT层内ROI Dice **{a.vessel_roi_dice*100:+.3f} pp**，Layer Dice **{a.layer_dice*100:+.3f} pp**，Precision **{a.precision*100:+.3f} pp**，Recall **{a.recall*100:+.3f} pp**。SABIDS的Recall并未更高。SABIDS−Noisy全图Dice {b.vessel_dice*100:+.3f} pp，ROI Dice {b.vessel_roi_dice*100:+.3f} pp。\n\n'
    text+='## 5. 全图Dice与ROI Dice\n\n定义outside_penalty = ROI Dice − 全图Dice。差值恒等式为delta_full_dice = delta_roi_dice − delta_outside_penalty。'
    text+=f'相对NAFNet，SABIDS的层外Dice惩罚减少{dm.outside_penalty_reduction*100:.3f} pp，抵消ROI Dice下降{abs(a.vessel_roi_dice*100):.3f} pp，使全图Dice净增加{a.vessel_dice*100:.3f} pp。这与层外错误抑制、Precision–Recall权衡和Layer定位差异一致，**是指标分解，不是已证明的因果机制**。\n\n'
    text+='## 6. 位置与种子稳定性\n\n'
    gp=g[g.comparison=='sabids_current - nafnet_paired']
    tab=gp[['position_id','layer_dice','vessel_dice','vessel_roi_dice','precision','recall']].copy()
    for c in tab.columns[1:]:tab[c]=tab[c].map(lambda x:f'{100*x:+.3f} pp')
    text+=tab.to_markdown(index=False)+'\n\n'
    text+=f"三seed全图Dice正方向为{int((sn.vessel_dice>0).sum())}/3，ROI Dice正方向为{int((sn.vessel_roi_dice>0).sum())}/3。完整逐seed和逐位置正负方向另表保留。优势主要集中于pku_0031；leave-one-position-out只是位置敏感性，不删除不利位置、不替换主结果。\n\n"
    text+='## 7. Precision–Recall、层外FP与Layer定位\n\n层外预测血管占比=层外预测血管/全部预测血管；层外血管FP率=层外预测血管/有效层外像素。两者不可混用。必须同时看全图和GT层内ROI Dice、Precision、Recall、Layer Dice、上下边界与厚度MAE。预测导出保留原始0.5二值结果，不以GT或预测Layer裁剪正式Vessel mask。\n\n'
    text+='## 8. 训练选模与后期动态\n\n'
    text+=f"{audit['best_metadata_verified']}/30个best checkpoint已直接读取元数据，{audit['best_epochs_5_to_15']}/30位于第5–15轮；best epoch范围{int(tr.best_epoch.min())}–{int(tr.best_epoch.max())}，不是把总预算33120误写成best更新数。逐组保存best checkpoint实际updates、验证ROI/Layer Dice、第60轮分数、下降量及AMP重放次数。\n\n"
    text+=f"平均best→60轮验证ROI Dice下降{tr.best_to_last_roi_drop.mean()*100:.3f} pp；同选模epoch的Layer Dice相对第60轮变化为{tr.best_to_last_layer_drop.mean()*100:+.3f} pp。早期最佳与后期ROI下降支持血管泛化退化/过拟合解释，但Layer训练动态可能不同，不能仅凭test指标选择新epoch或重调超参数。本轮不重新选模。\n\n"
    text+='## 9. 降噪与分割对应\n\n联合表只使用本次三个测试位置及实际深度降噪primary seed42。PSNR/SSIM/EPI/高频能量与分割的关系是描述性对应，不能混入历史六位置或三降噪seed平均。更多高频能量不等于更多真实血管，需要联合边缘、错误图及GT解释。Clean无来源降噪像素指标时保留不适用，不虚构PSNR值。\n\n'
    text+='## 10. 限制\n\n只有3个独立测试位置，不声称统计显著性或总体优越性。strict Boundary Dice按严格像素边界重合，不是容差边界Dice。无GT尺寸分组的NaN保留不适用。pku_0017、pku_0031预登记small_vessel_rich角色小组件数为0，图册标记“预登记样本，未检出小组件”，小血管证据不足。Clean仅参考上限。实验不是完整SABIDS Joint验证。\n\n'
    text+='## 11. 下一步\n\n补齐尚未标注的pku_0025、pku_0038、pku_0043测试位置，按冻结checkpoint/阈值进行独立扩展评估，之后再扩大独立病例数。潜在早停、阈值或后处理改变只能另做预登记实验，不能反向改变本次结果。本轮不启动训练或新推理。\n'
    (out/'downstream_seg_stage_summary.md').write_text(text,encoding='utf-8')
    takeaways=[f'SABIDS非Clean全图Dice {m.loc["sabids_current","vessel_dice_mean"]*100:.2f}%、Layer Dice {m.loc["sabids_current","layer_dice_mean"]*100:.2f}%，均领先。',
        f'相对NAFNet全图Dice {a.vessel_dice*100:+.2f} pp，但GT层内ROI Dice {a.vessel_roi_dice*100:+.2f} pp。',
        f'层外Dice惩罚减少{dm.outside_penalty_reduction*100:.2f} pp；这是指标分解而非因果证明。',
        'SABIDS Recall没有更高，应联合Precision、层外FP及Layer定位解释。',
        '收益具有位置异质性，主要集中于pku_0031，保留全部位置与种子结果。',
        f'{audit["best_epochs_5_to_15"]}/30个选定best位于第5–15轮，验证ROI后期下降需进一步研究。',
        'TCFL为unpaired，Clean为参考上限；更多高频不代表更多真血管。',
        '仅3个独立测试位置，不声称显著性，不等同完整SABIDS Joint验证。']
    (out/'presentation_takeaways.md').write_text('# 组会结论\n\n'+'\n'.join('- '+x for x in takeaways)+'\n',encoding='utf-8')
    (out/'limitations_and_next_steps.md').write_text('# 限制与下一步\n\n'+text.split('## 10. 限制\n\n')[1],encoding='utf-8')


def finish_immutable(out):
    before=read_json(out/'immutable_before.json');after={p:sha(p) for p in before}
    changes=[p for p in before if before[p]!=after[p]]
    write_json(out/'immutable_after.json',after)
    write_json(out/'original_run_mutation_audit.json',dict(protected_files=len(before),changed=changes,passed=not changes,
        scope='Original config, weights, curves, metric tables, locks, shared randomness and test access log'))
    require(not changes,'Original experiment files changed')


def zip_package(out,filename,files):
    files=sorted(set(Path(p) for p in files))
    require(not (out/filename).exists(),'Package exists; use a new review directory or explicit verified resume')
    manifest=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    buf=io.StringIO();pd.DataFrame(manifest).to_csv(buf,index=False);content=buf.getvalue().encode('utf-8')
    path=out/filename
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for p in files:z.write(p,p.relative_to(out).as_posix())
        z.writestr('PACKAGE_MANIFEST.csv',content)
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None,'ZIP CRC failure')
        for item in manifest:
            info=z.getinfo(item['path']);require(info.file_size==item['bytes'],'ZIP size mismatch')
            h=hashlib.sha256()
            with z.open(info) as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            require(h.hexdigest()==item['sha256'],'ZIP member SHA mismatch')
    receipt=dict(path=str(path.resolve()),bytes=path.stat().st_size,mib=path.stat().st_size/2**20,sha256=sha(path),
        files=len(manifest)+1,crc_passed=True,member_sizes_verified=True,member_sha256_verified=True)
    write_json(out/'package_receipts'/(filename+'.json'),receipt)
    return receipt


def packages(out,resume=False,predictions=True):
    finish_immutable(out)
    receipts=[]
    def make(name,files):
        receipt=out/'package_receipts'/(name+'.json')
        if resume and receipt.exists():
            d=read_json(receipt);require(sha(out/name)==d['sha256'],'Existing package changed');receipts.append(d);return
        print('package',name,flush=True);receipts.append(zip_package(out,name,files))
    if predictions:
        require((out/'prediction_export_audit.json').exists(),'Prediction export not completed')
        exports=out/'exports';mf=pd.read_csv(out/'prediction_export_manifest.csv')
        primary=mf[mf.segmentation_seed.isin([0,42])]
        write_csv(out/'prediction_export_primary_seed42.csv',primary)
        paths=[out/p for p in primary.relative_path.unique()]
        indexfiles=[out/n for n in ['prediction_export_primary_seed42.csv','public_asset_manifest.csv','prediction_export_audit.json','prediction_export_coverage.csv','prediction_export_missing.csv']]
        make('downstream_predictions_primary_seed42.zip',paths+indexfiles)
        for pos in POSITIONS:
            subset=primary[primary.position_id==pos];name='prediction_export_'+pos+'.csv';write_csv(out/name,subset)
            make('downstream_predictions_'+pos+'_seed42.zip',[out/p for p in subset.relative_path.unique()]+[out/name,out/'prediction_export_audit.json'])
    make('downstream_atlas_all_seeds.zip',[p for p in (out/'qualitative_atlas').rglob('*') if p.is_file()])
    # A portable analysis source snapshot is intentionally separate from original run code.
    snapshot=out/'analysis_source';snapshot.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*'):
        if p.suffix in ['.py','.mjs']:shutil.copyfile(p,snapshot/p.name)
    write_csv(out/'download_inventory.csv',receipts)
    download='# 下载路径\n\n'
    for d in receipts:download+=f"- `{d['path']}`：{d['mib']:.2f} MiB；SHA256 `{d['sha256']}`\n"
    download+=f'\n全量导出根目录：`{out.resolve()}/exports`。seed42存储在primary_seed42，seed123/2026在all_seeds；seed42的all_seeds索引引用主目录，不重复概率文件。\n'
    source=read_json(out/'review_scope.json')['source_run']
    download+='\n可选另打包其余种子（含通过manifest引用的公共输入和GT，不影响原run）：\n\n```bash\npython -m tools.downstream_seg_review.cli --run-dir "'+source+'" --output-dir "'+str(out.resolve())+'" --stage package-other-seeds --resume\n```\n'
    (out/'download_paths.md').write_text(download,encoding='utf-8')
    require((out/'downstream_seg_stage_summary.xlsx').exists(),'Artifact-tool workbook required before GPT packaging')
    files=[]
    excluded={'exports','qualitative_atlas','workbook_preview','package_receipts','node_modules','__pycache__'}
    for p in out.rglob('*'):
        rel=p.relative_to(out)
        if p.is_file() and not any(x in excluded for x in rel.parts) and p.suffix in ['.csv','.json','.md','.xlsx','.py','.mjs','.yaml','.png']:
            files.append(p)
    name='GPT_light_'+out.name+'.zip'
    make(name,files)
    require(receipts[-1]['bytes']<=50*2**20,'GPT light package exceeds 50 MiB; no quantitative rows may be dropped')
    write_csv(out/'download_inventory.csv',receipts)
    (out/'download_paths.md').write_text(download+f"\nGPT轻量包：`{receipts[-1]['path']}`，{receipts[-1]['mib']:.2f} MiB；SHA256 `{receipts[-1]['sha256']}`。\n",encoding='utf-8')
    write_json(out/'review_completion.json',dict(status='complete',packages=receipts,original_run_read_only=True,
        predictions_exported=predictions,workbook='downstream_seg_stage_summary.xlsx',no_new_inference=True,no_training=True))
    return receipts


def package_other_seeds(out):
    finish_immutable(out)
    mf=pd.read_csv(out/'prediction_export_manifest.csv');mf=mf[mf.segmentation_seed.isin([0,123,2026])]
    write_csv(out/'prediction_export_other_seeds.csv',mf)
    return zip_package(out,'downstream_predictions_other_seeds.zip',
        [out/p for p in mf.relative_path.unique()]+[out/'prediction_export_other_seeds.csv',out/'prediction_export_audit.json'])
