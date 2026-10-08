from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, label


def assd(a,b,valid):
    a=(a ^ binary_erosion(a)) & binary_erosion(valid,border_value=0)
    b=(b ^ binary_erosion(b)) & binary_erosion(valid,border_value=0)
    if not a.any() and not b.any():return 0.
    if not a.any() or not b.any():return float(np.hypot(*a.shape))
    return float((distance_transform_edt(~b)[a].sum()+distance_transform_edt(~a)[b].sum())/(a.sum()+b.sum()))


def component_recall(pred,gt,small=64,medium=256):
    components,n=label(gt,structure=np.ones((3,3)))
    area=np.bincount(components.ravel(),minlength=n+1)
    found=np.bincount(components[pred].ravel(),minlength=n+1)
    result={}
    for name,mask in [('small',(area<small)),('medium',(area>=small)&(area<medium)),('large',area>=medium)]:
        mask[0]=False
        # Pixel-weighted recall within GT connected-component size bins. No GT pixels => undefined.
        result[name+'_vessel_recall']=float(found[mask].sum()/area[mask].sum()) if area[mask].sum() else float('nan')
    return result


def dice(a, b):
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else 1.0


def metrics(lp, vp, layer, vessel, valid, vv):
    l, v = (lp >= .5) & valid, (vp >= .5) & vv
    gt_l, gt_v = layer & valid, vessel & vv
    tp, fp, fn = (v & gt_v).sum(), (v & ~gt_v).sum(), (~v & gt_v).sum()
    edge_valid = binary_erosion(vv, border_value=0)
    edge_v = (v ^ binary_erosion(v)) & edge_valid
    edge_gt = (gt_v ^ binary_erosion(gt_v)) & edge_valid
    boundaries = []
    for j in range(layer.shape[1]):
        a, b = np.flatnonzero(l[:, j]), np.flatnonzero(gt_l[:, j])
        if valid[:, j].all() and len(b):
            # Missing predicted boundary is penalized by the full image height.
            boundaries.append((abs(int(a[0])-int(b[0])), abs(int(a[-1])-int(b[-1])), abs(int(a[-1]-a[0])-int(b[-1]-b[0]))) if len(a) else (layer.shape[0],)*3)
    boundary = np.mean(boundaries, axis=0) if boundaries else [float('nan')]*3
    return dict(layer_dice=dice(l, gt_l), layer_iou=float((l&gt_l).sum()/max((l|gt_l).sum(),1)),
                layer_assd=assd(l,gt_l,valid),vessel_assd=assd(v,gt_v,vv),
                vessel_f1=float(2*tp/max(2*tp+fp+fn,1)),
                roi_fp=float((v&~gt_v&gt_l).sum()/max((gt_l&~gt_v&vv).sum(),1)),
                roi_fn=float((~v&gt_v&gt_l).sum()/max((gt_v&gt_l).sum(),1)),
                pred_layer_vessel_dice=dice(v&l,gt_v),
                vessel_outside_gt_layer_fp=float((v&~gt_l&vv).sum()/max((~gt_l&vv).sum(),1)),
                **component_recall(v,gt_v),vessel_dice=dice(v, gt_v),
                vessel_roi_dice=dice(v & gt_l, gt_v & gt_l),
                precision=float(tp / max(tp+fp, 1)), recall=float(tp / max(tp+fn, 1)),
                iou=float(tp / max(tp+fp+fn, 1)), boundary_dice=dice(edge_v, edge_gt),
                layer_outside_fp=float((l & ~gt_l & valid).sum()/max((~gt_l & valid).sum(), 1)),
                vessel_outside_gt_layer_fraction=float((v & ~gt_l).sum()/max(v.sum(), 1)),
                upper_boundary_mae=float(boundary[0]), lower_boundary_mae=float(boundary[1]), thickness_mae=float(boundary[2]))


METRICS = ['layer_dice', 'vessel_dice', 'vessel_roi_dice', 'precision', 'recall', 'iou', 'boundary_dice',
           'layer_outside_fp', 'vessel_outside_gt_layer_fraction', 'upper_boundary_mae', 'lower_boundary_mae', 'thickness_mae',
           'layer_iou','layer_assd','vessel_assd','vessel_f1','roi_fp','roi_fn','pred_layer_vessel_dice',
           'vessel_outside_gt_layer_fp','small_vessel_recall','medium_vessel_recall','large_vessel_recall']


def formal_tables(frame,run):
    from .common import write_csv,require
    position_seed,summary,gains=aggregate(frame)
    expected=frame.segmentation_seed.nunique()
    require((position_seed.groupby(['method_id','position_id']).size()==expected).all(),'Incomplete seed/position pairing')
    position=position_seed.groupby(['method_id','position_id'],as_index=False)[METRICS].mean()
    seed=position_seed.groupby(['method_id','segmentation_seed'],as_index=False)[METRICS].mean()
    gain_position=gains.groupby(['comparison','position_id'],as_index=False)[METRICS].mean()
    gain_seed=gains.groupby(['comparison','segmentation_seed'],as_index=False)[METRICS].mean()
    write_csv(run/'per_position_seed_metrics.csv',position_seed)
    write_csv(run/'per_position_metrics.csv',position)
    write_csv(run/'per_seed_metrics.csv',seed)
    write_csv(run/'per_method_metrics.csv',summary)
    write_csv(run/'paired_gains_by_position.csv',gain_position)
    write_csv(run/'paired_gains_by_seed.csv',gain_seed)
    rng=np.random.default_rng(2026);intervals=[]
    for comparison,group in gain_position.groupby('comparison',sort=True):
        indexes=rng.integers(len(group),size=(10000,len(group)))
        for metric in METRICS:
            values=group[metric].to_numpy(float)
            if not np.isfinite(values).all():
                intervals.append(dict(comparison=comparison,metric=metric,mean=None,ci_low=None,ci_high=None,
                                      positions=len(group),scope='undefined_size_bin_or_metric'))
                continue
            boot=values[indexes].mean(axis=1)
            low,high=np.quantile(boot,[.025,.975])
            intervals.append(dict(comparison=comparison,metric=metric,mean=values.mean(),ci_low=low,ci_high=high,
                                  positions=len(group),scope='descriptive_position_cluster_after_seed_mean'))
    write_csv(run/'bootstrap_confidence_intervals.csv',intervals)
    write_csv(run/'paired_method_differences.csv',intervals)
    return summary


def aggregate(frame):
    position = frame.groupby(['method_id', 'segmentation_seed', 'position_id'], as_index=False)[METRICS].mean()
    seed = position.groupby(['method_id', 'segmentation_seed'], as_index=False)[METRICS].mean()
    summary = seed.groupby('method_id')[METRICS].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    differences = []
    for method in sorted(set(position.method_id) - {'sabids_current'}):
        a = position[position.method_id == 'sabids_current'].set_index(['segmentation_seed', 'position_id'])
        b = position[position.method_id == method].set_index(['segmentation_seed', 'position_id'])
        if not a.index.equals(b.index):
            raise ValueError('Unpaired evaluation positions')
        for key, values in (a[METRICS]-b[METRICS]).iterrows():
            differences.append(dict(comparison='sabids_current - '+method, segmentation_seed=key[0], position_id=key[1], **values.to_dict()))
    return position, summary.reset_index(), pd.DataFrame(differences)


def classify(gains):
    """Frozen precedence resolves overlapping requested rules; no p-values."""
    if gains.empty:
        return 'not_evaluated'
    d = gains[METRICS].mean()
    mean, positive = d.vessel_roi_dice, int((gains.vessel_roi_dice > 0).sum())
    # 'Clearly worse FP' is preregistered as an absolute increase > 1 percentage point.
    harmful = d.precision < -.01 or d.vessel_outside_gt_layer_fraction > .01 or d.layer_outside_fp > .01
    if mean < -.005 or (d.recall > 0 and harmful):
        return 'not_supportive'
    if mean >= .005 and positive >= int(np.ceil(2*len(gains)/3)) and d.recall > 0 and not harmful and d.boundary_dice >= 0:
        return 'supportive'
    mixed = (gains.vessel_roi_dice > 0).any() and (gains.vessel_roi_dice < 0).any()
    if abs(mean) <= .005 and mixed:
        return 'inconclusive'
    return 'weak_support' if mean > 0 else 'inconclusive'
