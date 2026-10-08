"""Preregister atlas roles from Noisy and canonical GT, never method metrics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label, binary_erosion

from .common import KEY, require
from .data import image_array,mask_array


def fixed_samples(data):
    noisy=data[(data.method_id=='noisy_identity')&(data.split=='test')]
    require(len(noisy)>0,'Noisy required for atlas registration')
    selected=[]
    for position,group in noisy.groupby('position_id',sort=True):
        group=group.sort_values(['frame_id','sample_id'])
        scores=[]
        for row in group.to_dict('records'):
            layer=mask_array(row['layer_mask_path'])>0
            vessel=mask_array(row['vessel_mask_path'])>0
            if row.get('label_valid_mask_path'):vessel &= mask_array(row['label_valid_mask_path'])>0
            components,n=label(vessel,structure=np.ones((3,3)))
            areas=np.bincount(components.ravel(),minlength=n+1)
            small=int(((areas[1:]>0)&(areas[1:]<64)).sum())
            x=image_array(row['path']);gy,gx=np.gradient(x)
            edge=layer^binary_erosion(layer)
            strength=float(np.hypot(gx,gy)[edge].mean()) if edge.any() else float('inf')
            scores.append(dict(row=row,small=small,edge_strength=strength))
        middle=scores[len(scores)//2]
        rich=sorted(scores,key=lambda s:(-s['small'],s['row']['sample_id']))[0]
        weak=sorted(scores,key=lambda s:(s['edge_strength'],s['row']['sample_id']))[0]
        for role,item in [('middle',middle),('small_vessel_rich',rich),('weak_boundary',weak)]:
            row=item['row'];mask=mask_array(row['layer_mask_path'])>0
            yy,xx=np.nonzero(mask);size=min(128,int(row['height'])//4)
            cy=int(np.median(yy)) if len(yy) else int(row['height'])//2
            cx=int(np.median(xx)) if len(xx) else int(row['width'])//2
            selected.append({**{k:row[k] for k in KEY},'role':role,'crop_size':size,
                'crop_x':int(np.clip(cx-size//2,0,int(row['width'])-size)),
                'crop_y':int(np.clip(cy-size//2,0,int(row['height'])-size)),
                'small_component_count':item['small'],'noisy_boundary_gradient':item['edge_strength'],
                'selection_source':'Noisy+GT only before method scoring','gamma':1.,'gray_min':0.,'gray_max':1.,'resize':'nearest'})
    return selected
