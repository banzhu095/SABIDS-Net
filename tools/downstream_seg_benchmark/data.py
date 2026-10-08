from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from .common import require
from tools.denoise_result_review.image_io import decode_lossless


def image_array(path):
    a = decode_lossless(path)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / np.iinfo(a.dtype).max
    else:
        a = a.astype(np.float32)
    require(np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1, f'Image outside fixed [0,1]: {path}')
    return a


def mask_array(path):
    with Image.open(path) as image:
        return np.asarray(image).copy()


class Inputs(Dataset):
    def __init__(self, rows, size, plan=None, allow_test=False):
        require(allow_test or not any(r['split'] == 'test' for r in rows), 'Test access requires checkpoint lock')
        self.rows, self.size, self.plan = rows, size, plan

    def __len__(self):
        return len(self.plan) if self.plan is not None else len(self.rows)

    def __getitem__(self, index):
        item = self.plan[index] if self.plan is not None else None
        row = self.rows[item['index'] if item else index]
        x = image_array(row['path'])
        layer = mask_array(row['layer_mask_path']) > 0
        vessel = mask_array(row['vessel_mask_path']) > 0
        valid = np.ones_like(layer)
        if row.get('multiclass_label_path'):
            valid &= mask_array(row['multiclass_label_path']) != 255
        if row.get('label_valid_mask_path'):
            valid &= mask_array(row['label_valid_mask_path']) > 0
        vv = valid.copy()
        if row.get('vessel_valid_mask_path'):
            vv &= mask_array(row['vessel_valid_mask_path']) > 0
        arrays = [x, layer, vessel, valid, vv]
        if item and item['crop_size']:
            y, xx, size = item['y'], item['x'], item['crop_size']
            arrays = [a[y:y+size, xx:xx+size] for a in arrays]
        tensors = []
        for i, a in enumerate(arrays):
            t = torch.from_numpy(a.astype(np.float32))[None, None]
            if self.size and tuple(t.shape[-2:]) != (self.size,self.size):
                require(not item or not item.get('crop_size'),'Native training patch must not be resampled')
                # PKU37 audit requires square images. No padding is introduced.
                t = F.interpolate(t, size=(self.size, self.size), mode='bilinear' if i == 0 else 'nearest', **({'align_corners': False} if i == 0 else {}))
            t = t[0]
            if item:
                if item['flip_x']: t = t.flip(-1)
                if item['flip_y']: t = t.flip(-2)
            tensors.append(t)
        return (*tensors, row['sample_id'], row['position_id'])
