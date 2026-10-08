from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

KEY = ['dataset', 'split', 'position_id', 'frame_id', 'sample_id']
PILOT = ['noisy_identity', 'nafnet_paired', 'sabids_current', 'clean_oracle']
FORMAL = ['noisy_identity', 'clean_oracle', 'ksvd_self', 'tv_chambolle', 'nlm',
          'bm3d_standard', 'dncnn_paired', 'tcfl_dncnn', 'nafnet_paired', 'sabids_current']
DEEP = {'nafnet_paired', 'dncnn_paired', 'tcfl_dncnn', 'sabids_current'}
TRAIN_ORDER = ['noisy_identity','clean_oracle','nafnet_paired','sabids_current','dncnn_paired',
               'bm3d_standard','tv_chambolle','nlm','ksvd_self','tcfl_dncnn']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    temporary=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temporary,index=False)
    os.replace(temporary,path)


def source_hash():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(Path(__file__).parent.glob('*.py')) + [
        root / 'sabids/models/blocks.py', root / 'sabids/losses/common.py']
    return digest({str(p.relative_to(root)): sha(p) for p in paths})


def require(condition, reason):
    if not condition:
        raise ValueError(reason)
