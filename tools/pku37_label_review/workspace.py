from __future__ import annotations

import base64
import contextlib
import csv
import io
import json
import math
import os
import shutil
import threading
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label
import yaml

from sabids.experiments.attenuation_aux import (
    CACHE_VERSION, StableACConfig, array_sha256, file_sha256, stable_ac_v1,
)
from sabids.data.io import read_gray
from .display import resolve_parameters,compute_intermediates,render,auto_window,DISPLAY_CACHE_VERSION,LEGACY,WINDOW

STATUSES = ('not_started', 'modified', 'pending', 'confirmed', 'uncertain')
LOCK = threading.RLock()


class Conflict(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp_' + uuid.uuid4().hex)
    with temp.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def png_bytes(array):
    out = io.BytesIO(); Image.fromarray(np.asarray(array, dtype=np.uint8)).save(out, format='PNG')
    return out.getvalue()


def read_plane(path):
    with Image.open(path) as handle:
        return np.asarray(handle).copy()


def encode_mask(mask):
    return base64.b64encode(np.asarray(mask, dtype=np.uint8).tobytes()).decode('ascii')


def decode_mask(text, shape):
    raw = base64.b64decode(text, validate=True)
    if len(raw) != math.prod(shape):
        raise ValueError('Mask byte count does not match native dimensions')
    array = np.frombuffer(raw, np.uint8).reshape(shape).copy()
    if not np.isin(array, (0, 1)).all():
        raise ValueError('Mask must contain only0/1 bytes')
    return array.astype(bool)


def components(mask):
    return int(label(mask, structure=np.ones((3, 3)))[1])


def parameters(value, defaults):
    return resolve_parameters(value,defaults)


@contextlib.contextmanager
def transaction(root):
    """OS advisory lock releases on process exit, including a crashed server."""
    with LOCK:
        with (Path(root)/'.workspace.lock').open('a+b') as handle:
            if handle.tell() == 0: handle.write(b'0'); handle.flush()
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try: yield
            finally:
                handle.seek(0)
                if os.name == 'nt': msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize(project_root, atlas, destination, expected_positions=37):
    project_root, atlas, destination = map(lambda p: Path(p).resolve(), (project_root,atlas,destination))
    for protected_root in (project_root/'Data',project_root/'Label',project_root/'Manifests',project_root/'runs',atlas):
        if destination==protected_root or protected_root in destination.parents:
            raise ValueError(f'Workspace may not be placed inside protected assets: {protected_root}')
    if destination.exists(): raise FileExistsError(f'Workspace already exists: {destination}; use serve to resume')
    with (atlas/'selection_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_positions or len({row['position_id'] for row in rows}) != expected_positions:
        raise ValueError(f'Expected {expected_positions} distinct reference positions')
    config = yaml.safe_load((atlas/'resolved_ac_config.yaml').read_text(encoding='utf-8'))
    if config['version'] != CACHE_VERSION: raise ValueError('Requires validated ac_r2 atlas')
    # Fingerprint all requested original roots and the complete r2 atlas before
    # initialization, then verify them again. Never write through their paths.
    protected = []
    for folder in (project_root/'Data',project_root/'Label',project_root/'Manifests',atlas):
        protected += [{'path':str(p),'sha256':file_sha256(p)} for p in sorted(folder.rglob('*')) if p.is_file()]
    destination.mkdir(parents=True)
    for name in ('source_images','original_masks','edited_masks','validity','uncertain_regions',
                 'observation_images','labelme_working','revisions','candidate_labels','conflicts'):
        (destination/name).mkdir()
    atomic(destination/'source_manifest.csv',(atlas/'selection_manifest.csv').read_bytes())
    atomic(destination/'r2_config.yaml',(atlas/'resolved_ac_config.yaml').read_bytes())
    atomic(destination/'r2_source_inventory.json',(atlas/'source_file_inventory.json').read_bytes())
    atomic(destination/'protected_assets.json',json_bytes(protected))
    defaults = dict(beta=config['stable_ac_v1']['beta'], W=config['stable_ac_v1']['output_window'],gamma=1.,opacity=.35)
    index = dict(version='pku37_vessel_ac_review_v1',created_at=now(),project_root=str(project_root),
                 atlas=str(atlas),r2=CACHE_VERSION,default_parameters=defaults,
                 algorithm=config['stable_ac_v1'],positions={},states={},training_protocol_modified=False,
                 historical_test_access=read_json(atlas/'asset_inventory.json'))
    for row in rows:
        pid=row['position_id']; source=Path(row['reference_path'])
        native=read_plane(source)
        if native.dtype!=np.uint8 or native.ndim not in (2,3): raise ValueError(f'Unsupported reference gray domain: {source}')
        # Four real references are RGB TIFFs. Reuse precisely the same OpenCV
        # gray conversion and uint8 scaling as the validated r2 atlas; no resize.
        plane=np.rint(read_gray(source)*255).astype(np.uint8)
        shape=plane.shape; label_path=Path(row['multiclass_label_path']) if row['multiclass_label_path'] else None
        gt=read_plane(label_path) if label_path else None
        if gt is not None and (gt.shape!=shape or not np.isin(gt,(0,1,2,255)).all()):
            raise ValueError(f'Label geometry/class mismatch (no automatic stretch): {pid}')
        layer=np.isin(gt,(1,2)) if gt is not None else np.zeros(shape,bool)
        vessel=(gt==2) if gt is not None else np.zeros(shape,bool)
        valid=(gt!=255) if gt is not None else np.zeros(shape,bool)
        uncertain=(gt==255) if gt is not None else np.zeros(shape,bool)
        orig=destination/'original_masks'/pid;orig.mkdir()
        for name,mask in dict(layer=layer,vessel=vessel,valid=valid,uncertain=uncertain).items():
            atomic(orig/f'{name}.png',png_bytes(mask.astype(np.uint8)*255))
        if gt is not None: atomic(orig/'multiclass.png',png_bytes(gt))
        images=destination/'source_images'/pid;images.mkdir()
        atomic(images/'reference.png',png_bytes(plane))
        atomic(images/'reference_native.png',png_bytes(native))
        original_ref_sha=file_sha256(source)
        metadata=dict(**row,shape=list(shape),source_image_sha256=original_ref_sha,
                      native_geometry=dict(width=shape[1],height=shape[0],origin=[0,0],
                                           x_axis='right',y_axis='depth_down',transform='identity'),
                      had_original_label=gt is not None,original_label_sha256=file_sha256(label_path) if label_path else None,
                      correspondence='same native shape and filename; registration not proven',
                      validity_source='multiclass !=255' if gt is not None else 'unannotated: no coverage',
                      image_reading='sabids.data.io.read_gray; native geometry; uint8/255; RGB converted with OpenCV',
                      original_image_shape=list(native.shape),
                      repeat_transfer_allowed=False,source_labelme_json=None)
        metadata['binary_asset_comparison']={}
        for key,expected in (('layer_mask_path',layer),('vessel_mask_path',vessel)):
            if row.get(key):
                binary=read_plane(row[key])
                if binary.shape!=shape: raise ValueError(f'Binary label geometry mismatch: {pid}')
                metadata['binary_asset_comparison'][key]=dict(sha256=file_sha256(row[key]),pixel_difference=int(((binary>0)!=expected).sum()))
        # Locate an original JSON only by a matching image basename, never by
        # sorted pairing. Ambiguous JSON candidates are reported rather than used.
        candidates=list((project_root/'Label').rglob(f'{source.stem}.json'))
        if len(candidates)==1:
            original_json=read_json(candidates[0])
            if (original_json.get('imageWidth'),original_json.get('imageHeight'))==(shape[1],shape[0]):
                atomic(orig/'original_labelme.json',candidates[0].read_bytes())
                metadata['source_labelme_json']=str(candidates[0])
        metadata['labelme_json_candidates']=[str(p) for p in candidates]
        if row['label_reference_path']:
            label_ref=Path(row['label_reference_path']);with_ref=read_plane(label_ref)
            if with_ref.shape[:2]!=shape: raise ValueError(f'Annotation reference geometry mismatch: {pid}')
            with Image.open(label_ref) as handle: grey=np.asarray(handle.convert('L')).astype(float)
            metadata['label_reference_comparison']=dict(sha256=file_sha256(label_ref),
                mean_abs_gray_difference=float(np.abs(grey-plane).mean()),
                correlation=float(np.corrcoef(grey.ravel(),plane.ravel())[0,1]),
                proves_registration=False)
            atomic(images/'annotation_reference.png',png_bytes(with_ref))
        metadata['immutable_workspace_snapshots']={str(p.relative_to(destination)):file_sha256(p) for p in list(orig.glob('*'))+list(images.glob('*')) if p.is_file()}
        index['positions'][pid]=metadata
        state=dict(position_id=pid,revision=0,status='not_started',parameters=defaults.copy(),notes='',
                   layer_needs_review=False,confirmed_negative=False,annotation_origin='existing' if gt is not None else 'unannotated',
                   created_at=now(),operation='initialize',candidate=False)
        folder=destination/'revisions'/pid/'000000';folder.mkdir(parents=True)
        for name in ('vessel','valid','uncertain'): atomic(folder/f'{name}.png',(orig/f'{name}.png').read_bytes())
        state['files']={name:dict(path=str((folder/f'{name}.png').relative_to(destination)),sha256=file_sha256(folder/f'{name}.png')) for name in ('vessel','valid','uncertain')}
        atomic(folder/'state.json',json_bytes(state));index['states'][pid]=str((folder/'state.json').relative_to(destination))
    atomic(destination/'index.json',json_bytes(index))
    atomic(destination/'annotation_access_log.json',json_bytes(dict(time=now(),purpose='explicit_user_requested_37_position_annotation_review',
        positions=[dict(position_id=r['position_id'],local_split=r['split'],reference_opened=True,label_opened=bool(r['multiclass_label_path'])) for r in rows],
        historical_inventory=read_json(atlas/'asset_inventory.json'),formal_test_evaluation=False,formal_configuration_tuning=False)))
    workspace=Workspace(destination);workspace.refresh_summaries();workspace.verify_sources()
    atomic(destination/'initialization_audit.json',json_bytes(dict(status='passed',positions=len(rows),
        existing_labels=sum(m['had_original_label'] for m in index['positions'].values()),
        unannotated=sum(not m['had_original_label'] for m in index['positions'].values()),
        confirmed=0,protected_files=len(protected),native_geometry='identity',test_access_logged=True)))
    return workspace


class Workspace:
    def __init__(self,root):
        self.root=Path(root).resolve();self.cache={};self.intermediate_cache={};self.display_lock=threading.RLock();self.ac_compute_count=0
        self.index=read_json(self.root/'index.json')
        self.default=self.index['default_parameters']
        self.config=StableACConfig(**self.index['algorithm'])

    def position(self,pid):
        if pid not in self.index['positions']: raise ValueError('Unknown position')
        return self.index['positions'][pid]

    def load(self,pid):
        metadata=self.position(pid);index=read_json(self.root/'index.json')
        state=read_json(self.root/index['states'][pid])
        masks={}
        for name,entry in state['files'].items():
            path=self.root/entry['path']
            if file_sha256(path)!=entry['sha256']: raise ValueError(f'Revision file SHA changed: {path}')
            masks[name]=read_plane(path)>0
        return metadata,state,masks

    def image(self,pid):
        self.position(pid)
        if pid not in self.cache:
            # Final03's independent MATLAB/r2 review divides uint8 pixels by255
            # in float64. Normalizing in float32 first changes up to1 ULP and can
            # move a handful of PNG pixels across rounding ties.
            self.cache[pid]=read_plane(self.root/'source_images'/pid/'reference.png').astype(np.float64)/255
        return self.cache[pid]

    def original(self,pid,name):
        self.position(pid);return read_plane(self.root/'original_masks'/pid/f'{name}.png')>0

    def display_parameters(self,pid,value):
        params=parameters(value,self.default)
        metadata=self.position(pid)
        auto=params.get('auto_window')
        if auto and (auto.get('position_id',pid)!=pid or auto.get('source_image_sha256',metadata['source_image_sha256'])!=metadata['source_image_sha256']):
            raise ValueError('Auto-window provenance belongs to a different reference position')
        rectangle=params.get('observation_roi')
        if rectangle is not None:
            x0,y0,x1,y1=map(int,rectangle);h,w=metadata['shape']
            if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Observation ROI outside native grid or empty')
        return params

    def preview(self,pid,value):
        params=self.display_parameters(pid,value)
        ac,display,maps=render(self.intermediates(pid),params)
        return params,ac,display,maps

    def intermediates(self,pid):
        with self.display_lock:
            if pid in self.intermediate_cache:return self.intermediate_cache[pid]
            identity=dict(cache_version=DISPLAY_CACHE_VERSION,source_image_sha256=self.position(pid)['source_image_sha256'],
                input_array_sha256=array_sha256(self.image(pid)),algorithm=asdict(self.config),input_source='reference',
                algorithm_file_sha256=file_sha256(Path(__file__).resolve().parents[2]/'sabids/experiments/attenuation_aux.py'))
            import hashlib
            key=hashlib.sha256(json_bytes(identity)).hexdigest()
            folder=self.root/'display_cache'/DISPLAY_CACHE_VERSION/f'{pid}_{key}'
            meta_path=folder/'metadata.json'
            if meta_path.is_file():
                meta=read_json(meta_path)
                if meta['identity']!=identity:raise ValueError('Display cache identity mismatch')
                for name,sha in meta['arrays'].items():
                    if file_sha256(folder/f'{name}.npy')!=sha:raise ValueError('Display cache array SHA mismatch')
                intermediate={n:np.load(folder/f'{n}.npy',allow_pickle=False) for n in meta['arrays']}
                intermediate.update(reference_W=self.config.output_window,gain_cap=self.config.gain_cap)
            else:
                if folder.exists():folder=folder.with_name(folder.name+'_recovery_'+uuid.uuid4().hex)
                folder.mkdir(parents=True)
                intermediate=compute_intermediates(self.image(pid),self.config);self.ac_compute_count+=1
                arrays={}
                for name,array in intermediate.items():
                    if not isinstance(array,np.ndarray):continue
                    stream=io.BytesIO();np.save(stream,array,allow_pickle=False);atomic(folder/f'{name}.npy',stream.getvalue());arrays[name]=file_sha256(folder/f'{name}.npy')
                atomic(folder/'metadata.json',json_bytes(dict(identity=identity,arrays=arrays,created_at=now(),purpose='annotation_display_only')))
            self.intermediate_cache[pid]=intermediate
            if len(self.intermediate_cache)>4:self.intermediate_cache.pop(next(iter(self.intermediate_cache)))
            return intermediate

    def automatic_window(self,pid,value,rectangle=None):
        params=self.display_parameters(pid,value);shape=tuple(self.position(pid)['shape'])
        layer=self.original(pid,'layer')
        if rectangle is None:
            if not layer.any():raise ValueError('No original Layer ROI. Select the observation ROI tool and draw a rectangle first.')
            roi=layer;description=dict(kind='original_layer',mask_sha256=file_sha256(self.root/'original_masks'/pid/'layer.png'))
        else:
            values=np.asarray(rectangle,dtype=float)
            if values.shape!=(4,) or not np.isfinite(values).all():raise ValueError('ROI needs four finite native coordinates')
            if not np.equal(values,np.floor(values)).all():raise ValueError('ROI coordinates must be native integers')
            x0,y0,x1,y1=map(int,values)
            if not (0<=x0<x1<=shape[1] and 0<=y0<y1<=shape[0]):raise ValueError('ROI rectangle outside native grid or empty')
            roi=np.zeros(shape,bool);roi[y0:y1,x0:x1]=True;description=dict(kind='manual_rectangle',xyxy=[x0,y0,x1,y1],coordinate_grid='native',ends='exclusive')
        valid=self.intermediates(pid)['valid'].copy()
        if self.position(pid)['had_original_label']:valid&=self.original(pid,'valid')
        result=auto_window(self.intermediates(pid),params,roi,valid)
        result['observation_roi']=list(map(int,rectangle)) if rectangle is not None else None
        result['auto_window'].update(roi=description,position_id=pid,source_image_sha256=self.position(pid)['source_image_sha256'],created_at=now())
        return parameters(result,self.default)

    def save_observation(self,pid,value):
        params,ac,display,maps=self.preview(pid,value)
        target=self.root/'observation_images'/pid/uuid.uuid4().hex
        target.mkdir(parents=True)
        for name,array in dict(ac=ac,display=display,display_input_A=maps['A'],protected_raw=maps['protected_raw']).items():
            stream=io.BytesIO();np.save(stream,array,allow_pickle=False);atomic(target/f'{name}.npy',stream.getvalue())
        atomic(target/'observation.png',png_bytes(np.rint(display*255)))
        metadata=dict(position_id=pid,reference=self.position(pid)['reference_path'],
            source_image_sha256=self.position(pid)['source_image_sha256'],algorithm_version=CACHE_VERSION,
            input_dtype=str(self.image(pid).dtype),input_normalization='uint8 pixels /255 in float64; matches r2 final03',
            input_array_sha256=array_sha256(self.image(pid)),
            algorithm_config=asdict(replace(self.config,beta=params['beta'],output_window=params['W'] if params.get('display_mode',LEGACY)==LEGACY else self.config.output_window)),
            display_mode=params.get('display_mode',LEGACY),display_version=params.get('display_version',1),
            input_source='clean/reference',frame_id=self.position(pid).get('position',pid),native_geometry=self.position(pid)['native_geometry'],
            raw_array_sha256=file_sha256(target/'protected_raw.npy'),display_input_sha256=file_sha256(target/'display_input_A.npy'),
            protected_reference_W=self.config.output_window,
            display_order='legacy: r2 clip/cap/reliability -> beta -> gamma' if params.get('display_mode',LEGACY)==LEGACY else 'protected pre-display raw -> raw beta mix -> black/white window -> gamma',
            parameters=params,created_at=now(),purpose='annotation_review_only',
            float_sha256=file_sha256(target/'ac.npy'),display_float_sha256=file_sha256(target/'display.npy'),
            output_sha256=file_sha256(target/'observation.png'),shape=self.position(pid)['shape'])
        atomic(target/'metadata.json',json_bytes(metadata))
        self.log(dict(operation='save_observation',**metadata,path=str(target)))
        return target,metadata

    def log(self,event):
        with (self.root/'edit_log.jsonl').open('ab') as handle:
            handle.write(json.dumps(event,ensure_ascii=False,allow_nan=False).encode()+b'\n');handle.flush();os.fsync(handle.fileno())

    def save(self,pid,base_revision,masks,params,status,notes='',negative=False,layer_needs_review=False,operation='browser_save',extra=None):
        params=self.display_parameters(pid,params)
        if status not in STATUSES: raise ValueError('Unknown review status')
        with transaction(self.root):
            metadata,old,old_masks=self.load(pid)
            if int(base_revision)!=old['revision']:
                conflict=self.root/'conflicts'/f'{pid}_{uuid.uuid4().hex}'
                conflict.mkdir()
                for name,mask in masks.items(): atomic(conflict/f'{name}.png',png_bytes(np.asarray(mask,dtype=np.uint8)*255))
                atomic(conflict/'conflict.json',json_bytes(dict(base_revision=base_revision,current_revision=old['revision'],operation=operation,extra=extra,parameters=params,status=status,notes=notes,negative=negative,layer_needs_review=layer_needs_review)))
                raise Conflict(f'Version conflict; both preserved. Incoming draft: {conflict}')
            shape=tuple(metadata['shape'])
            if set(masks)!={'vessel','valid','uncertain'} or any(np.asarray(v).shape!=shape for v in masks.values()):
                raise ValueError('Invalid native mask set or geometry')
            masks={k:np.asarray(v,dtype=bool) for k,v in masks.items()}
            changed=any(not np.array_equal(masks[n],old_masks[n]) for n in masks)
            if changed and status=='not_started': status='modified'
            valid=masks['valid'] & ~masks['uncertain']
            if status=='confirmed':
                if not valid.any(): raise ValueError('Confirmation needs explicitly annotated coverage')
                if not masks['vessel'].any() and not negative:
                    raise ValueError('Empty mask needs explicit confirmed-negative checkbox')
            revision=old['revision']+1
            folder=self.root/'revisions'/pid/f'{revision:06d}'
            if folder.exists():
                # A previous crash may have created an unreferenced revision.
                # Preserve it rather than overwriting it.
                folder=folder.with_name(folder.name+'_'+uuid.uuid4().hex)
            folder.mkdir(parents=True)
            before=old_masks['vessel'];after=masks['vessel'];original=self.original(pid,'vessel')
            outside=after & ~self.original(pid,'layer')
            origin=('existing' if all(np.array_equal(masks[n],self.original(pid,n)) for n in masks) else 'existing_revised') if metadata['had_original_label'] else ('new_annotation' if any(m.any() for m in masks.values()) else 'unannotated')
            state=dict(position_id=pid,revision=revision,parent_revision=old['revision'],status=status,
                parameters=params,notes=str(notes),layer_needs_review=bool(layer_needs_review or outside.any()),
                outside_original_layer_pixels=int(outside.sum()),confirmed_negative=bool(negative),
                annotation_origin=origin,
                created_at=now(),operation=operation,candidate=status=='confirmed',extra=extra)
            for name,mask in masks.items(): atomic(folder/f'{name}.png',png_bytes(mask.astype(np.uint8)*255))
            atomic(folder/'effective_valid.png',png_bytes(valid.astype(np.uint8)*255))
            state['files']={name:dict(path=str((folder/f'{name}.png').relative_to(self.root)),sha256=file_sha256(folder/f'{name}.png')) for name in masks}
            difference=np.zeros((*shape,3),np.uint8)
            difference[after & original]=(150,150,150)
            difference[after & ~original]=(0,255,0)
            difference[original & ~after]=(255,0,180)
            atomic(folder/'diff_from_original.png',png_bytes(difference))
            state['changes']=dict(added_pixels=int((after & ~before).sum()),deleted_pixels=int((before & ~after).sum()),
                original_added_pixels=int((after & ~original).sum()),original_deleted_pixels=int((original & ~after).sum()),
                components_before=components(before),components_after=components(after),
                old_label_sha256=old['files']['vessel']['sha256'],new_label_sha256=state['files']['vessel']['sha256'])
            atomic(folder/'state.json',json_bytes(state))
            index=read_json(self.root/'index.json');index['states'][pid]=str((folder/'state.json').relative_to(self.root))
            # The single atomic pointer publication commits the whole immutable
            # snapshot. Derived summaries can be rebuilt after a crash.
            atomic(self.root/'index.json',json_bytes(index));self.index=index
            self.log(dict(position_id=pid,image_identity=metadata['source_image_sha256'],time=now(),
                revision=revision,status=status,parameters=params,notes=notes,operation=operation,**state['changes']))
            self.refresh_summaries()
            return state

    def refresh_summaries(self):
        states=[];parameters_by_position={}
        for pid in self.index['positions']:
            _,state,_=self.load(pid);states.append(state);parameters_by_position[pid]=state['parameters']
        atomic(self.root/'review_parameters.json',json_bytes(parameters_by_position))
        columns=['position_id','revision','status','annotation_origin','candidate','layer_needs_review','confirmed_negative','notes']
        text=io.StringIO();writer=csv.DictWriter(text,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(states)
        atomic(self.root/'review_status.csv',text.getvalue().encode('utf-8-sig'))
        text=io.StringIO();columns=['position_id','revision','status','added_pixels','deleted_pixels','original_added_pixels','original_deleted_pixels','components_before','components_after']
        writer=csv.DictWriter(text,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for state in states: writer.writerow(dict(**{k:state[k] for k in ('position_id','revision','status')},**state.get('changes',{})))
        atomic(self.root/'label_diff_summary.csv',text.getvalue().encode('utf-8-sig'))
        # Compatibility copies are derived, never the revision authority.
        for state in states:
            pid=state['position_id']
            for name,target in (('vessel','edited_masks'),('valid','validity'),('uncertain','uncertain_regions')):
                atomic(self.root/target/f'{pid}.png',(self.root/state['files'][name]['path']).read_bytes())

    def verify_sources(self):
        entries=read_json(self.root/'protected_assets.json')
        changed=[row['path'] for row in entries if not Path(row['path']).is_file() or file_sha256(row['path'])!=row['sha256']]
        snapshot_count=0
        for metadata in self.index['positions'].values():
            for name,expected in metadata.get('immutable_workspace_snapshots',{}).items():
                snapshot_count+=1;path=self.root/name
                if not path.is_file() or file_sha256(path)!=expected:changed.append(str(path))
            # Also validate snapshot pixels against their SHA-verified sources;
            # supports the initial v2 workspace created before snapshot hashes
            # were added. No filename pairing or resampling is involved.
            pid=metadata['position_id'];source=Path(metadata['reference_path'])
            image=read_plane(self.root/'source_images'/pid/'reference.png')
            native=read_plane(self.root/'source_images'/pid/'reference_native.png')
            if not np.array_equal(image,np.rint(read_gray(source)*255).astype(np.uint8)) or not np.array_equal(native,read_plane(source)):
                changed.append(f'{pid}: reference snapshot pixels')
            size=tuple(metadata['shape']);gt=read_plane(metadata['multiclass_label_path']) if metadata['had_original_label'] else None
            expected_masks=dict(layer=np.isin(gt,(1,2)) if gt is not None else np.zeros(size,bool),
                vessel=gt==2 if gt is not None else np.zeros(size,bool),valid=gt!=255 if gt is not None else np.zeros(size,bool),uncertain=gt==255 if gt is not None else np.zeros(size,bool))
            for name,expected in expected_masks.items():
                if not np.array_equal(self.original(pid,name),expected):changed.append(f'{pid}: original {name} snapshot')
                snapshot_count+=1
        if changed: raise ValueError(f'Protected assets changed: {changed[:10]}')
        return dict(status='passed',checked_files=len(entries),checked_workspace_snapshots=snapshot_count,changed_files=[])

    def candidate_export(self):
        with transaction(self.root):
            folder=self.root/'candidate_labels'/('version_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);folder.mkdir()
            rows=[]
            for pid in self.index['positions']:
                metadata,state,masks=self.load(pid)
                if state['status']!='confirmed': continue
                target=folder/pid;target.mkdir()
                for name in ('vessel','valid','uncertain'): atomic(target/f'{name}.png',png_bytes(masks[name].astype(np.uint8)*255))
                layer=self.original(pid,'layer')
                atomic(target/'original_layer_readonly.png',png_bytes(layer.astype(np.uint8)*255))
                # A semantic union is explicitly a proposal, not a reviewed
                # change to the read-only Layer annotation.
                atomic(target/'layer_union_proposal.png',png_bytes((layer | masks['vessel']).astype(np.uint8)*255))
                valid=masks['valid'] & ~masks['uncertain']
                atomic(target/'effective_valid.png',png_bytes(valid.astype(np.uint8)*255))
                rows.append(dict(position_id=pid,revision=state['revision'],source_sha256=metadata['source_image_sha256'],
                    vessel_sha256=file_sha256(target/'vessel.png'),layer_requires_review=state['layer_needs_review'],
                    confirmed_negative=state['confirmed_negative'],notes=state['notes']))
            result=dict(purpose='candidate_label_version_only',created_at=now(),positions=rows,
                        confirmed_count=len(rows),training_protocol_modified=False,repeat_transfer_allowed=False)
            atomic(folder/'manifest.json',json_bytes(result));self.log(dict(operation='candidate_export',path=str(folder),confirmed_count=len(rows),time=now()))
            return folder,result
