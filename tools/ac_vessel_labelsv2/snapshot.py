import json
import re
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes,label

from .common import *


def discover(project,workspace=None,evidence=None):
    project=Path(project).resolve()
    if evidence is not None:
        require(workspace is not None,'Explicit workspace required with service evidence')
        active=Path(workspace).resolve();result=read(evidence)
        require(sha(active/'index.json')==result['index_sha256'],'Explicit workspace differs from prior service evidence')
        return active,dict(result,workspace=str(active),basis='explicit workspace + SHA-verified prior service evidence')
    try:
        with urlopen('http://127.0.0.1:8787/api/bootstrap',timeout=3) as f:live=json.load(f)
        active=Path(live['workspace']).resolve()
        require(workspace is None or Path(workspace).resolve()==active,'Requested workspace differs from active service')
        result=dict(workspace=str(active),index_sha256=sha(active/'index.json'),
                    basis='live loopback bootstrap + runtime/start record',runtime=read(active/'runtime.json'))
        result['runtime'].pop('token',None)
    except (OSError,TimeoutError):
        require(workspace is not None and evidence is not None,'No active service: explicit workspace and prior service evidence required; never infer latest by timestamp')
        active=Path(workspace).resolve();result=read(evidence)
        require(sha(active/'index.json')==result['index_sha256'],'Explicit workspace differs from service evidence snapshot')
        result=dict(result,workspace=str(active),basis='explicit mirrored workspace + SHA-verified prior service evidence')
    require(active.exists(),'Workspace missing')
    return active,result


def folds(positions):
    groups=sorted({p['group_id'] for p in positions});require(len(groups)>=3,'Need >=3 independent groups for train/inner/outer')
    rng=np.random.default_rng(20261016);rng.shuffle(groups);k=min(4,len(groups));result=[]
    for fold,assessment in enumerate(np.array_split(groups,k)):
        outer=sorted(assessment.tolist());remaining=[g for g in groups if g not in outer]
        inner=sorted(remaining[:max(1,int(np.ceil(.2*len(remaining))))]);train=sorted(set(remaining)-set(inner))
        require(train and inner and outer,'Empty grouped split')
        result.append(dict(fold=fold,train=train,inner=inner,outer=outer))
    return result


def freeze(project,workspace,noisy_root,output,evidence=None):
    from tools.pku37_label_review.workspace import Workspace,transaction
    project=Path(project).resolve();workspace,service=discover(project,workspace,evidence)
    w=Workspace(workspace);noisy_root=Path(noisy_root).resolve();output=Path(output).resolve()
    for p in (workspace,project/'Data',project/'Label',project/'Manifests'):
        require(not (p==output or p in output.parents or output in p.parents),'Snapshot overlaps protected input')
    require(not output.exists(),'Snapshot directory exists; choose a new version')
    with transaction(workspace):
        index_sha=sha(workspace/'index.json');states=[];selected=[]
        protected={r['path'].replace('\\','/'):r['sha256'] for r in read(workspace/'protected_assets.json')}
        origin=w.index['project_root'].replace('\\','/').rstrip('/')
        def source_path(value):
            normalized=value.replace('\\','/');require(normalized.startswith(origin+'/'),'Source path outside original project root')
            return project/normalized[len(origin)+1:]
        for pid in sorted(w.index['positions']):
            meta,state,masks=w.load(pid)
            states.append(dict(position_id=pid,status=state['status'],revision=state['revision'],
                coverage_pixels=int(masks['valid'].sum()),uncertain_pixels=int(masks['uncertain'].sum()),
                confirmed_negative=bool(state.get('confirmed_negative')),vessel_pixels=int(masks['vessel'].sum())))
            if state['status']!='confirmed':continue
            for name,entry in state['files'].items():require(np.isin(plane(workspace/entry['path']),(0,255)).all(),pid+': nonbinary '+name)
            for key in ('reference_path','multiclass_label_path'):
                src=source_path(meta[key]);expected=protected.get(meta[key].replace('\\','/'))
                require(expected is not None and sha(src)==expected,pid+': original source changed: '+key)
            require(masks['valid'].any(),pid+': no explicit coverage; cannot assume background')
            require(masks['vessel'].any() or state.get('confirmed_negative'),pid+': empty is not confirmed negative')
            layer=w.original(pid,'layer');lv=w.original(pid,'valid');ref=plane(workspace/'source_images'/pid/'reference.png')
            original_gt=plane(source_path(meta['multiclass_label_path']))
            require(np.isin(original_gt,(0,1,2,255)).all(),pid+': unexpected original indexed classes')
            require(np.array_equal(layer,np.isin(original_gt,(1,2))) and np.array_equal(lv,original_gt!=255),pid+': original Layer/validity snapshot changed')
            from sabids.data.io import read_gray
            require(np.array_equal(ref,np.rint(read_gray(source_path(meta['reference_path']))*255).astype(np.uint8)),pid+': reference geometry/pixels changed')
            require(layer.shape==ref.shape==masks['vessel'].shape,pid+': native geometry mismatch')
            require(layer.any(),pid+': missing Layer annotation')
            pattern=re.compile(re.escape(str(meta['position']))+r'\d{2}$')
            frame_paths=sorted(p for p in noisy_root.iterdir() if p.suffix.lower() in ('.tif','.tiff','.png') and pattern.fullmatch(p.stem))
            require(bool(frame_paths),pid+': no raw noisy frames')
            group=str(meta.get('patient_id') or pid)
            vv=masks['valid'] & ~masks['uncertain'];conflict=masks['vessel'] & ~layer
            selected.append((pid,meta,state,masks,layer,lv,ref,frame_paths,group,conflict,vv))
        require(selected,'No confirmed annotations')
        frozen_index=read(workspace/'index.json')
    out=fresh(output);write(out/'service_evidence.json',service);write(out/'annotation_states.json',states)
    write(out/'manual_algorithm.json',frozen_index['algorithm'])
    write(out/'historical_test_access.json',frozen_index.get('historical_test_access',{}))
    positions=[];frames=[]
    for pid,meta,state,masks,layer,lv,ref,paths,group,conflict,vv in selected:
        for name,a in dict(layer=layer,vessel=masks['vessel'],layer_valid=lv,coverage=masks['valid'],uncertain=masks['uncertain'],vessel_valid=vv,conflict=conflict).items():png(out/'labels'/pid/f'{name}.png',a.astype(np.uint8)*255)
        png(out/'references'/f'{pid}.png',ref)
        rgb=np.repeat(ref[...,None],3,axis=2);rgb[conflict]=[255,0,255];png(out/'conflicts'/f'{pid}.png',rgb)
        coordinates=np.argwhere(conflict);write(out/'conflicts'/f'{pid}.json',dict(yx=coordinates.tolist(),pixels=len(coordinates)))
        write(out/'manual'/f'{pid}.json',dict(parameters=state['parameters'],source='clean/reference',
            overlay_available=True,actual_overlay_visibility='not recorded per parameter event; possible GT-assisted selection',
            algorithm_version=w.index['r2'],reference_sha256=meta['source_image_sha256']))
        write(out/'revisions'/f'{pid}.json',dict(metadata=meta,state=state))
        cc,n=label(masks['vessel'],np.ones((3,3)));holes=int(label(binary_fill_holes(masks['vessel']) & ~masks['vessel'])[1])
        positions.append(dict(position_id=pid,group_id=group,patient_id=meta.get('patient_id',''),
            grouping='patient' if meta.get('patient_id') else 'anatomical_position_only_patient_identity_unavailable',
            shape=list(layer.shape),revision=state['revision'],frames=len(paths),components=n,holes=holes,
            vessel_pixels=int(masks['vessel'].sum()),conflict_pixels=int(conflict.sum()),layer_needs_review=state.get('layer_needs_review',False),
            positive_known=int((masks['vessel']&vv).sum()),negative_known=int((~masks['vessel']&vv).sum()),
            uncertain=int(masks['uncertain'].sum()),uncovered=int((~masks['valid']).sum()),
            original_unknown=int(w.original(pid,'uncertain').sum()),confirmed_negative=bool(state.get('confirmed_negative')),
            historical_local_split=meta.get('split','unknown'),registration='unverified; no label transfer until explicit registration gate'))
        for p in paths:
            require(project in p.parents,'Raw noisy frame must be under project root for portable identity')
            frames.append(dict(sample_id=p.stem,position_id=pid,group_id=group,path=p.relative_to(project).as_posix(),sha256=sha(p)))
    require(sha(workspace/'index.json')==index_sha,'Workspace changed while snapshot was being exported')
    require(len({r['sample_id'] for r in frames})==len(frames),'Ambiguous duplicated raw frame IDs')
    write(out/'positions.json',positions);write(out/'frames.json',frames);write(out/'folds.json',folds(positions))
    files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    lock=dict(version=VERSION,files=files,positions=len(positions),frames=len(frames),group_count=len({p['group_id'] for p in positions}),
        folds=len(folds(positions)),input_workspace_index_sha256=index_sha,
        label_rule='RAW confirmed Vessel preserved, not the prior contained dataset; six exterior pixels remain if present',
        assessment='development grouped CV; prior human/test access disclosed; never blind test',
        manual_selection='clean/reference, GT overlay available, actual visibility not recorded',
        geometry='native grid; shape/mirror/registration diagnostics required; no implicit repeat transfer',training_protocol_modified=False)
    lock['identity']=digest(lock);write(out/'snapshot.json',lock)
    w.log(dict(operation='ac_vessel_labelsv2_snapshot',path=str(out),positions_opened=[p['position_id'] for p in positions],purpose='user_authorized_development_label_validation',test_evaluation=False))
    return lock


def registration_audit(project,snapshot,output):
    verify_lock(snapshot);snapshot=Path(snapshot);project=Path(project);out=fresh(output);positions={p['position_id']:p for p in read(snapshot/'positions.json')};rows=[]
    for r in read(snapshot/'frames.json'):
        path=project/r['path'];require(sha(path)==r['sha256'],'Raw frame changed: '+str(path));x=noisy(path)
        ref=plane(snapshot/'references'/f'{r["position_id"]}.png').astype(float)/255.
        shape_ok=x.shape==ref.shape
        result=dict(**r,shape=list(x.shape),shape_ok=shape_ok,shift_x=None,shift_y=None,phase_response=None,identity_corr=None,mirror_corr=None,numerical_gate=False)
        if shape_ok:
            a=(ref-ref.mean()).astype(np.float64);b=(x-x.mean()).astype(np.float64)
            shift,response=cv2.phaseCorrelate(a,b)
            def corr(z):return float(np.corrcoef(a.ravel(),z.ravel())[0,1]) if a.std()>0 and z.std()>0 else -1.
            identity=corr(b);mirror=max(corr(b[:,::-1]),corr(b[::-1,:]),corr(b[::-1,::-1]))
            result.update(shift_x=shift[0],shift_y=shift[1],phase_response=response,identity_corr=identity,mirror_corr=mirror,
                numerical_gate=bool(max(abs(shift[0]),abs(shift[1]))<=2. and identity>=.3 and identity>mirror+.02))
        rows.append(result)
    table(out/'per_frame_registration.csv',rows)
    report=dict(snapshot_identity=read(snapshot/'snapshot.json')['identity'],frames=rows,
        numerical_pass=all(r['numerical_gate'] for r in rows),
        approved=False,requirement='Source correspondence record (user_confirmation accepted); concrete geometry checks remain mandatory; numerical screening alone is not independent registration proof',
        threshold_rule='fixed engineering screen: abs translation<=2native px, identity corr>=.3 and > best mirror+.02; not fitted on segmentation results')
    write(out/'registration_audit.json',report);return dict(path=str(out),frames=len(rows),numerical_pass=report['numerical_pass'],approved=False)


def approve(snapshot,audit,output,reviewer,evidence):
    verify_lock(snapshot);a=read(Path(audit)/'registration_audit.json')
    require(a['snapshot_identity']==read(Path(snapshot)/'snapshot.json')['identity'],'Registration audit snapshot mismatch')
    require(a['numerical_pass'],'Numerical geometry screen failed; resolve acquisition/registration, do not bypass or silently remove frames')
    require(bool(reviewer.strip()) and len(evidence.strip())>=12,'Actual reviewer and verifiable registration evidence required')
    require(not Path(output).exists(),'Approval exists; preserve it')
    result=dict(approved=True,snapshot_identity=a['snapshot_identity'],audit_sha256=sha(Path(audit)/'registration_audit.json'),
        reviewer=reviewer,evidence=evidence,frame_identities={r['sample_id']:r['sha256'] for r in a['frames']},
        attestation='Human explicitly confirms native identity registration for ALL listed repeated noisy frames; no transform/label copying inferred from filename')
    write(output,result);return result


def confirm_source(snapshot,audit,output,evidence):
    """Record a human-provided dataset fact, NOT a fabricated acquisition log."""
    lock=verify_lock(snapshot);a=read(Path(audit)/'registration_audit.json')
    frames=read(Path(snapshot)/'frames.json')
    identities={r['sample_id']:r['sha256'] for r in frames}
    require(a['snapshot_identity']==lock['identity'],'Geometry audit snapshot mismatch')
    require({r['sample_id']:r['sha256'] for r in a['frames']}==identities,'Geometry audit frame identities mismatch')
    failures=[r for r in a['frames'] if not r['numerical_gate']]
    require(not failures,'Concrete geometry contradictions: '+str(failures))
    require(isinstance(evidence,str) and len(evidence.strip())>=12,'Actual user confirmation text required')
    require(not Path(output).exists(),'Source record exists; preserve it')
    from datetime import datetime,timezone
    result=dict(approved=True,source_type='user_confirmation',recorded_at=datetime.now(timezone.utc).isoformat(),
        scope='PKU37 registered version: same-position noisy and clean share native coordinates; only frozen file identities',
        evidence=evidence,snapshot_identity=lock['identity'],frame_identities=identities,
        audit_sha256=sha(Path(audit)/'registration_audit.json'),
        geometry_checks=dict(frames=len(a['frames']),all_shape_checks=all(r['shape_ok'] for r in a['frames']),
            numerical_screen_pass=True,independent_registration_proof=False),
        acquisition_log_supplied=False,historical_transform_supplied=False,per_frame_manual_review_claimed=False,
        historical_access_records_unchanged=True,coordinate_transform_applied=False)
    write(output,result);return result


def preflight(project,snapshot,approval=None):
    lock=verify_lock(snapshot);frames=read(Path(snapshot)/'frames.json');missing=[];changed=[]
    for r in frames:
        p=Path(project)/r['path']
        if not p.is_file():missing.append(r['path'])
        elif sha(p)!=r['sha256']:changed.append(r['path'])
    problems=[]
    if missing:problems.append('raw_noisy_missing')
    if changed:problems.append('raw_noisy_changed')
    if not approval or not Path(approval).is_file():problems.append('registration_approval_missing')
    else:
        a=read(approval)
        if not a.get('approved') or a.get('snapshot_identity')!=lock['identity'] or a.get('frame_identities')!={r['sample_id']:r['sha256'] for r in frames}:problems.append('registration_approval_mismatch')
        if a.get('source_type')=='user_confirmation':
            checks=a.get('geometry_checks',{})
            if checks.get('frames')!=len(frames) or not checks.get('all_shape_checks') or not checks.get('numerical_screen_pass'):
                problems.append('source_geometry_check_mismatch')
    return dict(status='blocked' if problems else 'passed',problems=problems,missing=missing,changed=changed,
        positions=lock['positions'],frames=lock['frames'],snapshot_identity=lock['identity'],development_not_blind_test=True,
        source_type=read(approval).get('source_type','legacy_reviewer_attestation') if approval and Path(approval).is_file() else None)
