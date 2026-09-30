from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
def sha(p):
 d=hashlib.sha256(); d.update(p.read_bytes()); return d.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--project-root',default='.'); p.add_argument('--run-id',required=True); p.add_argument('--output',required=True); a=p.parse_args()
 root=Path(a.project_root).resolve(); out=Path(a.output).resolve()
 run_root=root/'runs/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v2'/a.run_id/'seed42'
 complete=(run_root/'best_vessel_safe.pth').is_file()
 if not complete and 'incomplete' not in out.stem.lower(): out=out.with_name(out.stem+'_incomplete'+out.suffix)
 if out.exists(): raise FileExistsError(out)
 bases=[root/x/a.run_id for x in ('cache/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v2','runs/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v2','reports/adaptive_denoising/dual_task_adaptive_v2')]
 files=[]
 for b in bases:
  if b.exists(): files += [x for x in b.rglob('*') if x.is_file() and x.suffix.lower() in {'.json','.yaml','.csv','.md','.txt','.log','.png'} and x.stat().st_size<8*1024*1024]
 manifest=[{'path':str(x.relative_to(root)).replace('\\','/'),'size':x.stat().st_size,'sha256':sha(x)} for x in sorted(set(files))]
 if not manifest: raise RuntimeError('No lightweight v2 artifacts')
 out.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
  for x in sorted(set(files)): z.write(x,str(x.relative_to(root)).replace('\\','/'))
  z.writestr('PACKAGE_MANIFEST.json',json.dumps({'files':manifest,'test_assets_opened':0},indent=2))
 with zipfile.ZipFile(out) as z:
  if z.testzip(): raise RuntimeError('ZIP integrity failed')
 print(json.dumps({'status':'passed','path':str(out),'size_bytes':out.stat().st_size,'sha256':sha(out),'file_count':len(manifest)},indent=2))
if __name__=='__main__': main()
