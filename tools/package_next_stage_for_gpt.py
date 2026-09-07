from __future__ import annotations
import argparse, hashlib, json, tarfile
from datetime import datetime
from pathlib import Path

FORBIDDEN_SUFFIX={".pth",".pt",".ckpt",".npy",".npz"}; FORBIDDEN_PARTS={"Data","Label","test","test_results","cache"}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--project-root",default="."); p.add_argument("--output"); a=p.parse_args(); root=Path(a.project_root).resolve()
 output=Path(a.output).resolve() if a.output else root/"exports"/f"SABIDS_next_stage_GPT_{datetime.now():%Y%m%d_%H%M%S}.tar.gz"; output.parent.mkdir(parents=True,exist_ok=True)
 sources=[root/"configs/next_stage",root/"runs/reports"]
 files=[]
 for source in sources:
  if source.exists():
   for f in source.rglob("*"):
    rel=f.relative_to(root)
    if f.is_file() and f.suffix.lower() not in FORBIDDEN_SUFFIX and not any(part.lower() in {x.lower() for x in FORBIDDEN_PARTS} for part in rel.parts): files.append(f)
 with tarfile.open(output,"w:gz") as tar:
  for f in sorted(files): tar.add(f,arcname=f.relative_to(root))
 digest=hashlib.sha256(output.read_bytes()).hexdigest(); print(json.dumps({"path":str(output),"size_bytes":output.stat().st_size,"file_count":len(files),"sha256":digest,"included_test_count":0,"missing_assets":None},indent=2))
if __name__=="__main__": main()
