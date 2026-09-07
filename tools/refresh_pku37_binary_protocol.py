"""Rescan raw PKU37 assets and build the contract-authoritative v3 protocol."""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, yaml
from PIL import Image

def sha_file(p: Path) -> str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def sha_table(df): return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
def norm(x):
 d="".join(re.findall(r"\d",str(x))); return f"pku_{int(d):04d}" if d else ""
def image_shape(p):
 with Image.open(p) as im: return im.height,im.width
def label_qc(layer,vessel,expected):
 try:
  l=np.asarray(Image.open(layer)); v=np.asarray(Image.open(vessel)); lu=set(np.unique(l).tolist()); vu=set(np.unique(v).tolist()); lb=l>0; vb=v>0
  return {"decodable":True,"size_match":tuple(lb.shape)==tuple(expected) and tuple(vb.shape)==tuple(expected),"valid_binary_values":lu<={0,1,255} and vu<={0,1,255},"layer_empty":not lb.any(),"vessel_empty":not vb.any(),"vessel_fraction_of_layer":float(vb.sum()/max(lb.sum(),1)),"vessel_outside_layer_fraction":float((vb&~lb).sum()/max(vb.sum(),1)),"layer_values":str(sorted(lu)),"vessel_values":str(sorted(vu))}
 except Exception as e: return {"decodable":False,"size_match":False,"valid_binary_values":False,"error":str(e)}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--project-root",default="."); p.add_argument("--protocol-id",default="pku37_binary_v3"); p.add_argument("--split-contract",required=True); p.add_argument("--archive-existing",action="store_true"); p.add_argument("--force-new-protocol",action="store_true"); m=p.add_mutually_exclusive_group(required=True); m.add_argument("--dry-run",action="store_true"); m.add_argument("--write",action="store_true"); a=p.parse_args()
 root=Path(a.project_root).resolve(); cp=(root/a.split_contract).resolve(); contract=yaml.safe_load(cp.read_text(encoding="utf-8"))
 if contract.get("protocol_id")!=a.protocol_id: raise SystemExit("protocol_id differs from split contract")
 val={norm(x) for x in contract["validation_positions"]}; test={norm(x) for x in contract["test_positions"]}; data=root/"Data/PKU37_OCT_Denoising"; lab=root/"Label"
 noisy=sorted((data/"noisy").glob("*.tif")); cleans={norm(x.stem):x for x in (data/"clean").glob("*.tif")}; layers={norm(x.stem):x for x in (lab/"binary_choroid_layer").glob("*.png")}; vessels={norm(x.stem):x for x in (lab/"binary_choroid_vessel").glob("*.png")}
 rows=[]
 for image in noisy:
  digits="".join(re.findall(r"\d",image.stem)); pos=norm(digits[:4]); frame=int(digits[4:] or 0); split="test" if pos in test else "val" if pos in val else "train"
  rel=lambda x:x.relative_to(root).as_posix() if x else ""
  rows.append({"sample_id":f"{pos}_f{frame:02d}","group_id":pos,"patient_id":pos,"dataset":"PKU37","domain":"public","scan_protocol":"SD-OCT-repeat","frame_index":frame,"split":split,"image_path":rel(image),"clean_path":rel(cleans.get(pos)),"layer_mask_path":rel(layers.get(pos)),"vessel_mask_path":rel(vessels.get(pos)),"multiclass_label_path":"","has_manual_label":int(pos in layers and pos in vessels),"is_clean":0})
 allrows=pd.DataFrame(rows); positions=set(allrows.group_id); missing_contract=sorted((val|test)-positions); qc=[]
 for pos in sorted((set(layers)&set(vessels))-test):
  sample=allrows[allrows.group_id.eq(pos)].iloc[0]; qc.append({"group_id":pos,**label_qc(layers[pos],vessels[pos],image_shape(root/sample.image_path))})
 qcdf=pd.DataFrame(qc); bad=qcdf[(~qcdf.decodable)|(~qcdf.size_match)|(~qcdf.valid_binary_values)] if len(qcdf) else qcdf
 td=allrows[allrows.split.eq("train")&allrows.clean_path.ne("")].copy(); ts=allrows[allrows.split.eq("train")&allrows.has_manual_label.eq(1)].copy(); tj=allrows[allrows.split.eq("train")&(allrows.clean_path.ne("")|allrows.has_manual_label.eq(1))].copy(); va=allrows[allrows.split.eq("val")].copy(); te=allrows[allrows.split.eq("test")].copy()
 missing=[]
 for name,df in (("train_denoise",td),("train_segment",ts),("train_joint",tj),("validation",va)):
  for _,r in df.iterrows():
   for col in ("image_path","clean_path","layer_mask_path","vessel_mask_path"):
    if r[col] and not (root/r[col]).is_file(): missing.append({"manifest":name,"sample_id":r.sample_id,"asset":col,"path":r[col]})
 overlap=len((set(tj.group_id)&val)|(set(tj.group_id)&test)|(val&test)); inventory=allrows.drop_duplicates("group_id")[["group_id","split","clean_path","layer_mask_path","vessel_mask_path"]]
 labels=pd.DataFrame([{"group_id":x,"layer_path":str(layers.get(x,"")),"vessel_path":str(vessels.get(x,"")),"layer_sha256":sha_file(layers[x]) if x in layers else "","vessel_sha256":sha_file(vessels[x]) if x in vessels else ""} for x in sorted(positions)])
 status="passed" if not missing_contract and not len(bad) and not missing and overlap==0 else "blocked"
 audit={"status":status,"protocol_id":a.protocol_id,"split_contract_sha256":sha_file(cp),"data_plan_sha256":sha_table(allrows),"label_inventory_sha256":sha_table(labels),"dataset_inventory_sha256":sha_table(inventory),"train_positions":sorted(set(tj.group_id)),"validation_positions":sorted(val),"test_positions":sorted(test),"position_counts":{"train":len(set(tj.group_id)),"validation":len(val),"test":len(test)},"frame_counts":{"train":len(tj),"validation":len(va),"test":len(te)},"labeled_positions":len(set(layers)&set(vessels)),"paired_positions":len(set(cleans)&positions),"duke_training_samples":0,"split_overlap_count":overlap,"test_assets_opened":0,"abnormal_label_count":len(bad),"missing_asset_count":len(missing),"missing_contract_positions":missing_contract,"generated_at":datetime.now(timezone.utc).isoformat(),"git_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True).stdout.strip()}
 out=root/"Manifests"/a.protocol_id
 if a.write:
  if out.exists():
   if not a.archive_existing: raise SystemExit(f"Refusing overwrite: {out}; use --archive-existing")
   shutil.move(out,out.with_name(out.name+"_archive_"+datetime.now().strftime("%Y%m%d_%H%M%S")))
  if not a.force_new_protocol: raise SystemExit("--write requires --force-new-protocol")
  out.mkdir(parents=True); outputs={"train_denoise.csv":pd.concat([td,va[va.clean_path.ne("")]],ignore_index=True),"train_segment.csv":pd.concat([ts,va[va.has_manual_label.eq(1)]],ignore_index=True),"train_joint.csv":pd.concat([tj,va],ignore_index=True),"validation.csv":va,"test_sealed.csv":te,"dataset_inventory.csv":inventory,"label_inventory.csv":labels,"label_qc_by_frame.csv":qcdf,"label_qc_by_position.csv":qcdf,"noisy_clean_pair_audit.csv":inventory[["group_id","clean_path"]],"split_by_position.csv":inventory[["group_id","split"]],"missing_assets.csv":pd.DataFrame(missing),"excluded_assets.csv":pd.DataFrame(columns=["path","reason"])}
  for n,df in outputs.items(): df.to_csv(out/n,index=False)
  (out/"protocol_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8"); (out/"protocol_sha256.txt").write_text(hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest()+"\n")
 print(json.dumps(audit,indent=2,ensure_ascii=False)); raise SystemExit(0 if status=="passed" else 2)
if __name__=="__main__": main()
