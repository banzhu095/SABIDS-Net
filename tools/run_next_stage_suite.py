"""Fail-closed v3 suite launcher; sealed test is never selected."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
# When invoked as ``python tools/run_next_stage_suite.py`` Python adds only
# ``tools`` to sys.path.  Resolve the repository root before importing the
# package so the documented CLI works from any current directory.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
 sys.path.insert(0, str(_PROJECT_ROOT))
from sabids.config import load_config,save_config
SUITES={
 "d1_structure":["d1_d0_pku37_v3.yaml","d1_structure_pku37_v3.yaml"],
 "input_image":[f"input_{x}_pku37_v3.yaml" for x in ("noisy","d0","d1","clean")],
 "training_order":[f"order_{x}_pku37_v3.yaml" for x in ("ds","sd","alt")],
 "decoder_interaction_strength":[f"interaction_{x}_pku37_v3.yaml" for x in ("a00","ad05","ad10","ad20","as05","as10","as20")],
 "decoder_interaction_confirm":[f"interaction_j{x}_strong_pku37_v3.yaml" for x in ("00","10","01","11")],
}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--project-root",default="."); p.add_argument("--protocol-id",default="pku37_binary_v3"); p.add_argument("--suite",choices=SUITES,required=True); p.add_argument("--mode",choices=("pilot","full"),default="pilot"); p.add_argument("--folds",nargs="+",type=int,default=[0]); p.add_argument("--seeds",nargs="+",type=int,default=[42]); p.add_argument("--device",default="cuda"); p.add_argument("--resume",action="store_true"); p.add_argument("--save-fixed-predictions",action="store_true"); p.add_argument("--execute",action="store_true"); a=p.parse_args()
 root=Path(a.project_root).resolve(); audit_path=root/"Manifests"/a.protocol_id/"protocol_audit.json"
 if not audit_path.is_file(): raise SystemExit("BLOCKED: write the v3 protocol first")
 audit=json.loads(audit_path.read_text(encoding="utf-8"))
 if audit.get("status")!="passed" or audit.get("test_assets_opened")!=0: raise SystemExit("BLOCKED: protocol audit did not pass")
 commands=[]
 for fold in a.folds:
  for seed in a.seeds:
   for name in SUITES[a.suite]:
    cfg=load_config(root/"configs/next_stage_v3"/name); cfg["seed"]=seed; cfg["fold"]=fold; cfg["device"]=a.device
    for key in ("data_plan_sha256","label_inventory_sha256"):
     if cfg.get(key)!=audit.get(key): raise SystemExit(f"BLOCKED: {key} differs from current protocol")
    stem=Path(name).stem; cfg["train"]["output_dir"]=str(root/"runs/current"/stem.replace("fold0_seed42",f"fold{fold}_seed{seed}"))
    if a.mode=="pilot": cfg["train"]["epochs"]=2; cfg["data"]["max_train_samples"]=2; cfg["data"]["max_val_samples"]=2; cfg["train"]["num_workers"]=0
    if cfg["train"].get("schedule") and a.execute: raise SystemExit("BLOCKED: continuous order state machine remains incomplete")
    resolved=root/"runs/next_stage_v3_launch_configs"/f"{stem}_f{fold}_s{seed}.yaml"; save_config(cfg,resolved); cmd=[sys.executable,"train.py","--config",str(resolved)]; commands.append(" ".join(cmd))
    if a.execute: subprocess.run(cmd,cwd=root,check=True)
 print(json.dumps({"status":"executed" if a.execute else "planned","protocol_id":a.protocol_id,"run_count":len(commands),"commands":commands,"test_assets_opened":0},indent=2))
if __name__=="__main__": main()
