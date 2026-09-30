from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd, torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sabids.experiments.dual_task_adaptive import sha256_file
from sabids.experiments.dose_response import tensor_sha

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",required=True); p.add_argument("--output",required=True); p.add_argument("--engineering-check",action="store_true")
    a=p.parse_args(); run=Path(a.run_dir).resolve(); out=Path(a.output).resolve()
    history=pd.read_csv(run/"history.csv"); eligibility=pd.read_csv(run/"checkpoint_eligibility.csv")
    safe=run/"best_vessel_safe.pth"; unconstrained=run/"best_unconstrained.pth"; last=run/"last.pth"
    eligible=eligibility[eligibility["eligible"].astype(str).str.lower().isin(["true","1"])]
    engineering_ok = last.is_file() and unconstrained.is_file() and len(history) > 0
    status="passed" if (engineering_ok if a.engineering_check else safe.is_file() and not eligible.empty) else "blocked"
    result={"schema_version":"dual-task-adaptive-v2-audit","status":status,
            "blocked_message":None if status=="passed" else ("BLOCKED: CUDA CHECK INCOMPLETE" if a.engineering_check else "BLOCKED: NO VESSEL-SAFE CHECKPOINT"),
            "engineering_check":bool(a.engineering_check),
            "completed_epochs":int(history["epoch"].max()), "eligible_epochs":eligible["epoch"].astype(int).tolist(),
            "best_vessel_safe":str(safe), "best_vessel_safe_sha256":sha256_file(safe) if safe.is_file() else None,
            "best_unconstrained":str(unconstrained), "best_unconstrained_notice":"NOT FOR FORMAL CLAIMS",
            "last":str(last), "test_assets_opened":0}
    if safe.is_file():
        raw=torch.load(safe,map_location="cpu",weights_only=False)
        result["checkpoint_epoch"]=int(raw["epoch"])+1
        binding={"schema_version":"dual-task-adaptive-v2-safe-binding","status":"passed",
                 "checkpoint_path":str(safe),"checkpoint_sha256":sha256_file(safe),
                 "checkpoint_epoch":result["checkpoint_epoch"],"selection_rule":"vessel_safe_q_earliest_tie",
                 "test_assets_opened":0}
        (run/"best_vessel_safe_binding.json").write_text(json.dumps(binding,indent=2)+"\n")
    initial_path=run/"initialization_audit.json"
    selected=safe if safe.is_file() else last
    if initial_path.is_file() and selected.is_file():
        initial=json.loads(initial_path.read_text(encoding="utf-8")); raw=torch.load(selected,map_location="cpu",weights_only=False)
        changed=[]
        for name,value in raw["model"].items():
            if name in initial.get("tensor_sha256",{}) and tensor_sha(value) != initial["tensor_sha256"][name]: changed.append(name)
        frozen_changed=[n for n in changed if n.startswith("v1.")]
        trainable_changed=[n for n in changed if not n.startswith("v1.")]
        parameter={"status":"passed" if trainable_changed and not frozen_changed else "failed",
                   "changed_trainable_parameter_names":trainable_changed,"changed_frozen_parameter_names":frozen_changed}
        (run/"parameter_audit.json").write_text(json.dumps(parameter,indent=2)+"\n")
        (run/"frozen_parameter_audit.json").write_text(json.dumps({"status":"passed" if not frozen_changed else "failed","changed_v1_parameter_names":frozen_changed},indent=2)+"\n")
        result.update(parameter_audit=parameter)
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)); raise SystemExit(0 if status=="passed" else 3)
if __name__=="__main__": main()
