from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser(); p.add_argument("--project-root",default="."); p.add_argument("--suite",required=True); p.add_argument("--output"); a=p.parse_args()
 root=Path(a.project_root).resolve(); out=Path(a.output).resolve() if a.output else root/"runs/reports"/a.suite; out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for history in sorted((root/"runs/current").glob(f"*{a.suite.split('_')[0]}*pku37_v2*/history.csv")):
  try:
   table=pd.read_csv(history); last=table.iloc[-1].to_dict(); rows.append({"run_id":history.parent.name,**last})
  except Exception as e: rows.append({"run_id":history.parent.name,"status":f"unusable:{e}"})
 pd.DataFrame(rows).to_csv(out/"runs_summary.csv",index=False)
 (out/"SUMMARY.md").write_text("# Next-stage report\n\nValidation-only, fixed threshold 0.5, raw P0. Smoke runs are not scientific results.\n",encoding="utf-8")
 (out/"missing_assets.csv").write_text("asset,reason\n",encoding="utf-8")
 print(json.dumps({"output":str(out),"runs":len(rows),"test_assets_opened":0},indent=2))
if __name__=="__main__": main()
