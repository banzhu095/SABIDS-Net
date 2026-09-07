"""Guarded external-denoising entry point: Duke is evaluation-only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from sabids.config import load_config
def main():
 p=argparse.ArgumentParser(); p.add_argument("--project-root",default="."); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--dataset",choices=("Duke17","Duke28"),required=True); p.add_argument("--output",required=True); a=p.parse_args()
 cfg=load_config(a.config)
 if a.dataset in (cfg.get("data",{}).get("train_datasets") or []): raise SystemExit("Duke datasets are forbidden in training")
 out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=True)
 (out/"evaluation_request.json").write_text(json.dumps({"dataset":a.dataset,"checkpoint":str(Path(a.checkpoint).resolve()),"status":"blocked_missing_dedicated_external_manifest","test_assets_opened":0},indent=2),encoding="utf-8")
 raise SystemExit("BLOCKED: provide a development-only Duke external manifest; no implicit split is permitted")
if __name__=="__main__": main()
