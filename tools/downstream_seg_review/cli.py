from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tools.downstream_seg_benchmark.common import require,write_json
from .tables import audit,tables
from .images import export,atlas
from .reporting import figures,report,packages,finish_immutable,package_other_seeds


def main(argv=None):
    p=argparse.ArgumentParser(description='Read-only review; export saved probabilities, NEVER train/infer/tune')
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--stage',choices=['prepare','export','atlas','package','package-other-seeds'],default='prepare')
    p.add_argument('--resume',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--metrics-only',action='store_true',help='Explicitly omit unavailable remote predictions; never claim export completion')
    args=p.parse_args(argv);run=args.run_dir.resolve()
    out=(args.output_dir or args.project_root/'reports'/('downstream_seg_stage_summary_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    require(not out.is_relative_to(run) and out!=run,'Original run is read-only')
    require(args.workers in range(1,9),'Use 1 to 8 export workers')
    out.mkdir(parents=True,exist_ok=True)
    print('REVIEW_OUTPUT',out,flush=True)
    try:
        audit(run,out,args.resume)
        if args.stage=='prepare':tables(run,out);figures(out);report(out);finish_immutable(out)
        elif args.stage=='export':export(run,out,args.resume,args.workers);finish_immutable(out)
        elif args.stage=='atlas':atlas(run,out);finish_immutable(out)
        elif args.stage=='package':packages(out,args.resume,not args.metrics_only)
        elif args.stage=='package-other-seeds':package_other_seeds(out)
    except Exception as exc:
        write_json(out/'review_failure.json',dict(stage=args.stage,error=type(exc).__name__,reason=str(exc)))
        raise


if __name__=='__main__':main()
