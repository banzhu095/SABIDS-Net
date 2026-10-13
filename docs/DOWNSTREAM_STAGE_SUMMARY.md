# Read-only unified downstream segmentation review

`tools.downstream_seg_review` reviews a finished formal experiment without
training, checkpoint reselection, inference, threshold changes or postprocessing.
It writes only to a separate `reports/downstream_seg_stage_summary_<timestamp>`
directory. The original run is read-only and source hashes are checked before
and after each stage. All method/sample associations use manifest logical keys.

```bash
ROOT=/mnt/SABIDS-Net
RUN="$ROOT/runs/downstream_seg_formal_20261009_a30_amp_recovery"
OUT="$ROOT/reports/downstream_seg_stage_summary_$(date +%Y%m%d_%H%M%S)"
python -m tools.downstream_seg_review.cli --project-root "$ROOT" --run-dir "$RUN" --output-dir "$OUT" --stage prepare
python -m tools.downstream_seg_review.cli --project-root "$ROOT" --run-dir "$RUN" --output-dir "$OUT" --stage export --workers 4 --resume
python -m tools.downstream_seg_review.cli --project-root "$ROOT" --run-dir "$RUN" --output-dir "$OUT" --stage atlas --resume
```

Create `downstream_seg_stage_summary.xlsx` from `workbook_data.json` using
`build_workbook.mjs` and the documented artifact-tool runtime. Render and inspect
all sheets. On a cloud host without artifact-tool, create the workbook locally
from the same JSON and upload the verified workbook to the **review** directory,
never to the original run. Then package:

```bash
python -m tools.downstream_seg_review.cli --project-root "$ROOT" --run-dir "$RUN" --output-dir "$OUT" --stage package --resume
```

Preparation validates 30 complete method/seed tracks and 4500 unique frame
records, reaggregates position-within-seed then seed/position macro results,
compares the sealed summaries, checks common initialization/randomness hashes,
and reads each selected checkpoint's epoch/updates/validation metadata. A total
training budget is never labelled as best-checkpoint updates. Missing weight
access is reported as curve-inferred, not independently verified.

Export reads the frozen NPZ only, checks its sealed SHA and reproduces basic
frame metrics. Every PNG is round-trip checked. Quantitative probability is
`round(p*65535)` in uint16 PNG and the exact original float NPZ. Raw masks are
`p>=0.5`, including predictions in annotation-ignore pixels; error maps exclude
those pixels and mark them gray. Colors: TP green, FP red, FN blue, GT contours
cyan and predicted contours yellow. No GT/predicted-layer crop, morphology or
component filtering changes the masks. Seed42 assets occupy the primary tree;
the all-seeds seed42 index links there, while seeds123/2026 are stored separately.
Common GT/reference are stored once per position; method inputs once per frame.

Atlas selection/crops are copied from the locked registration, never rescored.
All three seeds use the same layout. A zero small-component registered role is
explicitly labelled as having no small components. Thumbnails are display-only;
full atlas and all quantitative prediction assets remain in download packages.

CSV retains unrounded fractions, space errors are px, workbook/report ratios
display as percentages and paired differences as pp. SD denotes segmentation-
seed SD, not patient variability. Only three independent test positions exist.
Do not claim significance, causal mechanisms, more true vessels from higher HF
energy, or full Joint/UGBI validation. Clean is a separate reference group;
TCFL is unpaired. Undefined component-size-bin recalls remain NaN, never zero.

Every ZIP embeds `PACKAGE_MANIFEST.csv` and has an external SHA/CRC/member-size
receipt. Optional other-seed packaging is an explicit separate stage documented
in the generated `download_paths.md`; its input/GT links are included.
