# Validation-only presentation archive

`tools/build_presentation_archive.py` creates the server result package without
training and without opening sealed test assets.  The workflow is deliberately
split so the audit can be reviewed before any model forward is run.

## Phase 1: audit

```bash
cd /mnt/SABIDS-Net
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="runs/presentation_archive_${STAMP}"

python tools/build_presentation_archive.py \
  --project-root . \
  --mode audit \
  --output "$OUT"
```

Review `audit/run_path_audit.csv`, `audit/checkpoint_audit.csv`, and
`audit/missing_assets.csv`.  The checkpoint audit hashes `best.pth` and
`last.pth` and reads the stored epoch; directory names alone are not accepted as
checkpoint evidence.

## Phase 2: validation inference only

```bash
python tools/build_presentation_archive.py \
  --project-root . \
  --mode evaluate \
  --output "$OUT" \
  --run-missing-validation \
  --device cuda \
  --num-workers 4
```

This phase calls only the existing Stage 1/2 exporter and the `evaluate`/`b0`
modes of the interaction and input-factorial runners.  It never calls their
training modes.  It uses Stage 2 metadata-selected `best.pth`, Joint fixed-final
`last.pth` at epoch 20, input fixed-final `last.pth` at epoch 60, threshold 0.5,
P0 for causal comparisons, original-grid restoration, and prediction export.

If a formal report already exists, provide it explicitly:

```bash
python tools/build_presentation_archive.py \
  --project-root . --mode evaluate --output "$OUT" \
  --run-missing-validation --device cuda \
  --stage12-report runs/reports/stage12_validation_EXACT_TIMESTAMP \
  --interaction-report runs/interaction_factorial_report \
  --input-report runs/input_factorial_report_v1
```

## Phase 3: assemble

```bash
python tools/build_presentation_archive.py \
  --project-root . \
  --mode assemble \
  --output "$OUT"
```

The assembler generates fixed triptychs, requested red-vessel/green-layer
overlays, TP/FP/FN maps, separated P2a/P2b postprocessing, 15 16:9 figures in
PNG/PDF, quantitative CSVs, source-required literature rows, summaries, and a
missing/failure checklist.  Missing checkpoints or predictions remain
`MISSING`; the assembler does not train replacements.

## Phase 4: workbooks and packages

Spreadsheet authoring uses `@oai/artifact-tool`.  In the Codex artifact runtime,
run its operation marker once, then:

```bash
node tools/build_presentation_workbooks.mjs "$OUT"

python tools/build_presentation_archive.py \
  --project-root . \
  --mode package \
  --output "$OUT"
```

The package phase requires `SABIDS_presentation_results.xlsx` unless
`--skip-workbooks` is explicitly supplied.  The latter is intended only for a
server lacking the artifact runtime; it preserves all CSVs and records the
workbook omission instead of substituting another Excel library.

Generated packages are:

- `packages/SABIDS_presentation_full_<timestamp>.tar.gz`
- `packages/SABIDS_presentation_for_GPT_<timestamp>.zip`

Neither contains checkpoints, float caches, original Data/Label assets, or
sealed test content.
