# PKU37 three-arm input-oracle cross-validation

This protocol compares `CV_I_NOISY`, `CV_I_CLEAN`, and `CV_I_DENOISED` while
changing only the image read by the segmentation network. D→S, S→D, RMAC,
reconstruction, identity, and pseudo-label losses remain disabled. The primary
analysis uses fixed-final epoch 60, threshold 0.5, P0, restored original
geometry, and anatomical position as the independent unit.

## Hard gate

A non-zero Phase 0 exit with `status: blocked` is a scientific safety result,
not a request to edit test IDs. Training is authorized only when the labelled
count and manifest test groups match the frozen protocol. The effective sealed
set is the union of configured and legacy-manifest test IDs; assets in that
union are never opened.

```bash
python tools/audit_pku37_positions.py --project-root . --resume
python tools/build_label_review_atlas.py --project-root .
python tools/build_input_oracle_folds.py --project-root .
python tools/audit_input_oracle_protocol.py --project-root . --fold 0 --seed 42
```

Require `status: passed`, `training_authorized: true`, and
`test_assets_opened: 0` in `runs/input_oracle_cv/audit/audit_summary.json`.

## One-fold pilot

Smoke artifacts use a separate namespace and cannot overwrite full runs.

```bash
python tools/train_fold_specific_d0.py --project-root . --fold 0 --mode train --device cuda --smoke-test
python tools/train_fold_specific_d0.py --project-root . --fold 0 --mode audit --smoke-test
python tools/prepare_input_oracle_fold.py --project-root . --fold 0 --mode cache --device cuda --smoke-test
python tools/prepare_input_oracle_fold.py --project-root . --fold 0 --mode initialize --smoke-test
python tools/run_input_oracle_cv.py --project-root . --fold 0 --seed 42 --mode audit --smoke-test
python tools/run_input_oracle_cv.py --project-root . --fold 0 --seed 42 --mode train --device cuda --smoke-test
python tools/run_input_oracle_cv.py --project-root . --fold 0 --seed 42 --mode evaluate --device cuda --smoke-test --save-predictions
python tools/analyze_input_oracle_cv.py --project-root . --folds 0 --seeds 42 --smoke-test --output runs/input_oracle_cv/report_smoke
python tools/build_input_oracle_atlas.py --project-root . --fold 0 --seed 42 --smoke-test --output runs/input_oracle_cv/report_smoke/atlas
python tools/package_input_oracle_analysis.py --project-root . --report runs/input_oracle_cv/report_smoke --dry-run
```

An interrupted exact run may continue with `--resume`; it is rejected when
`last.pth` is absent and is not an overwrite switch.

## Full fold sequence

For each fold, train one D0 and share it across all segmentation seeds:

```bash
for FOLD in 0 1 2 3; do
  python tools/train_fold_specific_d0.py --project-root . --fold "$FOLD" --mode train --device cuda
  python tools/train_fold_specific_d0.py --project-root . --fold "$FOLD" --mode audit
  python tools/prepare_input_oracle_fold.py --project-root . --fold "$FOLD" --mode cache --device cuda
  python tools/prepare_input_oracle_fold.py --project-root . --fold "$FOLD" --mode initialize
  python tools/run_input_oracle_cv.py --project-root . --fold "$FOLD" --seeds 42 43 44 --mode audit
  python tools/run_input_oracle_cv.py --project-root . --fold "$FOLD" --seeds 42 43 44 --mode train --epochs 60 --device cuda
done
```

Derive component bins from each fold's training labels. The Phase 0 64/256
original-pixel bins are descriptive fold features only.

```bash
for FOLD in 0 1 2 3; do
  python tools/derive_vessel_component_thresholds.py \
    --config "runs/input_oracle_cv/registry/fold${FOLD}_noisy_seed42.yaml" \
    --output "runs/input_oracle_cv/fold${FOLD}/component_thresholds.json"
done
```

For each fold, read `component_size_thresholds` from that JSON and evaluate:

```bash
FOLD=0
read SMALL_MAX MEDIUM_MAX < <(python -c "import json; v=json.load(open('runs/input_oracle_cv/fold0/component_thresholds.json'))['component_size_thresholds']; print(*v)")
python tools/run_input_oracle_cv.py --project-root . --fold "$FOLD" --seeds 42 43 44 \
  --mode evaluate --epochs 60 --device cuda --save-predictions \
  --component-size-thresholds "$SMALL_MAX" "$MEDIUM_MAX"
```

Repeat evaluation for folds 1–3, then:

```bash
python tools/analyze_input_oracle_cv.py --project-root . --folds 0 1 2 3 --seeds 42 43 44
python tools/build_input_oracle_atlas.py --project-root . --seed 42
python tools/package_input_oracle_analysis.py --project-root .
```

The report refuses partial inputs. Negative, mixed, or position-dependent
effects are retained. CLEAN is evaluated once per position and its repeat
stability is not interpreted.

## Fixed-final three-input visualization export

After all seed-42 `last.pth` checkpoints have reached epoch 60, the unified
exporter audits the exact initialization/data plan/optimizer parameter set,
reuses complete original-grid validation predictions or runs missing
validation inference, and writes fixed-color per-sample assets plus one panel
per external-validation position:

```bash
python tools/export_input_oracle_visualizations.py \
  --project-root /mnt/SABIDS-Net \
  --runs-root /mnt/SABIDS-Net/runs/input_oracle_cv \
  --output-root /mnt/SABIDS-Net/runs/input_oracle_visualization_seed42_fixed_final \
  --seed 42 \
  --checkpoint last.pth \
  --epoch 60 \
  --threshold 0.5 \
  --postprocess p0 \
  --all-outer-val-frames \
  --make-atlas \
  --make-gpt-bundle \
  --archive-existing \
  --device cuda \
  --num-workers 4
```

The command is deliberately fixed to seed 42, `last.pth`, epoch 60, threshold
0.5 and P0. It never reads a sealed test image. `I-CLEAN` is exported once per
position under `per_position/`; it is not duplicated into the repeat-frame
tree. Float32 probability arrays are retained in the full export but excluded
from the GPT ZIP.

The inherited config currently says `stage2_freeze_shared_encoder: true`, but
the implemented `input_segment` trainability rule freezes only the denoising
and interaction paths. Existing runs whose `parameter_audit.json` lists
`stem`, `encoder_blocks` or `downsamples` as trainable are therefore matched
fine-tuning experiments from the same Stage 1 initialization, not frozen-
encoder probes. The visualization audit reports this mismatch without
rewriting historical results.
