# Metric definitions

- Size strata: 8-connected GT vessel components on the 512×512 model grid;
  small ≤ train Q33, medium Q33–Q67, large > Q67. Optional fixed pixel bins are
  sensitivity analysis only.
- Local contrast: GT-layer-ring median minus vessel median, divided by ring IQR
  plus epsilon. Other GT vessels, invalid pixels, and layer exterior
  are excluded. The primary low-contrast threshold is train noisy-input Q25;
  clean contrast is sensitivity-only.
- Primary component recall: fraction of components with coverage ≥0.25. Also
  report any overlap, coverage ≥0.5, mean coverage, and complete misses.
- Residual structure leakage: mean absolute noisy-minus-denoised residual in GT
  vessel or its frozen model-grid boundary band. Separate vessel interior,
  boundary, train-frozen small, low-contrast, small×low, layer-stroma, and
  layer-exterior regions are written to `structure_leakage_metrics.csv`. It is
  diagnostic and is not interpreted as a noise label.
- Position metric source: `group_metrics.csv`; `group_id` is an anatomical
  position. Positions are equally weighted before seed summaries.
