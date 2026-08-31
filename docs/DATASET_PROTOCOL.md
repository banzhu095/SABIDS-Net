# SABIDS-Net dataset protocol

Last consolidated: 2026-08-27

This protocol defines how SABIDS-Net data are interpreted, split, prepared, and
audited. Changes to these rules can invalidate comparisons even when the model
code is unchanged.

## 1. Runtime storage

The repository contains code and configuration only. Runtime directories are:

```text
SABIDS-Net/
├── Data/            # public/private images
├── Label/           # public multiclass and derived binary labels
├── PreparedLabels/  # normalized private binary masks
├── Manifests/       # generated CSV splits and audit reports
└── runs/            # checkpoints, logs, metrics and predictions
```

All five directories are ignored by Git. Do not add raw OCT/OCTA images,
patient identifiers, manual labels, checkpoints, or prediction files to the
repository, including when the GitHub repository is private.

Known roots:

- Windows: `E:\1-脉络膜\OCT降噪\SABIDS-Net`
- Juchiyun: `/mnt/SABIDS-Net`

Generate manifests separately on each operating system so their paths resolve
under that runtime root. Do not reuse a CSV containing `E:\...` paths on Linux.

## 2. Last known data inventory

The following is a recorded preparation result from 2026-08-25, not a hardcoded
dataset constant:

| Dataset | Last known content | Main role |
| --- | ---: | --- |
| Duke17 | 17 noisy-clean pairs | paired denoising/external evaluation |
| Duke28 | 28 noisy-clean pairs | paired denoising/external evaluation |
| PKU37 | 1,734 noisy-clean rows from 37 positions | denoising, RMAC, labelled segmentation subset |
| PKU37 labelled subset | 13 anatomical positions | public layer/vessel supervision and five-fold evaluation |
| Private 12x12 SS-OCT | about 1,000 B-scans, all planned layer labels and sparse vessel labels | private segmentation adaptation |
| Private non-paired dataset 2 | about 100 51-line noisy scans and about 100 HD clean scans | optional domain/noise-bank study |

Before a new experiment, verify the current values in
`Manifests/dataset_report.json`; do not rely on this table if files or labels
have changed.

## 3. Expected source layout and pairing

```text
Data/
├── Duke17/
│   ├── noisy/   # e.g. 1_Raw Image.tif
│   └── clean/   # e.g. 1_Averaged Image.tif
├── Duke28/
│   ├── noisy/   # e.g. LL1.tif or sl1.tif
│   └── clean/   # e.g. HH1.tif or sh1.tif
└── PKU37_OCT_Denoising/
    ├── noisy/   # e.g. 000101.tif ... 000150.tif
    └── clean/   # e.g. 0001.tif

Label/
├── voc_seg/         # PKU37 indexed multiclass masks
├── voc_jpg/         # reference images used in label preparation
├── layer_binary/    # generated
└── vessel_binary/   # generated
```

Automatic public pairing rules:

| Dataset | Rule |
| --- | --- |
| Duke17 | `*_Raw Image` maps to `*_Averaged Image` |
| Duke28 | `LL*` maps to `HH*`; `sl*` maps to `sh*`, case-insensitive |
| PKU37 | first four noisy filename digits are position/clean ID; final two are repeat-frame ID |

Unmatched pairs should stop preparation by default. Use `--allow-unmatched` only
after inspecting and documenting every mismatch.

## 4. Label semantics

### 4.1 Public indexed label

```text
0   = background/outside choroid
1   = choroidal layer stroma/non-vessel tissue
2   = choroidal vessel lumen
255 = ignore, optional
```

Vessel pixels replace the layer class in the indexed mask. Binary targets must
therefore be derived as:

```python
layer_binary = (label == 1) | (label == 2)
vessel_binary = label == 2
```

Never derive the layer target as only `label == 1`, because that creates holes
where vessels lie. Never allow a later layer polygon to overwrite class 2 during
LabelMe conversion.

`tools/prepare_current_data.py` checks unique label values and normally stops on
unexpected classes. After preparation, inspect:

```text
Manifests/dataset_report.json
└── label_report
    ├── unique_values_by_label
    └── area_statistics_by_label
```

Flag masks whose `vessel_fraction_of_layer` is implausibly close to 1 or 0 for
manual review before training.

### 4.2 Private masks

Private source masks may use class-index values such as `0/1/2` or intensity
values such as `0/255`. `tools/prepare_private12x12_data.py` converts each layer
and vessel source independently to `0/255` masks and records:

- original unique values;
- binarization rule;
- foreground ratio;
- image/mask size consistency.

Missing layer labels are errors by default. Missing vessel masks are permitted
and indicate sparse supervision, not a negative vessel label.

## 5. Manifest contract

One CSV row represents one input B-scan. Core fields are:

| Field | Meaning |
| --- | --- |
| `sample_id` | unique row/image identifier |
| `group_id` | same registered anatomy/repeat group; primary anti-leakage unit for PKU37 |
| `patient_id` | patient/eye/volume grouping for patient-level protection |
| `dataset` | dataset identity, e.g. PKU37 or Duke17 |
| `domain` | public/private/private_synthetic |
| `scan_protocol` | acquisition protocol descriptor |
| `frame_index` | repeat or within-volume frame index |
| `split` | train/val/test |
| `image_path` | noisy or current input image |
| `clean_path` | paired clean target, blank when unavailable |
| `layer_mask_path` | binary layer mask, blank when unavailable |
| `vessel_mask_path` | manual binary vessel mask, blank when unavailable |
| `label_valid_mask_path` | optional binary validity for multiclass-derived layer/vessel labels; class 255 is zero |
| `vessel_valid_mask_path` | optional binary mask of pixels with known vessel labels; blank means every pixel is known when a vessel mask exists |
| `is_clean` | whether `image_path` is a clean-only sample |

Public preparation additionally records `multiclass_label_path` and
`has_manual_label`. Private preparation records disease, eye, scan identifiers,
source frame index, original shape, and filename for audit.

Paths may be project-relative. Relative paths are preferred when all files are
under the project root.

Validity masks describe annotation validity, not anatomy. Public multiclass
preparation writes `label_valid_mask_path` and `vessel_valid_mask_path` with
zero at class 255 (and any explicitly allowed unknown class), so ignored pixels
do not become background negatives for either task. A zero
pixel is ignored by vessel supervision and must never be silently interpreted
as a known non-vessel pixel. When the column is absent, a row with
`vessel_mask_path` is treated as fully labelled and a row without it has no
supervised vessel pixels. The Stage 2 ROI diagnostic further intersects this
validity mask with the ground-truth layer mask.

Before every Stage 2 pipeline run, the code compares Stage 1 training groups
with segmentation validation/test groups and writes
`stage1_isolation_audit.json`. A confirmed overlap stops training. Regenerate
this audit whenever newly labelled positions change a fold; an old Stage 1
checkpoint is not automatically valid for a new 20-position split.

Run metadata fingerprints the contents of unique layer, vessel, validity and
multiclass label assets in addition to the manifest text. The comparison tool
requires matching manifest, effective split, label-asset and Stage 1 checkpoint
SHA256 values before calling two runs a protocol-matched ablation. A history
CSV without these hashes is retained as historical evidence but cannot prove a
single-factor effect.

## 6. Split and independence rules

### 6.1 PKU37

- The 50 or fewer noisy frames for one four-digit position ID share one
  `group_id`, clean target, layer mask, vessel mask, and split.
- A repeat frame is an augmentation/consistency observation, not an independent
  patient or anatomical sample.
- Five-fold public vessel evaluation is generated over the 13 labelled
  positions. Each fold holds a position chunk out for test and a subset of the
  remaining labelled positions for validation.
- The Joint manifest contains all denoising rows, but labelled PKU positions are
  reassigned to the fold-specific segmentation split so held-out labels cannot
  enter Joint training.
- If future metadata show that several positions belong to the same patient,
  regenerate folds at `patient_id` level rather than position level.

### 6.2 Duke17 and Duke28

- Each pair currently has its own group.
- These datasets have paired denoising targets but no reliable vessel masks in
  the current project.
- They may support external denoising evaluation, but not public vessel Dice.

### 6.3 Private SS-OCT

- Default split unit is patient (`DYNxxxxx`), not B-scan.
- Both eyes, repeated scans, and adjacent slices from one patient must remain in
  the same split.
- Each ordinary B-scan receives a distinct `group_id`; adjacent B-scans are not
  the same anatomical repeat and must not be sampled by RMAC as a pair.
- Only registered repeated acquisitions of the same location may share a group.
- The split search targets approximately 70%/15%/15% train/val/test and tries
  to preserve disease distribution and place vessel-labelled patients into
  usable splits.
- Validation and test each require manual vessel labels for credible vessel
  model selection and final reporting. If there are too few labelled patients,
  acquire more labels instead of splitting one patient by frame.

## 7. Generated public manifests

`tools/prepare_current_data.py` produces:

```text
Manifests/
├── manifest_all.csv
├── manifest_denoise.csv
├── dataset_report.json
├── segmentation_folds/
│   ├── manifest_seg_fold0.csv
│   └── ... fold1-fold4
└── joint_folds/
    ├── manifest_joint_fold0.csv
    └── ... fold1-fold4
```

- `manifest_denoise.csv`: all valid Duke17, Duke28, and PKU37 noisy-clean rows.
- `manifest_seg_foldN.csv`: labelled PKU37 rows only, with fold-specific splits.
- `manifest_joint_foldN.csv`: all denoising rows plus the fold-specific split
  assignments for labelled PKU positions.

Generate on Windows:

```powershell
cd "E:\1-脉络膜\OCT降噪\SABIDS-Net"
python tools/prepare_current_data.py `
  --project-root "E:\1-脉络膜\OCT降噪\SABIDS-Net" `
  --overwrite-masks
```

Generate on Juchiyun:

```bash
cd /mnt/SABIDS-Net
python tools/prepare_current_data.py \
  --project-root /mnt/SABIDS-Net \
  --overwrite-masks
```

Validate a manifest and its files:

```bash
python tools/validate_manifest.py \
  --manifest Manifests/joint_folds/manifest_joint_fold0.csv \
  --root /mnt/SABIDS-Net \
  --check-files
```

## 8. Private preparation

Recommended Juchiyun layout:

```text
Data/Private12x12/
├── voc_jpg/
├── voc_seg/
└── voc_vessel_seg/
```

Preparation command:

```bash
python tools/prepare_private12x12_data.py \
  --project-root /mnt/SABIDS-Net \
  --image-dir /mnt/SABIDS-Net/Data/Private12x12/voc_jpg \
  --layer-dir /mnt/SABIDS-Net/Data/Private12x12/voc_seg \
  --vessel-dir /mnt/SABIDS-Net/Data/Private12x12/voc_vessel_seg \
  --overwrite-masks

python tools/validate_manifest.py \
  --manifest /mnt/SABIDS-Net/Manifests/manifest_private_seg.csv \
  --root /mnt/SABIDS-Net \
  --check-files
```

Review the generated summary and confirm vessel-labelled rows/patients in every
split before Stage 5.

## 9. Image geometry and normalization

| Experiment | Source geometry | Current target | Rule |
| --- | --- | --- | --- |
| Public main | PKU37 approximately 640x640 | 512x512 | preserve square geometry with resize/pad |
| Public fallback | same | 384x384 | OOM fallback, rerun all compared stages |
| Debug only | mixed | 256x256 | pipeline diagnosis, not final vessel result |
| Private 12x12 | approximately 500x1536 | 320x960 | preserve wide aspect ratio with resize/pad |

The joint transform uses aspect-preserving scaling and padding rather than
independent axial/lateral stretching. Masks use nearest-neighbor interpolation.

- Public paired data: use `normalization: fixed` when intensities are already
  mapped consistently to `[0,1]`.
- Private cross-device data: percentile normalization may be used, currently
  0.5th to 99.5th percentile.
- Do not independently apply nonlinear histogram enhancement to noisy and clean
  targets before paired PSNR/SSIM evaluation.
- Inference pads to the network stride and crops outputs back to original size.

Physical spacing must be updated after resize. If verified spacing is not
available, report boundary and distance metrics in pixels rather than labelling
them as micrometres.

## 10. Non-paired private dataset 2

The 51-line noisy scans and HD clean scans are not aligned pairs. Never place
one arbitrary noisy image and one arbitrary HD image in the same manifest row as
`image_path`/`clean_path` and train with paired reconstruction loss.

The current repository offers `tools/build_noise_bank.py` to estimate residuals
from noisy images and inject them into clean images. This is a synthetic domain
adaptation experiment, not a substitute for the public paired benchmark. Its
limitations include confounding by device, resolution, sampling density, scan
time, and anatomy.

## 11. Pre-training audit checklist

Before launching a long run, record all answers in the experiment log:

- [ ] `dataset_report.json` matches the actual file inventory.
- [ ] Unexpected and unmatched files are zero or manually justified.
- [ ] Label unique values are correct.
- [ ] No vessel mask has an implausible area fraction without visual review.
- [ ] Images and masks have matching original sizes.
- [ ] `sample_id` values are unique.
- [ ] `group_id` never crosses splits.
- [ ] `patient_id` never crosses splits where patient metadata are available.
- [ ] Validation and test contain the required manual labels.
- [ ] Public fold and target size match the checkpoint initialization chain.
- [ ] Paths resolve on the current OS/server.
- [ ] `git status --short` does not show data, labels, manifests, or runs.
# PKU37 input-oracle CV addendum (2026-08-31)

The three-arm NOISY/CLEAN/DENOISED experiment uses only labelled development
PKU37 positions accepted by `configs/current/input_oracle_cv/protocol.yaml`.
Configured test IDs and any legacy manifest test IDs form a conservative sealed
union until their discrepancy is resolved; no asset in that union may be opened.
Each development `group_id` is validation exactly once. Each fold-specific D0
excludes that fold's validation groups and the sealed union before generating
float32 inputs. One clean reference may be sampled repeatedly during paired
training but is evaluated once per position and is never treated as repeated
independent evidence.
