# PKU37 OCT denoising benchmark

This package is the active seven-method benchmark. The sibling `OCT_denoise`
directory is a historical third-party archive and is not the runtime repository.
Raw data, manifests, runs, checkpoints and predictions stay outside Git.

## Scientific contract

- Calibration and checkpoint selection use PKU37 validation only, aggregated
  frame -> position -> dataset. PKU37 test and Duke references stay sealed until
  `audit/config_lock.json` says `locked`.
- Deep models use seeds 42, 123 and 2026. Results report all seeds; seed 42 is
  preregistered as the primary downstream checkpoint and is never chosen from
  test performance.
- DnCNN is grayscale, 17-layer, 64-feature residual-noise prediction with
  residual MSE. AdamW plus cosine decay is an explicit OCT adaptation, not a
  claim of reproducing the original DnCNN training recipe.
- `NAFNet-paired-OCT` uses the official SIDD width-32 topology
  `enc=[2,2,4,8], middle=12, dec=[2,2,2,2]`, adapted to one channel and trained
  from random initialization on PKU37. No SIDD pretrained weights are used.
- Deep training targets 100,000 optimizer updates and effective batch 4.
  Fixed fast validation runs every 1,000 updates; complete 277-frame validation
  runs every 5,000 updates and alone can select a formal checkpoint.
- `ksvd_self` is a local K-SVD-style implementation. Its standard comparison
  fixes `noise_weight=1` and `aggregation_weight=0`; its finite fractional grid
  is not claimed to be a global optimum.

The official NAFBlock reference is
https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py.
The local order is norm1 -> conv1 -> depthwise conv2 -> SimpleGate -> SCA ->
conv3 -> beta residual.

## ModelWhale/Juchiyun tracks

Use a new run directory after checking out the final committed tag. The deep
track does not wait for classical calibration, and each track records its own
hash-bound state under `audit/tracks`.

```bash
cd /mnt/SABIDS-Net
RUN=/mnt/SABIDS-Net/runs/denoise_benchmark_pku_protocol_$(date +%Y%m%d_%H%M%S)
bash tools/oct_denoise_benchmark/scripts/run_modelwhale_protocol.sh --project-root /mnt/SABIDS-Net --run-dir "$RUN" --track preflight
bash tools/oct_denoise_benchmark/scripts/run_modelwhale_protocol.sh --project-root /mnt/SABIDS-Net --run-dir "$RUN" --track deep --resume
bash tools/oct_denoise_benchmark/scripts/run_modelwhale_protocol.sh --project-root /mnt/SABIDS-Net --run-dir "$RUN" --track classical --resume
bash tools/oct_denoise_benchmark/scripts/run_modelwhale_protocol.sh --project-root /mnt/SABIDS-Net --run-dir "$RUN" --track merge --resume
bash tools/oct_denoise_benchmark/scripts/run_modelwhale_protocol.sh --project-root /mnt/SABIDS-Net --run-dir "$RUN" --track evaluate --resume
bash tools/oct_denoise_benchmark/scripts/run_modelwhale_protocol.sh --project-root /mnt/SABIDS-Net --run-dir "$RUN" --track package --resume
```

`--track full` runs the same sequence. The deep track first performs a 1,000
update GPU smoke plus a resume load for each architecture, then starts the three
formal seeds. If physical batch 4 is not viable, rerun in a fresh formal run
with a preregistered `--batch-size`/`--accumulation-steps` pair whose product is
4. Do not change it after sealed evaluation starts.

Workbook generation is optional. Missing Node or artifact-tool marks that step
`skipped_optional`; CSV, JSON, images and checkpoints remain valid outputs.

## Protocol-locked file inference

Windows PowerShell, single file:

```powershell
python -m tools.oct_denoise_benchmark.cli denoise-file `
  --project-root "E:\1-脉络膜\OCT降噪\SABIDS-Net\SABIDS-Net" `
  --method nafnet_paired --input-file C:\input\scan.tif `
  --output-file C:\output\scan.tif `
  --config C:\run\configs\inference_registry.yaml `
  --checkpoint C:\run\checkpoints\nafnet_paired\seed_42\best_psnr.pth --device cuda:0
```

Linux, recursive folder:

```bash
python -m tools.oct_denoise_benchmark.cli denoise-folder \
  --project-root /mnt/SABIDS-Net --method tv_chambolle \
  --input-dir /path/input --output-dir /path/output \
  --config "$RUN/configs/inference_registry.yaml" --recursive --resume
```

Supported inputs are PNG, TIFF, JPEG and BMP. Processing is grayscale
`float32 [0,1]` with clipping only: no gamma, histogram equalization, resize or
crop. Output keeps source geometry and, for PNG/TIFF, source bit depth when the
codec supports it. Folder mode preserves relative paths. Existing output is
skipped only when input, adapter source, config, checkpoint and output hashes
all match `inference_manifest.csv`; failures are retained in `failures.csv`.

Full-frame inference is the default. If a 640x640 frame does not fit, use
`--tile-size 512 --tile-overlap 64`; tiled inference uses cosine feathering.
Compare overlaps 32/64/128 against full-frame output on validation and lock the
choice before opening test or Duke references.

## Local verification

```powershell
python -m compileall -q tools\oct_denoise_benchmark tests\test_denoise_protocol_v1.py
python -m pytest -q
git diff --check
```
