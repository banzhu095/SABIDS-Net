from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from sabids.engine.trainer import build_model
from tools.oct_denoise_benchmark.methods import AdapterContext, denoise
from tools.oct_denoise_benchmark.methods.tcfl_adapter import TCFLDiscriminator, TCFLGenerator
from tools.oct_denoise_benchmark.train_tcfl import TCFLUnpairedDataset, tcfl_discriminator_loss, tcfl_generator_loss
from tools.oct_denoise_benchmark.extension_protocol import init as init_extension, verify_base
from tools.oct_denoise_benchmark.statistics import bootstrap_confidence_intervals


def _write(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((32, 35), value, np.uint8))


def _train_rows(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for position in range(4):
        for frame in range(2):
            noisy, clean = tmp_path / f"n_{position}_{frame}.png", tmp_path / f"c_{position}_{frame}.png"
            _write(noisy, 20 + position); _write(clean, 200 + position)
            rows.append({"dataset": "PKU37", "split": "train", "position_id": f"p{position}", "sample_id": f"p{position}f{frame}", "image_path": str(noisy), "clean_path": str(clean)})
    return pd.DataFrame(rows)


def test_tcfl_unpaired_sampler_is_deterministic_but_never_paired(tmp_path: Path) -> None:
    dataset = TCFLUnpairedDataset(_train_rows(tmp_path), patch_size=16, seed=42, length=20)
    first = dataset[7]; repeated = dataset[7]
    assert first[3:] == repeated[3:]
    assert first[3].split("f")[0] != first[4].split("f")[0]
    assert first[3].split("f")[0] != first[5].split("f")[0]
    assert first[4].split("f")[0] != first[5].split("f")[0]
    assert all(tensor.shape == (1, 16, 16) for tensor in first[:3])


def test_tcfl_loss_and_adapter_checkpoint(tmp_path: Path) -> None:
    generator, discriminator = TCFLGenerator(num_layers=3, features=4), TCFLDiscriminator()
    a, b, c = (torch.rand(1, 1, 32, 32) for _ in range(3))
    loss_g, _, generated = tcfl_generator_loss(generator, discriminator, a, b, c)
    loss_d = tcfl_discriminator_loss(discriminator, c, generated)
    assert torch.isfinite(loss_g) and torch.isfinite(loss_d)
    loss_g.backward(retain_graph=True)
    checkpoint = tmp_path / "tcfl.pth"
    torch.save({"architecture": "tcfl_dncnn", "generator": generator.state_dict()}, checkpoint)
    image = np.random.default_rng(1).random((37, 53), dtype=np.float32)
    context = AdapterContext(checkpoint=checkpoint)
    config = {"method_id": "tcfl_dncnn", "num_layers": 3, "features": 4}
    output = denoise(image, config, context)
    assert output.shape == image.shape and output.dtype == np.float32 and np.isfinite(output).all()
    assert np.array_equal(output, denoise(image, config, context))


def test_sabids_embedded_config_arbitrary_shape_and_strict_checkpoint(tmp_path: Path) -> None:
    config = {"model": {"in_channels": 1, "channels": [4, 8, 16, 32], "encoder_depths": [1, 1, 1, 1], "decoder_depth": 1, "interaction_levels": [3, 2, 1], "enable_seg_to_denoise": True, "enable_denoise_to_seg": True, "use_uncertainty": True, "dropout": 0.0, "residual_scale": 0.5}}
    model = build_model(config)
    checkpoint = tmp_path / "sabids.pth"; torch.save({"config": config, "model": model.state_dict()}, checkpoint)
    image = np.random.default_rng(2).random((37, 53), dtype=np.float32)
    context = AdapterContext(checkpoint=checkpoint)
    output = denoise(image, {"method_id": "sabids_current"}, context)
    assert output.shape == image.shape and output.dtype == np.float32
    assert context.extras["last_padding"] == {"bottom": 3, "right": 3, "mode": "replicate"}
    assert np.array_equal(output, denoise(image, {"method_id": "sabids_current"}, context))


def test_extension_adapters_reject_missing_or_nonfinite_inputs() -> None:
    image = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError, match="locked"):
        denoise(image, {"method_id": "sabids_current"})
    image[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        denoise(image, {"method_id": "tcfl_dncnn"})


def test_split_aware_bootstrap_never_pools_pku_splits() -> None:
    rows = []
    for split, count in (("train", 3), ("val", 2), ("test", 1)):
        for index in range(count):
            for method, value in (("noisy_identity", 20.0), ("sabids_current", 22.0)):
                rows.append({"dataset": "PKU37", "split": split, "position_id": f"{split}_{index}", "method_id": method, "seed": 42, "psnr": value})
    result = bootstrap_confidence_intervals(pd.DataFrame(rows), iterations=100, seed=42)
    means = result[(result.comparison_type == "method_mean") & (result.method_id == "sabids_current")]
    assert means.set_index("split").n_positions.to_dict() == {"train": 3, "val": 2, "test": 1}


def test_extension_initialization_detects_base_run_mutation(tmp_path: Path) -> None:
    root, base, ext = tmp_path / "project", tmp_path / "project" / "runs" / "denoise_benchmark_pku_protocol_1", tmp_path / "project" / "runs" / "denoise_benchmark_plus_sabids_tcfl_1"
    (root / "Manifests").mkdir(parents=True); (base / "metrics").mkdir(parents=True); (base / "audit").mkdir()
    manifest_rows = []
    for index, (dataset, split) in enumerate((("PKU37", "train"), ("PKU37", "val"), ("PKU37", "test"), ("Duke17", "test"), ("Duke28", "test"))):
        noisy, clean = root / "Data" / f"n{index}.png", root / "Data" / f"c{index}.png"; _write(noisy, 20); _write(clean, 200)
        manifest_rows.append({"sample_id": f"s{index}", "group_id": f"g{index}", "dataset": dataset, "split": split, "image_path": noisy.relative_to(root).as_posix(), "clean_path": clean.relative_to(root).as_posix()})
    pd.DataFrame(manifest_rows).to_csv(root / "Manifests" / "manifest_denoise.csv", index=False)
    metric_rows = [{"method_id": method, "dataset": dataset, "status": "success"} for method in ("noisy_identity", "bm3d_standard", "tv_chambolle", "nlm", "ksvd_self", "dncnn_paired", "nafnet_paired") for dataset in ("PKU37", "Duke17", "Duke28")]
    pd.DataFrame(metric_rows).to_csv(base / "metrics" / "per_image_metrics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(base / "metrics" / "per_dataset_metrics.csv", index=False)
    (base / "audit" / "config_lock.json").write_text('{"status":"locked"}', encoding="utf-8")
    init_extension(root, ext, base); verify_base(ext)
    (base / "audit" / "config_lock.json").write_text('{"status":"changed"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="BASE_RUN changed"):
        verify_base(ext)
