from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tools.denoise_result_review.image_io import sha256_file
from tools.denoise_result_review.package_test_images import package_test_images
from tools.denoise_result_review.roi_evaluation import evaluate_rois
from tools.denoise_result_review.roi_registry import ROIRegistry
from tools.denoise_result_review.roi_visualization import build_panels
from tools.denoise_result_review.validation import audit_local, find_image_manifest


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True); encoded.tofile(path)


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    for directory in ("metrics", "configs", "audit", "manifests"): (run / directory).mkdir(parents=True, exist_ok=True)
    source = tmp_path / "中文源"; source.mkdir()
    noisy, reference = source / "noisy.tif", source / "reference.tif"
    _write(noisy, np.full((32, 40), 1000, np.uint16)); _write(reference, np.full((32, 40), 900, np.uint16))
    rows = []
    for method, seed in (("bm3d_standard", 0), ("tv_chambolle", 0), ("nlm", 0)):
        output = source / f"{method}.tif"; _write(output, np.full((32, 40), 950, np.uint16))
        rows.append({"dataset": "PKU37", "split": "test", "position_id": "pku_0006", "sample_id": "pku_0006_f26", "method_id": method, "seed": seed, "status": "success", "noisy_path": str(noisy), "reference_path": str(reference), "denoised_path": str(output), "checkpoint_sha256": "", "output_sha256": sha256_file(output), "width": 40, "height": 32, "bit_depth": 16})
    frame = pd.DataFrame(rows); frame.to_csv(run / "metrics" / "per_image_metrics.csv", index=False); frame.to_csv(run / "manifests" / "denoised_dataset_manifest.csv", index=False)
    (run / "configs" / "inference_registry.yaml").write_text("methods:\n  bm3d_standard: {seed: 0}\n  tv_chambolle: {seed: 0}\n  nlm: {seed: 0}\n", encoding="utf-8")
    return run


def test_position_package_is_lossless_zip64_crc_and_resumable(tmp_path: Path):
    run = _run(tmp_path); source_hashes = {path.name: sha256_file(path) for path in (tmp_path / "中文源").iterdir()}
    result = package_test_images(tmp_path, run, tmp_path / "packages", methods=["bm3d_standard", "tv_chambolle", "nlm"], include_noisy=True, include_reference=True, archive_by_position=True, archive_all=True)
    assert result["status"] == "completed" and result["positions"] == 1 and result["samples"] == 1
    package_root = Path(result["package_root"]); manifest = pd.read_csv(package_root / "manifests" / "image_manifest.csv")
    assert len(manifest) == 5 and manifest.sha256.map(len).eq(64).all()
    for row in manifest.itertuples(): assert sha256_file(package_root / row.packaged_path) == row.sha256
    for archive in list((package_root / "archives").glob("*.zip")) + list(package_root.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle: assert bundle.testzip() is None and "IMAGE_MANIFEST.csv" in bundle.namelist()
    assert source_hashes == {path.name: sha256_file(path) for path in (tmp_path / "中文源").iterdir()}
    resumed = package_test_images(tmp_path, run, tmp_path / "packages", methods=["bm3d_standard", "tv_chambolle", "nlm"], include_noisy=True, include_reference=True, archive_by_position=True, archive_all=True, resume=True)
    assert all(item["status"] == "resumed_existing" for item in resumed["archives"])

    # A changed package selection must rebuild rather than silently reuse an old valid ZIP.
    reduced = package_test_images(tmp_path, run, tmp_path / "packages", methods=["bm3d_standard", "tv_chambolle"], samples=["pku_0006_f26"], include_noisy=True, include_reference=True, archive_by_position=True, archive_all=True, resume=True)
    assert all(item["status"] == "created" for item in reduced["archives"])
    assert reduced["image_files"] == 4


def test_package_dry_run_does_not_create_output(tmp_path: Path):
    run = _run(tmp_path); output = tmp_path / "not-created"
    result = package_test_images(tmp_path, run, output, methods=["bm3d_standard", "tv_chambolle", "nlm"], dry_run=True)
    assert result["status"] == "dry_run" and not output.exists()


def test_extracted_archive_uppercase_manifest_is_discovered(tmp_path: Path):
    manifest = tmp_path / "IMAGE_MANIFEST.csv"
    manifest.write_text("sample_id\nexample\n", encoding="utf-8")
    assert find_image_manifest(tmp_path) == manifest.resolve()


def test_realistic_package_audit_locked_roi_common_crop_and_missing_panel(tmp_path: Path):
    run = _run(tmp_path)
    result = package_test_images(tmp_path, run, tmp_path / "packages", methods=["bm3d_standard", "tv_chambolle", "nlm"], include_noisy=True, include_reference=True, archive_by_position=True)
    package_root = Path(result["package_root"])
    audit = audit_local(package_root, tmp_path / "review")
    assert audit["status"] == "incomplete" and audit["issues"] == 0  # six formal methods are intentionally absent

    registry = ROIRegistry(tmp_path / "review")
    registry.add(dataset="PKU37", split="test", position_id="pku_0006", sample_id="pku_0006_f26", tissue="custom", roi_size=32, center_x=20, center_y=16, width=40, height=32, selection_image="noisy_and_reference_only", selection_reason="synthetic locked test", reference_sha256=sha256_file(tmp_path / "中文源" / "reference.tif"), noisy_sha256=sha256_file(tmp_path / "中文源" / "noisy.tif"))
    registry.lock()
    evaluation = evaluate_rois(package_root, registry.csv_path, tmp_path / "review", True)
    assert evaluation["analysis_scope"] == "exploratory_descriptive" and evaluation["records"] == 4
    per_roi = pd.read_csv(tmp_path / "review" / "metrics" / "per_roi_metrics.csv")
    assert per_roi[["x0", "y0", "x1", "y1"]].drop_duplicates().to_dict("records") == [{"x0": 4, "y0": 0, "x1": 36, "y1": 32}]
    panels = build_panels(package_root, registry.csv_path, tmp_path / "review")
    assert panels["roi_panels"] == 1
    roi_id = registry.frame.iloc[0].roi_id
    assert (tmp_path / "review" / "comparison_panels" / f"{roi_id}.png").is_file()
    assert not (tmp_path / "review" / "crops" / "raw" / f"{roi_id}__ksvd_self.tif").exists()
    crop = np.load(tmp_path / "review" / "crops" / "arrays" / f"{roi_id}__bm3d_standard.npy")
    np.testing.assert_allclose(crop, np.full((32, 32), 950 / 65535, np.float32))
