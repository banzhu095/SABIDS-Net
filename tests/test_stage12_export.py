import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import cv2
import torch

from sabids.metrics import binary_metrics, psnr
from tools.export_stage12_results import build_inventory, make_scope_manifest, predict_only


class _FakeStage2(torch.nn.Module):
    def forward(self, image, **kwargs):
        layer = torch.zeros_like(image); layer[:, :, 1:3, :] = 0.9
        vessel = torch.zeros_like(image); vessel[:, :, 1:3, 1:3] = 0.8
        return {"denoised": image * 0.9, "denoised_raw": image * 0.9,
                "layer_prob": layer, "vessel_prob": vessel}


class Stage12ExportTests(unittest.TestCase):
    def test_perfect_psnr_is_infinite_and_binary_counts_are_explicit(self):
        image = np.zeros((4, 4), dtype=np.float32)
        self.assertTrue(np.isinf(psnr(image, image)))
        metrics = binary_metrics(np.array([1, 0, 1]), np.array([1, 1, 0]))
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (1, 1, 1, 0))
        self.assertEqual(metrics["valid_pixels"], 3)

    def test_default_scope_excludes_test_and_sealed_scope_contains_only_test(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            pd.DataFrame([
                {"sample_id": "a", "group_id": "g1", "dataset": "D", "split": "train", "image_path": "Data/a.png"},
                {"sample_id": "b", "group_id": "g2", "dataset": "D", "split": "test", "image_path": "Data/b.png"},
            ]).to_csv(manifest, index=False)
            non_test = make_scope_manifest(manifest, root / "non_test.csv", include_test=False)
            sealed = make_scope_manifest(manifest, root / "sealed.csv", include_test=True)
            self.assertEqual(non_test["original_split"].tolist(), ["train"])
            self.assertEqual(sealed["original_split"].tolist(), ["test"])

    def test_test_linked_clean_is_excluded_without_opening_image(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Data").mkdir()
            for name in ("train.png", "test.png", "shared_clean.png"):
                (root / "Data" / name).write_bytes(b"not-an-image")
            manifest = root / "manifest.csv"
            pd.DataFrame([
                {"sample_id": "a", "group_id": "g1", "dataset": "D", "split": "train",
                 "image_path": "Data/train.png", "clean_path": "Data/shared_clean.png"},
                {"sample_id": "b", "group_id": "g2", "dataset": "D", "split": "test",
                 "image_path": "Data/test.png", "clean_path": "Data/shared_clean.png"},
            ]).to_csv(manifest, index=False)
            inventory, _ = build_inventory(root, manifest, skip_hashes=True)
            clean = inventory[inventory["relative_path"] == "Data/shared_clean.png"].iloc[0]
            self.assertEqual(clean["inference_status"], "excluded_reserved_test_and_linked_asset")

    def test_predict_only_writes_u16_independent_masks_and_float_arrays(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = {
                "image": torch.full((1, 1, 4, 4), 0.5),
                "valid_mask": torch.ones((1, 1, 4, 4)),
                "original_height": torch.tensor([4]), "original_width": torch.tensor([4]),
                "dataset": ["synthetic"], "source_split": ["val"], "group_id": ["g1"],
                "sample_id": ["s1"], "original_path": [str(root / "Data" / "s1.png")],
            }
            table = predict_only(_FakeStage2(), [batch], torch.device("cpu"),
                                 root / "predictions", root, root, "E3b", "segment", False, False)
            denoised = cv2.imdecode(np.fromfile(root / table.iloc[0]["denoised"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            layer = cv2.imdecode(np.fromfile(root / table.iloc[0]["layer_final"], dtype=np.uint8), cv2.IMREAD_UNCHANGED) > 0
            vessel = cv2.imdecode(np.fromfile(root / table.iloc[0]["vessel_p3"], dtype=np.uint8), cv2.IMREAD_UNCHANGED) > 0
            self.assertEqual(denoised.dtype, np.uint16)
            self.assertTrue(np.all(~vessel | layer))
            npz = root / "predictions" / "E3b" / "synthetic" / "val" / "g1" / "s1" / "raw_outputs_float32.npz"
            with np.load(npz) as arrays:
                self.assertEqual(arrays["layer_probability"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
