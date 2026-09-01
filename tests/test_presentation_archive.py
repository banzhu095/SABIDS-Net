import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_presentation_archive import audit_history_csv, error_map, forbidden, package_archive, prepare_output, rgb_overlay


class PresentationArchiveTests(unittest.TestCase):
    def test_test_paths_are_forbidden(self):
        self.assertTrue(forbidden(Path("runs/current/demo/test_results/frame_metrics.csv")))
        self.assertFalse(forbidden(Path("runs/current/demo/final_validation/frame_metrics.csv")))

    def test_overlay_applies_vessel_after_layer(self):
        noisy = np.full((2, 2), 0.5, np.float32)
        layer = np.ones((2, 2), np.float32)
        vessel = np.zeros((2, 2), np.float32); vessel[0, 0] = 1
        result = rgb_overlay(noisy, layer, vessel)
        self.assertGreater(result[0, 0, 0], result[0, 0, 1])
        self.assertGreater(result[1, 1, 1], result[1, 1, 0])

    def test_error_map_uses_requested_classes(self):
        pred = np.array([[1, 1], [0, 0]], np.float32)
        true = np.array([[1, 0], [1, 0]], np.float32)
        image = error_map(pred, true)
        np.testing.assert_array_equal(image[0, 0], [1, 1, 1])
        np.testing.assert_array_equal(image[0, 1], [0, 0, 1])
        np.testing.assert_allclose(image[1, 0], [1, .55, 0], atol=1e-7)

    def test_packages_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "presentation_archive_20260901_000000"
            (output / "packages").mkdir(parents=True)
            (output / "README_FIRST.md").write_text("validation only", encoding="utf-8")
            package_archive(output)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                package_archive(output)

    def test_malformed_history_is_inventoried_without_becoming_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.csv"
            history.write_text("epoch,loss,dice\n1,0.5,0.4\n2,0.4,0.5,unexpected\n", encoding="utf-8")
            result = audit_history_csv(history)
            self.assertEqual(result["history_parse_status"], "malformed_inventory_only")
            self.assertEqual(result["history_rows"], 2)
            self.assertEqual(result["last_history_epoch"], 2)
            self.assertEqual(result["history_bad_line_count"], 1)
            self.assertFalse(result["history_metrics_usable"])

    def test_partial_audit_is_archived_before_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "runs" / "presentation_archive_demo"
            output.mkdir(parents=True)
            (output / "partial.txt").write_text("preserve", encoding="utf-8")
            prepared = prepare_output(root, str(output), "audit", archive_existing=True)
            self.assertEqual(prepared, output)
            self.assertTrue(prepared.is_dir())
            archives = list(output.parent.glob("presentation_archive_demo_archive_*"))
            self.assertEqual(len(archives), 1)
            self.assertEqual((archives[0] / "partial.txt").read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
