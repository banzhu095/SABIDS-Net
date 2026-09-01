import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_presentation_archive import error_map, forbidden, package_archive, rgb_overlay


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


if __name__ == "__main__":
    unittest.main()
