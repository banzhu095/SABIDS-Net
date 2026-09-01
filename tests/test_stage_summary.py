import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.build_stage_summary import classify, is_forbidden, write_csv


class StageSummaryTests(unittest.TestCase):
    def test_current_stage2_is_ranked_separately(self):
        family, classification, ranked = classify("E3b", "current_matched_protocol", False)
        self.assertEqual((family, classification, ranked), ("C", "current_stage2_ablation", True))

    def test_smoke_is_never_ranked(self):
        family, classification, ranked = classify("stage2_segment_roi_outside", "", True)
        self.assertEqual(family, "A")
        self.assertFalse(ranked)

    def test_forbidden_test_paths_are_detected(self):
        self.assertTrue(is_forbidden(Path("runs/demo/test_results/summary.csv")))
        self.assertFalse(is_forbidden(Path("runs/demo/validation/summary.csv")))

    def test_empty_csv_has_explicit_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            write_csv(path, [], ["status", "reason"])
            self.assertEqual(list(pd.read_csv(path).columns), ["status", "reason"])


if __name__ == "__main__":
    unittest.main()
