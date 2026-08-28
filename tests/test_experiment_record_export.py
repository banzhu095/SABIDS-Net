import io
import tarfile
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from tools.export_experiment_record import SafeArchive, best_row, safe_relative_path
from tools.package_experiment_record import validate_val_only_csv


class ExperimentRecordExportTests(unittest.TestCase):
    def test_safe_archive_rejects_test_result_member(self):
        with TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "source.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                payload = b"split,metric\ntest,0.9\n"
                info = tarfile.TarInfo("runs/test/summary.csv")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(RuntimeError, "test result"):
                SafeArchive(archive_path)

    def test_best_row_prefers_metadata_epoch(self):
        history = pd.DataFrame({"epoch": [1, 2], "val_vessel_soft_dice": [0.9, 0.8]})
        row, status = best_row(history, {"best_epoch": 2})
        self.assertEqual(int(row["epoch"]), 2)
        self.assertEqual(status, "metadata_verified")

    def test_safe_relative_path_removes_absolute_prefix(self):
        self.assertEqual(safe_relative_path("/mnt/SABIDS-Net/runs/current/x.csv"), "runs/current/x.csv")

    def test_package_rejects_non_validation_csv(self):
        with TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "metrics.csv"
            csv_path.write_text("split,metric\ntest,0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-validation"):
                validate_val_only_csv(csv_path)


if __name__ == "__main__":
    unittest.main()
