import io
import tarfile
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from tools.export_experiment_record import SafeArchive, best_row, export_test_results, safe_relative_path
from tools.package_experiment_record import validate_scope_csv


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

    def test_explicit_test_archival_copies_without_aggregation(self):
        with TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            archive_path = temporary / "source.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                payload = b"split,dice\ntest,0.7\n"
                info = tarfile.TarInfo("runs/current/demo/test_results/summary.csv")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            archive = SafeArchive(archive_path, allow_test_results=True)
            output = temporary / "output"
            output.mkdir()
            index = export_test_results(archive, output)
            archive.close()
            self.assertEqual(len(index), 1)
            self.assertEqual(index.iloc[0]["selection_or_calibration_use"], "forbidden")
            self.assertTrue((output / index.iloc[0]["exported_path"]).is_file())

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
                validate_scope_csv(csv_path, include_test_results=False)


if __name__ == "__main__":
    unittest.main()
