from __future__ import annotations

from pathlib import Path

import pytest

from tools.summarize_input_factorial import prepare_output_directory


def test_summary_output_refuses_nonempty_directory_without_archive(tmp_path: Path):
    output = tmp_path / "report"
    output.mkdir()
    (output / "partial.csv").write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_output_directory(output, archive_existing=False)
    assert (output / "partial.csv").is_file()


def test_summary_output_archives_without_deleting_existing_files(tmp_path: Path):
    output = tmp_path / "report"
    output.mkdir()
    (output / "partial.csv").write_text("partial", encoding="utf-8")
    archive = prepare_output_directory(output, archive_existing=True)
    assert archive is not None
    assert archive.parent == output.parent
    assert archive.name.startswith("report_archive_")
    assert (archive / "partial.csv").read_text(encoding="utf-8") == "partial"
    assert output.is_dir()
    assert not any(output.iterdir())
