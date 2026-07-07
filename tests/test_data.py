# test/test_data.py

from danflow.data import (
    extract_zip,
)

import zipfile
from pathlib import Path

def test_extract_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "archive.zip"
    output_dir = tmp_path / "output"

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("file1.txt", "Hello")
        zip_file.writestr("folder/file2.txt", "World")

    extract_zip(zip_path, output_dir)

    assert (output_dir / "file1.txt").exists()
    assert (output_dir / "folder" / "file2.txt").exists()

    assert (output_dir / "file1.txt").read_text() == "Hello"
    assert (output_dir / "folder" / "file2.txt").read_text() == "World"