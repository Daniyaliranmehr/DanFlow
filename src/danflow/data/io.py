# data/io.py

from pathlib import Path
import zipfile


def extract_zip(
    zip_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Extract a ZIP archive to the specified directory.

    Parameters
    ----------
    zip_path
        Path to the ZIP archive.

    output_path
        Directory where the archive will be extracted.
    """

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)