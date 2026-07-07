# data/io.py

import pandas as pd
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


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    return pd.read_csv(path)