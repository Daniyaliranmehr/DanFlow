# data/io.py

import pandas as pd
from pathlib import Path
import zipfile
import csv


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


def delimited_to_csv(
    input_path: str | Path,
    output_path: str | Path,
    delimiter: str = " ",
    encoding: str = "utf-8",
) -> None:
    """
    Convert a delimited text file to a CSV file.

    Parameters
    ----------
    input_path
        Path to the input text file.

    output_path
        Path where the CSV file will be saved.

    delimiter
        Character separating values in the input file.

    encoding
        Encoding of the input file.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding=encoding) as infile:
        with output_path.open("w", newline="", encoding=encoding) as outfile:
            writer = csv.writer(outfile)

            for line in infile:
                line = line.strip()

                if not line:
                    continue

                writer.writerow(line.split(delimiter))
                