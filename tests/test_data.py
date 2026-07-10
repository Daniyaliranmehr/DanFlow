# test/test_data.py

from danflow.data import (
    extract_zip,
    load_csv,
    delimited_to_csv,
)

import pandas as pd
import zipfile
from pathlib import Path
import csv


def test_extract_zip(tmp_path: Path):
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


def test_load_csv(tmp_path: Path):
    csv_path = tmp_path / "data.csv"

    csv_path.write_text(
        "feature,label\n1,0\n2,1\n"
    )

    df = load_csv(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["feature", "label"]


def test_delimited_to_csv_converts_space_delimited_file(tmp_path):
    input_file = tmp_path / "input.trn"
    output_file = tmp_path / "output.csv"

    input_file.write_text(
        "1 2 3\n"
        "4 5 6\n",
        encoding="utf-8",
    )
    
    delimited_to_csv(input_file, output_file)

    with output_file.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    assert rows == [
        ["1", "2", "3"],
        ["4", "5", "6"],
    ]


def test_delimited_to_csv_skips_empty_lines(tmp_path):
    input_file = tmp_path / "input.trn"
    output_file = tmp_path / "output.csv"

    input_file.write_text(
        "1 2 3\n"
        "\n"
        "4 5 6\n"
        "\n",
        encoding="utf-8",
    )

    delimited_to_csv(input_file, output_file)

    with output_file.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    assert rows == [
        ["1", "2", "3"],
        ["4", "5", "6"],
    ]


def test_delimited_to_csv_uses_custom_delimiter(tmp_path):
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.csv"

    input_file.write_text(
        "1|2|3\n"
        "4|5|6\n",
        encoding="utf-8",
    )

    delimited_to_csv(
        input_file,
        output_file,
        delimiter="|",
    )

    with output_file.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    assert rows == [
        ["1", "2", "3"],
        ["4", "5", "6"],
    ]
    