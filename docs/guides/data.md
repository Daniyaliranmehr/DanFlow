# Working with Datasets

## Introduction

The `danflow.data` module provides simple utilities for preparing file-based datasets before they are used in machine learning and deep learning workflows.

Instead of writing repetitive file-handling code for common dataset formats, DanFlow provides utilities for:

* extracting ZIP archives,
* converting delimited text files to CSV,
* loading CSV files into pandas `DataFrame` objects.

The goal of this guide is not to describe every function parameter. Instead, it explains how to use the data module effectively, how to choose the appropriate operation for a dataset, and which common data-preparation mistakes to avoid.

A typical preparation workflow can be organized into three stages:

```text
Dataset source
    ZIP archive or text file

File preparation
    Extract archive
    Convert delimited data

Data loading
    Load CSV into DataFrame
```

Once the dataset has been loaded into a `DataFrame`, standard pandas operations can be used for inspection, cleaning, transformation, and preparation for model training.


## Understanding the Data Preparation Workflow

Before using any DanFlow data utility, first identify the format of the data you actually have.

For example, a dataset may be distributed as:

```text
dataset.zip
    dataset.txt
```

where the text file contains:

```text
feature_1 feature_2 label
1 10 0
2 20 1
3 30 0
4 40 1
5 50 1
```

In this situation, there are three different operations:

1. The ZIP archive must be extracted.
2. The space-delimited text file can be converted to CSV.
3. The resulting CSV can be loaded into pandas.

These operations solve different problems. They should not be treated as interchangeable steps.


## Extracting a Dataset

Many datasets are distributed as ZIP archives, especially when several files are packaged together.

Use `extract_zip()` when the data you need is stored inside a ZIP archive.

```python
from danflow.data import extract_zip

extract_zip(
    zip_path="dataset.zip",
    output_path="data",
)
```

After extraction, the project may contain:

```text
project/
    data/
        dataset.txt
    train.py
```

### When to Extract

Use ZIP extraction when the source dataset is actually archived.

Do not use `extract_zip()` for an ordinary CSV or text file that is already available on disk. In that case, move directly to the appropriate loading or conversion step.

### Keep Raw Data Separate

A useful project structure is to keep downloaded or extracted source data separate from generated files:

```text
project/
    data/
        raw/
            dataset.txt
        processed/
            dataset.csv
    train.py
```

This makes it easier to distinguish the original dataset from files generated during preparation.

The exact directory structure is a project-level choice; the important principle is to avoid overwriting the original source data unnecessarily.


## Converting Delimited Text Files

Not every dataset is distributed as CSV.

Text files may use spaces, tabs, pipes, or other delimiters:

```text
feature_1 feature_2 label
1 10 0
2 20 1
3 30 0
```

or:

```text
feature_1|feature_2|label
1|10|0
2|20|1
3|30|0
```

or:

```text
feature_1   feature_2   label
1   10  0
2   20  1
3   30  0
```

Use `delimited_to_csv()` when the source file is a plain text file whose values are separated by a known delimiter.

For a space-delimited file:

```python
from danflow.data import delimited_to_csv

delimited_to_csv(
    input_path="data/dataset.txt",
    output_path="data/dataset.csv",
    delimiter=" ",
)
```

For a pipe-delimited file:

```python
delimited_to_csv(
    input_path="data/dataset.txt",
    output_path="data/dataset.csv",
    delimiter="|",
)
```

For a tab-delimited file:

```python
delimited_to_csv(
    input_path="data/dataset.txt",
    output_path="data/dataset.csv",
    delimiter="\t",
)
```

### Choose the Delimiter Carefully

The delimiter must match the actual file format.

For example, a file containing:

```text
10|20|1
```

must be converted using:

```python
delimiter="|"
```

Using:

```python
delimiter=" "
```

would not correctly describe that file.

This matters because `delimited_to_csv()` splits each input line using the exact delimiter provided to the function.

### Spaces Are Not the Same as Arbitrary Whitespace

The default delimiter is a single space:

```python
delimiter=" "
```

This does not mean "any amount of whitespace."

For example, a file with inconsistent spacing such as:

```text
1  10  0
2    20    1
```

does not have the same structure as a consistently single-space-delimited file.

Before converting a dataset, inspect how values are actually separated instead of assuming that all whitespace is equivalent.


## Handling Empty Lines

Delimited datasets can contain blank lines between records.

For example:

```text
feature_1 feature_2 label
1 10 0

2 20 1

3 30 0
```

`delimited_to_csv()` strips each line and skips lines that become empty.

Therefore, blank lines do not become empty CSV rows during conversion.

This makes the function useful for simple text datasets that contain occasional empty lines.

However, empty lines should still be distinguished from missing values inside an actual record. For example:

```text
1 10
```

is not an empty line. It is an incomplete data record and should be handled as a data-quality issue rather than relying on the empty-line behavior.


## Handling File Encoding

Text files are not always encoded as UTF-8.

`delimited_to_csv()` uses UTF-8 by default:

```python
delimited_to_csv(
    input_path="data/dataset.txt",
    output_path="data/dataset.csv",
    delimiter="|",
)
```

When the source file uses another encoding, specify it explicitly:

```python
delimited_to_csv(
    input_path="data/dataset.txt",
    output_path="data/dataset.csv",
    delimiter="|",
    encoding="latin-1",
)
```

The same encoding is used when reading the input file and writing the resulting CSV.

### When Encoding Problems Appear

Encoding becomes particularly important when a dataset contains:

* non-English text,
* accented characters,
* legacy files,
* files created by older software,
* files exported from systems using a non-UTF-8 encoding.

A `UnicodeDecodeError` while reading a text dataset is often a sign that the selected encoding does not match the actual file.

Do not randomly change the encoding until the error disappears. Determine the source file's encoding when possible and use that encoding consistently.


## Loading the CSV into pandas

Once a CSV file is available, use `load_csv()` to load it into a pandas `DataFrame`.

```python
from danflow.data import load_csv

df = load_csv("data/dataset.csv")
```

The returned object is a standard pandas `DataFrame`:

```pycon
>>> type(df)
<class 'pandas.core.frame.DataFrame'>
```

From this point onward, pandas can be used normally:

```pycon
>>> df.shape
(5, 3)

>>> df.columns.tolist()
['feature_1', 'feature_2', 'label']
```

For example, inspect the first records with:

```pycon
>>> df.head(3)
   feature_1  feature_2  label
0          1         10      0
1          2         20      1
2          3         30      0
```

DanFlow's responsibility ends at loading the CSV. Operations such as filtering, missing-value handling, type conversion, feature engineering, splitting, and normalization can then be performed with pandas and the other tools in the project.


## Make Sure the CSV Has the Expected Header

One important detail when using `load_csv()` is that it delegates directly to pandas CSV loading.

A typical dataset should therefore contain a header:

```text
feature_1,feature_2,label
1,10,0
2,20,1
3,30,0
```

This is different from a headerless file such as:

```text
1,10,0
2,20,1
3,30,0
```

When a CSV does not contain a header, the default pandas CSV behavior treats the first row as column names.

That means the first record can be consumed as the header instead of being loaded as data.

Before calling `load_csv()`, determine whether the first row contains column names or actual observations.


## Keep Conversion and Loading as Separate Steps

It is useful to distinguish these two operations:

```text
Text file
    delimited_to_csv()

CSV file
    load_csv()
```

`delimited_to_csv()` changes the file representation.

`load_csv()` loads the resulting CSV into memory as a pandas `DataFrame`.

Keeping these responsibilities separate makes the workflow easier to inspect and debug.

For example:

```python
from danflow.data import delimited_to_csv, load_csv

delimited_to_csv(
    "data/dataset.txt",
    "data/dataset.csv",
    delimiter=" ",
)

df = load_csv("data/dataset.csv")
```

This also gives you the opportunity to inspect the generated CSV before loading it.


## Inspect the Dataset Immediately After Loading

Loading a dataset successfully does not mean that the data is ready for training.

After loading, perform a small set of basic checks:

```pycon
>>> df.shape
(5, 3)

>>> df.columns.tolist()
['feature_1', 'feature_2', 'label']

>>> df.head()
   feature_1  feature_2  label
0          1         10      0
1          2         20      1
2          3         30      0
3          4         40      1
4          5         50      1
```

At minimum, confirm:

* the expected number of rows and columns,
* the expected column names,
* the first few records,
* whether the values appear in the correct columns.

For machine-learning workflows, continue with checks such as data types, missing values, duplicated rows, and target distribution before training a model.


## Common Mistakes

### Using the Wrong Delimiter

A file separated by `|` should not be converted with the default space delimiter.

Incorrect:

```python
delimited_to_csv(
    "data/dataset.txt",
    "data/dataset.csv",
    delimiter=" ",
)
```

when the source actually contains:

```text
1|10|0
2|20|1
```

Use:

```python
delimiter="|"
```

instead.

### Assuming All Whitespace Is Equivalent

Because the conversion uses the exact delimiter supplied to the function, inconsistent spacing can produce unexpected columns.

Inspect the raw file before selecting the delimiter.

### Loading Headerless CSV Data Without Checking

A CSV without a header can cause its first record to be interpreted as column names.

Check the file structure before loading it.

### Using the Wrong Encoding

A file containing non-UTF-8 text may fail when read with the default encoding.

Use the source file's actual encoding when necessary.

### Assuming Generated Directories Are Created Automatically

`delimited_to_csv()` opens the specified output path directly. It does not create missing parent directories.

For example, this requires the destination directory to already exist:

```python
delimited_to_csv(
    "data/dataset.txt",
    "data/processed/dataset.csv",
)
```

If `data/processed/` does not exist, create it before calling the function.

```python
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
```

Then perform the conversion.

### Overwriting the Raw Dataset

Do not use the original raw file as the destination for a converted dataset.

Prefer separate input and output paths:

```text
data/
    raw/
        dataset.txt
    processed/
        dataset.csv
```

This keeps the original source available for verification or future reprocessing.


## Choosing the Right Data Operation

A simple decision process is useful when starting a new dataset.

| Dataset situation                         | Recommended operation                                |                                                  |
| ----------------------------------------- | ---------------------------------------------------- | ------------------------------------------------ |
| Data is stored inside a ZIP archive       | `extract_zip()`                                      |                                                  |
| Data is already a CSV file                | `load_csv()`                                         |                                                  |
| Data is a space-delimited text file       | `delimited_to_csv()`                                 |                                                  |
| Data uses `                               | `, tab, or another delimiter                         | `delimited_to_csv()` with the matching delimiter |
| Text file uses a non-UTF-8 encoding       | `delimited_to_csv()` with the appropriate `encoding` |                                                  |
| Dataset has already been converted to CSV | `load_csv()`                                         |                                                  |

The key question is not "Which DanFlow function should I call first?" but rather:

> What format is my dataset currently in, and what format do I need for the next stage?


## Recommended Data Preparation Workflow

For a common file-based dataset, a practical workflow is:

### 1. Identify the Source Format

Determine whether the dataset is:

* archived,
* delimited text,
* CSV,
* or another format outside the scope of the data module.

### 2. Preserve the Raw Source

Keep the original archive or text file unchanged whenever possible.

### 3. Extract Archives When Necessary

Use `extract_zip()` when the required files are inside a ZIP archive.

### 4. Convert Delimited Text When Necessary

Use `delimited_to_csv()` when the source is a plain delimited text file.

Choose the delimiter and encoding based on the actual file.

### 5. Load the CSV

Use `load_csv()` once a valid CSV file is available.

### 6. Inspect the Resulting DataFrame

Check:

```text
Shape
Column names
Data types
Missing values
Sample records
Target column
```

### 7. Continue with Preprocessing

After the data has been validated, continue with the preprocessing and modeling workflow appropriate for the project.


## A Typical Workflow

A small end-to-end data preparation workflow can look like this:

```python
from pathlib import Path

from danflow.data import (
    extract_zip,
    delimited_to_csv,
    load_csv,
)

Path("data/processed").mkdir(parents=True, exist_ok=True)

extract_zip(
    "dataset.zip",
    "data/raw",
)

delimited_to_csv(
    "data/raw/dataset.txt",
    "data/processed/dataset.csv",
    delimiter=" ",
)

df = load_csv(
    "data/processed/dataset.csv",
)
```

After loading:

```pycon
>>> type(df)
<class 'pandas.core.frame.DataFrame'>

>>> df.shape
(5, 3)

>>> df.columns.tolist()
['feature_1', 'feature_2', 'label']
```

At this stage, the file-handling part of the workflow is complete. The `DataFrame` can now be passed to the next stage of the machine-learning pipeline.


## Summary

Effective use of DanFlow's data module starts with understanding the source format rather than calling functions mechanically.

Use:

* `extract_zip()` for archived datasets,
* `delimited_to_csv()` for plain delimited text files,
* `load_csv()` for loading CSV data into pandas.

The most important practical considerations are selecting the correct delimiter, matching the file encoding, preserving the raw dataset, ensuring the CSV has the expected header structure, and inspecting the resulting `DataFrame` before moving into preprocessing or training.

For a complete executable example of this workflow, see [Data Preparation](../examples/data_preparation.md).