# Data

## extract_zip()

Extracts a ZIP archive to a specified directory.

### Parameters

#### `zip_path` : `str | Path`
Path to the ZIP archive.

#### `output_path` : `str | Path`
Directory where the archive will be extracted.

### Returns

`None`

### Example

```python
from danflow.data.io import extract_zip

extract_zip(
    "dataset.zip",
    "data/"
)
```

---

## load_csv()

Loads a CSV file into a pandas DataFrame.

### Parameters

#### `path` : `str | Path`

Path to the CSV file.

### Returns

`pd.DataFrame`


### Example

```python
from danflow.data.io import load_csv

df = load_csv("dataset.csv")
```

---

## delimited_to_csv()

Converts a delimited text file into a CSV file.

Each non-empty line from the input file is split using the specified delimiter and written as a row in the output CSV file.

### Parameters

#### `input_path` : `str | Path`

Path to the input text file.

#### `output_path` : `str | Path`

Path where the generated CSV file will be saved.

#### `delimiter` : `str`, default=`" "`

Character used to separate values in the input file.

#### `encoding` : `str`, default=`"utf-8"`

Encoding used to read the input file and write the output CSV file.

### Returns

`None`

### Example

Suppose `data.txt` contains:

```text
Ali;20;CS
Sara;21;Math
Reza;19;Physics
```

Convert it to a CSV file:

```python
from danflow.data.io import delimited_to_csv

delimited_to_csv(
    input_path="students.txt",
    output_path="students.csv",
    delimiter=";"
)
```

The generated `data.csv` will contain:

```text
Ali,20,CS
Sara,21,Math
Reza,19,Physics
```