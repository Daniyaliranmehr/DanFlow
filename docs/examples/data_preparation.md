# Data Preparation

This example demonstrates a complete data preparation workflow with DanFlow.

The workflow starts with a delimited text file stored inside a ZIP archive, extracts the archive, converts the text file to CSV, and loads the resulting CSV into a pandas DataFrame.

The same workflow can be used as a starting point before preprocessing, visualization, or model training.


## Input Data

The dataset is stored in `dataset.zip` and contains the following text file:

```text
feature_1 feature_2 label
1 10 0
2 20 1
3 30 0
4 40 1
5 50 1
```

The values are separated by a single space.


## Extracting the Dataset

Extract the archive into the `data` directory:

```python
from danflow.data import extract_zip

extract_zip(
    "dataset.zip",
    "data",
)
```

The extracted file contains:

```text
feature_1 feature_2 label
1 10 0
2 20 1
3 30 0
4 40 1
5 50 1
```


## Convert the Text File to CSV

The extracted file is space-delimited, so specify a single space as the delimiter.
```python
from danflow.data import delimited_to_csv

delimited_to_csv(
    "data/dataset.txt",
    "data/dataset.csv",
    delimiter=" ",
)
```

The resulting CSV file is:
```csv
feature_1,feature_2,label
1,10,0
2,20,1
3,30,0
4,40,1
5,50,1
```

## Loading the CSV

Once the CSV file has been created, load it into pandas with `load_csv()`:

```python
from danflow.data import load_csv

df = load_csv("data/dataset.csv")
```

The result can be inspected directly in the Python console:

```pycon
>>> df
   feature_1  feature_2  label
0          1         10      0
1          2         20      1
2          3         30      0
3          4         40      1
4          5         50      1
```

Because `load_csv()` returns a pandas `DataFrame`, standard pandas operations can be used immediately:

```pycon
>>> type(df)
<class 'pandas.core.frame.DataFrame'>

>>> df.shape
(5, 3)

>>> df.columns.tolist()
['feature_1', 'feature_2', 'label']
```

For example, `head()` can be used to inspect the first records:

```pycon
>>> df.head(3)
   feature_1  feature_2  label
0          1         10      0
1          2         20      1
2          3         30      0
```


## Using a Different Delimiter

The same conversion workflow can be used when the input file uses another delimiter.

For example, suppose a separate file uses `|`:
```text
feature_1|feature_2|label
1|10|0
2|20|1
3|30|0
4|40|1
5|50|1
```

Convert this file by specifying `|`:
```python
delimited_to_csv(
    "data/pipe_dataset.txt",
    "data/pipe_dataset.csv",
    delimiter="|",
)
```
The resulting CSV is:
```csv
feature_1,feature_2,label
1,10,0
2,20,1
3,30,0
4,40,1
5,50,1
```

The input and output paths are kept separate from the main example to make it clear that this is a different source file.


## Handling Empty Lines

Delimited files may contain empty lines between records.

For example:

```text
feature_1 feature_2 label
1 10 0

2 20 1

3 30 0
```

The same conversion call can be used:

```python
delimited_to_csv(
    "data/dataset.txt",
    "data/dataset.csv",
    delimiter=" ",
)
```

The empty lines are omitted from the resulting CSV:

```csv
feature_1,feature_2,label
1,10,0
2,20,1
3,30,0
```


## Complete Workflow

The complete workflow can be reduced to three DanFlow operations:
```python
import pandas as pd

from danflow.data import (
    extract_zip,
    delimited_to_csv,
    load_csv,
)

extract_zip(
    "dataset.zip",
    "data",
)

delimited_to_csv(
    "data/dataset.txt",
    "data/dataset.csv",
    delimiter=" ",
)

df = load_csv("data/dataset.csv")
```
The resulting `DataFrame` can then be passed to the next stage of the machine-learning workflow.
```pycon
>>> df.shape
(5, 3)
```

```pycon
>>> df.columns.tolist()
['feature_1', 'feature_2', 'label']
```

```pycon
>>> isinstance(df, pd.DataFrame)
True
```


At this point, the raw dataset has been extracted, converted into a standard CSV format, and loaded into a pandas DataFrame ready for further processing.