# Data Preparation

This example demonstrates a practical data preparation workflow with DanFlow and pandas.

The workflow starts with a delimited text file stored inside a ZIP archive, converts the raw data to CSV, and loads the result into a pandas `DataFrame`.

The DanFlow utilities used in this example are:

```python
from danflow.data import (
    extract_zip,
    delimited_to_csv,
    load_csv,
)
```

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


## Converting the Delimited File

The extracted file uses spaces as separators. Convert it to CSV by specifying the delimiter:

```python
from danflow.data import delimited_to_csv

delimited_to_csv(
    "data/dataset.txt",
    "data/dataset.csv",
    delimiter=" ",
)
```

### Input

```text
feature_1 feature_2 label
1 10 0
2 20 1
3 30 0
4 40 1
5 50 1
```

### Output

```csv
feature_1,feature_2,label
1,10,0
2,20,1
3,30,0
4,40,1
5,50,1
```

The same conversion can be performed for other delimiters.

For example, given:

```text
feature_1|feature_2|label
1|10|0
2|20|1
3|30|0
4|40|1
5|50|1
```

use:

```python
delimited_to_csv(
    "data/dataset.txt",
    "data/dataset.csv",
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

The complete workflow can be expressed in a single script:

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

print(df)
print(df.shape)
print(df.columns.tolist())

assert isinstance(df, pd.DataFrame)
```

Running the inspection code produces:

```pycon
>>> print(df)
   feature_1  feature_2  label
0          1         10      0
1          2         20      1
2          3         30      0
3          4         40      1
4          5         50      1

>>> print(df.shape)
(5, 3)

>>> print(df.columns.tolist())
['feature_1', 'feature_2', 'label']
```

The resulting `DataFrame` can now be passed directly to pandas operations or other DanFlow utilities for further analysis.
