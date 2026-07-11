## plot_correlation_heatmap()

Plots a correlation heatmap for all numerical columns in a DataFrame.

### Parameters

#### `df` : `pd.DataFrame`

Input DataFrame containing the data.

#### `save_path` : `str | Path | None`, default=`None`

Optional path where the generated figure will be saved.

If `None`, the figure is displayed without being saved.

#### `figsize` : `tuple[int, int]`, default=`(10, 8)`

Size of the matplotlib figure.

### Returns

`None`

### Example

```python
import pandas as pd

from danflow.visualization.data import plot_correlation_heatmap

df = pd.read_csv("dataset.csv")

plot_correlation_heatmap(
    df=df,
    save_path="plots/correlation_heatmap.png"
)
```

## plot_histogram()

Plots the histogram of a selected DataFrame column.

### Parameters

#### `df` : `pd.DataFrame`

Input DataFrame containing the data to visualize.

#### `column` : `str`

Name of the column to plot.

#### `bins` : `int`, default=`50`

Number of histogram bins.

#### `save_path` : `str | Path | None`, default=`None`

Optional path where the figure will be saved.

- If a filename is provided (e.g., `"plots/histogram.png"`), the figure is saved using that filename.
- If a directory is provided (e.g., `"plots/"`), the figure is saved in that directory using an automatically generated filename.
- If `None`, the figure is displayed without being saved.

#### `figsize` : `tuple[int, int]`, default=`(8, 5)`

Size of the matplotlib figure.

### Returns

`None`

### Example

```python
import pandas as pd

from danflow.visualization.data import plot_histogram

df = pd.read_csv("dataset.csv")

plot_histogram(
    df=df,
    column="age",
    bins=30,
    save_path="plots/"
)
```