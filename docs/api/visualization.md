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

## plot_multi_histograms()

Plots histograms for multiple DataFrame columns in a single figure.

### Parameters

#### `df` : `pd.DataFrame`

Input DataFrame containing the data.

#### `columns` : `list[str]`

List of column names to visualize.

#### `name` : `str`

Title of the figure.

#### `bins` : `int`, default=`50`

Number of histogram bins.

#### `save_path` : `str | Path | None`, default=`None`

Optional path where the generated figure will be saved.

If `None`, the figure is displayed without being saved.

#### `figsize` : `tuple[int, int] | None`, default=`None`

Size of the matplotlib figure.

If `None`, the figure size is determined automatically based on the number of subplots.

### Returns

`None`

### Example

```python
import pandas as pd

from danflow.visualization.data import plot_multi_histograms

df = pd.read_csv("dataset.csv")

plot_multi_histograms(
    df=df,
    columns=["feature1", "feature2", "feature3"],
    name="Feature Distributions",
    bins=30,
    save_path="plots/histograms.png"
)
```


## plot_boxplot()

Plots a boxplot for a selected DataFrame column.

### Parameters

#### `df` : `pd.DataFrame`

Input DataFrame containing the data.

#### `column` : `str`

Name of the column to visualize.

#### `save_path` : `str | Path | None`, default=`None`

Optional path where the generated figure will be saved.

If `None`, the figure is displayed without being saved.

#### `figsize` : `tuple[int, int]`, default=`(8, 5)`

Size of the matplotlib figure.

### Returns

`None`

### Example

```python
import pandas as pd

from danflow.visualization.data import plot_boxplot

df = pd.read_csv("dataset.csv")

plot_boxplot(
    df=df,
    column="feature1",
    save_path="plots/boxplot.png"
)
```

## plot_multi_boxplots()

Plots boxplots for multiple DataFrame columns in a single figure.

### Parameters

#### `df` : `pd.DataFrame`

Input DataFrame containing the data.

#### `columns` : `list[str]`

List of column names to visualize.

#### `save_path` : `str | Path | None`, default=`None`

Optional path where the generated figure will be saved.

If `None`, the figure is displayed without being saved.

#### `figsize` : `tuple[int, int]`, default=`(12, 4)`

Base size of the matplotlib figure. The figure height is automatically adjusted according to the number of subplot rows.

### Returns

`None`

### Example

```python
import pandas as pd

from danflow.visualization.data import plot_multi_boxplots

df = pd.read_csv("dataset.csv")

plot_multi_boxplots(
    df=df,
    columns=["feature1", "feature2", "feature3"],
    save_path="plots/boxplots.png"
)
```

## plot_training_history()

Plots training and validation loss over epochs, with optional training and validation metrics.

### Parameters

#### `history` : `Dict[str, List[float]]`

Dictionary containing the training history returned by `Trainer.fit()`.

Expected keys include:

- `"train_loss"`
- `"valid_loss"`
- `"train_metric"` *(optional)*
- `"valid_metric"` *(optional)*
- `"metric_name"` *(optional)*
- `"best_loss_epoch"` *(optional)*
- `"best_metric_epoch"` *(optional)*

#### `name` : `str`

Name of the experiment or model. Used as the figure title.

#### `save_path` : `str | Path | None`, default=`None`

Optional path where the generated figure will be saved.

If `None`, the figure is displayed without being saved.

#### `show_best_loss` : `bool`, default=`False`

Whether to highlight the best validation loss on the plot.

#### `show_best_metric` : `bool`, default=`False`

Whether to highlight the best validation metric on the plot.

#### `figsize` : `tuple[int, int]`, default=`(10, 5)`

Size of the matplotlib figure.

### Returns

`None`

### Example

```python
from danflow.visualization.training import plot_training_history

history = {
    "train_loss": [0.82, 0.64, 0.49, 0.38],
    "valid_loss": [0.88, 0.70, 0.55, 0.46],
    "train_metric": [0.71, 0.80, 0.87, 0.92],
    "valid_metric": [0.69, 0.77, 0.84, 0.89],
    "metric_name": "Accuracy",
    "best_loss_epoch": 4,
    "best_metric_epoch": 4,
}

plot_training_history(
    history=history,
    name="Model",
    show_best_loss=True,
    show_best_metric=True,
    save_path="plots/training_history.png"
)
```