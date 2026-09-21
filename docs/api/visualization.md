# Visualization

DanFlow provides visualization utilities for exploratory data analysis and training-history inspection.

The visualization API is organized into two modules:

* `danflow.visualization.data` contains plots for tabular datasets.
* `danflow.visualization.training` contains plots for model training history.

## Data Visualization

### plot_correlation_heatmap()


Plots the correlation matrix of numerical columns in a pandas `DataFrame`.


#### Parameters

##### `df` : `pd.DataFrame`
Input DataFrame.

##### `save_path` : `Optional[str | Path]`, default=`"None"`
Optional path where the figure will be saved.

- If a filename is provided (e.g., ``"plots/correlation_heatmap.png"``), the
        figure is saved using that filename.
- If a directory is provided (e.g., ``"plots/"`` or ``"plots"``), the
        figure is saved in that directory using an automatically generated
        filename.
- If ``None``, the figure is not saved.

##### `figsize` : `tuple[int, int]`, default=`(10, 8)`
Size of the matplotlib figure.


#### Returns

`None`


#### Example

```python
from danflow.visualization.data import plot_correlation_heatmap

plot_correlation_heatmap(
    df,
    save_path=None,
    figsize=(10, 8),
)
```


### plot_histogram()
---

Plots the distribution of a selected DataFrame column.


#### Parameters

##### `df` : `pd.DataFrame`
Input DataFrame.

##### `column` : `str`
Name of the column to visualize.

##### `bins` : `int` ,default=`"50"`
Number of histogram bins.

##### `save_path` : `Optional[str | Path]`, default=`"None"`
Optional path where the figure will be saved.

- If a filename is provided (e.g., ``"plots/histogram.png"``), the
        figure is saved using that filename.
- If a directory is provided (e.g., ``"plots/"`` or ``"plots"``), the
        figure is saved in that directory using an automatically generated
        filename.
- If ``None``, the figure is not saved.

##### `figsize` : `tuple[int, int]`, default=`(8, 5)`
Size of the matplotlib figure.


#### Returns

`None`

#### Example

```python
from danflow.visualization.data import plot_histogram

plot_histogram(
    df,
    column,
    bins=50,
    save_path=None,
    figsize=(8, 5),
)
```


### plot_multi_histograms()
---

Plots histograms for multiple DataFrame columns in a multi-subplot figure.


#### Parameters

##### `df` : `pd.DataFrame`
Input DataFrame.

##### `columns` : `list[str]`
List of column names to visualize.
ValueError: If `columns` is empty.

##### `name` : `Optional[str]`, default=`None`
Figure title.

##### `bins` : `int`, default=`50`
Number of histogram bins.

##### `save_path` : `Optional[str | Path]`, default=`None`
Optional path where the figure will be saved.

- If a filename is provided (e.g., ``"plots/histograms.png"``),
        the figure is saved using that filename.
- If a directory is provided (e.g., ``"plots/"`` or ``"plots"``),
        the figure is saved in that directory using the figure name
        (e.g., ``"multi_histograms.png"``).
- If ``None``, the figure is not saved.

##### `figsize` : `tuple[int, int] | None`, default=`None`
Size of the matplotlib figure. If None, the figure size is determined automatically.

#### Returns

`None`

#### Example

```python
from danflow.visualization.data import plot_multi_histograms

plot_multi_histograms(
    df,
    columns,
    name=None,
    bins=50,
    save_path=None,
    figsize=None,
)
```


### plot_boxplot()
---

Plots a box plot for a selected DataFrame column.


#### Parameters

##### `df` : `pd.DataFrame`
Input dataframe.

##### `column` : `str`
Name of the column to visualize.

##### `save_path` : `Optional[str | Path]`, default=`None`
Optional path where the figure will be saved.

- If a filename is provided (e.g., ``"plots/boxplot.png"``), the
        figure is saved using that filename.
- If a directory is provided (e.g., ``"plots/"`` or ``"plots"``), the
        figure is saved in that directory using an automatically generated
        filename.
- If ``None``, the figure is not saved.

##### `figsize` : `tuple[int, int]`, default=`(8, 5)`
Size of the matplotlib figure.


#### Returns

`None`

#### Example

```python
from danflow.visualization.data import plot_boxplot

plot_boxplot(
    df,
    column,
    save_path=None,
    figsize=(8, 5),
)
```


### plot_multi_boxplots()
---

Plots box plots for multiple DataFrame columns.

#### Parameters

##### `df` : `pd.DataFrame`
Input dataframe.

##### `column` : `str`
List of column names to visualize.

##### `name`: `Optional[str]`, default=`None`
Figure title.

##### `save_path` : `Optional[str | Path]`, default=`None`
Optional path where the figure will be saved.

- If a filename is provided (e.g., ``"plots/multi_boxplots.png"``),
        the figure is saved using that filename.
- If a directory is provided (e.g., ``"plots/"`` or ``"plots"``),
        the figure is saved in that directory using the figure name
        (e.g., ``"multi_boxplots.png"``).
- If ``None``, the figure is not saved.

##### `figsize` : `tuple[int, int]`, default=`(12, 4)`
Base size of the matplotlib figure. The height is automatically scaled according to the number of subplot rows.


#### Returns

`None`

#### Example
```python
from danflow.visualization.data import plot_multi_boxplots

plot_multi_boxplots(
    df,
    columns,
    name=None,
    save_path=None,
    figsize=(12, 4),
)
```


## Training Visualization

### plot_training_history()

Plots training and validation loss and, when available, training and validation metrics in one figure.

#### Parameters

##### `history` : `Dict[str, List[float]]`
Dictionary returned by Trainer.fit().

Expected keys:

- "train_loss"
- "valid_loss"
- "train_metric" (optional)
- "valid_metric" (optional)
- "metric_name" (optional)
- "best_loss_epoch" (optional)
- "best_metric_epoch" (optional) 

##### `name` : `str`
Name of the experiment or model (used in the plot title).

##### `save_path` : `Optional[str | Path]`, default=`None`
Optional path where the figure will be saved. If provided, the plot is saved to this location before being displayed. If None, the figure is not saved.


##### `show_best_loss` : `bool`, default=`False`
Whether to mark the epoch with the best validation loss on the plot. Requires "best_loss_epoch" to be present in history.

##### `show_best_metric` : `bool`, default=`False`
Whether to mark the epoch with the best validation metric on the plot. Requires "best_metric_epoch" to be present in history.

##### `figsize` : `tuple[int, int]`, default=`(10, 5)`
Size of the matplotlib figure.


#### Returns

`None`

#### Example

```python
from danflow.visualization.training import plot_training_history

plot_training_history(
    history,
    name,
    save_path=None,
    show_best_loss=False,
    show_best_metric=False,
    figsize=(10, 5),
)
```


### plot_loss_history()
---
Plots training and validation loss over epochs.

#### Parameters

##### `history` : `Dict[str, List[float]]` 
Dictionary returned by Trainer.fit().

Expected keys:

- "train_loss"
- "valid_loss"
- "best_loss_epoch" (optional)

##### `name` : `str`
Name of the experiment or model (used in title).

##### `save_path` : `Optional[str | Path]`, default=`None`
Optional path where the figure will be saved. If provided, the plot is saved to this location before being displayed. If None, the figure is not saved.

##### `show_best_loss` : `bool`, default=`False`
If True and "best_loss_epoch" is present in history, marks the best validation loss with a star.

##### `figsize` : `tuple[int, int]`, default=`(10, 5)`
Size of the matplotlib figure.

#### Returns

`None`

#### Example

```python
from danflow.visualization.training import plot_loss_history

plot_loss_history(
    history,
    name,
    save_path=None,
    show_best_loss=False,
    figsize=(10, 5),
)
```


### plot_metric_history()
---
Plots training and validation metric over epochs.

#### Parameters

##### `history` : `Dict[str, List[float]]`
Dictionary returned by Trainer.fit().

Expected keys:

- "train_metric"
- "valid_metric"
- "metric_name" (optional, used for axis label and legend)
- "best_metric_epoch" (optional)

##### `name` : `str`
Name of the experiment or model (used in title).

##### `save_path` : `Optional[str | Path]`, default=`None`
Optional path where the figure will be saved. If provided, the plot is saved to this location before being displayed. If None, the figure is not saved.

##### `show_best_metric` : `bool`, default=`False`
If True and "best_metric_epoch" is present in history, marks the best validation metric with a star.

##### `figsize` : `tuple[int, int]`, default=`(10, 5)`
Size of the matplotlib figure.

#### Returns

`None`

#### Example
```python
from danflow.visualization.training import plot_metric_history

plot_metric_history(
    history,
    name,
    save_path=None,
    show_best_metric=False,
    figsize=(10, 5),
)
```