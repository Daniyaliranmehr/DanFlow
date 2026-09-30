# API Reference

The DanFlow API provides a small set of public utilities for data preparation, loss calculation, model training, test-time evaluation, checkpoint-aware training, and visualization.

This page is the central index of the public API exposed by the current DanFlow package. It is intentionally concise: detailed parameters, return values, notes, and error behavior belong in the individual API pages.

## Public API at a Glance

| Area          | Public API                                                                                                                            | Documentation                             |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Data          | `extract_zip`, `load_csv`, `delimited_to_csv`                                                                                         | [Data API](api/data.md)                   |
| Losses        | `adaptive_loss`, `log_cosh_loss`                                                                                                      | [Losses API](api/losses.md)               |
| Training      | `Trainer`, `AverageMeter`                                                                                                             | [Trainer API](api/trainer.md)             |
| Evaluation    | `Evaluator`                                                                                                                           | [Evaluator API](api/evaluator.md)         |
| Visualization | `plot_training_history`, `plot_correlation_heatmap`, `plot_histogram`, `plot_multi_histograms`, `plot_boxplot`, `plot_multi_boxplots` | [Visualization API](api/visualization.md) |

These entries reflect the public exports of the current package.

## Public Import Paths

The recommended user-facing import style is to import public APIs from their package namespace rather than depending on implementation modules.

For example:

```python
from danflow import (
    Trainer,
    Evaluator,
    adaptive_loss,
    log_cosh_loss,
    extract_zip,
    load_csv,
    delimited_to_csv,
    plot_training_history,
    plot_correlation_heatmap,
    plot_histogram,
    plot_multi_histograms,
    plot_boxplot,
    plot_multi_boxplots,
)
```

`AverageMeter` is publicly exposed through `danflow.training`:

```python
from danflow.training import AverageMeter
```

The package root currently re-exports `Trainer` and `Evaluator`, the loss functions, the data utilities, and the visualization functions listed above.

## danflow.data

The data namespace provides small utilities for file-based and tabular data preparation.

```text
danflow.data
    extract_zip()
    load_csv()
    delimited_to_csv()
```

### `extract_zip()`

Extracts a ZIP archive into a specified directory.

```python
from danflow import extract_zip

extract_zip(
    zip_path="dataset.zip",
    output_path="data",
)
```

See the [Data API](api/data.md) for the complete parameter and behavior reference.

### `load_csv()`

Loads a CSV file into a pandas `DataFrame`.

```python
from danflow import load_csv

df = load_csv("dataset.csv")
```

See the [Data API](api/data.md).

### `delimited_to_csv()`

Converts a delimited text file into a CSV file.

```python
from danflow import delimited_to_csv

delimited_to_csv(
    input_path="dataset.txt",
    output_path="dataset.csv",
    delimiter=" ",
    encoding="utf-8",
)
```

The current API supports configurable delimiters and encoding.

See the [Data API](api/data.md).


## danflow.losses

The losses namespace contains reusable tensor-based loss functions.

```text
danflow.losses
    adaptive_loss()
    log_cosh_loss()
```

### `adaptive_loss()`

Computes DanFlow's adaptive loss for model outputs and targets.

```python
from danflow import adaptive_loss

loss = adaptive_loss(
    outputs,
    targets,
)
```

See the [Losses API](api/losses.md) for the exact signature and parameter behavior.

### `log_cosh_loss()`

Computes the log-cosh loss for model outputs and targets.

```python
from danflow import log_cosh_loss

loss = log_cosh_loss(
    outputs,
    targets,
)
```

See the [Losses API](api/losses.md).

Both functions return values intended to participate in standard PyTorch training workflows.


## danflow.training

The training namespace currently exposes:

```text
danflow.training
    AverageMeter
    Trainer
    Evaluator
```

The top-level package additionally re-exports `Trainer` and `Evaluator`.

## Trainer

`Trainer` is the main training abstraction in DanFlow.

```text
Trainer
    __init__()
    train_epoch()
    validate_epoch()
    fit()
```

It receives standard PyTorch training components and manages the training workflow around them.

```python
from danflow import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=metric,
)
```

### `train_epoch()`

Runs one complete training epoch and updates model parameters batch by batch.

```python
loss, metric_value = trainer.train_epoch(
    train_loader,
)
```

### `validate_epoch()`

Runs one validation epoch without updating model parameters.

The public parameter name is `valid_loader`.

```python
loss, metric_value = trainer.validate_epoch(
    valid_loader,
)
```

### `fit()`

Runs a multi-epoch training workflow with validation and optional best-checkpoint saving.

```python
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=100,
    save_best=True,
    checkpoint_path="best_model.pth",
)
```

The current implementation uses `valid_loader`, not `validation_loader`. It also returns the training and validation histories together with best-validation metadata.

See the [Trainer API](api/trainer.md) for the complete reference.

## AverageMeter

`AverageMeter` is a small utility for storing a current value and maintaining a running average.

```python
from danflow.training import AverageMeter

meter = AverageMeter()

meter.update(0.5)
meter.update(0.3)

print(meter.avg)
```

It is publicly exported by `danflow.training`.

See the [Trainer API](api/trainer.md).


## Evaluator

`Evaluator` provides test-time evaluation for a trained PyTorch model.

```text
Evaluator
    __init__()
    test()
```

A typical setup is:

```python
from danflow import Evaluator

evaluator = Evaluator(
    model=model,
    loss_fn=loss_fn,
    metric=metric,
)
```

### `test()`

Evaluates the model on test tensors without gradient computation.

```python
result = evaluator.test(
    x_test,
    y_test,
)
```

The returned dictionary can contain:

```text
"Metric"
"Loss"
```

depending on which optional evaluation components were supplied.

The current implementation restores the model's original training/evaluation mode after the test operation.

See the [Evaluator API](api/evaluator.md).


## danflow.visualization

The current public visualization API contains data-visualization utilities and the combined training-history plot.

```text
danflow.visualization
    plot_correlation_heatmap()
    plot_histogram()
    plot_multi_histograms()
    plot_boxplot()
    plot_multi_boxplots()
    plot_training_history()
```

The current package root re-exports all of these functions.

## Data Visualization

### `plot_correlation_heatmap()`

Plots the correlation matrix of numerical DataFrame columns.

```python
from danflow import plot_correlation_heatmap

plot_correlation_heatmap(
    df,
)
```

### `plot_histogram()`

Plots the distribution of one DataFrame column.

```python
from danflow import plot_histogram

plot_histogram(
    df,
    column="feature_1",
)
```

### `plot_multi_histograms()`

Plots histograms for multiple DataFrame columns.

```python
from danflow import plot_multi_histograms

plot_multi_histograms(
    df,
    columns=["feature_1", "feature_2"],
)
```

The current implementation requires at least one column in `columns`.

### `plot_boxplot()`

Plots a box plot for one DataFrame column.

```python
from danflow import plot_boxplot

plot_boxplot(
    df,
    column="feature_1",
)
```

### `plot_multi_boxplots()`

Plots box plots for multiple DataFrame columns.

```python
from danflow import plot_multi_boxplots

plot_multi_boxplots(
    df,
    columns=["feature_1", "feature_2"],
)
```

See the [Visualization API](api/visualization.md) for detailed parameters and saving behavior.

## Training Visualization

### `plot_training_history()`

Plots training and validation loss and, when a metric is available, training and validation metric values.

```python
from danflow import plot_training_history

plot_training_history(
    history,
    name="Experiment",
)
```

The function accepts the history produced by `Trainer.fit()` and supports optional best-loss and best-metric markers.

See the [Visualization API](api/visualization.md).

---

## API by Task

| Task                          | API                           |
| ----------------------------- | ----------------------------- |
| Extract a ZIP dataset         | `extract_zip()`               |
| Convert delimited text to CSV | `delimited_to_csv()`          |
| Load CSV data                 | `load_csv()`                  |
| Use an adaptive loss          | `adaptive_loss()`             |
| Use log-cosh loss             | `log_cosh_loss()`             |
| Train one epoch               | `Trainer.train_epoch()`       |
| Validate one epoch            | `Trainer.validate_epoch()`    |
| Run multi-epoch training      | `Trainer.fit()`               |
| Save the best checkpoint      | `Trainer.fit(save_best=True)` |
| Evaluate test data            | `Evaluator.test()`            |
| Plot feature correlations     | `plot_correlation_heatmap()`  |
| Plot one feature distribution | `plot_histogram()`            |
| Plot multiple distributions   | `plot_multi_histograms()`     |
| Plot one feature's spread     | `plot_boxplot()`              |
| Plot multiple feature spreads | `plot_multi_boxplots()`       |
| Plot training history         | `plot_training_history()`     |

## API Reference vs Guides vs Examples

DanFlow documentation separates API information from workflow guidance.

### API Reference

Answers:

> What public APIs exist, how are they imported, and where is their exact reference?

The individual API pages under `docs/api/` contain signatures, parameters, return values, notes, and API-specific behavior.

### Guides

Answers:

> How should this functionality be used as part of a real workflow?

See the [Guides](guides/training.md).

### Examples

Answers:

> What does a complete practical workflow look like?

See the [Examples](examples/end_to_end.md).

This separation keeps API descriptions from being duplicated throughout the documentation.

## Public API vs Internal Modules

The current source tree contains implementation modules such as:

```text
danflow.training.trainer
danflow.data.io
danflow.visualization.data
danflow.visualization.training
```

These implementation paths explain where functionality is implemented internally.

User-facing code should prefer the documented public package imports where available.

For example:

```python
from danflow import Trainer
```

is preferable to coupling application code to:

```python
from danflow.training.trainer import Trainer
```

unless direct access to the implementation module is specifically required.

## Detailed API Pages

### Data

[Data API](api/data.md)

### Losses

[Losses API](api/losses.md)

### Training

[Trainer API](api/trainer.md)

### Evaluation

[Evaluator API](api/evaluator.md)

### Visualization

[Visualization API](api/visualization.md)

## Related Documentation

For installation and the first working example, see the [Quick Start](getting_started/quickstart.md).

For the overall package design, see [Architecture](architecture.md).

For workflow-oriented explanations, see the [Guides](guides/training.md).

For complete workflows, see the [Examples](examples/end_to_end.md).