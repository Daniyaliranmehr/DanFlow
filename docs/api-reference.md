# API Reference

DanFlow provides utilities for data preparation, loss functions, model training, model validation, evaluation, hyperparameter search, and visualization.

This page provides an overview of the available public APIs and links to their detailed API documentation.

For complete parameter descriptions, return values, exceptions, and method-level behavior, see the corresponding pages under [`docs/api/`](api/).


## Data

Module:

```python
danflow.data
```

Data utilities are provided for common file-preparation workflows.

### `extract_zip`

Extract a ZIP archive into a target directory.

Detailed documentation: [`api/data.md`](api/data.md)

### `delimited_to_csv`

Convert a delimiter-separated text file into a CSV file.

Detailed documentation: [`api/data.md`](api/data.md)

### `load_csv`

Load a CSV file into a pandas DataFrame.

Detailed documentation: [`api/data.md`](api/data.md)


## Losses

Module:

```python
danflow.losses
```

DanFlow provides custom loss functions for regression and robust optimization workflows.

### `adaptive_loss`

Compute DanFlow's adaptive loss function.

Detailed documentation: [`api/losses.md`](api/losses.md)

### `log_cosh_loss`

Compute the logarithmic hyperbolic cosine loss.

Detailed documentation: [`api/losses.md`](api/losses.md)


## Training

Module:

```python
danflow.training
```

The training package provides the core training loop, metric tracking, model checking, evaluation, and hyperparameter-search utilities.

### `AverageMeter`

Track the running average of scalar values during training.

Detailed documentation: [`api/trainer.md`](api/trainer.md)

### `Trainer`

Manage model training, validation, metric tracking, training history, and optional best-checkpoint saving.

Main methods include:

* `train_epoch()`
* `validate_epoch()`
* `fit()`

Example import:

```python
from danflow.training import Trainer
```

Detailed documentation: [`api/trainer.md`](api/trainer.md)

### `Evaluator`

Evaluate a trained model on test data using an optional loss function and metric.

Example import:

```python
from danflow.training import Evaluator
```

Detailed documentation: [`api/evaluator.md`](api/evaluator.md)

### `ModelChecker`

Validate a model's forward path and verify that it can learn from a small subset of training data.

The checker supports:

* forward-path validation
* backward-path overfitting checks
* optional loss targets
* optional metric targets
* continued backward training

Example import:

```python
from danflow.training import ModelChecker
```

Detailed documentation: [`api/checker.md`](api/checker.md)


## Hyperparameter Tuning

Module:

```python
danflow.training.tuner
```

DanFlow provides utilities for comparing training configurations before the final training run.

### `LearningRateSelector`

Evaluate multiple learning-rate candidates using short training experiments.

Example import:

```python
from danflow.training.tuner import LearningRateSelector
```

Detailed documentation: [`api/tuner.md`](api/tuner.md)

### `SmallGrid`

Evaluate combinations of learning rate and weight decay using a small grid search.

Example import:

```python
from danflow.training.tuner import SmallGrid
```

Detailed documentation: [`api/tuner.md`](api/tuner.md)


## Visualization

DanFlow provides utilities for visualizing both tabular data and training history.

### Data Visualization

Module:

```python
danflow.visualization.data
```

Available functions:

* `plot_correlation_heatmap()`
* `plot_histogram()`
* `plot_multi_histograms()`
* `plot_boxplot()`
* `plot_multi_boxplots()`

These functions are intended for inspecting feature relationships, distributions, and spread.

Detailed documentation: [`api/visualization.md`](api/visualization.md)

### Training Visualization

Module:

```python
danflow.visualization.training
```

Available functions:

* `plot_loss_history()`
* `plot_metric_history()`
* `plot_training_history()`

These functions visualize training and validation behavior across epochs and can optionally highlight the best loss or metric.

Detailed documentation: [`api/visualization.md`](api/visualization.md)


## API Documentation Map

| Area                  | Detailed API                                   |
| --------------------- | ---------------------------------------------- |
| Data preparation      | [`api/data.md`](api/data.md)                   |
| Loss functions        | [`api/losses.md`](api/losses.md)               |
| Training              | [`api/trainer.md`](api/trainer.md)             |
| Evaluation            | [`api/evaluator.md`](api/evaluator.md)         |
| Model checking        | [`api/checker.md`](api/checker.md)             |
| Hyperparameter tuning | [`api/tuner.md`](api/tuner.md)                 |
| Visualization         | [`api/visualization.md`](api/visualization.md) |

The detailed API pages are the authoritative source for individual signatures, parameters, return values, and behavior.