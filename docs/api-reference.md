# API Reference

The DanFlow API is organized into a small set of focused namespaces covering data preparation, loss functions, training, evaluation, model checking, hyperparameter search, and visualization.

This page is the central index for the public API. It does not replace the detailed API pages. Use the links in each section to find parameters, return values, exceptions, and implementation-specific behavior for an individual API.

## Public API at a Glance

| Area                  | Main APIs                                                   | Detailed documentation                    |
| --------------------- | ----------------------------------------------------------- | ----------------------------------------- |
| Data                  | `extract_zip`, `delimited_to_csv`, `load_csv`               | [Data API](api/data.md)                   |
| Losses                | `adaptive_loss`, `log_cosh_loss`                            | [Losses API](api/losses.md)               |
| Training              | `Trainer`, `AverageMeter`                                   | [Trainer API](api/trainer.md)             |
| Evaluation            | `Evaluator`                                                 | [Evaluator API](api/evaluator.md)         |
| Model Checking        | `ModelChecker`, `ForwardCheckResult`, `BackwardCheckResult` | [Checker API](api/checker.md)             |
| Hyperparameter Search | `LearningRateSelector`, `SmallGrid`                         | [Tuner API](api/tuner.md)                 |
| Visualization         | Data and training-history plotting functions                | [Visualization API](api/visualization.md) |

The API is intentionally centered around standard PyTorch objects. DanFlow provides workflow utilities around those objects rather than introducing a replacement model, tensor, optimizer, or data-loading abstraction.

## Importing the Public API

DanFlow exposes its public functionality through package namespaces.

For user-facing code, prefer the public package or subpackage import paths documented for each API rather than importing implementation details from private or internal modules.

For example:

```python
from danflow.training import Trainer
from danflow.data import load_csv
from danflow.visualization import plot_histogram
```

The detailed API pages show the appropriate import for each component.

The repository currently contains several internal modules that implement these APIs. Those internal module paths are useful when reading the source code, but application code should normally depend on the documented public namespace.

## API Namespaces

### `danflow.data`

The data namespace provides file and tabular-data preparation utilities.

```text
danflow.data
    extract_zip()
    delimited_to_csv()
    load_csv()
```

#### `extract_zip`

Extracts a ZIP archive to a target directory.

See the [Data API](api/data.md) for parameters, path behavior, and errors.

#### `delimited_to_csv`

Converts delimited text data into CSV format.

The API supports common delimiters such as spaces, pipes, and tabs. The detailed documentation also describes handling of empty lines, encoding, and output paths.

See the [Data API](api/data.md).

#### `load_csv`

Loads CSV data into a pandas `DataFrame`.

See the [Data API](api/data.md).


### `danflow.losses`

The losses namespace contains reusable tensor-based loss functions.

```text
danflow.losses
    adaptive_loss()
    log_cosh_loss()
```

#### `adaptive_loss`

Computes the DanFlow adaptive loss function for model outputs and targets.

See the [Losses API](api/losses.md) for the exact signature and parameter behavior.

#### `log_cosh_loss`

Computes the log-cosh loss for model outputs and targets.

See the [Losses API](api/losses.md).

The loss functions can be used wherever a standard PyTorch loss function can be supplied.


### `danflow.training`

The training namespace contains DanFlow's primary training and evaluation classes.

```text
danflow.training
    Trainer
    Evaluator
    AverageMeter
    ModelChecker
    LearningRateSelector
    SmallGrid
```

These components do not have identical responsibilities.

`Trainer` manages training execution.

`Evaluator` performs test-time evaluation.

`ModelChecker` validates the model's forward and backward behavior.

`LearningRateSelector` and `SmallGrid` perform lightweight hyperparameter searches.

`AverageMeter` is a training utility used for tracking averaged values and is exposed through the training namespace.

See the detailed pages:

* [Trainer API](api/trainer.md)
* [Evaluator API](api/evaluator.md)
* [Checker API](api/checker.md)
* [Tuner API](api/tuner.md)

## Trainer

`Trainer` is the main multi-epoch training abstraction.

Its public interface includes:

```text
Trainer
    __init__()
    train_epoch()
    validate_epoch()
    fit()
```

Use `Trainer` when the model, optimizer, loss function, and optional metric are already defined and a standard training loop is required.

The detailed API page documents:

* constructor parameters,
* training and validation behavior,
* training history,
* metric handling,
* checkpoint saving,
* method parameters and return values,
* error behavior.

See the [Trainer API](api/trainer.md).

### Training history

`Trainer.fit()` produces the training history consumed by DanFlow's training-visualization utilities.

The history can contain training and validation loss values and, when a metric is configured, training and validation metric values.

See:

* [Trainer API](api/trainer.md)
* [Visualization API](api/visualization.md)

### Checkpoints

`Trainer.fit()` can optionally save the best checkpoint during training.

Checkpoint format and restoration behavior are documented separately in the [Checkpoints Guide](guides/checkpoints.md).


## Evaluator

`Evaluator` provides test-time evaluation for a trained model.

```text
Evaluator
    __init__()
    test()
```

The current `Evaluator.test()` interface accepts test input and target tensors directly:

```python
result = evaluator.test(
    x_test,
    y_test,
)
```

The returned value is a dictionary containing the configured evaluation metric and loss.

The exact result structure, constructor parameters, and metric contract are documented in the [Evaluator API](api/evaluator.md).

For conceptual guidance about separating training, validation, and test data, see the [Evaluation Guide](guides/evaluation.md).


## AverageMeter

`AverageMeter` is exposed through the training namespace and is used by the training implementation for tracking averaged values.

It is currently grouped with the training APIs rather than treated as a separate subsystem.

```python
from danflow.training import AverageMeter
```

See the [Trainer API](api/trainer.md) for its current API documentation.


## ModelChecker

`ModelChecker` is the pre-training validation utility for checking whether a model and its training components are compatible.

```text
ModelChecker
    __init__()
    forward_check()
    backward_check()
    continue_backward()
```

### `forward_check()`

Checks the model's forward path using a `DataLoader`.

The check validates compatibility between:

```text
inputs
    |
    v
model
    |
    v
outputs
    |
    +---- targets
    |
    v
loss function
```

The returned `ForwardCheckResult` contains information about processed batches, the average loss, and the observed input, target, and output shapes.

### `backward_check()`

Runs a small-subset training experiment to determine whether the model can learn from a limited training sample.

The returned `BackwardCheckResult` records information such as:

```text
initial loss
final loss
final metric
epochs trained
target loss
target metric
success
automatic extension state
```

### `continue_backward()`

Continues an existing backward check using the current checker state and training subset.

This is different from starting a new backward check.

Detailed signatures, parameter requirements, state behavior, and result definitions belong in the [Checker API](api/checker.md).


## Hyperparameter Search

DanFlow provides two lightweight search utilities in `danflow.training.tuner`.

```text
danflow.training.tuner
    LearningRateSelector
    SmallGrid
```

### `LearningRateSelector`

`LearningRateSelector` compares candidate learning rates using short independent training experiments.

Its search space is the list of candidate learning rates.

```python
from danflow.training import LearningRateSelector

selector = LearningRateSelector(
    model=model,
    optimizer_cls=optimizer_cls,
    loss_fn=loss_fn,
    learning_rates=[0.01, 0.001, 0.0001],
    epochs=5,
)
```

### `SmallGrid`

`SmallGrid` searches combinations of learning rates and weight decays.

```python
from danflow.training import SmallGrid

grid = SmallGrid(
    model=model,
    optimizer_cls=optimizer_cls,
    loss_fn=loss_fn,
    learning_rates=[0.01, 0.001],
    weight_decays=[0.0, 1e-4],
    epochs=5,
)
```

Both utilities use the training infrastructure rather than implementing a separate optimization engine.

Their exact constructors, search behavior, metric requirements, and result handling are documented in the [Tuner API](api/tuner.md).

## Metric Requirements

Metric handling differs between training-oriented components and evaluation.

For:

```text
Trainer
ModelChecker
LearningRateSelector
SmallGrid
```

the metric is stateful and follows the interface:

```python
metric.reset()
metric.update(outputs, targets)
metric.compute()
```

For `Evaluator`, the metric is a callable that receives model outputs and targets.

```python
metric(outputs, targets)
```

This distinction is important when moving a metric from the training workflow into evaluation.

For the complete metric contract and examples, see the [Metrics Guide](guides/metrics.md).


## danflow.visualization

The visualization namespace contains two groups of plotting functions.

```text
danflow.visualization
    Data visualization
        plot_correlation_heatmap()
        plot_histogram()
        plot_multi_histograms()
        plot_boxplot()
        plot_multi_boxplots()

    Training visualization
        plot_training_history()
        plot_loss_history()
        plot_metric_history()
```

### Data visualization

These functions operate on pandas `DataFrame` objects.

| Function                     | Purpose                                                |
| ---------------------------- | ------------------------------------------------------ |
| `plot_correlation_heatmap()` | Visualize numerical-feature correlations               |
| `plot_histogram()`           | Visualize one column's distribution                    |
| `plot_multi_histograms()`    | Visualize multiple column distributions                |
| `plot_boxplot()`             | Visualize spread and potential outliers for one column |
| `plot_multi_boxplots()`      | Visualize spread across multiple columns               |

See the [Visualization API](api/visualization.md).

### Training visualization

These functions consume the history produced by `Trainer.fit()`.

| Function                  | Purpose                             |
| ------------------------- | ----------------------------------- |
| `plot_loss_history()`     | Plot training and validation loss   |
| `plot_metric_history()`   | Plot training and validation metric |
| `plot_training_history()` | Combine loss and metric histories   |

The visualization API also documents the `save_path` behavior and optional best-result markers.

See the [Visualization API](api/visualization.md).


## API by Task

Use this section when you know what you want to accomplish rather than which class or function you need.

| Task                                             | API                                           |
| ------------------------------------------------ | --------------------------------------------- |
| Extract a ZIP dataset                            | `extract_zip()`                               |
| Convert a delimited text file to CSV             | `delimited_to_csv()`                          |
| Load CSV data                                    | `load_csv()`                                  |
| Use a DanFlow loss                               | `adaptive_loss()`, `log_cosh_loss()`          |
| Train a PyTorch model                            | `Trainer`                                     |
| Train a single epoch manually                    | `Trainer.train_epoch()`                       |
| Validate a single epoch                          | `Trainer.validate_epoch()`                    |
| Run a complete training process                  | `Trainer.fit()`                               |
| Save the best training checkpoint                | `Trainer.fit(save_best=True)`                 |
| Check a model's forward path                     | `ModelChecker.forward_check()`                |
| Check whether a model can overfit a small subset | `ModelChecker.backward_check()`               |
| Continue a backward check                        | `ModelChecker.continue_backward()`            |
| Compare learning rates                           | `LearningRateSelector`                        |
| Search learning rate and weight decay            | `SmallGrid`                                   |
| Evaluate a trained model on test data            | `Evaluator.test()`                            |
| Plot feature distributions                       | `plot_histogram()`, `plot_multi_histograms()` |
| Plot feature relationships                       | `plot_correlation_heatmap()`                  |
| Inspect outliers and spread                      | `plot_boxplot()`, `plot_multi_boxplots()`     |
| Plot training loss                               | `plot_loss_history()`                         |
| Plot training metric                             | `plot_metric_history()`                       |
| Plot loss and metric together                    | `plot_training_history()`                     |

## API Documentation vs Guides vs Examples

DanFlow documentation separates API information from conceptual and practical documentation.

### API Reference

This page answers:

> **Which public API should I use, and where is it documented?**

The pages under `docs/api/` answer:

> **What exactly does this API accept, return, and raise?**

### Guides

Guides answer:

> **How should I use this functionality in a real workflow, and what should I watch out for?**

See the [Guides](guides/training.md) for workflow-level explanations.

### Examples

Examples answer:

> **What does a complete DanFlow workflow look like?**

See the [Examples](examples/end_to_end.md) for complete runnable-style workflows.

Keeping these responsibilities separate avoids duplicating detailed parameter documentation throughout the rest of the documentation.

## Public API vs Internal Implementation

The public API should be treated as the interface exposed by the documented DanFlow namespaces.

The implementation currently contains internal module paths such as:

```text
danflow.training.trainer
danflow.training.checker
danflow.training.tuner
danflow.data.io
danflow.losses.loss
danflow.visualization.data
danflow.visualization.training
```

These modules contain the implementation of the public API.

Users should generally avoid coupling application code to internal module organization when an equivalent public package import is available.

This distinction also allows DanFlow to reorganize implementation modules without unnecessarily changing user-facing code.

## API Index

### Data

[Data API](api/data.md)

### Losses

[Losses API](api/losses.md)

### Training

[Trainer API](api/trainer.md)

### Evaluation

[Evaluator API](api/evaluator.md)

### Model Checking

[Checker API](api/checker.md)

### Hyperparameter Search

[Tuner API](api/tuner.md)

### Visualization

[Visualization API](api/visualization.md)

## Related Documentation

For installation and the first working example, start with the [Quick Start](getting_started/quickstart.md).

For the overall design and dependency relationships between components, see [Architecture](architecture.md).

For workflow-oriented explanations, see the [Guides](guides/training.md).

For complete practical workflows, see the [Examples](examples/end_to_end.md).