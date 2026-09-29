# Metrics

Metrics provide a way to measure model performance during training and evaluation.

DanFlow uses two different metric contracts depending on where the metric is used. Understanding this distinction is essential when creating or passing a metric to DanFlow components.

This guide focuses specifically on those metric contracts, how to implement a compatible training metric, and how metric results are handled by DanFlow.

## Metric Contracts in DanFlow

DanFlow currently uses two different interfaces for metrics.

| Component              | Metric type     | Required interface                                 |
| ---------------------- | --------------- | -------------------------------------------------- |
| `Trainer`              | Stateful metric | `reset()`, `update(outputs, targets)`, `compute()` |
| `ModelChecker`         | Stateful metric | `reset()`, `update(outputs, targets)`, `compute()` |
| `LearningRateSelector` | Stateful metric | `reset()`, `update(outputs, targets)`, `compute()` |
| `SmallGrid`            | Stateful metric | `reset()`, `update(outputs, targets)`, `compute()` |
| `Evaluator`            | Callable        | `metric(outputs, targets)`                         |

The distinction exists because the training-related components accumulate metric state across batches, while `Evaluator` evaluates the supplied tensors directly.

For a complete evaluation workflow, see the [Evaluation Guide](evaluation.md).

## Stateful Metrics

Training-related DanFlow components expect a metric object that maintains its state while batches are processed.

The required lifecycle is:

```text id="5t3m9p"
reset()
    Clear accumulated state

update(outputs, targets)
    Process one batch

compute()
    Return the final metric value
```

A metric used with `Trainer`, `ModelChecker`, `LearningRateSelector`, or `SmallGrid` must provide these operations.

### `reset()`

`reset()` clears all accumulated state.

A metric should be ready to process a new evaluation period immediately after `reset()`.

For example, an accuracy metric may reset its counters:

```python id="s4r1y6"
def reset(self):
    self.correct = 0
    self.total = 0
```

### `update(outputs, targets)`

`update()` receives the model outputs and the corresponding targets for one batch.

The method should update the internal state without returning the final metric value.

For example:

```python id="3x5m2a"
def update(self, outputs, targets):
    predictions = outputs.argmax(dim=1)

    self.correct += (predictions == targets).sum().item()
    self.total += targets.numel()
```

### `compute()`

`compute()` returns the metric accumulated from the processed batches.

For metrics used by DanFlow's training components, the returned value must support `.item()` because DanFlow converts the computed metric to a Python scalar.

A safe implementation is therefore:

```python id="7c1v9e"
def compute(self):
    return torch.tensor(self.correct / self.total)
```

The returned tensor should contain a single scalar value.

## Creating a Custom Stateful Metric

A complete minimal accuracy metric can be implemented as follows:

```python id="n5j7q2"
import torch


class SimpleAccuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        predictions = outputs.argmax(dim=1)

        self.correct += (
            predictions == targets
        ).sum().item()

        self.total += targets.numel()

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0)

        return torch.tensor(
            self.correct / self.total
        )
```

The metric can be tested independently before passing it to DanFlow:

```python id="9b6x4k"
metric = SimpleAccuracy()

outputs = torch.tensor([
    [2.0, 1.0],
    [0.5, 1.5],
    [3.0, 1.0],
    [0.2, 2.0],
])

targets = torch.tensor([0, 1, 0, 0])

metric.update(outputs, targets)
result = metric.compute()
```

```pycon id="m8z1c4"
>>> result
tensor(0.7500)
>>> result.item()
0.75
```

The metric has accumulated three correct predictions out of four samples.

Before starting a new evaluation period, reset its state:

```pycon id="w2h7k5"
>>> metric.reset()
>>> metric.total
0
>>> metric.correct
0
```

This state-reset behavior is especially important when the same metric object is reused across multiple training or validation phases.

## Using TorchMetrics

DanFlow's stateful metric interface is compatible with metric objects that expose the required lifecycle.

TorchMetrics provides ready-made metric implementations with this style of state management.

For example:

```python id="p7d4m2"
from torchmetrics.classification import MulticlassAccuracy

metric = MulticlassAccuracy(
    num_classes=2,
)
```

This metric can be passed to training-related DanFlow components that require a stateful metric.

For installation instructions, see the [Installation Guide](../getting_started/installation.md).

Using TorchMetrics can be preferable to implementing common metrics manually because the metric implementation and state management are already provided.

## Metric Naming in `Trainer`

When a stateful metric is supplied to `Trainer`, DanFlow derives the metric name from the metric object's class name.

For example:

```python id="e8c2v5"
metric = SimpleAccuracy()
```

The resulting metric name is based on:

```text
SimpleAccuracy
```

This name is stored in the training history under:

```python id="r5h6n3"
history["metric_name"]
```

It is also used when the training results are displayed.

Because the name is derived from the class name, using a descriptive metric class name makes training output easier to interpret.


## Metrics in Training Components

The same stateful metric contract is used by several DanFlow components.

### `Trainer`

`Trainer` uses the metric while processing training and validation batches and records the resulting values in:

```text id="7e2r6u"
train_metric
valid_metric
metric_name
```

When no metric is supplied, `train_metric` and `valid_metric` remain empty lists.

The complete structure and behavior of the training history are documented in the [Training Guide](training.md).

### `ModelChecker`

`ModelChecker.backward_check()` accepts the same type of stateful metric because its internal training process uses `Trainer`.

A metric can also be used as part of a backward check target.

For example:

```python id="f4k9r1"
checker.backward_check(
    train_dataset=train_dataset,
    metric=metric,
    target_metric=0.95,
    epochs=50,
)
```

The metric target is interpreted as a minimum value: the final metric must reach or exceed the requested target.

The details of backward checking, including its training side effects, are documented in the Model Checking example and API reference.

### `LearningRateSelector` and `SmallGrid`

The tuning utilities also use the stateful metric contract.

This means the same `SimpleAccuracy` or TorchMetrics object can be used with:

```python id="r9n5w2"
LearningRateSelector(...)
```

or:

```python id="d6p3k8"
SmallGrid(...)
```

The metric is evaluated during the short training runs performed by these components.

The complete tuning workflow belongs in the Hyperparameter Tuning example rather than this guide.

## Callable Metrics in `Evaluator`

`Evaluator` uses a different interface.

Instead of a stateful metric object, it accepts a callable that receives the outputs and targets and returns a scalar result:

```python id="u3v8q1"
def metric(outputs, targets):
    ...
```

For example:

```python id="k5s2m7"
def accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean()
```

The callable does not need `reset()`, `update()`, or `compute()`.

This distinction is intentional and is important when moving a metric implementation between training and evaluation code.

The full `Evaluator` workflow is documented in the [Evaluation Guide](evaluation.md).

## Choosing a Metric

The metric should represent the property you actually want to monitor.

Examples include:

| Task                      | Possible metric                 |
| ------------------------- | ------------------------------- |
| Binary classification     | Accuracy, Precision, Recall, F1 |
| Multiclass classification | Accuracy, Macro F1              |
| Regression                | MAE, RMSE, R²                   |
| Imbalanced classification | Precision, Recall, F1           |

The appropriate metric depends on the task and the objective of the experiment.

For example, accuracy can be misleading when one class is much more common than another. In such cases, class-sensitive metrics such as precision, recall, or F1 may provide more useful information.

The metric should therefore be selected independently from the loss function: the loss determines how the model is optimized, while the metric determines how its performance is measured.

## Metric Direction

DanFlow's training history stores metric values, but the interpretation of whether a larger or smaller value is preferable depends on the metric itself.

For example:

```text id="r6k2m9"
Accuracy
    Higher is better

F1
    Higher is better

MAE
    Lower is better

RMSE
    Lower is better
```

This distinction is particularly important when metric values are used to identify a best result.

The current `Trainer` implementation treats a larger validation metric as better when updating `best_valid_metric`. Therefore, loss-like metrics such as MAE or RMSE should not be passed as training metrics when the expected selection behavior is "lower is better" unless this behavior is intentionally accounted for.

For loss-like quantities, use the training loss for model selection or verify the selection behavior before using the metric as a best-metric criterion.

## Metric State and Reuse

A stateful metric stores information from previous batches.

That state must represent only the current evaluation period.

The metric lifecycle should therefore be conceptually separated:

```text id="1c9v7s"
Training period
    Reset
    Update for training batches
    Compute

Validation period
    Reset
    Update for validation batches
    Compute
```

Do not manually accumulate values from unrelated evaluation periods in the same metric state.

When implementing a custom metric, keeping all accumulated state inside the metric object makes this lifecycle explicit.

## Common Mistakes

### Passing a Function to `Trainer`

This is not the same as the callable metric accepted by `Evaluator`.

This will not satisfy the stateful metric contract:

```python id="v4r8n2"
def accuracy(outputs, targets):
    return (outputs.argmax(dim=1) == targets).float().mean()
```

A training metric must provide:

```python id="c7w3m5"
reset()
update(outputs, targets)
compute()
```

### Returning a Python Float from `compute()`

For training-related DanFlow components, `compute()` should return a scalar value that supports `.item()`.

Prefer:

```python id="m1k8p4"
return torch.tensor(value)
```

rather than relying on a raw Python float.

### Forgetting to Reset Internal State

If the metric keeps counters between evaluation periods, results can include values from earlier data.

A correct `reset()` implementation should clear every piece of accumulated state.

### Using an Uninformative Class Name

Because `Trainer` derives the metric name from the metric object's class name, a descriptive class name improves the readability of training results and history.

## Summary

DanFlow has two metric interfaces:

```text id="h5d2q7"
Training-related components
    Trainer
    ModelChecker
    LearningRateSelector
    SmallGrid

    Required metric lifecycle
        reset()
        update(outputs, targets)
        compute()
```

and:

```text id="z8f4m1"
Evaluation
    Evaluator

    Required metric interface
        metric(outputs, targets)
```

For training metrics, the implementation must maintain state across batches and return a scalar value from `compute()` that supports `.item()`.

For evaluation, a simple callable is sufficient.

Keep metric selection aligned with the task, and be aware that DanFlow currently treats larger validation metric values as better when tracking `best_valid_metric`.