# Checker

Provides utilities for verifying the forward and backward paths of PyTorch models before or during training.

The `ModelChecker` class can be used to validate model input/output compatibility, verify loss calculation, and check whether a model can overfit a small training subset.

## forward_check()

Checks the model's forward path using batches from a training `DataLoader`.

The method:

- Iterates over the provided DataLoader.
- Passes several batches through the model.
- Checks input, target, and output compatibility.
- Checks the model output size when provided.
- Calculates the loss.
- Raises a helpful error if the loss cannot be calculated.
- Computes the average initial loss.


### Parameters

#### `train_loader` : `torch.utils.data.DataLoader`

DataLoader containing the input and target tensors used for the forward-path check.

#### `expected_output_size` : `int | None`, default=`None`

Expected size of the model's final output dimension.

If provided, the last dimension of the model output must match this value.

For example, for a 7-class classification model:

```pycon
>>> expected_output_size=7
```

#### `num_batches` : `int, default=5`

Maximum number of batches used for the forward check and average initial loss calculation.

The value must be at least `1`.

### Returns

`ForwardCheckResult`

Information about the verified forward path.

### Example

```pycon
>>> import torch
>>> from torch import nn
>>> from torch.utils.data import DataLoader, TensorDataset
>>> from danflow.training import ModelChecker

>>> x = torch.randn(20, 4)
>>> y = torch.randint(0, 3, (20,))

>>> dataset = TensorDataset(x, y)
>>> train_loader = DataLoader(dataset, batch_size=4)

>>> model = nn.Sequential(
...     nn.Linear(4, 3)
... )

>>> optimizer = torch.optim.Adam(model.parameters())
>>> loss_fn = nn.CrossEntropyLoss()

>>> checker = ModelChecker(
...     model=model,
...     optimizer=optimizer,
...     loss_fn=loss_fn,
... )

>>> result = checker.forward_check(
...     train_loader=train_loader,
...     expected_output_size=3,
...     num_batches=2,
... )
Input shape:  torch.Size([4, 4])
Target shape: torch.Size([4])
Output shape: torch.Size([4, 3])
Average initial loss (2 batches): 1.2...

>>> result.num_batches
2

>>> result.input_shape
(4, 4)

>>> result.target_shape
(4,)

>>> result.output_shape
(4, 3)
```


## backward_check()

Checks whether a PyTorch model can overfit a small subset of the training dataset.

The method:

- Selects a random subset of the provided training dataset.
- Creates a mini `DataLoader` for the selected subset.
- Trains the model on the subset using `Trainer`.
- Tracks the loss and optional metric during training.
- Checks whether the requested loss and metric targets are reached.
- Automatically extends training when `epochs=None` and the requested target is not reached.
- Preserves the same subset and optimizer state when training is continued.

This check can help identify problems in the backward path, such as incorrect gradient flow, optimizer configuration, loss calculation, or model parameter updates.

### Parameters

#### `train_dataset`

Training dataset from which the subset used for the overfitting experiment is selected.

The dataset must implement `__len__()` and `__getitem__()` and must contain at least one sample.

#### `num_samples` : `int`, default=`1000`

Number of samples selected from the training dataset for the overfitting experiment.

The value must be at least `1` and cannot be greater than the size of the training dataset.

#### `batch_size` : `int | None`, default=`None`

Batch size used by the `DataLoader`.

If `None`, the checker automatically selects a batch size that produces approximately 5 batches.

For example:

```pycon
>>> num_samples=1000
>>> batch_size=200
```

#### `metric`

Optional TorchMetrics metric used during training.

If a metric is provided, its value is tracked and can be used as an overfitting target.

#### `target_metric` : `float | None`, default=`None`

Minimum metric value required for the overfitting check to succeed.

A `metric` must be provided when `target_metric` is specified.

For example, for an accuracy metric:

```pycon
>>> target_metric=0.99
```
The check succeeds when the final metric is greater than or equal to the specified value.

#### `target_loss` : `float | None`, default=`None`

Maximum loss value required for the overfitting check to succeed.

The value must be non-negative.

For example:

```pycon
>>> target_loss=0.01
```

The check succeeds when the final loss is less than or equal to the specified value.

#### `epochs` : `int | None`, default=`None`

Number of epochs used for the first training attempt.

If `None`, the checker uses `500` epochs.

When `None` and an overfitting target is provided but not reached, the checker automatically trains for another `500` epochs.

If an explicit value is provided, automatic extension is not performed.

The value must be at least `1`.

#### `seed` : `int | None`, default=`None`

Optional random seed used when selecting the subset from the training dataset.

Providing a seed makes the subset selection reproducible.

### Returns

A backward-check result containing:

- Final loss recorded after the most recently completed epoch.
- Final metric, if a metric was provided.
- Total number of epochs trained.
- Requested target loss.
- Requested target metric.
- Whether the requested overfitting target was reached.
- Whether automatic training extension was used.

### Example

```pycon
>>> import torch
>>> from torch import nn
>>> from torch.utils.data import TensorDataset
>>> from torchmetrics.classification import MulticlassAccuracy
>>> from danflow.training import ModelChecker

>>> x = torch.randn(100, 4)
>>> y = torch.randint(0, 3, (100,))

>>> train_dataset = TensorDataset(x, y)

>>> model = nn.Sequential(
...     nn.Linear(4, 16),
...     nn.ReLU(),
...     nn.Linear(16, 3),
... )

>>> optimizer = torch.optim.Adam(
...     model.parameters(),
...     lr=0.01,
... )

>>> loss_fn = nn.CrossEntropyLoss()

>>> metric = MulticlassAccuracy(
...     num_classes=3,
... )

>>> checker = ModelChecker(
...     model=model,
...     optimizer=optimizer,
...     loss_fn=loss_fn,
... )

>>> result = checker.backward_check(
...     train_dataset=train_dataset,
...     num_samples=20,
...     batch_size=5,
...     metric=metric,
...     target_metric=0.99,
...     target_loss=0.05,
...     epochs=100,
...     seed=42,
... )

Backward check:   ...%|...| 100/100 [...]
Initial loss: 1.2...
Final loss:   0.0...
Final metric: 1.0000
Result: The model successfully reached the requested overfitting target.
```

## continue_backward()

Continues the backward-path overfitting check on the same subset of the training dataset.

The method:

- Continues training from the current model state.
- Uses the same subset that was created by `backward_check()`.
- Preserves the current optimizer state.
- Trains the model for the specified number of additional epochs.
- Updates the total number of epochs trained.
- Re-evaluates the requested overfitting targets.
- Returns the updated backward-check results.

`backward_check()` must be called before using this method.

### Parameters

#### `epochs` : `int`

Number of additional epochs used to continue the backward check.

The value must be at least `1`.

### Returns

A backward-check result containing the updated:

- Initial loss.
- Final loss.
- Final metric, if a metric was provided.
- Total number of epochs trained.
- Target loss.
- Target metric.
- Overfitting success status.

The `automatic_extension_used` value is `False` for results returned by `continue_backward()`.

### Example

```pycon
>>> result = checker.backward_check(
...     train_dataset=train_dataset,
...     num_samples=20,
...     batch_size=5,
...     metric=metric,
...     target_metric=0.99,
...     target_loss=0.05,
...     epochs=100,
...     seed=42,
... )

Backward check:   ...%|...| 100/100 [...]
Initial loss: 1.2...
Final loss:   0.2...
Final metric: 0.9500
Result: The model did not reach the requested overfitting target.

>>> result = checker.continue_backward(
...     epochs=100
... )

Backward check:   ...%|...| 100/100 [...]
Initial loss: 1.2...
Final loss:   0.0...
Final metric: 1.0000
Result: The model successfully reached the requested overfitting target.
```

## ForwardCheckResult

Stores the results of a forward-path check.

The result contains information about the batches processed during the check, the average loss, and the shapes of the inputs, targets, and model outputs.

### Attributes

#### `num_batches` : `int`

Number of batches processed during the forward check.

#### `average_loss` : `float`

Average loss calculated across the processed batches.

#### `input_shape` : `tuple[int, ...]`

Shape of the input tensor from the first processed batch.

#### `target_shape` : `tuple[int, ...]`

Shape of the target tensor from the first processed batch.

#### `output_shape` : `tuple[int, ...]`

Shape of the model output tensor from the first processed batch.

## BackwardCheckResult

Stores the results of a backward-path overfitting check.

The result contains information about the initial and final loss, the final metric, the number of epochs trained, the requested overfitting targets, and whether those targets were reached.

### Attributes

#### `initial_loss` : `float`

Loss recorded after the first training epoch.

#### `final_loss` : `float`

Loss recorded after the most recently completed training epoch.

#### `final_metric` : `float | None`

Metric value recorded after the most recently completed training epoch.

None if no metric was provided.

#### `epochs_trained` : `int`

Total number of epochs trained during the backward check.

This includes any additional epochs from automatic extension or calls to continue_backward().

#### `target_loss` : `float | None`

Maximum loss specified as the overfitting target.

None if no loss target was provided.

#### `target_metric` : `float | None`

Minimum metric value specified as the overfitting target.

None if no metric target was provided.

#### `success` : `bool | None`

Indicates whether the requested overfitting targets were reached.

True if all requested targets were reached.
False if at least one requested target was not reached.
None if no overfitting target was provided.

#### `automatic_extension_used` : `bool`

Indicates whether backward_check() automatically extended the training by another DEFAULT_EPOCHS because the requested target was not reached during the first training phase.