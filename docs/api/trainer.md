## AverageMeter

Computes and tracks the current value and running average of a metric.

### Example

```python
from danflow.training import AverageMeter

meter = AverageMeter()

meter.update(0.42)
meter.update(0.38)

print(meter.avg)
```


## reset()

Resets all tracked statistics.

### Parameters

None

### Returns

`None`

### Example

```python
from danflow.training import AverageMeter

meter = AverageMeter()

meter.update(0.42)
meter.reset()
```


## update()

Updates the running statistics with a new value.

### Parameters

#### `val` : `float`

Value to add.

#### `n` : `int`, default=`1`

Number of samples represented by `val`.

### Returns

`None`

### Example

```python
from danflow.training import AverageMeter

meter = AverageMeter()

meter.update(
    val=0.42,
    n=32,
)
```

---

## Trainer

Trains and validates PyTorch models.

The `Trainer` class manages the complete training workflow, including training, validation, metric tracking, and optional checkpoint saving.

### Example

```python
from danflow.training import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```


## train_epoch()

Trains the model for a single epoch.

The model is switched to training mode and updated using all batches from the training dataset.

### Parameters

#### `train_loader` : `torch.utils.data.DataLoader`

DataLoader providing the training dataset.

### Returns

`tuple[float, float | None]`

A tuple containing:

- Average training loss.
- Computed metric value if a metric is provided; otherwise `None`.

### Example

```python
from danflow.training import Trainer

train_loss, train_metric = trainer.train_epoch(
    train_loader=train_loader
)
```

## validate_epoch()

Evaluates the model for a single validation epoch.

The model is switched to evaluation mode and performs inference without updating its parameters.

### Parameters

#### `valid_loader` : `torch.utils.data.DataLoader`

DataLoader providing the validation dataset.

### Returns

`tuple[float, float | None]`

A tuple containing:

- Average validation loss.
- Computed metric value if a metric is provided; otherwise `None`.

### Example

```python
from danflow.training import Trainer

valid_loss, valid_metric = trainer.validate_epoch(
    valid_loader=valid_loader
)
```

## fit()

Trains the model for multiple epochs.

This method performs the complete training workflow, including training, validation, history tracking, optional checkpoint saving, and best model monitoring.

### Parameters

#### `train_loader` : `torch.utils.data.DataLoader`

DataLoader providing the training dataset.

#### `valid_loader` : `torch.utils.data.DataLoader`

DataLoader providing the validation dataset.

#### `epochs` : `int`, default=`100`

Number of training epochs.

#### `save_best` : `bool`, default=`False`

Whether to save the model checkpoint corresponding to the best validation loss.

#### `checkpoint_path` : `str`, default=`"best_model.pth"`

Path where the best model checkpoint will be saved.

### Returns

`dict`

Dictionary containing the training history and the best validation results.

### Example

```python
from danflow.training import Trainer

history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
    save_best=True,
    checkpoint_path="checkpoint.pth",
)
```
---

## Evaluator

Evaluates a trained PyTorch model on test data.

The `Evaluator` class provides a simple interface for running inference on a test dataset and computing optional evaluation metrics and loss.

### Example

```python
from danflow.training import Evaluator

evaluator = Evaluator(
    model=model,
    loss_fn=loss_fn,
    metric=metric,
)
```

---

## test()

Runs model evaluation on test data.

The model is switched to evaluation mode and inference is performed without gradient computation. The model's original training state is restored after evaluation.

### Parameters

#### `x_test` : `torch.Tensor`

Input test features.

#### `y_test` : `torch.Tensor`

Ground-truth target values.

### Returns

`dict[str, float]`

Dictionary containing the evaluation results.

Possible keys include:

- `"Loss"`
- `"Metric"`

### Example

```python
from danflow.training import Evaluator

results = evaluator.test(
    x_test=x_test,
    y_test=y_test,
)

print(results)
```