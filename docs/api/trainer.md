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
train_loss, train_metric = trainer.train_epoch(
    train_loader=train_loader
)
```