# Trainer

Trains and validates PyTorch models.

The `Trainer` class manages the complete training workflow, including training, validation, metric tracking, and optional checkpoint saving.

### Example

```pycon
>>> from danflow.training import Trainer

>>> trainer = Trainer(
...     model=model,
...     optimizer=optimizer,
...     loss_fn=loss_fn,
... )
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

```pycon
>>> train_loss, train_metric = trainer.train_epoch(
...     train_loader=train_loader,
... )

>>> train_loss
0.4238

>>> train_metric
0.9184
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

```pycon
>>> valid_loss, valid_metric = trainer.validate_epoch(
...     valid_loader=valid_loader,
... )

>>> valid_loss
0.3972

>>> valid_metric
0.9261
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

```pycon
>>> history = trainer.fit(
...     train_loader=train_loader,
...     valid_loader=valid_loader,
...     epochs=20,
...     save_best=True,
...     checkpoint_path="checkpoint.pth",
... )

>>> history.keys()
dict_keys([
    'train_loss',
    'valid_loss',
    'train_metric',
    'valid_metric',
    'metric_name',
    'best_valid_loss',
    'best_loss_epoch',
    'best_valid_metric',
    'best_metric_epoch'
])
```