# Tuner

Provides utilities for selecting a suitable learning rate by comparing multiple training experiments.

The `LearningRateSelector` class evaluates different learning rates by creating independent copies of the provided model, initializing a new optimizer for each learning rate, and comparing the final loss and metric values.

## LearningRateSelector

Selects a suitable learning rate by running multiple training experiments.

The selector evaluates different learning rates by:

- Creating independent copies of the provided model.
- Initializing a new optimizer for each learning rate.
- Training each model for a fixed number of epochs.
- Comparing final loss and metric values.

This helps identify promising learning rate ranges before performing more detailed hyperparameter searches.

### Parameters

#### `model` : `torch.nn.Module`

Neural network model to evaluate.

#### `trainer_cls` : `type`

Trainer class used for training each model.

#### `optimizer_cls` : `type[torch.optim.Optimizer]`

Optimizer class used to update model parameters.

#### `loss_fn` : `Callable`

Loss function used during training.

#### `metric` : `object | None`, default=`None`

Optional metric used to evaluate model performance.

#### `learning_rates` : `list[float] | None`, default=`None`

Learning rates to evaluate.

If `None`, the following learning rates are used:

```pycon
>>> LearningRateSelector.DEFAULT_LEARNING_RATES
[0.1, 0.01, 0.001, 0.0001]
```

#### `weight_decay` : `float`, default=`1e-4`

Weight decay value passed to the optimizer.

#### `epochs` : `int`, default=`5`

Number of training epochs performed for each learning rate.

### Example

```pycon
>>> import torch
>>> from torch import nn
>>> from torch.utils.data import DataLoader, TensorDataset
>>> from torch.optim import Adam
>>> from danflow.training import Trainer
>>> from danflow.training.tuner import LearningRateSelector

>>> x = torch.randn(100, 4)
>>> y = torch.randint(0, 3, (100,))

>>> dataset = TensorDataset(x, y)
>>> train_loader = DataLoader(
...     dataset,
...     batch_size=10,
... )

>>> model = nn.Sequential(
...     nn.Linear(4, 16),
...     nn.ReLU(),
...     nn.Linear(16, 3),
... )

>>> selector = LearningRateSelector(
...     model=model,
...     trainer_cls=Trainer,
...     optimizer_cls=Adam,
...     loss_fn=nn.CrossEntropyLoss(),
...     learning_rates=[0.01, 0.001],
...     epochs=5,
... )

>>> results = selector.search(train_loader)

LR=0.01
...
LR=0.001
...

Final Results
+---------------+----------+--------+
| Learning Rate |  Metric  |  Loss  |
+---------------+----------+--------+
|      0.01     |   ...    |  ...   |
|     0.001     |   ...    |  ...   |
+---------------+----------+--------+

Best learning rate: 0.01 (Final loss: ...)
```

## search()

Evaluates the configured learning rates using the provided training DataLoader.

For each learning rate, the method:

### Parameters

#### `train_loader` : `torch.utils.data.DataLoader`

DataLoader containing the training data used for the learning rate experiments.

### Returns

`list[dict[str, Any]]`

A list containing the results of each learning rate experiment.

Each result dictionary contains:

- `"learning_rate"` : `float`
- `"loss"` : `float` 
- `"metric"` : `Any`

### Example

```pycon
>>> results = selector.search(
...     train_loader=train_loader
... )

LR=0.01
...
LR=0.001
...

Final Results
+---------------+----------+--------+
| Learning Rate |  Metric  |  Loss  |
+---------------+----------+--------+
|      0.01     |   ...    |  ...   |
|     0.001     |   ...    |  ...   |
+---------------+----------+--------+

Best learning rate: 0.01 (Final loss: ...)

>>> results[0].keys()
dict_keys(['learning_rate', 'loss', 'metric'])

>>> results[0]["learning_rate"]
0.01

>>> results[0]["loss"]
...

>>> results[0]["metric"]
...
```