# Custom Loss

This example demonstrates how to define and use a custom PyTorch loss function with DanFlow.

The example uses a small regression problem and compares two training setups:

```text
MSE Loss
    Model
    Trainer
    Training

Custom Huber Loss
    Model
    Trainer
    Training

Comparison
    Common Test Metric
    Training History
    Validation Performance
```

The custom loss is implemented directly in the example to demonstrate how user-defined objectives can be passed to DanFlow's `Trainer`.

## Prepare the Dataset

Create a small synthetic regression dataset.

The target follows a simple linear relationship with some noise.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)

x = torch.linspace(-2, 2, 240).unsqueeze(1)
y = 3 * x + 1 + 0.2 * torch.randn_like(x)

dataset = TensorDataset(x, y)

train_dataset, valid_dataset = random_split(
    dataset,
    [192, 48],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
)
```

Check the dataset shapes:

```pycon
>>> len(dataset)
240

>>> next(iter(train_loader))[0].shape
torch.Size([32, 1])

>>> next(iter(train_loader))[1].shape
torch.Size([32, 1])
```

The same training and validation data will be used for both loss functions.

## Define the Model

Use a small feed-forward regression model.

```python
import torch.nn as nn


def build_model():
    return nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
```

A function is used so that both experiments can start from the same model architecture.

## Define the Custom Loss

For this example, implement a Huber-style loss manually.

The loss behaves quadratically for small errors and grows more slowly for larger errors.

```python
def custom_huber_loss(outputs, targets, delta=1.0):
    errors = outputs - targets
    absolute_errors = errors.abs()

    quadratic = 0.5 * errors.pow(2)
    linear = delta * (absolute_errors - 0.5 * delta)

    loss = torch.where(
        absolute_errors <= delta,
        quadratic,
        linear,
    )

    return loss.mean()
```

The function accepts model outputs and targets and returns one scalar tensor.

Before using it for training, verify that it produces a scalar and supports backpropagation:

```python
sample_outputs = torch.tensor(
    [[1.0], [2.0]],
    requires_grad=True,
)

sample_targets = torch.tensor(
    [[1.2], [1.5]],
)

loss = custom_huber_loss(
    sample_outputs,
    sample_targets,
)

loss.backward()
```

```pycon
>>> loss.shape
torch.Size([])

>>> sample_outputs.grad is not None
True
```

The scalar shape confirms that the loss can be consumed by the training loop, while the non-`None` gradient confirms that the computation remains differentiable.

## Train a Model with MSE

First, train a baseline model with PyTorch's built-in `MSELoss`.

Create the initial model once so both experiments can start from identical parameters.

```python
import copy
import torch.optim as optim

base_model = build_model()

mse_model = copy.deepcopy(base_model)
custom_model = copy.deepcopy(base_model)
```

Create the optimizer and DanFlow `Trainer` for the MSE experiment:

```python
from danflow.training import Trainer

mse_loss = nn.MSELoss()

mse_optimizer = optim.Adam(
    mse_model.parameters(),
    lr=0.01,
)

mse_trainer = Trainer(
    model=mse_model,
    optimizer=mse_optimizer,
    loss_fn=mse_loss,
)
```

Train the baseline model:

```python
mse_history = mse_trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
)
```

The returned history contains the training and validation losses:

```pycon
>>> list(mse_history.keys())
['train_loss', 'valid_loss', 'train_metric', 'valid_metric', 'metric_name', 'best_valid_loss', 'best_loss_epoch', 'best_valid_metric', 'best_metric_epoch']
```

## Train a Model with the Custom Loss

Create a separate optimizer and trainer for the custom-loss experiment.

```python
custom_optimizer = optim.Adam(
    custom_model.parameters(),
    lr=0.01,
)

custom_trainer = Trainer(
    model=custom_model,
    optimizer=custom_optimizer,
    loss_fn=custom_huber_loss,
)
```

Train the second model using the custom objective:

```python
custom_history = custom_trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
)
```

The training workflow remains unchanged. Only the loss function supplied to `Trainer` has changed.

```pycon
>>> list(custom_history.keys())
['train_loss', 'valid_loss', 'train_metric', 'valid_metric', 'metric_name', 'best_valid_loss', 'best_loss_epoch', 'best_valid_metric', 'best_metric_epoch']
```

This is the main point of the example: a custom callable can be supplied anywhere the training workflow expects a loss function.

## Compare the Two Models

The two models were trained with different optimization objectives, so their training losses are not directly comparable when the loss functions are different.

Instead, evaluate both models using the same loss and metric.

Define mean absolute error as a common evaluation metric:

```python
def mean_absolute_error(outputs, targets):
    return (outputs - targets).abs().mean()
```

Create an evaluator for the MSE-trained model.

```python
from danflow.training import Evaluator

mse_evaluator = Evaluator(
    model=mse_model,
    loss_fn=nn.MSELoss(),
    metric=mean_absolute_error,
)
```

Evaluate it on the validation tensors:

```python
x_valid = torch.cat(
    [batch[0] for batch in valid_loader],
)

y_valid = torch.cat(
    [batch[1] for batch in valid_loader],
)

mse_result = mse_evaluator.test(
    x_valid,
    y_valid,
)
```

Create a second evaluator for the custom-loss model, but keep the evaluation criteria identical:

```python
custom_evaluator = Evaluator(
    model=custom_model,
    loss_fn=nn.MSELoss(),
    metric=mean_absolute_error,
)

custom_result = custom_evaluator.test(
    x_valid,
    y_valid,
)
```

Both results use the same evaluation criteria:

```pycon
>>> list(mse_result.keys())
['Metric', 'Loss']

>>> list(custom_result.keys())
['Metric', 'Loss']
```

The numerical values depend on the training run and should be read directly from the returned dictionaries.

```python
print("MSE model:")
print(f"Validation MSE: {mse_result['Loss']:.4f}")
print(f"Validation MAE: {mse_result['Metric']:.4f}")

print("\nCustom-loss model:")
print(f"Validation MSE: {custom_result['Loss']:.4f}")
print(f"Validation MAE: {custom_result['Metric']:.4f}")
```

Using the same evaluation loss and metric makes the comparison meaningful even though the models were optimized with different training objectives.

## Visualize the Training Histories

The training histories can be inspected with DanFlow's visualization utilities.

Plot the MSE training history:

```python
from danflow.visualization.training import plot_loss_history

plot_loss_history(
    mse_history,
    name="MSE Loss",
)
```

Plot the custom-loss training history:

```python
plot_loss_history(
    custom_history,
    name="Custom Huber Loss",
)
```

These plots show how the training and validation objectives evolved during the two experiments.

## Complete Workflow

The complete example can now be summarized as two parallel training experiments followed by a common evaluation.

```python
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)

from danflow.training import Evaluator, Trainer


# Prepare data
torch.manual_seed(42)

x = torch.linspace(-2, 2, 240).unsqueeze(1)
y = 3 * x + 1 + 0.2 * torch.randn_like(x)

dataset = TensorDataset(x, y)

train_dataset, valid_dataset = random_split(
    dataset,
    [192, 48],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
)


# Model
def build_model():
    return nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


# Custom loss
def custom_huber_loss(outputs, targets, delta=1.0):
    errors = outputs - targets
    absolute_errors = errors.abs()

    quadratic = 0.5 * errors.pow(2)
    linear = delta * (absolute_errors - 0.5 * delta)

    loss = torch.where(
        absolute_errors <= delta,
        quadratic,
        linear,
    )

    return loss.mean()


# Start both experiments from identical parameters
base_model = build_model()

mse_model = copy.deepcopy(base_model)
custom_model = copy.deepcopy(base_model)


# MSE training
mse_trainer = Trainer(
    model=mse_model,
    optimizer=optim.Adam(
        mse_model.parameters(),
        lr=0.01,
    ),
    loss_fn=nn.MSELoss(),
)

mse_history = mse_trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
)


# Custom-loss training
custom_trainer = Trainer(
    model=custom_model,
    optimizer=optim.Adam(
        custom_model.parameters(),
        lr=0.01,
    ),
    loss_fn=custom_huber_loss,
)

custom_history = custom_trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
)


# Prepare common evaluation tensors
x_valid = torch.cat(
    [batch[0] for batch in valid_loader],
)

y_valid = torch.cat(
    [batch[1] for batch in valid_loader],
)


# Common evaluation criteria
def mean_absolute_error(outputs, targets):
    return (outputs - targets).abs().mean()


mse_evaluator = Evaluator(
    model=mse_model,
    loss_fn=nn.MSELoss(),
    metric=mean_absolute_error,
)

custom_evaluator = Evaluator(
    model=custom_model,
    loss_fn=nn.MSELoss(),
    metric=mean_absolute_error,
)

mse_result = mse_evaluator.test(
    x_valid,
    y_valid,
)

custom_result = custom_evaluator.test(
    x_valid,
    y_valid,
)

print("MSE model:")
print(f"Validation MSE: {mse_result['Loss']:.4f}")
print(f"Validation MAE: {mse_result['Metric']:.4f}")

print("\nCustom-loss model:")
print(f"Validation MSE: {custom_result['Loss']:.4f}")
print(f"Validation MAE: {custom_result['Metric']:.4f}")
```

The important part of the workflow is that the training infrastructure does not change when the loss changes:

```text
Data
    Dataset
    DataLoader

Experiment A
    MSELoss
    Trainer
    Training

Experiment B
    Custom Loss
    Trainer
    Training

Comparison
    Common Evaluation Loss
    Common Evaluation Metric
    Training History
```

This pattern can be adapted to other regression objectives by replacing `custom_huber_loss` with a different differentiable function that accepts model outputs and targets and returns a scalar tensor.
