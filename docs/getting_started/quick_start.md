# Quick Start

This quick start demonstrates the minimum workflow required to train a PyTorch model with DanFlow.

The example creates a small binary classification dataset, defines a PyTorch model, creates a `Trainer`, and runs training with separate training and validation data.

For a more complete training workflow, see the [Training Example](../examples/training.md).

## 1. Import the Required Modules

Start by importing PyTorch and DanFlow.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from danflow.training import Trainer
```

## 2. Create the Dataset

For this quick start, create a small synthetic binary classification dataset.

```python
torch.manual_seed(42)

x = torch.randn(200, 8)
y = (x.sum(dim=1) > 0).long()

dataset = TensorDataset(x, y)

train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset,
    [160, 40],
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

The training and validation loaders now provide batches of input tensors and target tensors.

```pycon
>>> len(train_dataset)
160
>>> len(valid_dataset)
40
>>> next(iter(train_loader))[0].shape
torch.Size([32, 8])
>>> next(iter(train_loader))[1].shape
torch.Size([32])
```

## 3. Define the Model

Create a small feed-forward neural network for the two-class problem.

```python
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
```

The final layer produces two class scores, one for each class.

## 4. Define the Loss Function and Optimizer

Use cross-entropy loss and the Adam optimizer.

```python
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)
```

## 5. Create the Trainer

Pass the model, optimizer, and loss function to `Trainer`.

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```

At this point, the training configuration is ready.

## 6. Train the Model

Call `fit()` with the training and validation loaders.

```python
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=5,
)
```

`valid_loader` is the validation loader used to evaluate the model after each training epoch.

The returned `history` dictionary contains the recorded training information.

```pycon
>>> history.keys()
dict_keys(['train_loss', 'valid_loss', 'train_metric', 'valid_metric', 'metric_name', 'best_valid_loss', 'best_loss_epoch', 'best_valid_metric', 'best_metric_epoch'])
```

Because no metric was provided in this example, the metric-related lists remain empty.

```pycon
>>> history["train_metric"]
[]
>>> history["valid_metric"]
[]
```

## 7. Inspect the Training Loss

The loss recorded for each epoch is available through `train_loss` and `valid_loss`.

```pycon
>>> len(history["train_loss"])
5
>>> len(history["valid_loss"])
5
```

You can inspect the loss from the final epoch:

```pycon
>>> history["train_loss"][-1]
<value produced by the current run>
>>> history["valid_loss"][-1]
<value produced by the current run>
```

The exact numerical values depend on the current model initialization, data, PyTorch version, and execution environment.

## 8. Complete Quick Start

The complete example can be reduced to the following script:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from danflow.training import Trainer


torch.manual_seed(42)

# Data
x = torch.randn(200, 8)
y = (x.sum(dim=1) > 0).long()

dataset = TensorDataset(x, y)

train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset,
    [160, 40],
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
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

# Training configuration
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)

# Training
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=5,
)

print("Final training loss:", history["train_loss"][-1])
print("Final validation loss:", history["valid_loss"][-1])
```

## Next Steps

The Quick Start intentionally omits advanced DanFlow features.

Continue with the [Training Guide](../guides/training.md) to learn how to build a more complete training workflow, including model checking, learning-rate selection, grid search, metrics, validation, and checkpoints.