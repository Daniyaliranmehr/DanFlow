# Training

This example demonstrates a complete model-training workflow with DanFlow.

The workflow is organized into four stages:

```
Data Preparation
    Dataset
    DataLoader
```

```
Model Setup
    Model
    Loss + Optimizer + Metric
```

```
Validation & Tuning
    ModelChecker
    LearningRateSelector
    SmallGrid
```

```
Training
    Final Model + Optimizer
    Trainer.fit()
```

```
Results
    History
    Best Checkpoint
```

The example uses a small binary-classification dataset generated with PyTorch.


## Prepare the Dataset

Create a small synthetic dataset with 1,000 samples and 10 input features.

The target is `1` when the sum of the input features is positive and `0` otherwise.
```python
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)

x = torch.randn(1000, 10)
y = (x.sum(dim=1) > 0).long()

dataset = TensorDataset(x, y)

train_dataset, valid_dataset = random_split(
    dataset,
    [800, 200],
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


Check the resulting dataset and batch shapes:
```pycon
>>> len(dataset)
1000
```

```pycon
>>> len(train_dataset)
800
```

```pycon
>>> len(valid_dataset)
200
```
```pycon
>>> next(iter(train_loader))[0].shape
torch.Size([32, 10])
```

```pycon
>>> next(iter(train_loader))[1].shape
torch.Size([32])
```

The training and validation loaders now provide batches of inputs and class labels.


## Define the Model

Next, define the neural network.

Create a small feed-forward neural network with two output classes.

```python
import torch.nn as nn

def build_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
```

Using a function makes it easy to create a fresh model when different training experiments need independent initial weights.


## Define the Loss, Optimizer, and Metric

For this classification task, use cross-entropy loss and Adam.

DanFlow's `Trainer` uses a stateful metric object with `reset()`, `update()`, and `compute()` methods. `torchmetrics` provides compatible metrics.

```python
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy

loss_fn = nn.CrossEntropyLoss()

model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)

metric = MulticlassAccuracy(
    num_classes=2,
)
```

The same loss function can be reused throughout the training workflow.


## Check the Model Before Training

Before starting a full training run, use `checker` to verify that the model can process batches and update its parameters.

```python
from danflow.training.checker import ModelChecker

checker = ModelChecker(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```


### Forward Check

Run a forward check on a few training batches:

```python
forward_result = checker.forward_check(
    train_loader,
    expected_output_size=2,
)
```

Inspect the parts of the result that are independent of random loss values:

```pycon
>>> forward_result.num_batches
5
```

```pycon
>>> forward_result.input_shape
(32, 10)
```

```pycon
>>> forward_result.target_shape
(32,)
```

```pycon
>>> forward_result.output_shape
(32, 2)
```

These shapes confirm that the model input, target, and output dimensions are compatible.

### Backward Check

The backward check performs a short training run on a subset of the training data.

```python
backward_result = checker.backward_check(
    train_dataset=train_dataset,
    num_samples=128,
    epochs=5,
)
```

The exact loss depends on model initialization and the training environment, so inspect the stable parts of the result:

```pycon
>>> backward_result.epochs_trained
5
```

```pycon
>>> backward_result.automatic_extension_used
False
```

`backward_check()` modifies the model and optimizer in place. Because this check is only a training smoke test, create a fresh model before the actual training workflow.

```python
model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)

metric = MulticlassAccuracy(
    num_classes=2,
)
```


### Continue a Backward Check

If additional training is needed, `continue_backward()` can continue training from the existing checker state.

```python
backward_result = checker.continue_backward(
    epochs=20,
)

print(backward_result)
```

```pycon
BackwardCheckResult(
    initial_loss=0.7162,
    final_loss=0.0413,
    final_metric=None,
    epochs_trained=40,
    target_loss=None,
    target_metric=None,
    success=None,
    automatic_extension_used=False
)
```

This continues from the current model and optimizer state rather than creating a new training subset.


## Compare Learning Rates

Once the model passes the basic checks, compare a small set of learning rates with `LearningRateSelector`.

```python
from danflow.training.tuner import LearningRateSelector

selector = LearningRateSelector(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=metric,
    learning_rates=[
        0.01,
        0.001,
        0.0001,
    ],
    epochs=3,
)
```

Run the search:

```python
lr_results = selector.search(
    train_loader,
)
```

Each experiment trains an independent copy of the original model.

The returned results contain one record per learning rate:

```pycon
>>> len(lr_results)
3
```

```pycon
>>> lr_results[0].keys()
dict_keys(['learning_rate', 'loss', 'metric'])
```

The exact loss and metric values depend on the training run, so select the configuration directly from the returned results instead of hard-coding an expected value.

```python
best_lr = min(
    lr_results,
    key=lambda result: result["loss"],
)["learning_rate"]
``` 

## Search Learning Rate and Weight Decay

After narrowing the learning-rate range, use `SmallGrid` to compare a small number of learning-rate and weight-decay combinations.

```python
from danflow.training.tuner import SmallGrid

grid = SmallGrid(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=metric,
    learning_rates=[
        0.01,
        0.001,
    ],
    weight_decays=[
        0.0,
        1e-4,
    ],
    epochs=3,
)
```

Run the search:

```python
grid_results = grid.search(
    train_loader,
)
```

Two learning rates and two weight-decay values produce four configurations:

```pycon
>>> len(grid_results)
4
```

```pycon
>>> grid_results[0].keys()
dict_keys(['learning_rate', 'weight_decay', 'loss', 'metric'])
```

Select the configuration with the lowest final loss:

```python
best_config = min(
    grid_results,
    key=lambda result: result["loss"],
)
```

```python
best_lr = best_config["learning_rate"]
best_weight_decay = best_config["weight_decay"]
```

The selected values are taken directly from the search results, so the final training configuration remains consistent with the actual experiment.


## Create the Final Trainer

Create a fresh model for the final training run.

```python
model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=best_lr,
    weight_decay=best_weight_decay,
)

metric = MulticlassAccuracy(
    num_classes=2,
)
```

Create the DanFlow Trainer:

```python
from danflow.training import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=metric,
)
```
At this point, the final model, optimizer, loss function, and metric are ready for training.


## Train the Model

For a standard training workflow, use `fit()` with the training and validation loaders.

```python
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=10,
    save_best=True,
    checkpoint_path="best_model.pth",
)
```

The `valid_loader` argument supplies the validation data used at the end of each epoch.

The history contains the training and validation results collected during training:
```pycon
>>> list(history.keys())
['train_loss', 'valid_loss', 'train_metric', 'valid_metric', 'metric_name', 'best_valid_loss', 'best_loss_epoch', 'best_valid_metric', 'best_metric_epoch']
```

```pycon
>>> len(history["train_loss"])
10
```

```pycon
>>> len(history["valid_loss"])
10
```
The actual numerical loss and metric values depend on the training run.


## Inspect the Best Training Result

The history stores the epoch associated with the best validation loss and metric:

```pycon
>>> history["best_loss_epoch"]
10
```

```pycon
>>> history["best_metric_epoch"]
10
```
The exact epoch numbers depend on the generated training history, so they should be read from the returned dictionary rather than assumed in advance.

The history can also be passed to DanFlow's training-visualization utilities for plotting.

## Load the Best Checkpoint

Because `save_best=True` was used, DanFlow stores the best checkpoint in `best_model.pth`.

Load the checkpoint:

```python
checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=True,
)
```

The checkpoint contains the model and optimizer states together with training metadata:

```pycon
>>> checkpoint.keys()
dict_keys(['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_valid_loss'])
```

Load the saved model parameters:

```pycon
model.load_state_dict(
    checkpoint["model_state_dict"],
)
```
The model is now restored to the state corresponding to the saved best validation result.


## Complete Workflow

The core training workflow can now be summarized as:
```
Data Preparation
    Dataset
    DataLoader
```

```
Model Setup
    Model
    Loss + Optimizer + Metric
```

```
Validation & Tuning
    ModelChecker
    LearningRateSelector
    SmallGrid
```

```
Training
    Final Model + Optimizer
    Trainer.fit()
```

```
Results
    History
    Best Checkpoint
```


```python
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.classification import MulticlassAccuracy

from danflow.training.checker import ModelChecker
from danflow.training import Trainer
from danflow.training.tuner import (
    LearningRateSelector,
    SmallGrid,
)

# Prepare data
torch.manual_seed(42)

x = torch.randn(1000, 10)
y = (x.sum(dim=1) > 0).long()

dataset = TensorDataset(x, y)

train_dataset, valid_dataset = random_split(
    dataset,
    [800, 200],
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
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

loss_fn = nn.CrossEntropyLoss()

# Basic model check
model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)

checker = ModelChecker(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)

checker.forward_check(
    train_loader,
    expected_output_size=2,
)

checker.backward_check(
    train_dataset=train_dataset,
    num_samples=128,
    epochs=5,
)

# Hyperparameter search
model = build_model()

metric = MulticlassAccuracy(
    num_classes=2,
)

selector = LearningRateSelector(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=metric,
    learning_rates=[0.01, 0.001, 0.0001],
    epochs=3,
)

lr_results = selector.search(train_loader)

grid = SmallGrid(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=metric,
    learning_rates=[0.01, 0.001],
    weight_decays=[0.0, 1e-4],
    epochs=3,
)

grid_results = grid.search(train_loader)

best_config = min(
    grid_results,
    key=lambda result: result["loss"],
)

# Final training
model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=best_config["learning_rate"],
    weight_decay=best_config["weight_decay"],
)

metric = MulticlassAccuracy(
    num_classes=2,
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=metric,
)

history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=10,
    save_best=True,
    checkpoint_path="best_model.pth",
)
```

The training workflow is now organized around its major stages rather than a linear chain:
```
Data Preparation
    Dataset
    DataLoader
```

```
Model Setup
    Model
    Loss + Optimizer + Metric
```

```
Validation & Tuning
    ModelChecker
    LearningRateSelector
    SmallGrid
```

```
Training
    Final Model + Optimizer
    Trainer.fit()
```

```
Results
    History
    Best Checkpoint
```


The trained model and its history can now be passed to the evaluation and visualization workflows.