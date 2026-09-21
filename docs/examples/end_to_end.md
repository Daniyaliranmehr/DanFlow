# End-to-End

This example demonstrates a complete machine-learning workflow with DanFlow, from dataset preparation to final model evaluation and training-history visualization.

The example uses a synthetic binary-classification dataset and combines DanFlow's model checking, hyperparameter tuning, training, evaluation, and visualization utilities in one workflow.

The workflow is organized into the following stages:

```text
Data Preparation
    Dataset
    Train / Validation / Test Split

Model Setup
    Model
    Loss
    Optimizer
    Metric

Validation & Tuning
    ModelChecker
    LearningRateSelector
    SmallGrid

Final Training
    Final Model
    Trainer.fit()
    Best Checkpoint

Evaluation
    Load Checkpoint
    Evaluator
    Test Results

Visualization
    Loss History
    Metric History
    Training History
```

## Prepare the Dataset

Create a synthetic binary-classification dataset with four numerical features.

The target is generated from a linear combination of the input features.

```python id="0p2x7d"
import torch

torch.manual_seed(42)

x = torch.randn(600, 4)

score = (
    1.5 * x[:, 0]
    + 0.8 * x[:, 1]
    - 0.5 * x[:, 2]
    + 0.2 * x[:, 3]
)

y = (score > 0).long()
```

Create deterministic train, validation, and test splits:

```python id="8ohqis"
indices = torch.randperm(
    len(x),
    generator=torch.Generator().manual_seed(42),
)

train_indices = indices[:360]
valid_indices = indices[360:480]
test_indices = indices[480:]

x_train = x[train_indices]
y_train = y[train_indices]

x_valid = x[valid_indices]
y_valid = y[valid_indices]

x_test = x[test_indices]
y_test = y[test_indices]
```

Check the resulting splits:

```pycon id="vw1pxv"
>>> x_train.shape
torch.Size([360, 4])

>>> x_valid.shape
torch.Size([120, 4])

>>> x_test.shape
torch.Size([120, 4])

>>> y_train.shape
torch.Size([360])

>>> y_valid.shape
torch.Size([120])

>>> y_test.shape
torch.Size([120])
```

The test set is kept separate from the training and validation data and will only be used after the final model has been trained.

## Create the DataLoaders

Wrap the training and validation tensors in PyTorch datasets and data loaders.

```python id="s6gfns"
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(
    x_train,
    y_train,
)

valid_dataset = TensorDataset(
    x_valid,
    y_valid,
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

The test tensors are kept available directly because `Evaluator.test()` accepts tensors rather than a `DataLoader`.

## Define the Model

Create a small feed-forward classifier with two output classes.

```python id="48l5n0"
import torch.nn as nn


def build_model():
    return nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
```

Using a builder function makes it possible to create fresh model instances for checking, tuning, and final training.

## Define the Loss and Metric

Use cross-entropy loss for the two-class classification task.

DanFlow's training workflow expects a stateful metric object with `reset()`, `update()`, and `compute()` methods, so define a small metric class for this example.

```python id="8l4q8q"
import torch
import torch.optim as optim


class BinaryAccuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        predictions = outputs.argmax(dim=1)
        self.correct += (predictions == targets).sum().item()
        self.total += targets.numel()

    def compute(self):
        return torch.tensor(
            self.correct / self.total
        )


loss_fn = nn.CrossEntropyLoss()
```

## Check the Model Before Training

Create a model, optimizer, and `ModelChecker` for the initial validation.

```python id="x73l1l"
from danflow.training.checker import ModelChecker

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
```

Run a forward check:

```python id="wts08n"
forward_result = checker.forward_check(
    train_loader,
    expected_output_size=2,
)
```

Inspect the structural parts of the result:

```pycon id="47f4cr"
>>> forward_result.num_batches
5

>>> forward_result.input_shape
(32, 4)

>>> forward_result.target_shape
(32,)

>>> forward_result.output_shape
(32, 2)
```

The model can also be tested on a small subset to verify that the loss decreases during backpropagation:

```python id="qj9e2b"
backward_result = checker.backward_check(
    train_dataset=train_dataset,
    num_samples=128,
    epochs=5,
)
```

Because `backward_check()` updates the model and optimizer in place, create a fresh model before starting hyperparameter search.

```python id="y6d7m3"
model = build_model()
```

## Search for a Suitable Learning Rate

Start with a small set of candidate learning rates.

```python id="4p0h2p"
from danflow.training.tuner import LearningRateSelector

metric = BinaryAccuracy()

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

```python id="qndt1t"
lr_results = selector.search(
    train_loader,
)
```

The results contain one record for each learning rate:

```pycon id="xwhby0"
>>> len(lr_results)
3

>>> lr_results[0].keys()
dict_keys(['learning_rate', 'loss', 'metric'])
```

Select the learning rate with the lowest final loss:

```python id="j1q8fj"
best_lr = min(
    lr_results,
    key=lambda result: result["loss"],
)["learning_rate"]
```

The selected value comes directly from the actual search results.

## Search Learning Rate and Weight Decay

Use `SmallGrid` to compare a small number of learning-rate and weight-decay combinations.

```python id="qcnx4f"
from danflow.training.tuner import SmallGrid

grid = SmallGrid(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=BinaryAccuracy(),
    learning_rates=[
        best_lr,
        0.0005,
    ],
    weight_decays=[
        0.0,
        1e-4,
    ],
    epochs=3,
)
```

Run the grid search:

```python id="w2r8cu"
grid_results = grid.search(
    train_loader,
)
```

The result list contains one record for every parameter combination:

```pycon id="be2y4t"
>>> len(grid_results)
4

>>> grid_results[0].keys()
dict_keys(['learning_rate', 'weight_decay', 'loss', 'metric'])
```

Select the configuration with the lowest final loss:

```python id="z5c1xg"
best_config = min(
    grid_results,
    key=lambda result: result["loss"],
)
```

The selected configuration can be inspected directly:

```pycon id="n4z9gq"
>>> best_config.keys()
dict_keys(['learning_rate', 'weight_decay', 'loss', 'metric'])
```

## Create the Final Model

Create a fresh model for the final training stage.

```python id="3q51sa"
model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=best_config["learning_rate"],
    weight_decay=best_config["weight_decay"],
)

metric = BinaryAccuracy()
```

Create the DanFlow `Trainer`:

```python id="9zixqp"
from danflow.training import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=metric,
)
```

At this point, the final training configuration has been selected and a fresh model is ready for training.

## Train the Final Model

Train the model using both the training and validation loaders.

Save the best validation-loss checkpoint during training.

```python id="p1r3yq"
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
    save_best=True,
    checkpoint_path="best_model.pth",
)
```

Inspect the returned history:

```pycon id="3n7tt4"
>>> list(history.keys())
['train_loss', 'valid_loss', 'train_metric', 'valid_metric', 'metric_name', 'best_valid_loss', 'best_loss_epoch', 'best_valid_metric', 'best_metric_epoch']

>>> len(history["train_loss"])
20

>>> len(history["valid_loss"])
20
```

The exact numerical values depend on the training run and should be read from the actual returned history.

## Load the Best Checkpoint

The final evaluation should use the saved best checkpoint rather than assuming that the last training epoch is the best model.

Load the checkpoint:

```python id="h2n9qk"
checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=True,
)
```

The checkpoint contains the model and optimizer states together with training metadata:

```pycon id="t2l4m0"
>>> checkpoint.keys()
dict_keys(['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_valid_loss'])
```

Create a fresh model and restore its parameters:

```python id="4c3wno"
evaluation_model = build_model()

evaluation_model.load_state_dict(
    checkpoint["model_state_dict"],
)
```

The evaluation model now represents the saved best checkpoint.

## Evaluate the Final Model

For final evaluation, use the test tensors that were not involved in training or hyperparameter selection.

The `Evaluator` uses a callable metric, so define an accuracy function for the test stage:

```python id="o4n86p"
def accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean()
```

Create the evaluator:

```python id="50vkqw"
from danflow.training import Evaluator

evaluator = Evaluator(
    model=evaluation_model,
    loss_fn=nn.CrossEntropyLoss(),
    metric=accuracy,
)
```

Run the evaluation:

```python id="w6x4cv"
test_result = evaluator.test(
    x_test,
    y_test,
)
```

The returned object is a dictionary:

```pycon id="r5k0zn"
>>> list(test_result.keys())
['Metric', 'Loss']
```

The numerical results depend on the trained checkpoint:

```python id="bcs6f2"
print(f"Test loss: {test_result['Loss']:.4f}")
print(f"Test accuracy: {test_result['Metric']:.4f}")
```

The final test values should be taken from the actual evaluation run rather than hard-coded into the example.

## Visualize the Training History

The same history returned by `Trainer.fit()` can be used with DanFlow's training visualization utilities.

Plot the loss history:

```python id="8f1t2r"
from danflow.visualization.training import plot_loss_history

plot_loss_history(
    history,
    name="Binary Classifier",
    show_best_loss=True,
)
```

Plot the metric history:

```python id="6js7hw"
from danflow.visualization.training import plot_metric_history

plot_metric_history(
    history,
    name="Binary Classifier",
    show_best_metric=True,
)
```

The combined view can also be used:

```python id="6q0y83"
from danflow.visualization.training import plot_training_history

plot_training_history(
    history,
    name="Binary Classifier",
    show_best_loss=True,
    show_best_metric=True,
)
```

These plots provide a visual record of the model's training and validation behavior.

## Complete Workflow

The main workflow can now be summarized in one script:

```python id="be0qfu"
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from danflow.training import (
    Evaluator,
    ModelChecker,
    Trainer,
)
from danflow.training.tuner import (
    LearningRateSelector,
    SmallGrid,
)
from danflow.visualization.training import (
    plot_loss_history,
    plot_metric_history,
    plot_training_history,
)


# Data

torch.manual_seed(42)

x = torch.randn(600, 4)

score = (
    1.5 * x[:, 0]
    + 0.8 * x[:, 1]
    - 0.5 * x[:, 2]
    + 0.2 * x[:, 3]
)

y = (score > 0).long()

indices = torch.randperm(
    len(x),
    generator=torch.Generator().manual_seed(42),
)

train_indices = indices[:360]
valid_indices = indices[360:480]
test_indices = indices[480:]

x_train, y_train = x[train_indices], y[train_indices]
x_valid, y_valid = x[valid_indices], y[valid_indices]
x_test, y_test = x[test_indices], y[test_indices]

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=32,
    shuffle=True,
)

valid_loader = DataLoader(
    TensorDataset(x_valid, y_valid),
    batch_size=32,
    shuffle=False,
)


# Model

def build_model():
    return nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )


loss_fn = nn.CrossEntropyLoss()


# Metric for training

class BinaryAccuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        predictions = outputs.argmax(dim=1)
        self.correct += (predictions == targets).sum().item()
        self.total += targets.numel()

    def compute(self):
        return torch.tensor(
            self.correct / self.total
        )


# Model checking

model = build_model()

checker = ModelChecker(
    model=model,
    optimizer=optim.Adam(
        model.parameters(),
        lr=0.001,
    ),
    loss_fn=loss_fn,
)

checker.forward_check(
    train_loader,
    expected_output_size=2,
)

checker.backward_check(
    train_dataset=TensorDataset(
        x_train,
        y_train,
    ),
    num_samples=128,
    epochs=5,
)


# Hyperparameter search

model = build_model()

selector = LearningRateSelector(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=BinaryAccuracy(),
    learning_rates=[0.01, 0.001, 0.0001],
    epochs=3,
)

lr_results = selector.search(
    train_loader,
)

best_lr = min(
    lr_results,
    key=lambda result: result["loss"],
)["learning_rate"]

grid = SmallGrid(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=BinaryAccuracy(),
    learning_rates=[best_lr, 0.0005],
    weight_decays=[0.0, 1e-4],
    epochs=3,
)

grid_results = grid.search(
    train_loader,
)

best_config = min(
    grid_results,
    key=lambda result: result["loss"],
)


# Final training

model = build_model()

trainer = Trainer(
    model=model,
    optimizer=optim.Adam(
        model.parameters(),
        lr=best_config["learning_rate"],
        weight_decay=best_config["weight_decay"],
    ),
    loss_fn=loss_fn,
    metric=BinaryAccuracy(),
)

history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
    save_best=True,
    checkpoint_path="best_model.pth",
)



# Load best checkpoint

checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=True,
)

evaluation_model = build_model()

evaluation_model.load_state_dict(
    checkpoint["model_state_dict"],
)


# Final evaluation

def accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean()


evaluator = Evaluator(
    model=evaluation_model,
    loss_fn=nn.CrossEntropyLoss(),
    metric=accuracy,
)

test_result = evaluator.test(
    x_test,
    y_test,
)

print(f"Test loss: {test_result['Loss']:.4f}")
print(f"Test accuracy: {test_result['Metric']:.4f}")


# Visualization

plot_loss_history(
    history,
    name="Binary Classifier",
    show_best_loss=True,
)

plot_metric_history(
    history,
    name="Binary Classifier",
    show_best_metric=True,
)

plot_training_history(
    history,
    name="Binary Classifier",
    show_best_loss=True,
    show_best_metric=True,
)
```

The completed project now connects the main DanFlow components into one reproducible workflow:

```text
Data Preparation
    Dataset
    Train / Validation / Test Split

Model Setup
    Model
    Loss
    Metric

Validation & Tuning
    ModelChecker
    LearningRateSelector
    SmallGrid

Final Training
    Trainer.fit()
    Best Checkpoint

Evaluation
    Evaluator
    Test Loss
    Test Metric

Visualization
    Loss History
    Metric History
    Training History
```

This example focuses on how the components work together. Detailed parameter descriptions, return values, and error handling belong in the corresponding `docs/api/` pages.