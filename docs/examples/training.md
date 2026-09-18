# Training

Demonstrates a complete model training workflow using DanFlow's training utilities.

This example shows how to:

* Create a PyTorch `Dataset` and `DataLoader`.
* Define a PyTorch model.
* Define a loss function and optimizer.
* Validate the model with `ModelChecker`.
* Train a model for a single epoch with `train_epoch()`.
* Validate a model with `validate_epoch()`.
* Train a model for multiple epochs with `fit()`.
* Select a suitable learning rate with `LearningRateSelector`.
* Perform a small grid search over learning rate and weight decay with `SmallGrid`.


## Create a Dataset and DataLoader

The first step is to prepare the training data.

For this example, a small synthetic binary classification dataset is created using PyTorch tensors.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

x = torch.randn(1000, 10)

y = (x.sum(dim=1) > 0).long()

dataset = TensorDataset(x, y)

train_size = 800
validation_size = 200

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
)
```

```pycon
>>> len(dataset)
1000
>>> len(train_dataset)
800
>>> len(validation_dataset)
200
>>> next(iter(train_loader))[0].shape
torch.Size([32, 10])
>>> next(iter(train_loader))[1].shape
torch.Size([32])
```

The resulting `DataLoader` objects provide batches of input tensors and target tensors during training and validation.


## Define a PyTorch Model

Next, define the neural network.

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(in_features=10, out_features=32, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=2, bias=True),
)
```


## Define the Loss Function and Optimizer

Define the loss function and optimizer used during training.

```python
import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)
```

```pycon
>>> loss_fn
CrossEntropyLoss()

>>> optimizer
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    initial_lr: 0.001
    lr: 0.001
    maximize: False
    weight_decay: 0
)
```

`CrossEntropyLoss` is appropriate for this two-class classification example because the model outputs class scores (logits).


## Validate the Model with `ModelChecker`

Before performing a full training run, it is useful to verify that the model can execute a valid forward pass and that its parameters can be updated during backpropagation.

```python
from danflow.training import ModelChecker

checker = ModelChecker(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```


### Forward Check

`forward_check()` verifies that the model can process batches correctly.

```python
forward_result = checker.forward_check(
    train_loader,
    expected_output_size=2,
)

print(forward_result)
```

```pycon
ForwardCheckResult(
    num_batches=5,
    average_loss=0.7241,
    input_shape=(32, 10),
    target_shape=(32,),
    output_shape=(32, 2)
)
```

The result contains information about:

* Number of checked batches.
* Average loss.
* Input tensor shape.
* Target tensor shape.
* Output tensor shape.

For example, the output shapes should be consistent with the model:

```pycon
>>> forward_result.input_shape
(32, 10)
>>> forward_result.target_shape
(32,)
>>> forward_result.output_shape
(32, 2)
```

The method also validates that the output is a tensor, that the batch dimension matches the target, and that the computed loss is a finite scalar.

---

### Backward Check

After verifying the forward pass, `backward_check()` can be used to determine whether the model is capable of learning from the training data.

```python
backward_result = checker.backward_check(
    train_dataset=train_dataset,
    num_samples=200,
    epochs=20,
)

print(backward_result)
```

```pycon
BackwardCheckResult(
    initial_loss=0.7162,
    final_loss=0.0827,
    final_metric=None,
    epochs_trained=20,
    target_loss=None,
    target_metric=None,
    success=None,
    automatic_extension_used=False
)
```

The result reports information such as:

* Initial loss.
* Final loss.
* Final metric, when a metric is provided.
* Number of epochs trained.
* Target loss or target metric, when specified.
* Whether the requested target was reached.
* Whether automatic extension was used.

Targets can also be supplied when a specific training condition is expected.

```python
backward_result = checker.backward_check(
    train_dataset=train_dataset,
    num_samples=200,
    target_loss=0.10,
)
```

```pycon
>>> backward_result.final_loss
0.0938
>>> backward_result.target_loss
0.1
>>> backward_result.success
True
```

When a target is not reached, the result indicates whether the check succeeded.

---

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


## Create a `Trainer`

Once the model and training components have been validated, create a `Trainer`.

```python
from danflow.training import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```


A metric can optionally be provided.


## Train for One Epoch with `train_epoch()`

The `train_epoch()` method performs one complete pass over the training `DataLoader`.

```python
loss, metric_value = trainer.train_epoch(
    train_loader,
)

print("Loss:", loss)
print("Metric:", metric_value)
```

```pycon
Loss: 0.4217
Metric: 0.8563
```

This method is useful when training needs to be controlled manually, for example when implementing a custom training loop or applying additional logic between epochs.


## Validate with `validate_epoch()`

Validation can be performed using a separate validation `DataLoader`.

```python
loss, metric_value = trainer.validate_epoch(
    validation_loader,
)

print("Validation loss:", loss)
print("Validation metric:", metric_value)
```

```pycon
Validation loss: 0.3982
Validation metric: 0.8700
```

This evaluates the model without performing parameter updates.

Using a separate validation dataset helps monitor generalization during training.


## Train for Multiple Epochs with `fit()`

For a standard training workflow, `fit()` can be used instead of manually calling `train_epoch()` for every epoch.

```python
history = trainer.fit(
    train_loader=train_loader,
    validation_loader=validation_loader,
    epochs=10,
)
```

```pycon
>>> history
<training history returned by Trainer.fit()>
```

The returned training history can be used for further analysis or visualization.

> The exact structure of `history` depends on the current `Trainer.fit()` implementation.


## Select a Learning Rate with `LearningRateSelector`

Choosing a suitable learning rate is often an important first step in hyperparameter tuning.

DanFlow provides `LearningRateSelector` for comparing several learning rates using short independent training experiments.

```python
from danflow.training.tuner import LearningRateSelector
```

Create the selector:

```python
selector = LearningRateSelector(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    learning_rates=[
        0.1,
        0.01,
        0.001,
        0.0001,
    ],
    epochs=5,
)
```

```pycon
>>> selector.learning_rates
[0.1, 0.01, 0.001, 0.0001]

>>> selector.weight_decay
0.0001

>>> selector.epochs
5
```

The search can then be performed using the training `DataLoader`.

```python
results = selector.search(
    train_loader,
)
```

```pycon
LR=0.1
Epoch 0: 100%|██████████| 1/1 [00:00<00:00, ...]
Epoch 1: 100%|██████████| 1/1 [00:00<00:00, ...]
Epoch 2: 100%|██████████| 1/1 [00:00<00:00, ...]
Epoch 3: 100%|██████████| 1/1 [00:00<00:00, ...]
Epoch 4: 100%|██████████| 1/1 [00:00<00:00, ...]

LR=0.01
...

LR=0.001
...

LR=0.0001
...

Final Results
+--------------+--------+--------+
| Learning Rate| Metric | Loss   |
+--------------+--------+--------+
| 0.1          | 0.5000 | 0.6931 |
| 0.01         | 0.8200 | 0.4028 |
| 0.001        | 0.8700 | 0.3184 |
| 0.0001       | 0.7600 | 0.5217 |
+--------------+--------+--------+

Best learning rate: 0.001 (Final loss: 0.3184)
```

Each experiment starts from an independent copy of the original model, ensuring that every learning-rate comparison begins from the same initial weights.

The returned results contain:

```python
[
    {
        "learning_rate": ...,
        "loss": ...,
        "metric": ...,
    },
]
```

```pycon
>>> results
[
    {
        'learning_rate': 0.1,
        'loss': 0.6931,
        'metric': 0.5000
    },
    {
        'learning_rate': 0.01,
        'loss': 0.4028,
        'metric': 0.8200
    },
    {
        'learning_rate': 0.001,
        'loss': 0.3184,
        'metric': 0.8700
    },
    {
        'learning_rate': 0.0001,
        'loss': 0.5217,
        'metric': 0.7600
    }
]
```

### Custom Learning Rates

A different set of learning rates can be provided:

```python
selector = LearningRateSelector(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    learning_rates=[
        0.005,
        0.001,
        0.0005,
    ],
    epochs=5,
)
```

```pycon
>>> selector.learning_rates
[0.005, 0.001, 0.0005]
```

The default learning-rate candidates are:

```text
0.1
0.01
0.001
0.0001
```


## Perform a Small Grid Search with `SmallGrid`

After identifying a promising learning-rate range, `SmallGrid` can be used to evaluate combinations of learning rate and weight decay.

```python
from danflow.training.tuner import SmallGrid
```

Create a grid-search object:

```python
grid = SmallGrid(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=metric,
    learning_rates=[
        0.01,
        0.001,
        0.0001,
    ],
    weight_decays=[
        0.0,
        1e-4,
        1e-5,
        1e-6,
    ],
    epochs=5,
)
```

```pycon
>>> grid.learning_rates
[0.01, 0.001, 0.0001]

>>> grid.weight_decays
[0.0, 0.0001, 1e-05, 1e-06]

>>> grid.epochs
5
```

Then run the search:

```python
results = grid.search(
    train_loader,
)
```

```pycon
LR=0.01 | WD=0.0
Epoch 0: 100%|██████████| 1/1 [00:00<00:00, ...]
...

LR=0.01 | WD=0.0001
...

LR=0.001 | WD=0.0
...

Final Results:
+--------------+--------------+--------+--------+
| Learning Rate| Weight Decay | Metric | Loss   |
+--------------+--------------+--------+--------+
| 0.01         | 0.0          | 0.8200 | 0.4051 |
| 0.01         | 0.0001       | 0.8300 | 0.3924 |
| 0.001        | 0.0          | 0.8700 | 0.3189 |
| 0.001        | 0.0001       | 0.8800 | 0.3017 |
| 0.0001       | 0.0          | 0.7600 | 0.5201 |
| ...          | ...          | ...    | ...    |
+--------------+--------------+--------+--------+

Best configuration: Learning Rate=0.001, Weight Decay=0.0001 (Final loss: 0.3017)
```

For every combination, `SmallGrid`:

1. Creates an independent copy of the original model.
2. Creates a new optimizer with the selected learning rate and weight decay.
3. Trains the copied model for the requested number of epochs.
4. Records the final loss and metric.

The returned result contains:

```python
[
    {
        "learning_rate": ...,
        "weight_decay": ...,
        "loss": ...,
        "metric": ...,
    },
]
```

```pycon
>>> results[0]
{
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'loss': 0.4051,
    'metric': 0.8200
}
```

A summary table is printed after the search, including:

* Learning rate.
* Weight decay.
* Metric.
* Loss.

The configuration with the lowest final loss is reported as the best configuration.


### Using a Single Learning Rate

`SmallGrid` can also test a single learning rate against multiple weight-decay values:

```python
grid = SmallGrid(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=metric,
    learning_rate=0.001,
    weight_decays=[
        0.0,
        1e-4,
        1e-5,
    ],
    epochs=5,
)
```

```pycon
>>> grid.learning_rates
[0.001]

>>> grid.weight_decays
[0.0, 0.0001, 1e-05]
```

At least one of `learning_rates` or `learning_rate` must be provided.


## Complete Training Workflow

A typical DanFlow training workflow can combine model validation, hyperparameter selection, and final training.

### Prepare the Data

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
)
```

```pycon
>>> len(train_loader)
25

>>> len(validation_loader)
7
```

### Define the Model

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)
```

```pycon
>>> model
Sequential(
  (0): Linear(in_features=10, out_features=32, bias=True)
  (1): ReLU()
  (2): Linear(in_features=32, out_features=2, bias=True)
)
```

### Define the Loss and Optimizer

```python
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)
```

```pycon
>>> optimizer.param_groups[0]["lr"]
0.001
```

### Validate the Model

```python
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
    num_samples=200,
    epochs=20,
)
```

```pycon
ForwardCheckResult(
    num_batches=5,
    average_loss=0.7183,
    input_shape=(32, 10),
    target_shape=(32,),
    output_shape=(32, 2)
)

BackwardCheckResult(
    initial_loss=0.7183,
    final_loss=0.0841,
    final_metric=None,
    epochs_trained=20,
    target_loss=None,
    target_metric=None,
    success=None,
    automatic_extension_used=False
)
```

### Search for a Suitable Learning Rate

```python
selector = LearningRateSelector(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    learning_rates=[
        0.01,
        0.001,
        0.0001,
    ],
    epochs=5,
)

lr_results = selector.search(
    train_loader,
)
```

```pycon
Final Results
+--------------+--------+--------+
| Learning Rate| Metric | Loss   |
+--------------+--------+--------+
| 0.01         | 0.8300 | 0.3864 |
| 0.001        | 0.8700 | 0.3197 |
| 0.0001       | 0.7600 | 0.5178 |
+--------------+--------+--------+

Best learning rate: 0.001 (Final loss: 0.3197)
```

### Run a Small Grid Search

```python
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
        1e-5,
    ],
    epochs=5,
)

grid_results = grid.search(
    train_loader,
)
```

```pycon
Final Results:
+--------------+--------------+--------+--------+
| Learning Rate| Weight Decay | Metric | Loss   |
+--------------+--------------+--------+--------+
| 0.01         | 0.0          | 0.8300 | 0.3869 |
| 0.01         | 0.0001       | 0.8400 | 0.3742 |
| 0.01         | 1e-05        | 0.8300 | 0.3811 |
| 0.001        | 0.0          | 0.8700 | 0.3197 |
| 0.001        | 0.0001       | 0.8800 | 0.3015 |
| 0.001        | 1e-05        | 0.8700 | 0.3148 |
+--------------+--------------+--------+--------+

Best configuration: Learning Rate=0.001, Weight Decay=0.0001 (Final loss: 0.3015)
```

### Create the Final Trainer

After selecting suitable hyperparameters, create the optimizer and trainer for the final model.

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=metric,
)
```

```pycon
>>> optimizer.param_groups[0]["lr"]
0.001

>>> optimizer.param_groups[0]["weight_decay"]
0.0001

>>> trainer
<danflow.training.trainer.Trainer object at 0x...>
```

### Train the Final Model

```python
history = trainer.fit(
    train_loader=train_loader,
    validation_loader=validation_loader,
    epochs=20,
)
```

```pycon
>>> history
<training history returned by Trainer.fit()>
```

This workflow separates the stages of model validation, hyperparameter exploration, and final training, making the training process easier to inspect and reproduce.