# Training Models with DanFlow

## Introduction

The `danflow.training` module provides utilities for structuring, validating, tuning, and running PyTorch model training workflows.

A useful training workflow is more than calling a training loop. Before a model is trained for many epochs, you need to make sure that:

* the data is provided in a usable form,
* the model produces the expected outputs,
* the loss function matches the task,
* the optimizer is configured correctly,
* the metric is compatible with the training components,
* the model can actually learn from the data,
* and the important hyperparameters have reasonable values.

This guide explains how to organize these decisions with DanFlow and how to move from an initial model definition to a final training run.

A typical workflow can be organized as:

```text
Data
    Dataset
    DataLoader
    Training / validation split

Model setup
    Model
    Loss
    Optimizer
    Metric

Model validation
    Forward check
    Backward check

Training configuration
    Learning-rate search
    Hyperparameter search

Final training
    Trainer
    Validation
    Checkpoint

Training analysis
    History
    Loss
    Metric
```

The exact workflow depends on the project, but the order of these decisions helps prevent common training problems.


## Preparing Training and Validation Data

Training starts with a dataset that can provide pairs of inputs and targets.

For a PyTorch model, DanFlow works with standard PyTorch `Dataset` and `DataLoader` objects.

A simple example is:

```python
import torch

from torch.utils.data import DataLoader, TensorDataset, random_split

x = torch.randn(1000, 10)
y = (x.sum(dim=1) > 0).long()

dataset = TensorDataset(x, y)

train_dataset, valid_dataset = random_split(
    dataset,
    [800, 200],
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

The important distinction is between the dataset and the loaders:

* the `Dataset` represents the samples,
* the `DataLoader` controls how those samples are provided during training.

### Training and Validation Splits

The training and validation datasets serve different purposes.

The training set is used to update model parameters.

The validation set is used to monitor how the current model performs on data that is not used for parameter updates.

For the training loader, shuffling is usually appropriate:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
```

For validation, deterministic ordering is generally preferable:

```python
valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
)
```

The important principle is that validation data should not participate in the optimizer's parameter updates.

### Choose the Split Before Tuning

The validation set should already exist before learning-rate or hyperparameter experiments begin.

Do not repeatedly change the validation set while comparing configurations. Otherwise, differences between experiments can become difficult to interpret.

For datasets where class balance or temporal structure matters, the splitting strategy should also preserve those properties when appropriate.


## Defining the Model

Once the data pipeline is ready, define the PyTorch model.

For the example above, a small classifier might be:

```python
import torch.nn as nn

def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

model = build_model()
```

A useful practice is to put model construction inside a function.

This becomes particularly important when using DanFlow's tuning utilities because experiments should be able to start from the same initial model configuration.

For example:

```python
model = build_model()
```

and later:

```python
final_model = build_model()
```

This makes it easier to distinguish an experimental model from the final model that will be saved and evaluated.


## Choosing the Loss Function

The loss function defines the quantity that the optimizer tries to minimize.

For a two-class classification model that outputs two logits, a suitable choice is:

```python
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
```

The loss must match both the model outputs and the target representation.

For example, `CrossEntropyLoss` expects class-index targets rather than one-hot encoded targets.

The important question is not which loss is popular, but:

> What does this loss expect as input, and does that match my model and target format?

Before using a loss in a long training run, verify:

* output shape,
* target shape,
* target dtype,
* output semantics,
* and whether the loss returns a scalar.

`ModelChecker` can help identify several of these problems before full training begins.


## Choosing the Optimizer

The optimizer determines how model parameters are updated from the gradients produced by the loss.

For example:

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)
```

The optimizer should be created from the same model that will actually be trained.

When the model is rebuilt, the optimizer must also be rebuilt:

```python
model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)
```

Do not reuse an optimizer created for a different model instance.

### Separate the Initial Configuration from the Final Configuration

During experimentation, you may try several learning rates or weight-decay values.

It is usually clearer to treat those as experimental configurations rather than modifying the final training configuration repeatedly.

This becomes especially important when moving from tuning to final training.


## Choosing a Metric

A metric answers a different question from the loss.

The loss is used for optimization.

The metric is used to measure model performance in a way that is meaningful for the task.

For training-related DanFlow components, the metric follows a stateful interface.

It needs to support operations such as:

```python
metric.reset()
metric.update(outputs, targets)
metric.compute()
```

A simple metric implementation for binary classification can be written as:

```python
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
            if self.total
            else 0.0
        )
```

Then create a metric object:

```python
metric = BinaryAccuracy()
```

This distinction is important because DanFlow does not use exactly the same metric contract everywhere.

`Trainer`, `ModelChecker`, `LearningRateSelector`, and `SmallGrid` use the stateful metric interface.

`Evaluator` uses a callable metric interface instead.

Do not assume that a simple function such as:

```python
def accuracy(outputs, targets):
    ...
```

can be passed interchangeably to every DanFlow training component.


## Checking the Model Before Training

A common mistake in deep-learning projects is starting a long training run before verifying that the model can perform a valid forward and backward pass.

`ModelChecker` is intended to catch these problems early.

Create a checker with the current model, optimizer, and loss:

```python
from danflow.training.checker import ModelChecker

checker = ModelChecker(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```

### Forward Check

The forward check verifies that the model can process batches from the training loader and that the resulting loss can be computed.

```python
forward_result = checker.forward_check(
    train_loader,
    expected_output_size=2,
)
```

Use this check when you want to verify basic compatibility between:

```text
Input
    Model
    Output
    Target
    Loss
```

A forward check is especially useful after changing:

* the model architecture,
* the output layer,
* the target representation,
* or the loss function.

### Backward Check

The backward check tests whether the model can overfit a small subset of the training dataset.

```python
backward_result = checker.backward_check(
    train_dataset=train_dataset,
    num_samples=200,
    epochs=20,
)
```

The purpose is not to produce the final model.

Instead, it asks a practical diagnostic question:

> Can this model learn the training examples when the problem is made deliberately small?

A model that cannot reduce the loss on a small subset often has a problem with its architecture, loss, optimizer, data, or training setup.

### Use the Checker Before Tuning

Model checking should happen before expensive hyperparameter experiments.

There is little value in testing many learning rates when the model cannot perform a correct forward or backward pass in the first place.

### Important State Consideration

`backward_check()` performs actual training on the current model and optimizer.

Therefore, the model and optimizer used by the backward check should not automatically be treated as the untouched starting point for final training.

A clean workflow is:

```python
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
    num_samples=200,
    epochs=20,
)
```

Then rebuild the model and optimizer before starting tuning or final training:

```python
model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)
```

This keeps the subsequent experiments independent of the diagnostic training performed by the checker.


## Choosing a Learning Rate

The learning rate is one of the most important training hyperparameters.

If it is too small, optimization may progress very slowly.

If it is too large, training can become unstable or fail to converge effectively.

Rather than guessing a single value, compare a small set of candidates.

DanFlow provides `LearningRateSelector` for this purpose:

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
    epochs=5,
)
```

Then run the search:

```python
lr_results = selector.search(
    train_loader,
)
```

### Keep the Search Small

The purpose of an initial learning-rate search is not to explore every possible value.

A small set of values that spans a useful range is usually enough to identify a promising region.

For example:

```text
0.01
0.001
0.0001
```

can provide a useful first comparison.

Once a promising range is found, more focused experiments can be performed.

### Use a Metric with the Selector

The current DanFlow training tuner expects a metric during these training experiments.

Pass a stateful metric object rather than a plain metric function:

```python
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
    epochs=5,
)
```

This keeps the selector compatible with the same training metric contract used by `Trainer`.

### Understand What the Search Is Comparing

Each learning-rate experiment should be compared under otherwise consistent conditions.

Do not change the model architecture, dataset, or loss between learning-rate candidates.

The purpose of the search is to isolate the effect of the learning rate.

DanFlow's selector trains independent copies for the candidate experiments, so one candidate does not continue training from another candidate's learned weights.


## Searching Multiple Hyperparameters

Once a useful learning-rate range has been identified, the next question may be:

> How do learning rate and weight decay behave together?

For this situation, DanFlow provides `SmallGrid`.

For example:

```python
from danflow.training.tuner import SmallGrid

grid = SmallGrid(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=BinaryAccuracy(),
    learning_rates=[
        0.001,
        0.0005,
    ],
    weight_decays=[
        0.0,
        1e-4,
    ],
    epochs=5,
)
```

Then:

```python
grid_results = grid.search(
    train_loader,
)
```

### Keep the Grid Small

A grid search grows quickly as the number of parameters increases.

For example:

```text
2 learning rates
2 weight-decay values

4 experiments
```

Adding more parameters multiplies the number of experiments.

Start with a small grid that answers a specific question.

Do not search every possible hyperparameter at once.

### Use Lists for Search Values

Use:

```python
learning_rates=[0.001, 0.0005]
```

rather than passing a single floating-point value to `learning_rates`.

`SmallGrid` is designed to iterate over the supplied candidate values.

### Keep the Metric Stateful

`SmallGrid` passes its metric into `Trainer`, so the metric must support the training metric interface:

```python
reset()
update(outputs, targets)
compute()
```

A simple function is not interchangeable with this object.


## Interpreting Tuning Results

Tuning should not be treated as a search for the configuration with the most impressive single number.

Instead, ask what the experiment actually measured.

For learning-rate experiments, compare:

* final loss,
* metric,
* stability of training,
* and whether the results are consistent with the expected behavior of the task.

For grid searches, consider whether an apparent improvement is meaningful enough to justify the additional configuration.

A tuning result is a candidate configuration, not automatically the final model.

The final configuration should still be trained and evaluated in the normal workflow.


## Creating the Final Training Configuration

After model checks and tuning, create a fresh model for the final run.

For example:

```python
model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

metric = BinaryAccuracy()
```

Then create the final `Trainer`:

```python
from danflow.training import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=metric,
)
```

The important principle is that the final model should start from the intended initialization rather than from a temporary diagnostic or tuning state.


## Training One Epoch or Many Epochs

DanFlow provides both lower-level epoch methods and the higher-level `fit()` workflow.

Use `train_epoch()` when you need direct control over a single training epoch.

```python
loss, metric_value = trainer.train_epoch(
    train_loader,
)
```

Use `validate_epoch()` when you want to inspect the current model on the validation set without updating model parameters.

```python
valid_loss, valid_metric = trainer.validate_epoch(
    valid_loader,
)
```

These methods are useful when implementing custom training logic or when you need to perform additional operations between epochs.

For a normal multi-epoch training workflow, `fit()` is the more convenient interface.


## Training with `fit()`

For standard training, use:

```python
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
)
```

Notice the parameter name:

```python
valid_loader
```

not:

```python
validation_loader
```

The exact parameter names used by the DanFlow API should be followed consistently.

### When to Use `fit()`

Use `fit()` when:

* training follows a standard epoch-based workflow,
* validation should be performed regularly,
* you want a training history,
* and you want optional best-model checkpointing.

Use `train_epoch()` and `validate_epoch()` directly when the training procedure requires custom logic between individual epochs.


## Monitoring Training

The history returned by `fit()` contains the information needed to inspect training progress.

The current history structure contains:

```text
train_loss
valid_loss
train_metric
valid_metric
metric_name
best_valid_loss
best_loss_epoch
best_valid_metric
best_metric_epoch
```

You can inspect the available keys with:

```pycon
>>> sorted(history.keys())
['best_loss_epoch',
 'best_metric_epoch',
 'best_valid_loss',
 'best_valid_metric',
 'metric_name',
 'train_loss',
 'train_metric',
 'valid_loss',
 'valid_metric']
```

### Training Loss vs Validation Loss

The most useful comparison is usually between training and validation behavior.

A typical healthy pattern is:

```text
Training loss decreases
Validation loss decreases
```

A potential overfitting pattern is:

```text
Training loss continues to decrease
Validation loss stops improving or begins increasing
```

The exact interpretation depends on the task and the training setup, but divergence between training and validation behavior is an important signal to investigate.

### Training Metric vs Validation Metric

Metrics should also be examined together rather than in isolation.

For example:

```text
Training metric increases
Validation metric increases
```

suggests that improvement is occurring on both datasets.

A widening gap between the two can indicate that the model is fitting the training data more strongly than it generalizes.


## Saving the Best Checkpoint

For many training workflows, the model from the final epoch is not necessarily the model you want to keep.

DanFlow can save the best validation-loss checkpoint during `fit()`:

```python
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
    save_best=True,
    checkpoint_path="artifacts/best_model.pth",
)
```

When `save_best=True`, the checkpoint contains:

```text
model_state_dict
optimizer_state_dict
epoch
best_valid_loss
```

The checkpoint is therefore more than a raw `state_dict`.

When loading it, access the model state explicitly:

```python
checkpoint = torch.load(
    "artifacts/best_model.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
)
```

### Keep Checkpoints Separate from Source Data

A useful project structure is:

```text
project/
    data/
        raw/
        processed/
    artifacts/
        best_model.pth
    train.py
```

This keeps generated training artifacts separate from the dataset itself.


## Training Best Practices

### Validate During Development

Do not use only the training loss to decide whether the model is improving.

The validation set provides a separate signal about generalization.

### Keep Tuning Experiments Consistent

When comparing hyperparameters, keep the following stable:

* dataset split,
* model architecture,
* loss function,
* evaluation method,
* number of training epochs used for the comparison.

Otherwise, you may be comparing multiple changes at once.

### Rebuild After Diagnostic Training

`ModelChecker.backward_check()` changes model parameters.

Rebuild the model and optimizer before final training when you want the final experiment to begin from a clean initialization.

### Start with Small Searches

A small learning-rate search followed by a small grid is usually easier to understand than a large hyperparameter sweep.

The goal is to answer focused questions, not to maximize the number of experiments.

### Save the Important Model

When checkpointing is important, use:

```python
save_best=True
```

and keep the checkpoint path explicit.

This makes the training artifact reproducible and easier to use later during evaluation.


## Common Mistakes

### Using `validation_loader` with `Trainer.fit()`

Incorrect:

```python
trainer.fit(
    train_loader=train_loader,
    validation_loader=valid_loader,
    epochs=20,
)
```

Use:

```python
trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
)
```

### Passing a Plain Function as a Training Metric

A callable such as:

```python
def accuracy(outputs, targets):
    ...
```

does not satisfy the stateful metric interface expected by `Trainer` and the tuning utilities.

Use a metric object that supports:

```python
reset()
update(outputs, targets)
compute()
```

### Reusing the Model After `backward_check()`

`backward_check()` performs real optimization steps.

Do not assume that the model is still in its original initialization state afterward.

Rebuild the model and optimizer before final training.

### Using a Single Float for `SmallGrid.learning_rates`

Do not treat:

```python
learning_rates=0.001
```

as a list of candidates.

Use:

```python
learning_rates=[0.001]
```

and preferably provide multiple candidate values when performing an actual search.

### Tuning on the Test Set

The test set should be reserved for the final evaluation stage.

Do not repeatedly use test performance to select the learning rate or hyperparameters.

Otherwise, the test set is no longer an unbiased final measurement.

### Changing Several Components at Once

If the learning-rate experiment also changes:

* model architecture,
* loss,
* batch size,
* dataset split,
* and optimizer,

the result cannot be interpreted as a learning-rate comparison.

Change one experimental dimension at a time when possible.

### Training for Too Many Epochs Without Monitoring

More epochs do not automatically produce a better model.

Inspect validation behavior while training and use checkpointing when the best validation state matters.


## Recommended Training Workflow

A practical DanFlow training workflow can be organized into these stages.

### 1. Prepare the Data

Create the training and validation datasets and their loaders.

### 2. Define the Model

Create the model architecture and keep model construction reproducible.

### 3. Choose the Loss

Make sure the loss is compatible with the model outputs and target representation.

### 4. Choose the Optimizer

Create the optimizer from the current model parameters.

### 5. Define the Metric

Use a stateful metric object for training-related DanFlow components.

### 6. Check the Model

Run a forward check and a small backward check before expensive experiments.

### 7. Rebuild the Model

After backward checking, create a fresh model and optimizer for subsequent experiments.

### 8. Search the Learning Rate

Use `LearningRateSelector` with a focused set of candidate learning rates.

### 9. Tune Additional Hyperparameters

Use `SmallGrid` when a small combination search is useful.

### 10. Create the Final Trainer

Build a fresh final model, optimizer, and metric using the selected configuration.

### 11. Train and Validate

Use:

```python
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
    save_best=True,
    checkpoint_path="artifacts/best_model.pth",
)
```

### 12. Inspect the Training History

Use the recorded loss and metric histories to understand how training progressed.

### 13. Evaluate Separately

Once training is complete, load the selected checkpoint and evaluate the model on held-out test data.

The final evaluation workflow is covered in [Evaluating Models](evaluation.md).


## Training and Visualization

Training history becomes much more useful when it is visualized.

The values returned by `Trainer.fit()` are designed to work with DanFlow's training visualization utilities.

For example:

```python
from danflow.visualization import (
    plot_loss_history,
    plot_metric_history,
    plot_training_history,
)

plot_loss_history(
    history=history,
    name="Binary Classifier",
)

plot_metric_history(
    history=history,
    name="Binary Classifier",
)

plot_training_history(
    history=history,
    name="Binary Classifier",
)
```

Use these plots to inspect:

* convergence,
* divergence between training and validation,
* possible overfitting,
* and the relationship between loss and metric over time.

For detailed guidance on choosing and interpreting these plots, see [Visualizing Data and Training](visualization.md).


## Final Perspective

A reliable training workflow separates three different activities:

```text
Model validation
    Is the model configured correctly?

Hyperparameter tuning
    Which configuration should be tested further?

Final training
    Which model should be kept for evaluation?
```

Keeping these stages separate makes experiments easier to understand and reduces accidental reuse of state between diagnostic, tuning, and final training runs.

DanFlow provides utilities for each stage, but the quality of the workflow still depends on choosing appropriate data splits, losses, metrics, and experimental settings.

For a complete executable training workflow, see [Training](../examples/training.md).