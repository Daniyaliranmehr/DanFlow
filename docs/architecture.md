# DanFlow Architecture

DanFlow is a lightweight, PyTorch-oriented library that provides utilities around the model development workflow.

The library does not replace PyTorch's core abstractions. Models, datasets, data loaders, optimizers, and the underlying tensor operations remain PyTorch components. DanFlow adds higher-level utilities for data preparation, loss functions, training orchestration, model validation, lightweight hyperparameter search, evaluation, checkpointing, and visualization.

This page describes the architectural boundaries between those components and how they work together.

## Architectural Overview

At a high level, DanFlow is organized around several complementary responsibilities:

```text
DanFlow
    Data
        File and tabular data preparation

    Losses
        Loss-function utilities

    Training
        Trainer
        Evaluator
        ModelChecker
        LearningRateSelector
        SmallGrid

    Visualization
        Data visualization
        Training-history visualization
```

The package is intentionally compositional. A user can keep using standard PyTorch objects and introduce DanFlow only where a higher-level utility is useful.


DanFlow then provides the orchestration layer around those objects:

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```

This design keeps the library close to the PyTorch programming model instead of introducing a separate model or data abstraction.

## Package Structure

The current package is divided into the following functional areas:

```text
danflow/
    data/
        io.py

    losses/
        loss.py

    training/
        trainer.py
        checker.py
        tuner.py

    visualization/
        data.py
        training.py
```

The package root also exposes public functionality through re-exports, allowing users to access important objects without depending on every internal module layout. The exact public import paths should be taken from the corresponding API documentation.

## Module Responsibilities

### `danflow.data`

The data package contains utilities for preparing and loading data.

The current I/O implementation is centered on `danflow.data.io`, which provides operations such as:

* extracting compressed datasets,
* converting delimited text data to CSV,
* loading CSV files into pandas `DataFrame` objects.

The data package does not define a replacement for `torch.utils.data.Dataset` or `DataLoader`.

Instead, its responsibility ends at preparing usable tabular or file-based data. The PyTorch data pipeline remains the user's responsibility.

This separation keeps file handling independent from model training.

See the [Data Guide](guides/data.md) and [Data API](api/data.md).

### `danflow.losses`

The losses package contains reusable loss-function utilities.

Loss functions remain compatible with the normal PyTorch training model: the model produces outputs, the loss function compares those outputs with targets, and the resulting scalar loss participates in backpropagation.

A user can therefore choose between a standard PyTorch loss and a DanFlow-provided loss without changing the surrounding training architecture.

See the [Losses API](api/losses.md) and the [Custom Loss Example](examples/custom_loss.md).

### `danflow.training`

The training package contains the main model-development utilities.

It is the central part of the current DanFlow architecture.

```text
training/
    trainer.py
        Trainer
        Evaluator
        AverageMeter

    checker.py
        ModelChecker
        ForwardCheckResult
        BackwardCheckResult

    tuner.py
        LearningRateSelector
        SmallGrid
```

The three modules serve different purposes while sharing the same PyTorch-based training primitives.

## Trainer: Training Orchestration

`Trainer` is the primary training engine in DanFlow.

It receives the model, optimizer, loss function, and optionally a metric, then provides methods for:

* training one epoch,
* validating one epoch,
* running a multi-epoch training process,
* recording training history,
* tracking validation results,
* saving the best checkpoint.

Conceptually, `Trainer` sits between the user's PyTorch objects and the higher-level training workflow.

```text
User-defined PyTorch components
    Model
    Optimizer
    Loss
    Metric
    DataLoader

            Trainer

Training state and results
    Epoch execution
    Validation
    History
    Best-checkpoint metadata
```

`Trainer` does not create the model architecture. The model remains completely user-defined.

This is an important architectural boundary: DanFlow controls the training procedure, while PyTorch controls the neural-network implementation.

See the [Training Guide](guides/training.md) and [Trainer API](api/trainer.md).

## ModelChecker: Pre-training Validation

`ModelChecker` is designed to detect common model and training incompatibilities before committing to a full training run.

It provides two complementary checks.

### Forward-path checking

`forward_check()` verifies that the model can successfully process batches from a `DataLoader` and that the resulting outputs are compatible with the targets and loss function.

It can also verify an expected output dimension and reports the observed input, target, and output shapes.

### Backward-path checking

`backward_check()` performs a small-subset training experiment to test whether the model is capable of learning and overfitting a limited amount of training data.

This is a diagnostic operation rather than the final training stage.

An important consequence of the current implementation is that backward checking trains the supplied model and optimizer. After using it for validation, the model should therefore be rebuilt or otherwise reset before the final training configuration is created.

`ModelChecker` depends on the training machinery provided by `Trainer`, rather than implementing a completely independent training engine.


The result dataclasses returned by the checker are implementation details of the checking process and are intentionally separate from the main training history.

See the [Training Guide](guides/training.md) and [Checker API](api/checker.md).

## LearningRateSelector and SmallGrid

The tuner module provides lightweight hyperparameter-search utilities.

### `LearningRateSelector`

`LearningRateSelector` compares several candidate learning rates using short training experiments.

Its purpose is intentionally narrow: it provides a convenient way to inspect learning-rate candidates without requiring the user to build a separate experiment loop manually.

### `SmallGrid`

`SmallGrid` extends this idea to a small Cartesian search over learning rate and weight decay.

Both utilities use the same underlying training abstractions rather than defining a separate optimization framework.

```text
Tuning configuration
    Candidate hyperparameters
            |
            v
    Independent training runs
            |
            v
        Trainer
            |
            v
    Loss / Metric results
```

The tuner should therefore be understood as an orchestration layer around training, not as an alternative training engine.

See the [Training Guide](guides/training.md) and [Tuner API](api/tuner.md).

## Evaluator: Test-time Evaluation

`Evaluator` provides the test-time evaluation boundary.

Conceptually, it is separate from training:

```text
Training
    Trainer
        |
        v
    Trained model

Evaluation
    Evaluator
        |
        v
    Test inputs and targets
        |
        v
    Loss / Metric result
```

The current implementation defines `Evaluator` in `danflow.training.trainer`, even though evaluation has a separate conceptual responsibility and a separate API documentation page. This is an implementation-level organization detail, not a requirement that users treat training and evaluation as the same operation.

`Evaluator.test()` operates on test tensors rather than training or validation `DataLoader` objects. It returns a dictionary containing the evaluation metric and loss when configured.

The evaluation stage is intentionally isolated from model fitting. It does not retrain the model.

See the [Evaluation Guide](guides/evaluation.md) and [Evaluator API](api/evaluator.md).

## danflow.visualization

The visualization package is divided into two responsibilities:

```text
visualization/
    data.py
        DataFrame-oriented plots

    training.py
        Training-history plots
```

### Data visualization

`danflow.visualization.data` operates primarily on pandas `DataFrame` objects and provides plots such as:

* histograms,
* grouped histograms,
* boxplots,
* grouped boxplots,
* correlation heatmaps.

These functions are independent of model training.

### Training visualization

`danflow.visualization.training` consumes training-history data produced by the training process and provides:

* loss history plots,
* metric history plots,
* combined training-history plots.

The visualization layer therefore depends on the structure of the produced data, rather than owning the training process itself.

This creates a useful architectural separation:

```text
Training
    produces history

Visualization
    consumes history
```

The visualization functions do not need to know how the model was implemented or how the optimizer performed its parameter updates.

See the [Visualization Guide](guides/visualization.md), [Visualization Example](examples/visualization.md), and [Visualization API](api/visualization.md).

## Core Dependency Relationships

The current implementation can be understood through the following dependency relationships:

| Component                | Primary input                              | Main responsibility          | Relationship                     |
| ------------------------ | ------------------------------------------ | ---------------------------- | -------------------------------- |
| `data`                   | Files / tabular data                       | Data preparation and loading | Independent utility layer        |
| `losses`                 | Model outputs and targets                  | Loss calculation             | PyTorch-compatible utility layer |
| `Trainer`                | Model, optimizer, loss, metric, loaders    | Training orchestration       | Core training engine             |
| `ModelChecker`           | Model, optimizer, loss, data               | Forward/backward validation  | Uses training machinery          |
| `LearningRateSelector`   | Model, optimizer class, loss, candidates   | Learning-rate search         | Uses training machinery          |
| `SmallGrid`              | Model, optimizer class, loss, search space | Small hyperparameter search  | Uses training machinery          |
| `Evaluator`              | Model, loss, optional metric, test tensors | Final evaluation             | Test-time evaluation boundary    |
| `visualization.data`     | pandas `DataFrame`                         | Data plots                   | Independent from training        |
| `visualization.training` | Training history                           | Training plots               | Consumes training results        |

The most important dependency direction is that higher-level training utilities build on the same training primitives rather than maintaining unrelated implementations.

## Data and State Ownership

A useful way to understand DanFlow is to identify which component owns each piece of state.

| State                        | Owner                           |
| ---------------------------- | ------------------------------- |
| Model parameters             | PyTorch `nn.Module`             |
| Optimizer state              | PyTorch optimizer               |
| Metric state during training | User-provided metric object     |
| Training history             | `Trainer.fit()` result          |
| Best-checkpoint state        | `Trainer.fit()` checkpoint      |
| Backward-check state         | `ModelChecker`                  |
| Test evaluation result       | `Evaluator.test()` return value |
| Plot configuration           | Visualization function call     |

DanFlow generally avoids hiding these objects behind a large global state manager.

The user creates the principal PyTorch objects and passes them into the appropriate DanFlow component.

## Metric Interfaces

Metrics are an important architectural boundary because DanFlow currently uses two different metric protocols.

Training-oriented components such as `Trainer`, `ModelChecker`, `LearningRateSelector`, and `SmallGrid` expect a stateful metric object with operations equivalent to:

```python
metric.reset()
metric.update(outputs, targets)
metric.compute()
```

`Evaluator`, by contrast, accepts a callable metric that receives outputs and targets directly.

```text
Training metric
    reset()
    update(...)
    compute()

Evaluation metric
    metric(outputs, targets)
```

This distinction matters when moving a metric object from the training workflow into the evaluation workflow.

The complete metric contract is documented in the [Metrics Guide](guides/metrics.md).

## Training Lifecycle

The major components form a coherent model-development lifecycle.

```text
Data preparation
    danflow.data

Model definition
    PyTorch

Loss + optimizer + metric
    PyTorch / DanFlow

Pre-training validation
    ModelChecker

Hyperparameter exploration
    LearningRateSelector
    SmallGrid

Final training
    Trainer

Checkpoint
    Trainer.fit()

Test evaluation
    Evaluator

Result inspection
    visualization.training
```

This lifecycle is intentionally not implemented as a single monolithic pipeline.

Each stage can be used independently.


## Checkpoint Architecture

Checkpoint management currently belongs to `Trainer` rather than to a separate checkpoint manager.

When best-checkpoint saving is enabled, `Trainer.fit()` writes a checkpoint containing the model state, optimizer state, epoch information, and best validation-loss metadata.

The checkpoint therefore captures training state needed for model restoration, while the training history remains a separate in-memory result.

```text
Trainer.fit()
    |
    +-- training history
    |
    +-- best checkpoint
            |
            +-- model state
            +-- optimizer state
            +-- epoch
            +-- best validation loss
```

The current implementation does not provide a separate automatic training-resume abstraction. Restoring a checkpoint and continuing training is therefore a manual reconstruction process.

See the [Checkpoints Guide](guides/checkpoints.md).

## Visualization Boundary

Visualization is deliberately downstream from data preparation and training.

The training engine does not need to know how its history will eventually be plotted.

Similarly, the plotting utilities do not need access to the live model or optimizer.



That boundary also means visualization can be used after training has finished, including in a separate analysis process, provided the expected history structure is available.

## Architectural Principles

The current implementation follows several consistent principles.

### PyTorch as the foundation

DanFlow uses PyTorch's established abstractions instead of replacing them.

Models are `nn.Module` objects, data loading uses PyTorch data utilities, optimizers are standard PyTorch optimizers, and loss functions operate on tensors.

### Composition over abstraction

The library provides utilities that compose with existing PyTorch code.

A user can adopt one DanFlow component without adopting an entirely different application architecture.

### Separation of workflow stages

Data preparation, training, validation, evaluation, tuning, and visualization are represented as distinct responsibilities.

This keeps each module focused on a specific stage of model development.

### Explicit state

Important objects such as models, optimizers, metrics, histories, and checkpoints remain visible to the user.

DanFlow does not attempt to hide the complete training state inside a single opaque experiment object.

### Lightweight orchestration

DanFlow focuses on convenience and workflow orchestration rather than attempting to become a full machine-learning platform.

The current architecture does not include a model registry, experiment-tracking backend, callback framework, or separate checkpoint-management subsystem.

Those responsibilities remain outside the current package surface.

## Extension Model

The architecture is designed so that users can provide their own PyTorch components.

A custom project can supply:

```text
Custom model
    +
Custom loss
    +
PyTorch optimizer
    +
Compatible metric
    +
PyTorch Dataset / DataLoader
```

and still use DanFlow's training and evaluation utilities around those components.

This means extending a project does not require extending DanFlow itself unless a new workflow abstraction is needed.

For example, a custom neural-network architecture belongs in the user's project, while a reusable training utility that solves a library-wide problem may belong in DanFlow.

## Current Implementation Notes

Several responsibilities are conceptually separate even when their implementation is currently co-located.

`Trainer`, `Evaluator`, and `AverageMeter` are currently implemented in `danflow.training.trainer`, while `Evaluator` has a separate conceptual and documentation role.

Similarly, the visualization API groups two modules into one documentation area because `visualization.data` and `visualization.training` represent closely related presentation functionality even though they operate on different inputs.

These implementation details are useful when navigating the source tree, but they should not be interpreted as requiring users to depend on internal module organization.

## What DanFlow Owns

DanFlow currently owns the workflow utilities around the PyTorch model-development process:

```text
Data preparation
Loss utilities
Training orchestration
Model validation
Lightweight hyperparameter search
Test evaluation
Checkpoint saving
Visualization
```

## What PyTorch Owns

The underlying machine-learning primitives remain PyTorch responsibilities:

```text
Tensor operations
Neural-network modules
Dataset / DataLoader abstractions
Optimizers
Automatic differentiation
Parameter updates
Device execution
```

This boundary is intentional.

DanFlow adds workflow-level structure without replacing the underlying deep-learning framework.

## Related Documentation

For a practical introduction, see the [Quick Start](getting_started/quickstart.md).

For workflow-level explanations, see the [Training Guide](guides/training.md), [Evaluation Guide](guides/evaluation.md), and [Visualization Guide](guides/visualization.md).

For exact API behavior, parameters, and return values, use the [API Reference](api/).

For a complete workflow showing the components together, see the [End-to-End Example](examples/end_to_end.md).