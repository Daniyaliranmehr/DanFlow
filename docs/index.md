# DanFlow Documentation

Welcome to the DanFlow documentation.

DanFlow is a PyTorch-based library for data preparation, model training, evaluation, loss functions, hyperparameter tuning, model checking, and visualization.

The documentation is organized by purpose so that you can start quickly, learn a specific workflow, follow a complete example, or look up the exact API reference.

## Getting Started

Start here if you are new to DanFlow.

### Installation

Set up DanFlow and its required environment dependencies.

[Installation Guide](getting_started/installation.md)

### Quick Start

Train your first model with DanFlow using a minimal workflow.

[Quick Start](getting_started/quickstart.md)

## Guides

Guides explain how to use DanFlow concepts and features as part of a practical workflow.

### Data

Learn how to prepare datasets with DanFlow, including archive extraction, delimiter-based conversion, CSV loading, and common data-preparation issues.

[Data Guide](guides/data.md)

### Training

Learn how to structure a model training workflow with DanFlow, including model validation, training, metrics, and hyperparameter utilities.

[Training Guide](guides/training.md)

### Metrics

Understand the metric interfaces used throughout DanFlow and how to implement and use compatible metrics.

[Metrics Guide](guides/metrics.md)

### Evaluation

Learn how to evaluate an already-trained model on test data and interpret the resulting loss and metric values.

[Evaluation Guide](guides/evaluation.md)

### Visualization

Learn how to visualize tabular data and training history using DanFlow's visualization utilities.

[Visualization Guide](guides/visualization.md)

### Checkpoints

Learn how DanFlow saves, loads, and restores model and optimizer state using training checkpoints.

[Checkpoints Guide](guides/checkpoints.md)

## Examples

Examples demonstrate complete, practical workflows using DanFlow.

### Data Preparation

A complete dataset preparation workflow using archive extraction, delimiter conversion, and CSV loading.

[Data Preparation Example](examples/data_preparation.md)

### Training

A complete model training workflow combining dataset preparation, model setup, model checking, tuning, and training.

[Training Example](examples/training.md)

### Evaluation

A complete evaluation workflow for loading a trained model and evaluating it on test data.

[Evaluation Example](examples/evaluation.md)

### Visualization

A practical workflow for visualizing tabular data and training history.

[Visualization Example](examples/visualization.md)

### Custom Loss

A complete example showing how to define and use a custom loss function in a training workflow.

[Custom Loss Example](examples/custom_loss.md)

### End-to-End

A complete workflow combining data preparation, model training, validation, tuning, checkpointing, evaluation, and visualization.

[End-to-End Example](examples/end_to_end.md)

## API Reference

The API Reference documents the public DanFlow interfaces, including their parameters, return values, and behavior.

### Data

Functions for dataset extraction, delimited-file conversion, and CSV loading.

[Data API](api/data.md)

### Losses

Loss functions provided by DanFlow.

[Losses API](api/losses.md)

### Trainer

The `Trainer` API for model training and training history.

[Trainer API](api/trainer.md)

### Evaluator

The `Evaluator` API for evaluating trained models on test data.

[Evaluator API](api/evaluator.md)

### Model Checker

The `ModelChecker` API for validating model forward and backward behavior.

[Model Checker API](api/checker.md)

### Tuner

The `LearningRateSelector` and `SmallGrid` APIs for hyperparameter search.

[Tuner API](api/tuner.md)

### Visualization

Functions for data and training-history visualization.

[Visualization API](api/visualization.md)

## Choosing a Starting Point

| I want to...                       | Start with                                      |
| ---------------------------------- | ----------------------------------------------- |
| Install DanFlow                    | [Installation](getting_started/installation.md) |
| Run my first DanFlow workflow      | [Quick Start](getting_started/quickstart.md)    |
| Prepare a dataset                  | [Data Guide](guides/data.md)                    |
| Build a training workflow          | [Training Guide](guides/training.md)            |
| Create a compatible metric         | [Metrics Guide](guides/metrics.md)              |
| Evaluate a trained model           | [Evaluation Guide](guides/evaluation.md)        |
| Work with checkpoints              | [Checkpoints Guide](guides/checkpoints.md)      |
| Visualize data or training history | [Visualization Guide](guides/visualization.md)  |
| See a complete training workflow   | [Training Example](examples/training.md)        |
| See a complete evaluation workflow | [Evaluation Example](examples/evaluation.md)    |
| Use a custom loss                  | [Custom Loss Example](examples/custom_loss.md)  |
| See the complete workflow          | [End-to-End Example](examples/end_to_end.md)    |
| Check exact API behavior           | [API Reference](#api-reference)                 |

## Documentation Structure

DanFlow documentation follows four levels:

```text
Getting Started
    Installation
    Quick Start

Guides
    Learn how a specific DanFlow workflow works

Examples
    Follow a complete practical workflow

API Reference
    Look up exact public interfaces and behavior
```

Each section has a separate purpose so that concepts, practical workflows, and API reference material remain clearly separated.