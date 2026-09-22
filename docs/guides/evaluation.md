# Evaluating Models

## Introduction

Model training does not end when the training loss becomes small or the validation metric improves.

After the training workflow is complete, the model should be evaluated on data that was not used to update its parameters and was not repeatedly used to make training decisions.

DanFlow provides the `Evaluator` class for running this final evaluation.

This guide explains how to organize the evaluation stage, how to prepare the test data, how to load a trained checkpoint, how to choose a metric, and how to interpret the resulting loss and metric values.

A typical evaluation workflow can be organized as:

```text
Trained model
    Load the intended checkpoint

Test data
    Prepare test inputs
    Prepare test targets

Evaluation
    Select loss
    Select metric
    Run Evaluator.test()

Interpretation
    Test loss
    Test metric
    Compare with training and validation
```

The central principle is that the test set should remain separate from the model-development process until the final evaluation stage.


## Evaluation vs Training

Training and evaluation answer different questions.

During training, the model parameters are updated using the training data.

During validation, the current model is measured on data that is not used for those parameter updates. Validation results can influence decisions such as:

* selecting a learning rate,
* choosing hyperparameters,
* deciding when training should stop,
* selecting the checkpoint to keep.

The test set should have a different role.

It should provide a final measurement of the selected model on data that was not used to make those development decisions.

A useful separation is:

```text
Training set
    Used to update model parameters

Validation set
    Used during model development

Test set
    Used for final evaluation
```

### Do Not Tune on the Test Set

A common mistake is checking test performance repeatedly and then changing the model based on those results.

For example:

```text
Train model
    Evaluate test set

Change learning rate
    Evaluate test set

Change architecture
    Evaluate test set

Choose the configuration with the best test result
```

In this situation, the test set has become part of the model-selection process.

Keep the test set untouched until the final evaluation stage whenever possible.


## Preparing Test Data

Before using `Evaluator`, the test inputs and targets should already be prepared.

The current `Evaluator.test()` interface accepts two tensors:

```python id="5l8h6d"
results = evaluator.test(
    x_test,
    y_test,
)
```

The inputs and targets should correspond sample by sample:

```text
x_test[i]
    corresponds to
y_test[i]
```

The test tensors should use the same input representation expected by the trained model.

For example, if the model was trained on normalized floating-point features, the test features should be transformed using the same preprocessing logic.

Do not fit a new scaler or other preprocessing transform on the test set.

### Keep Test Preprocessing Consistent

Suppose the training workflow used a preprocessing transformation:

```text
Raw training data
    Fit preprocessing
    Transform training data

Validation data
    Transform using the same fitted preprocessing

Test data
    Transform using the same fitted preprocessing
```

The test set should not be treated as an independent dataset from which preprocessing parameters are learned.

The goal is to evaluate the trained model under the same input representation used during development.


## Loading the Trained Model

The evaluation stage should use the model state that you actually intend to evaluate.

If the model was trained with:

```python id="nquf32"
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=20,
    save_best=True,
    checkpoint_path="artifacts/best_model.pth",
)
```

the resulting checkpoint is a dictionary containing:

```text id="o8rt3f"
model_state_dict
optimizer_state_dict
epoch
best_valid_loss
```

The model architecture must be recreated before loading the saved parameters.

For example:

```python id="1r9os7"
def build_model():
    return nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )

model = build_model()
```

Then load the model state:

```python id="z2dh0f"
checkpoint = torch.load(
    "artifacts/best_model.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
)
```

### Recreate the Same Architecture

The architecture used during evaluation must match the architecture used when the checkpoint was created.

For example, if the trained model contains:

```text
Input
    Linear(4, 16)

Hidden
    ReLU

Output
    Linear(16, 2)
```

the evaluation model must use the same parameter structure.

Changing the architecture before loading the checkpoint can result in incompatible parameter shapes or missing/unexpected parameters.

### You Do Not Need the Optimizer to Calculate Test Metrics

For final evaluation, the model parameters are needed.

The optimizer state stored in the checkpoint is useful when continuing training, but it is not required simply to calculate test loss and test metrics.

The evaluation workflow can therefore load:

```python id="d3o5vh"
checkpoint["model_state_dict"]
```

into a fresh model instance.


## Choosing the Evaluation Loss

The evaluation loss should normally match the loss used during model development.

For example, if a classification model was trained using:

```python id="2u3l4t"
loss_fn = nn.CrossEntropyLoss()
```

the same loss can be used for final evaluation:

```python id="f8x8pc"
evaluator = Evaluator(
    model=model,
    loss_fn=loss_fn,
    metric=accuracy,
)
```

Using the same loss makes it easier to compare training, validation, and test behavior.

A different loss can also be useful when the project needs to report a separate evaluation criterion, but the distinction should be intentional rather than accidental.


## Choosing an Evaluation Metric

The metric should reflect the question you actually care about.

For a classification example, an accuracy function might be:

```python id="nq3m8k"
def accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean()
```

Then pass the callable metric to `Evaluator`:

```python id="0zi7eq"
evaluator = Evaluator(
    model=model,
    loss_fn=loss_fn,
    metric=accuracy,
)
```

This is different from the metric interface used by `Trainer`, `ModelChecker`, and the tuning utilities.

For `Evaluator`, the metric is a callable:

```text id="m5h4fc"
metric(outputs, targets)
```

It does not need the stateful:

```text id="zg7q1n"
reset()
update()
compute()
```

interface used by the training components.

This distinction is important when moving a metric from the training workflow to the evaluation workflow.


## Choosing More Than One Metric

A single metric may not fully describe model behavior.

For example, accuracy can be useful for a balanced classification problem but less informative when class frequencies are highly uneven.

Before evaluating the final model, decide what performance characteristics matter for the task.

Depending on the project, these may include:

```text id="j4q7d1"
Accuracy
Precision
Recall
F1
Task-specific metrics
```

The appropriate metric depends on the model's objective and the consequences of different types of errors.

The important principle is:

> Choose metrics based on the task, not simply because a metric is commonly used.


## Running the Evaluation

Once the model, loss, metric, and test tensors are ready, create the evaluator:

```python id="z2so1f"
from danflow.training import Evaluator

evaluator = Evaluator(
    model=model,
    loss_fn=loss_fn,
    metric=accuracy,
)
```

Then evaluate the test tensors:

```python id="0xyk8k"
results = evaluator.test(
    x_test,
    y_test,
)
```

The method expects the test inputs and targets directly.

It does not expect a `DataLoader`:

```python id="6bfgf7"
evaluator.test(
    test_loader,
)
```

is not the correct interface.

Use:

```python id="dcbjch"
evaluator.test(
    x_test,
    y_test,
)
```

instead.

### Understand the Returned Result

`Evaluator.test()` returns a dictionary containing the metric and loss:

```python id="76lpgu"
results["Metric"]
results["Loss"]
```

The result structure is:

```text id="teqjwf"
{
    "Metric": ...,
    "Loss": ...
}
```

Do not unpack the dictionary as though the method returned a tuple:

```python id="vlzhm0"
loss, metric = evaluator.test(
    x_test,
    y_test,
)
```

A dictionary is returned, so access the values by their keys.


## Interpreting Test Loss

Test loss measures how well the model's predictions agree with the targets according to the selected loss function.

The absolute value of a loss is only meaningful in the context of the loss definition and task.

For example, a cross-entropy value should not be interpreted in exactly the same way as an MSE value.

The most useful comparison is usually:

```text
Training loss
Validation loss
Test loss
```

Look for consistency across these measurements rather than relying on the test-loss value alone.

### Test Loss Higher Than Validation Loss

A higher test loss than validation loss is not automatically a problem.

The test set may simply be more difficult or have a somewhat different sample distribution.

However, a large difference should motivate an investigation into:

* the data split,
* distribution differences,
* preprocessing consistency,
* model selection,
* and possible overfitting.


## Interpreting the Test Metric

The test metric should be interpreted according to the metric itself.

For example, with accuracy:

```text
Test metric
    Proportion of correctly classified samples
```

For metrics where larger values indicate better performance, a larger value generally represents better performance.

For metrics where smaller values are better, the interpretation is reversed.

Do not compare numerical values from different metrics as though they were directly comparable.

A metric should always be interpreted within its definition and task context.


## Comparing Training, Validation, and Test Results

A final evaluation is more informative when viewed together with the training history.

For example:

```text id="wvd5rf"
Training
    Train loss
    Train metric

Validation
    Validation loss
    Validation metric

Test
    Test loss
    Test metric
```

The goal is not to force these values to be identical.

Instead, look for relationships between them.

### Similar Training, Validation, and Test Performance

When the three sets show broadly consistent behavior, the model may be generalizing reasonably well.

The exact acceptable difference depends on the dataset and task.

### Strong Training Performance but Weaker Validation and Test Performance

A large gap between training and unseen-data performance can indicate that the model has learned the training data more strongly than it generalizes.

This is commonly associated with overfitting, although the gap should be investigated together with the dataset and experimental setup.

### Validation and Test Behave Differently

A large difference between validation and test performance can indicate:

* different data distributions,
* a small validation or test set,
* inconsistent preprocessing,
* or instability in the model-selection process.

Do not immediately attribute every difference to the model itself.


## When Should the Test Set Be Used?

The test set should normally appear near the end of the model-development process.

A practical sequence is:

```text
Training data
    Train model

Validation data
    Select configuration
    Monitor training
    Select checkpoint

Test data
    Final evaluation
```

Once test results have been used to make decisions about the model, the test set is no longer purely a final holdout.

If repeated experimentation is necessary, consider whether the evaluation protocol needs a separate validation or cross-validation strategy.


## Common Evaluation Mistakes

### Evaluating a Training Checkpoint That Was Not Intended for Final Use

A checkpoint may represent a particular training epoch rather than the model state you actually selected.

When `save_best=True` is used, load:

```python id="pkmw7e"
checkpoint["model_state_dict"]
```

from the checkpoint produced for the selected validation criterion.

### Passing a DataLoader to `Evaluator.test()`

`Evaluator.test()` expects:

```python id="x_test"
x_test
y_test
```

as tensors.

It does not consume a `DataLoader`.

### Unpacking the Result

Incorrect:

```python id="sb8z8d"
loss, metric = evaluator.test(
    x_test,
    y_test,
)
```

Correct:

```python id="g1lcf6"
results = evaluator.test(
    x_test,
    y_test,
)

loss = results["Loss"]
metric = results["Metric"]
```

### Using the Wrong Metric Type

The evaluation metric should be a callable:

```python id="l8f8hw"
metric(outputs, targets)
```

Do not automatically reuse the stateful metric object used by the training utilities without adapting it to the evaluator's interface.

### Fitting Preprocessing on the Test Set

Do not calculate normalization, scaling, or other learned preprocessing parameters using the test set.

The test data should pass through the preprocessing pipeline established during model development.

### Using Test Results to Tune the Model

Do not change the learning rate, architecture, or hyperparameters repeatedly based on test performance.

That turns the test set into another validation set.

### Loading the Entire Checkpoint as a State Dictionary

A checkpoint produced by `Trainer.fit(save_best=True)` is a dictionary containing multiple fields.

Use:

```python id="uy2n8v"
model.load_state_dict(
    checkpoint["model_state_dict"],
)
```

rather than:

```python id="dt4hf6"
model.load_state_dict(checkpoint)
```


## Recommended Evaluation Workflow

A practical evaluation workflow can be organized into these stages.

### 1. Freeze the Development Decisions

Finish the model-selection process using the training and validation data.

### 2. Identify the Model to Evaluate

Select the intended trained checkpoint.

### 3. Recreate the Model

Build the same architecture used when the checkpoint was created.

### 4. Load the Model State

Load:

```python id="x26x2t"
checkpoint["model_state_dict"]
```

into the evaluation model.

### 5. Prepare the Test Tensors

Prepare:

```python id="ar92q5"
x_test
y_test
```

using the same preprocessing pipeline used during training.

### 6. Choose the Evaluation Loss

Use the appropriate loss for the task and model outputs.

### 7. Choose the Evaluation Metric

Use a callable metric that directly evaluates model outputs against targets.

### 8. Create the Evaluator

```python id="5uz5wu"
evaluator = Evaluator(
    model=model,
    loss_fn=loss_fn,
    metric=accuracy,
)
```

### 9. Run the Test

```python id="6tsw4w"
results = evaluator.test(
    x_test,
    y_test,
)
```

### 10. Inspect the Results

```python id="bq3h7y"
test_loss = results["Loss"]
test_metric = results["Metric"]
```

### 11. Compare with Training and Validation

Interpret the test result together with the training history rather than in isolation.

### 12. Record the Final Evaluation

Keep the final model, evaluation configuration, and test results together so that the evaluation can be reproduced later.


## Evaluation and Visualization

Numerical evaluation results are only one part of model analysis.

Training history can help explain why a particular test result occurred.

For example, the training history can reveal:

```text
Training loss decreases
Validation loss decreases
Validation loss begins increasing
```

while the final test result provides a measurement of how the selected model performs on the held-out test data.

Use the training visualization tools to inspect these patterns:

```python id="2fgrcs"
from danflow.visualization import (
    plot_loss_history,
    plot_metric_history,
    plot_training_history,
)

plot_loss_history(
    history,
    name="Binary Classifier",
)

plot_metric_history(
    history,
    name="Binary Classifier",
)

plot_training_history(
    history,
    name="Binary Classifier",
)
```

Visualization is especially useful when a test result appears unexpectedly high or low because it can provide context from the model-development process.

For detailed guidance on interpreting training and data plots, see [Visualizing Data and Training](visualization.md).


## Final Perspective

Evaluation should answer a specific question:

> How well does the selected model perform on data that was not used to fit its parameters or choose its configuration?

A reliable evaluation process therefore keeps the test set separate from model development, uses the intended checkpoint, applies consistent preprocessing, selects meaningful metrics, and interprets test performance together with training and validation behavior.

DanFlow's `Evaluator` provides the final evaluation step, while the surrounding workflow determines whether that evaluation is meaningful.

For a complete executable evaluation example, see [Evaluation](../examples/evaluation.md).
