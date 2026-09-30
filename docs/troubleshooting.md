# Troubleshooting

This page provides practical solutions for common problems encountered when using DanFlow.

Use the symptom that matches the problem you are seeing, then check the corresponding cause and solution.

For complete API details, see the [API Reference](api-reference.md).


## Installation and Environment

### `ModuleNotFoundError` when importing DanFlow

If `import danflow` fails because a dependency is missing, verify that the required runtime packages are installed in the active Python environment.

Common missing dependencies include:

```text
pandas
matplotlib
tqdm
rich
prettytable
```

Some workflows may also require:

```text
torchmetrics
```

Install the missing package in the same environment where DanFlow is being used.

For example:

```bash
pip install pandas matplotlib tqdm rich prettytable
```

For TorchMetrics-based workflows:

```bash
pip install torchmetrics
```

Then verify the installation:

```bash
python -c "import danflow; print('DanFlow installed successfully')"
```

If the command still fails, confirm that the shell and IDE are using the same Python environment.

See [Installation](getting_started/installation.md) for the installation workflow.


### DanFlow installs successfully, but importing it immediately fails

If installation completes but `import danflow` raises a missing-dependency error, the package metadata may not yet declare every library imported by the source code.

Check the missing package in the error message and install it explicitly.

Do not assume that a successful `pip install -e .` means every runtime dependency is available.


## Import and API Errors

### `ModuleNotFoundError: No module named 'danflow.evaluation'`

`Evaluator` is part of DanFlow's training package.

Use:

```python
from danflow.training import Evaluator
```

Do not use:

```python
from danflow.evaluation import Evaluator
```

The evaluation API is documented in [Evaluator API](api/evaluator.md).


### `TypeError: Trainer.fit() got an unexpected keyword argument 'validation_loader'`

The validation argument is named `valid_loader`.

Use:

```python
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=10,
)
```

Do not use:

```python
history = trainer.fit(
    train_loader=train_loader,
    validation_loader=validation_loader,
    epochs=10,
)
```

The same naming convention should be used consistently for validation DataLoaders throughout a project.

See [Trainer API](api/trainer.md).


### `TypeError: LearningRateSelector.__init__() got an unexpected keyword argument 'trainer_cls'`

`trainer_cls` is not part of the current `LearningRateSelector` interface covered by the approved API documentation.

Remove the argument:

```python
selector = LearningRateSelector(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    learning_rates=[0.01, 0.001, 0.0001],
    epochs=5,
)
```

See [Tuner API](api/tuner.md).


## Data and File Errors

### `FileNotFoundError` when reading or writing a file

Check that:

1. The input path actually exists.
2. The path is relative to the current working directory.
3. The output directory exists or can be created.
4. The filename and extension are correct.

Check the current working directory with:

```python
from pathlib import Path

print(Path.cwd())
```

For file-related APIs, prefer explicit paths when debugging.

See [Data API](api/data.md).


### ZIP extraction fails with an invalid archive error

When `extract_zip()` fails while opening an archive, verify that the input file is a valid ZIP archive and is not partially downloaded or corrupted.

Also confirm that the path points to the archive itself rather than to its parent directory.


### Permission error while reading or writing a file

A permission error usually means the current process cannot access the specified location.

Check whether:

* the file is open in another application,
* the directory is writable,
* the current user has access to the location,
* the output path points to a protected system directory.

Using a project-local output directory is usually easier to debug.


### `load_csv()` produces unexpected columns or missing data

Check the structure of the CSV file before loading it.

In particular, verify whether the first row is a header. A headerless CSV file can be interpreted differently from a file that explicitly contains column names.

For converted text files, verify the delimiter used by `delimited_to_csv()` before calling `load_csv()`.

For example:

```text
feature_1 feature_2 label
1.2 3.4 0
2.1 4.2 1
```

requires a space delimiter, while:

```text
feature_1|feature_2|label
1.2|3.4|0
2.1|4.2|1
```

requires:

```python
delimiter="|"
```

When debugging data conversion, inspect the generated CSV before loading it.


## Model and Tensor Shape Errors

### Model output batch size does not match target batch size

A forward check may report an error similar to:

```text
Model output batch size does not match target batch size.
outputs.shape = ...
targets.shape = ...
```

This means the first dimension of the model output is different from the target batch dimension.

Check that:

```python
outputs.shape[0] == targets.shape[0]
```

For a standard classification task:

```text
outputs: [batch_size, num_classes]
targets: [batch_size]
```

Verify the model architecture and the target construction before changing the loss function.


### `Incorrect model output size`

When using `ModelChecker.forward_check()` with `expected_output_size`, the final model dimension must match the expected value.

For example:

```python
forward_result = checker.forward_check(
    train_loader,
    expected_output_size=7,
)
```

requires the model to produce seven output values per sample.

Check the actual model output with:

```python
outputs = model(inputs)

print(outputs.shape)
```

Make sure the final model layer matches the number of classes or output values expected by the task.


### The model output is not a `torch.Tensor`

`ModelChecker.forward_check()` expects the model output to be a PyTorch tensor.

If a custom model returns another object, dictionary, tuple, or Python value, inspect the model's `forward()` method and ensure the value passed to the loss function is the intended tensor.


### Loss calculation fails during a model check

If the checker reports that the loss could not be calculated, verify all three components together:

```text
model output
target tensor
loss function
```

For classification with `CrossEntropyLoss`, for example:

```text
outputs: [batch_size, num_classes]
targets: [batch_size]
targets dtype: torch.long
```

Do not change only the loss function without first checking the shapes and dtypes of the tensors.


### Loss is non-finite

If the checker reports a non-finite loss, inspect the tensors and training configuration for:

```text
NaN
Inf
-Infinity
```

Check the model inputs first:

```python
print(torch.isnan(inputs).any())
print(torch.isinf(inputs).any())
```

Then inspect the model outputs:

```python
outputs = model(inputs)

print(torch.isnan(outputs).any())
print(torch.isinf(outputs).any())
```

If the inputs and outputs are finite, inspect the loss function and optimization settings.


## Metric Errors

### `Trainer` fails when using a metric

Metrics passed to `Trainer` are expected to follow the stateful metric interface used by the training loop.

The metric should provide:

```python
metric.reset()
metric.update(outputs, targets)
metric.compute()
```

The value returned by `compute()` must be compatible with:

```python
compute().item()
```

TorchMetrics objects are suitable for this interface.

A plain function such as:

```python
def accuracy(outputs, targets):
    ...
```

should not be assumed to satisfy the same contract.

See [Metrics Guide](guides/metrics.md).


### `target_metric` is provided but no metric was supplied

`ModelChecker.backward_check()` requires an actual metric when `target_metric` is specified.

This combination is invalid:

```python
checker.backward_check(
    train_dataset=train_dataset,
    target_metric=0.99,
)
```

Provide a metric:

```python
checker.backward_check(
    train_dataset=train_dataset,
    metric=accuracy,
    target_metric=0.99,
)
```

A metric target cannot be evaluated without a metric.


### `Evaluator` behaves differently from `Trainer` when using a metric

`Evaluator` uses a different metric contract from the training utilities.

For `Evaluator`, the metric is a callable that receives predictions and targets and returns a scalar value.

The training utilities use the stateful metric interface described above.

Do not copy a metric implementation between `Trainer` and `Evaluator` without checking the expected interface.

See [Metrics Guide](guides/metrics.md).


## Training Errors

### `Trainer.fit()` uses the wrong validation argument

The correct parameter is:

```python
valid_loader
```

A complete call is:

```python
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=10,
)
```

See [Trainer API](api/trainer.md).


### Training returns no useful metric values

If no metric is supplied to `Trainer`, metric history is empty.

For example:

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```

does not create training or validation metric values.

In that case, the history contains loss information but metric lists remain empty.

Provide a compatible stateful metric when metric tracking is required.


### Training appears not to improve

First inspect the training and validation loss separately.

Typical checks include:

```python
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=10,
)

print(history["train_loss"])
print(history["valid_loss"])
```

Then inspect:

* whether the loss decreases,
* whether training loss and validation loss behave differently,
* whether the learning rate is appropriate,
* whether the model output and targets have compatible shapes,
* whether the input data is correctly normalized or encoded.

Do not use a low validation metric alone to diagnose the problem. Verify the full training pipeline.

See [Training Guide](guides/training.md).


### `train_epoch()` works, but `fit()` does not

Run the same components explicitly:

```python
train_loss, train_metric = trainer.train_epoch(train_loader)
valid_loss, valid_metric = trainer.validate_epoch(valid_loader)
```

If these calls work, compare the arguments passed to `fit()` with the arguments used by the individual methods.

This is especially useful for detecting incorrect loader names or incorrect assumptions about metric configuration.


### Training DataLoader produces no batches

An empty DataLoader cannot provide training data.

Check:

```python
print(len(train_loader))
```

and verify the underlying dataset:

```python
print(len(train_dataset))
```

For a custom dataset, also verify that `__len__()` and `__getitem__()` return the expected values.


## ModelChecker Errors

### `backward_check()` has already been initialized

`ModelChecker.backward_check()` creates and stores an overfitting subset.

Calling `backward_check()` again on the same checker instance is not the way to continue that experiment.

Use:

```python
checker.continue_backward(
    epochs=20,
)
```

when additional training is needed on the same subset.

Start a new checker instance when a completely new overfitting experiment is intended.


### `num_samples` cannot be greater than the dataset size

The requested overfitting subset must fit inside the training dataset.

For example:

```python
checker.backward_check(
    train_dataset=train_dataset,
    num_samples=1000,
)
```

requires:

```python
len(train_dataset) >= 1000
```

Reduce `num_samples` or provide a larger dataset.


### `batch_size must be at least 1`

`batch_size` must be a positive integer.

For example:

```python
batch_size=0
```

is invalid.

When `batch_size` is omitted, DanFlow calculates a batch size automatically from the requested subset size.


### `target_loss must be non-negative`

A loss target represents a maximum acceptable loss and therefore cannot be negative.

Use:

```python
target_loss=0.01
```

rather than:

```python
target_loss=-0.01
```


### `backward_check()` appears to train longer than expected

When `epochs` is omitted and an explicit overfitting target is provided, `ModelChecker` can automatically perform an additional training phase if the target is not reached.

For predictable training duration, specify `epochs` explicitly:

```python
checker.backward_check(
    train_dataset=train_dataset,
    epochs=20,
)
```


## Checkpoint Errors

### `load_state_dict()` fails after loading a checkpoint

DanFlow's checkpoint is a dictionary containing multiple training-related values.

The model weights are stored under:

```python
checkpoint["model_state_dict"]
```

Load them with:

```python
checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
```

Do not pass the entire checkpoint dictionary directly to `load_state_dict()`.

See [Checkpoints Guide](guides/checkpoints.md).


### Checkpoint keys are not what you expected

A checkpoint created by `Trainer.fit(save_best=True)` contains information such as:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "best_valid_loss": ...,
}
```

Inspect the file before loading it:

```python
print(checkpoint.keys())
```

If a project requires additional state such as scheduler state, metric state, or random-number-generator state, that information is not automatically included in the DanFlow checkpoint format.


### Loading a checkpoint on the wrong device

When a checkpoint was created on one device and loaded on another, explicitly specify the target device when loading.

For CPU-side inspection:

```python
checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=True,
)
```

Then load the model state normally.


## Evaluation Errors

### `Evaluator.test()` reports a missing `y_test` argument

`Evaluator.test()` expects two tensors:

```python
result = evaluator.test(
    x_test,
    y_test,
)
```

Do not pass a single `DataLoader`:

```python
evaluator.test(test_loader)
```

The method operates on the supplied test tensors directly.

See [Evaluator API](api/evaluator.md) and [Evaluation Guide](guides/evaluation.md).


### `Evaluator.test()` result cannot be unpacked as `(loss, metric)`

The evaluation result is a dictionary.

Use:

```python
result = evaluator.test(
    x_test,
    y_test,
)

print(result)
```

Then access the available values by key.

For example:

```python
result["Loss"]
```

or:

```python
result["Metric"]
```

only when the corresponding component was supplied to `Evaluator`.


## Visualization Errors

### `KeyError` while plotting a DataFrame column

If a plotting function reports a missing column, verify the exact column name:

```python
print(df.columns.tolist())
```

Then pass an existing column:

```python
plot_histogram(
    df,
    column="feature_1",
)
```

Column names are case-sensitive.


### `ValueError` when using a multi-column plotting function

Functions such as:

```python
plot_multi_histograms()
plot_multi_boxplots()
```

require at least one column.

This is invalid:

```python
plot_multi_histograms(
    df,
    columns=[],
)
```

Provide a non-empty list:

```python
plot_multi_histograms(
    df,
    columns=[
        "feature_1",
        "feature_2",
    ],
)
```


### Training-history plot fails because keys are missing

Training-history visualization expects the history structure produced by `Trainer.fit()`.

At minimum, the loss history should contain:

```python
history["train_loss"]
history["valid_loss"]
```

Metric-related keys are relevant when a metric is used.

Inspect the available keys:

```python
print(history.keys())
```

Use the same history object returned by `Trainer.fit()` rather than manually reconstructing it unless there is a specific reason to do so.


### Best-loss or best-metric markers do not appear

Best-value markers depend on the corresponding information being present in the training history.

For example:

```python
plot_training_history(
    history,
    name="Demo Model",
    show_best_loss=True,
    show_best_metric=True,
)
```

requires the relevant best-epoch information to be available in `history`.

Also note that best metric tracking only makes sense when a metric was actually supplied during training.


### A visualization is saved to an unexpected location

Check the interpretation of `save_path` for the function being used.

For the data-visualization functions, a directory-like path can be used to save a file with DanFlow's generated filename.

For example:

```python
plot_histogram(
    df,
    column="feature_1",
    save_path="plots",
)
```

Training-history plotting functions use `save_path` as the output file path.

When debugging file output, use an explicit filename:

```python
save_path="plots/training_history.png"
```

See [Visualization API](api/visualization.md).


## Hyperparameter Tuning Errors

### `LearningRateSelector` fails before training starts

Check the constructor arguments against the approved tuner API.

In particular, do not pass obsolete arguments such as:

```python
trainer_cls=...
```

Use the supported optimizer class directly:

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
```


### Tuning fails because of metric handling

`LearningRateSelector` and `SmallGrid` use the training metric contract rather than the callable metric contract used by `Evaluator`.

The metric should therefore support:

```python
reset()
update(outputs, targets)
compute()
```

Check the metric first before debugging the tuner itself.

See [Metrics Guide](guides/metrics.md) and [Tuner API](api/tuner.md).


### A tuner result does not match the final model training

The tuner performs independent experiments to compare configurations.

After identifying a candidate configuration, create a fresh optimizer for the final training run rather than assuming that the optimizer state from one search experiment is the final training state.

For example:

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
)
```

Then run the final training workflow with the selected configuration.


## Debugging Checklist

When the source of an error is unclear, inspect the pipeline in this order:

```text
1. Python environment
2. DanFlow imports
3. Input file paths
4. Dataset length and sample structure
5. DataLoader batches
6. Model input shape
7. Model output shape
8. Target shape and dtype
9. Loss function compatibility
10. Metric interface
11. Training call arguments
12. Checkpoint loading
13. Evaluation inputs
14. Visualization inputs
```

For model-related problems, a useful first diagnostic is to run:

```python
forward_result = checker.forward_check(
    train_loader,
    expected_output_size=expected_output_size,
)
```

and then, when appropriate:

```python
backward_result = checker.backward_check(
    train_dataset=train_dataset,
    epochs=20,
)
```

These checks can identify shape, loss, and basic learning-path problems before a longer training run.

When an error remains unclear, inspect the exact exception message first and compare it with the corresponding API page before changing multiple parts of the pipeline at once.


## Related Documentation

* [Getting Started](getting_started/quickstart.md)
* [Data Guide](guides/data.md)
* [Training Guide](guides/training.md)
* [Metrics Guide](guides/metrics.md)
* [Evaluation Guide](guides/evaluation.md)
* [Checkpoints Guide](guides/checkpoints.md)
* [Visualization Guide](guides/visualization.md)
* [API Reference](api-reference.md)