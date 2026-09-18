# Evaluation

This example demonstrates how to evaluate an already-trained PyTorch model on a separate test set using DanFlow.

The training process is intentionally omitted. The workflow starts with a trained model, its saved parameters, and a test dataset.

The typical evaluation workflow is:

1. Load the trained model.
2. Restore its learned parameters.
3. Prepare the test `DataLoader`.
4. Create an `Evaluator`.
5. Run `test()`.
6. Inspect the evaluation results.


## Start with a Trained Model

Assume that the model architecture has already been defined and trained.

For this example, use the same binary classification architecture from the training workflow.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(in_features=10, out_features=32, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=2, bias=True),
)
```


At this point, the model object only defines the architecture. The learned parameters must be restored from the trained checkpoint.


## Load the Trained Model Parameters

Assume that the trained model was saved previously.

```python
checkpoint = torch.load(
    "artifacts/model.pth",
    map_location="cpu",
)

model.load_state_dict(
    checkpoint,
)
```

```pycon
>>> checkpoint.keys()
odict_keys([
    '0.weight',
    '0.bias',
    '2.weight',
    '2.bias'
])

>>> model.training
True
```

The model now contains the parameters learned during training.

The optimizer used during training is not required when performing a standard test-set evaluation.


## Prepare the Test Dataset

The test set must be kept separate from the training data.

Assume that `test_dataset` has already been prepared using the same preprocessing and feature representation used during training.

```python
from torch.utils.data import DataLoader

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
)
```

```pycon
>>> len(test_loader)
7

>>> next(iter(test_loader))[0].shape
torch.Size([32, 10])

>>> next(iter(test_loader))[1].shape
torch.Size([32])
```

For evaluation, `shuffle=False` is generally preferred because the order of samples does not need to be randomized.


## Create the `Evaluator`

Now create an evaluator for the already-trained model.

```python
from danflow.evaluation import Evaluator

evaluator = Evaluator(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
)
```

The evaluator is now ready to process the test `DataLoader`.


## Evaluate the Model on the Test Set

Run the evaluation using `test()`.

```python
result = evaluator.test(
    test_loader,
)

print(result)
```

```pycon
TestResult(
    loss=0.2847,
    metric=0.8914
)
```

The test run evaluates the trained model on the complete test set without updating its parameters.

The returned result provides the final evaluation values produced by the evaluator.


## Inspect the Evaluation Results

The returned result can be assigned to variables when the individual values are needed.

```python
loss, metric = evaluator.test(
    test_loader,
)

print("Test loss:", loss)
print("Test metric:", metric)
```

```pycon
Test loss: 0.2847
Test metric: 0.8914
```

This gives two important pieces of information:

* **Test loss** measures the model's prediction error according to the configured loss function.
* **Test metric** measures the model's performance according to the configured metric.

A lower loss is generally preferable, while the interpretation of the metric depends on the metric itself.


## Evaluate with a Metric

For a more informative evaluation, a metric can be supplied when creating the evaluator.

For example, define a simple accuracy metric:

```python
def accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean()
```

```pycon
>>> accuracy(
...     torch.tensor([[2.0, 1.0], [0.5, 2.0]]),
...     torch.tensor([0, 1]),
... )
tensor(1.)
```

Create the evaluator with the metric:

```python
evaluator = Evaluator(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    metric=accuracy,
)
```


Run the test again:

```python
loss, accuracy_value = evaluator.test(
    test_loader,
)

print("Test loss:", loss)
print("Test accuracy:", accuracy_value)
```

```pycon
Test loss: 0.2847
Test accuracy: 0.8914
```

The metric now provides a more directly interpretable measure of test-set performance.


## Compare Test Performance with Training Performance

The test set should be used only after model development and training are complete.

For example, suppose the final training and validation results were:

```text
Training accuracy:   0.9420
Validation accuracy: 0.9135
Test accuracy:       0.8914
```

The test result is lower than the training and validation performance.

```pycon
>>> 0.9420 - 0.8914
0.05059999999999998
```

This gap can indicate that the model performs better on data it has seen during development than on completely held-out data.

The important point is that the test set provides an independent estimate of how the trained model performs on unseen data.


## A Complete Evaluation Workflow

The complete evaluation stage can be kept very small because the model has already been trained.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from danflow.evaluation import Evaluator

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

checkpoint = torch.load(
    "artifacts/model.pth",
    map_location="cpu",
)

model.load_state_dict(
    checkpoint,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
)

evaluator = Evaluator(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    metric=accuracy,
)

loss, metric_value = evaluator.test(
    test_loader,
)

print("Test loss:", loss)
print("Test metric:", metric_value)
```

```pycon
Test loss: 0.2847
Test metric: 0.8914
```

At this stage, the training workflow is complete. The reported values represent the model's performance on the held-out test set.


## Practical Interpretation

The evaluation stage should answer a simple question:

> How well does the final trained model perform on data that was not used for training?

For example:

```pycon
>>> print(f"Test loss: {loss:.4f}")
Test loss: 0.2847

>>> print(f"Test accuracy: {metric_value:.4f}")
Test accuracy: 0.8914
```

These values can then be used when reporting the final model performance, comparing different trained models, or deciding whether further model development is necessary.

The test set should remain untouched during training and hyperparameter selection so that its final evaluation remains an unbiased estimate of generalization performance.