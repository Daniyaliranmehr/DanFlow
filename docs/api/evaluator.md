# Evaluator

Evaluates a trained PyTorch model on test data.

The `Evaluator` class provides a simple interface for running inference on a test dataset and computing optional evaluation metrics and loss.

### Example

```pycon
>>> from danflow.training import Evaluator

>>> evaluator = Evaluator(
...     model=model,
...     loss_fn=loss_fn,
...     metric=metric,
... )
```

## test()

Runs model evaluation on test data.

The model is switched to evaluation mode and inference is performed without gradient computation. The model's original training state is restored after evaluation.

### Parameters

#### `x_test` : `torch.Tensor`

Input test features.

#### `y_test` : `torch.Tensor`

Ground-truth target values.

### Returns

`dict[str, float]`

Dictionary containing the evaluation results.

Possible keys include:

- `"Loss"`
- `"Metric"`

### Example

```pycon
>>> results = evaluator.test(
...     x_test=x_test,
...     y_test=y_test,
... )

>>> results
{'Loss': 0.2458, 'Metric': 0.9375}
```