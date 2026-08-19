# Losses

Loss functions for regression tasks.

## adaptive_loss()

Computes an adaptive robust loss for regression tasks with multiple formulations controlled by the `alpha` parameter.

### Parameters

#### `outputs` : `torch.Tensor`

Predicted values produced by the model.

#### `targets` : `torch.Tensor`

Ground-truth target values.

#### `c` : `float`, default=`1.0`

Scale parameter used to normalize the prediction error.

#### `alpha` : `float`, default=`1.0`

Controls the formulation of the adaptive loss.

Special values include:

- `2` for the quadratic formulation.
- `0` for the Cauchy formulation.
- `-torch.inf` for the Welsch formulation.

Other values use the general formulation implemented by the function.

### Returns

`torch.Tensor`

Mean loss computed over all elements.

### Example

```pycon
>>> import torch
>>> from danflow.losses.loss import adaptive_loss

>>> outputs = torch.tensor([1.2, 2.4, 3.1])
>>> targets = torch.tensor([1.0, 2.0, 3.0])

>>> loss = adaptive_loss(
...     outputs=outputs,
...     targets=targets,
...     c=1.0,
...     alpha=2,
... )

>>> loss
tensor(0.0700)
```

## log_cosh_loss()

Computes the Log-Cosh loss for regression tasks.

### Parameters

#### `utputs` : `torch.Tensor`

Predicted values produced by the model.

#### `targets` : `torch.Tensor`

Ground-truth target values.

### Returns

`torch.Tensor`

Mean Log-Cosh loss computed over all elements.

### Example

```pycon
>>> import torch
>>> from danflow.losses.loss import log_cosh_loss

>>> outputs = torch.tensor([1.2, 2.4, 3.1])
>>> targets = torch.tensor([1.0, 2.0, 3.0])

>>> loss = log_cosh_loss(
...     outputs=outputs,
...     targets=targets,
... )

>>> loss
tensor(0.0350)
```