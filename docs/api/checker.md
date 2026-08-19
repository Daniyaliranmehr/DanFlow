# Checker

Provides utilities for verifying the forward and backward paths of PyTorch models before or during training.

The `ModelChecker` class can be used to validate model input/output compatibility, verify loss calculation, and check whether a model can overfit a small training subset.

## forward_check()

Checks the model's forward path using batches from a training `DataLoader`.

The method:

- Iterates over the provided DataLoader.
- Passes several batches through the model.
- Checks input, target, and output compatibility.
- Checks the model output size when provided.
- Calculates the loss.
- Raises a helpful error if the loss cannot be calculated.
- Computes the average initial loss.


### Parameters

#### `train_loader` : `torch.utils.data.DataLoader`

DataLoader containing the input and target tensors used for the forward-path check.

#### `expected_output_size` : `int | None`, default=`None`

Expected size of the model's final output dimension.

If provided, the last dimension of the model output must match this value.

For example, for a 7-class classification model:

```pycon
>>> expected_output_size=7
```

#### `num_batches` : `int, default=5`

Maximum number of batches used for the forward check and average initial loss calculation.

The value must be at least `1`.

### Returns

`ForwardCheckResult`

Information about the verified forward path.

### Example

```pycon
>>> import torch
>>> from torch import nn
>>> from torch.utils.data import DataLoader, TensorDataset
>>> from danflow.training import ModelChecker

>>> x = torch.randn(20, 4)
>>> y = torch.randint(0, 3, (20,))

>>> dataset = TensorDataset(x, y)
>>> train_loader = DataLoader(dataset, batch_size=4)

>>> model = nn.Sequential(
...     nn.Linear(4, 3)
... )

>>> optimizer = torch.optim.Adam(model.parameters())
>>> loss_fn = nn.CrossEntropyLoss()

>>> checker = ModelChecker(
...     model=model,
...     optimizer=optimizer,
...     loss_fn=loss_fn,
... )

>>> result = checker.forward_check(
...     train_loader=train_loader,
...     expected_output_size=3,
...     num_batches=2,
... )
Input shape:  torch.Size([4, 4])
Target shape: torch.Size([4])
Output shape: torch.Size([4, 3])
Average initial loss (2 batches): 1.2...

>>> result.num_batches
2

>>> result.input_shape
(4, 4)

>>> result.target_shape
(4,)

>>> result.output_shape
(4, 3)
```