# Evaluation

This example demonstrates how to evaluate an already-trained PyTorch model on a separate test set using DanFlow.

The training process is intentionally omitted. The example assumes that the training workflow has already produced a checkpoint such as `best_model.pth`.

The evaluation workflow is:

Trained Checkpoint:
-    Model Architecture
-    Saved Parameters


Test Data:
-    Input Tensors
-    Target Tensors


Evaluation
-    Evaluator
-   Test Results


## Recreate the Model Architecture

The model architecture must match the architecture that was used during training.

For this example, use the same binary-classification model from the training workflow.

```python
import torch.nn as nn

def build_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

model = build_model()
```

At this point, `model` contains the network structure but not the parameters learned during training.


## Load the Trained Checkpoint

Assume that the training workflow saved the best model to `best_model.pth`.

Load the checkpoint:

```python
import torch

checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=True,
)
```

The checkpoint contains the model state together with training metadata:

```pycon
>>> checkpoint.keys()
dict_keys(['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_valid_loss'])
```

Load the saved model parameters:

```python
model.load_state_dict(
    checkpoint["model_state_dict"],
)
```

The model now contains the parameters saved during training.

The optimizer state is not required for a standard test-set evaluation.


## Prepare the Test Dataset

Create a separate test set that was not used during training or hyperparameter selection.

For this example, generate a deterministic synthetic test set using the same feature representation as the training data.

```python
import torch

torch.manual_seed(123)

x_test = torch.randn(200, 10)
y_test = (x_test.sum(dim=1) > 0).long()
```

Check the tensor shapes:

```pycon
>>> x_test.shape
torch.Size([200, 10])
```

```pycon
>>> y_test.shape
torch.Size([200])
```

The test inputs have the same 10-feature representation expected by the model.


## Create the Evaluator

Create an accuracy function for the test metric.

The evaluator accepts a callable metric that receives model outputs and target labels.

```python
def accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean()
```

Create the evaluator:

```python
from danflow.training import Evaluator

evaluator = Evaluator(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    metric=accuracy,
)
```

The evaluator is now ready to evaluate the model on the test tensors.


## Evaluate the Model

Run the test evaluation using the test inputs and targets.

```python
result = evaluator.test(
    x_test,
    y_test,
)
```

`Evaluator.test()` returns a dictionary containing the configured metric and the loss.

Inspect its keys:

```pycon
>>> list(result.keys())
['Metric', 'Loss']
```

The individual values can be accessed directly:

```pycon
>>> result["Metric"]
0.0
```

```pycon
>>> result["Loss"]
0.0
```

The numerical values depend on the trained checkpoint and therefore should be read from the actual evaluation run rather than hard-coded in the documentation.

For readable reporting:
```python
print(f"Test loss: {result['Loss']:.4f}")
print(f"Test accuracy: {result['Metric']:.4f}")
```


## Complete Evaluation Workflow

The complete evaluation stage can now be kept small because the model has already been trained.

```python
import torch
import torch.nn as nn

from danflow.training import Evaluator


def build_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean()


# Recreate the model architecture
model = build_model()

# Load the trained checkpoint
checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
)

# Prepare the test set
torch.manual_seed(123)

x_test = torch.randn(200, 10)
y_test = (x_test.sum(dim=1) > 0).long()

# Create the evaluator
evaluator = Evaluator(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    metric=accuracy,
)

# Evaluate
result = evaluator.test(
    x_test,
    y_test,
)

print(f"Test loss: {result['Loss']:.4f}")
print(f"Test accuracy: {result['Metric']:.4f}")
```

The result is a dictionary containing the evaluation metric and loss:

```pycon
>>> list(result.keys())
['Metric', 'Loss']
```

The numerical values depend on the trained checkpoint used for the example.

## Evaluation Workflow

The complete evaluation process consists of these stages:

Model Preparation:
-    Recreate the model architecture
-    Load the trained checkpoint

Test Data:
-    Prepare test inputs
-    Prepare test targets

Evaluation:
-    Create Evaluator
-    Run test()

Results:
-    Test Loss
-    Test Metric

At this point, the model has been evaluated on data that was not used during training or hyperparameter selection.

The resulting test loss and metric can be reported alongside the training and validation results when documenting the final model performance.