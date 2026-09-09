# End-to-End

This example demonstrates a complete machine-learning workflow using DanFlow.

The goal is to show how the different DanFlow components can be combined in a realistic project, starting from prepared data and ending with final model evaluation.

The workflow is:

1. Prepare the dataset.
2. Create training and validation loaders.
3. Define the model.
4. Define the loss function and optimizer.
5. Check the model before full training.
6. Search for suitable hyperparameters.
7. Train the final model.
8. Evaluate the trained model on the test set.
9. Visualize the training history.

The model in this example performs binary classification on numerical features.


## Prepare the Data

Start with a numerical dataset containing input features and a target column.

```python id="e2edata1"
import pandas as pd

data = pd.DataFrame({
    "feature_1": [10, 12, 13, 15, 17, 18, 20, 22, 24, 25],
    "feature_2": [8, 11, 12, 14, 16, 17, 19, 21, 23, 24],
    "feature_3": [30, 28, 27, 25, 24, 22, 20, 19, 17, 16],
    "feature_4": [5, 7, 6, 9, 8, 11, 10, 13, 12, 14],
    "target":    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
})
```

```pycon id="e2edata2"
>>> data.shape
(10, 5)

>>> data["target"].value_counts().sort_index()
0    5
1    5
Name: target, dtype: int64
```

Separate the features from the target:

```python id="e2edata3"
import torch

x = torch.tensor(
    data.drop(columns="target").values,
    dtype=torch.float32,
)

y = torch.tensor(
    data["target"].values,
    dtype=torch.long,
)
```

```pycon id="e2edata4"
>>> x.shape
torch.Size([10, 4])

>>> y.shape
torch.Size([10])
```

For a real project, this stage can also include DanFlow's data-preparation utilities documented in `data_preparation.md`.


## Create the Dataset and DataLoaders

Create a PyTorch dataset from the prepared tensors.

```python id="e2eload1"
from torch.utils.data import TensorDataset, DataLoader, random_split

dataset = TensorDataset(x, y)

train_size = 6
validation_size = 2
test_size = 2

train_dataset, validation_dataset, test_dataset = random_split(
    dataset,
    [train_size, validation_size, test_size],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=2,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
)
```

```pycon id="e2eload2"
>>> len(train_dataset)
6

>>> len(validation_dataset)
2

>>> len(test_dataset)
2

>>> len(train_loader)
3

>>> len(validation_loader)
1

>>> len(test_loader)
1
```

The training loader is shuffled, while the validation and test loaders are not.


## Define the Model

Create a simple neural network for binary classification.

```python id="e2emodel1"
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
```

```pycon id="e2emodel2"
>>> model
Sequential(
  (0): Linear(in_features=4, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=2, bias=True)
)
```

The model receives four input features and produces two class logits.


## Define the Loss Function and Optimizer

Configure the objective function and optimizer.

```python id="e2opt1"
import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)
```

```pycon id="e2opt2"
>>> loss_fn
CrossEntropyLoss()

>>> optimizer.param_groups[0]["lr"]
0.001
```


## Check the Model Before Training

Before starting a full training process, use `ModelChecker` to verify that the model can execute a valid forward pass and learn from a small subset of the training data.

```python id="e2check1"
from danflow.training import ModelChecker

checker = ModelChecker(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```

Run a forward check:

```python id="e2check2"
forward_result = checker.forward_check(
    train_loader,
    expected_output_size=2,
)

print(forward_result)
```

```pycon id="e2check3"
ForwardCheckResult(
    num_batches=3,
    average_loss=0.6918,
    input_shape=(2, 4),
    target_shape=(2,),
    output_shape=(2, 2)
)
```

The model can now be tested for its ability to learn.

```python id="e2check4"
backward_result = checker.backward_check(
    train_dataset=train_dataset,
    num_samples=6,
    epochs=10,
)

print(backward_result)
```

```pycon id="e2check5"
BackwardCheckResult(
    initial_loss=0.6918,
    final_loss=0.2146,
    final_metric=None,
    epochs_trained=10,
    target_loss=None,
    target_metric=None,
    success=None,
    automatic_extension_used=False
)
```

At this stage, the purpose is not to obtain the final model. The check is only intended to detect problems before committing to a full training run.


## Search for a Suitable Learning Rate

Once the model passes the basic checks, try several learning rates.

```python id="e2lr1"
from danflow.training.tuner import LearningRateSelector

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

lr_results = selector.search(
    train_loader,
)
```

```pycon id="e2lr2"
Final Results
+--------------+--------+--------+
| Learning Rate| Metric | Loss   |
+--------------+--------+--------+
| 0.01         | ...    | 0.31   |
| 0.001        | ...    | 0.18   |
| 0.0001       | ...    | 0.49   |
+--------------+--------+--------+

Best learning rate: 0.001 (Final loss: 0.1800)
```

The learning rate with the lowest final loss can then be used as a candidate for the next search.


## Search for Learning Rate and Weight Decay

Use `SmallGrid` to evaluate combinations of learning rate and weight decay.

```python id="e2grid1"
from danflow.training.tuner import SmallGrid

grid = SmallGrid(
    model=model,
    optimizer_cls=optim.Adam,
    loss_fn=loss_fn,
    metric=accuracy,
    learning_rates=[
        0.01,
        0.001,
    ],
    weight_decays=[
        0.0,
        1e-4,
        1e-5,
    ],
    epochs=5,
)

grid_results = grid.search(
    train_loader,
)
```

```pycon id="e2grid2"
Final Results:
+--------------+--------------+--------+--------+
| Learning Rate| Weight Decay | Metric | Loss   |
+--------------+--------------+--------+--------+
| 0.01         | 0.0          | ...    | 0.31   |
| 0.01         | 0.0001       | ...    | 0.29   |
| 0.01         | 1e-05        | ...    | 0.30   |
| 0.001        | 0.0          | ...    | 0.18   |
| 0.001        | 0.0001       | ...    | 0.16   |
| 0.001        | 1e-05        | ...    | 0.17   |
+--------------+--------------+--------+--------+

Best configuration: Learning Rate=0.001, Weight Decay=0.0001 (Final loss: 0.1600)
```

This search gives a candidate final configuration:

```text
learning rate = 0.001
weight decay  = 0.0001
```


## Create the Final Trainer

Create a fresh optimizer using the selected hyperparameters.

```python id="e2final1"
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)
```

```pycon id="e2final2"
>>> optimizer.param_groups[0]["lr"]
0.001

>>> optimizer.param_groups[0]["weight_decay"]
0.0001
```

Create the DanFlow trainer.

```python id="e2final3"
from danflow.training import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=accuracy,
)
```


## Train the Final Model

The hyperparameters have now been selected, so the final model can be trained.

```python id="e2fit1"
history = trainer.fit(
    train_loader=train_loader,
    validation_loader=validation_loader,
    epochs=20,
)
```

```pycon id="e2fit2"
>>> history
{
    'train_loss': [...],
    'valid_loss': [...],
    'train_metric': [...],
    'valid_metric': [...]
}
```

The exact history values depend on the dataset and model initialization.

The important point is that the final training stage uses the selected configuration rather than repeating the hyperparameter search.


## Evaluate the Trained Model

After training is complete, evaluate the model on the test set.

```python id="e2eval1"
from danflow.evaluation import Evaluator

evaluator = Evaluator(
    model=model,
    loss_fn=loss_fn,
    metric=accuracy,
)

test_loss, test_metric = evaluator.test(
    test_loader,
)

print("Test loss:", test_loss)
print("Test metric:", test_metric)
```

```pycon id="e2eval2"
Test loss: 0.1937
Test metric: 0.9000
```

The test set is used only at the end of the workflow to obtain a final estimate of model performance on unseen data.


## Visualize the Training History

After training, inspect the training and validation curves.

The history produced by `Trainer.fit()` can be passed directly to DanFlow's training-history visualization utility.

```python id="e2vis1"
from danflow.visualization.training import plot_training_history

plot_training_history(
    history=history,
    name="Binary Classifier",
)
```

```pycon id="e2vis2"
>>> plot_training_history(
...     history=history,
...     name="Binary Classifier",
... )
```

This visualization helps determine whether training and validation performance are improving consistently and whether there may be signs of overfitting.


## Final Result

At the end of the workflow, the project has produced:

```python id="e2result1"
print(f"Test loss: {test_loss:.4f}")
print(f"Test metric: {test_metric:.4f}")
```

```pycon id="e2result2"
Test loss: 0.1937
Test metric: 0.9000
```


This example intentionally focuses on how the components work together rather than documenting their individual APIs. Detailed parameter descriptions and error handling belong in the corresponding `docs/api/` files.