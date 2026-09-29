# Checkpoints

Checkpoints preserve a model's training state so that a selected training state can be saved and used later.

In DanFlow, checkpoints are created by `Trainer.fit()` when `save_best=True`. DanFlow stores the best model according to validation loss rather than saving a checkpoint after every epoch.

This guide focuses on the checkpoint lifecycle and does not cover the general training workflow or test evaluation.

## Saving the Best Checkpoint

Enable checkpoint saving with `save_best=True`:

```python id="a7k3p1"
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=10,
    save_best=True,
    checkpoint_path="artifacts/best_model.pth",
)
```

When checkpoint saving is enabled, DanFlow monitors the validation loss after each epoch.

A checkpoint is written when the current validation loss becomes better than the previously recorded best validation loss.

The path is controlled by `checkpoint_path`.

For example:

```text id="m2r8v4"
artifacts/
    best_model.pth
```

The filename and directory are not special to DanFlow. They are determined by the value passed to `checkpoint_path`.

## Checkpoint Contents

A checkpoint created by `Trainer.fit(save_best=True)` is a dictionary containing four entries:

```python id="k5d1q8"
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "best_valid_loss": ...,
}
```

Each entry serves a different purpose.

| Key                    | Purpose                                              |
| ---------------------- | ---------------------------------------------------- |
| `model_state_dict`     | Model parameters and persistent buffers              |
| `optimizer_state_dict` | Optimizer state required for continued optimization  |
| `epoch`                | Epoch at which the checkpoint was saved              |
| `best_valid_loss`      | Validation loss associated with the saved checkpoint |

The checkpoint is therefore more than a collection of model weights. It also preserves the optimizer state and information about the validation result that caused the checkpoint to be selected.

## Inspecting a Checkpoint

A checkpoint can be loaded as a dictionary and inspected before loading anything into a model:

```python id="h4v9c2"
import torch

checkpoint = torch.load(
    "artifacts/best_model.pth",
    map_location="cpu",
    weights_only=True,
)
```

Inspect its keys:

```pycon id="n6s2x7"
>>> checkpoint.keys()
dict_keys(['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_valid_loss'])
```

The saved metadata can also be inspected directly:

```pycon id="q8c3m5"
>>> checkpoint["epoch"]
<checkpoint epoch>
>>> checkpoint["best_valid_loss"]
<best validation loss>
```

The exact numerical values depend on the training run.

## Loading Model Weights

The model parameters are stored under `model_state_dict`.

They should be loaded into a model instance using that entry:

```python id="v3m7k1"
model.load_state_dict(
    checkpoint["model_state_dict"]
)
```

Do not pass the complete checkpoint dictionary directly to `load_state_dict()`:

```python id="e1q9r4"
model.load_state_dict(checkpoint)
```

The complete checkpoint contains metadata and optimizer information in addition to the model state dictionary, so the model state must be selected explicitly.

The model architecture must also match the architecture used when the checkpoint was created.

For the full model-loading and evaluation workflow, see the [Evaluation Guide](evaluation.md).

## Restoring the Optimizer

The optimizer state is stored separately:

```python id="c6k2p8"
optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)
```

Restoring the optimizer state is important when the goal is to continue optimization from the saved training state rather than starting the optimizer with a fresh internal state.

For example, optimizers such as Adam maintain additional state for their parameters. That state is preserved in `optimizer_state_dict`.

## Continuing Training from a Checkpoint

DanFlow does not provide a separate checkpoint-resume API.

A checkpoint can nevertheless be used for manual continuation by:

1. Recreating the model architecture.
2. Recreating the optimizer.
3. Loading `model_state_dict`.
4. Loading `optimizer_state_dict`.
5. Calling `Trainer.fit()` with the restored model and optimizer.

For example:

```python id="r5w8n3"
import torch
import torch.optim as optim

from danflow.training import Trainer


checkpoint = torch.load(
    "artifacts/best_model.pth",
    map_location="cpu",
    weights_only=True,
)

model = build_model()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```

The restored model and optimizer can then be passed to a new training run:

```python id="p4m6x9"
history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=5,
)
```

This restores the model and optimizer states, but `Trainer.fit()` itself starts a new `fit()` call. The previous training history is not automatically resumed as a continuation of the original history.

The `epoch` value stored in the checkpoint is metadata that identifies when the checkpoint was created; it does not cause `fit()` to automatically start counting from that epoch.

## Best Checkpoint and Final Epoch

The checkpoint saved by DanFlow represents the epoch with the best validation loss encountered during that `fit()` call.

This means:

```text id="x7p2m5"
Final epoch
    May not be the checkpoint epoch

Checkpoint
    Corresponds to the best validation loss
```

For example, a model may continue training after its best validation loss has already been reached. The checkpoint still contains the earlier model state if that earlier state had the lower validation loss.

This is the main reason to use `save_best=True` when the saved model should represent the best validation-loss state rather than simply the final epoch.

## What the Checkpoint Does Not Store

The checkpoint created by `Trainer.fit()` contains exactly the following state:

```text id="s4k8q2"
model_state_dict
optimizer_state_dict
epoch
best_valid_loss
```

It does not include other training information such as:

```text id="d9m3v6"
Training history
Metric state
Learning-rate scheduler state
Random-number-generator state
```

Such information must be managed separately when a project requires exact experiment reproducibility or a more complete training-resume mechanism.

## Loading on CPU

A checkpoint can be loaded on a CPU even when it was originally created in another environment by specifying `map_location="cpu"`:

```python id="y6p2r8"
checkpoint = torch.load(
    "artifacts/best_model.pth",
    map_location="cpu",
    weights_only=True,
)
```

This is useful when a checkpoint is being inspected, evaluated, or transferred to a machine without the original compute device.

The model can then be moved to the appropriate device after loading.

## Managing Checkpoint Files

For projects with multiple experiments, keeping checkpoints in a dedicated directory makes them easier to manage:

```text id="z3v8n1"
project/
    artifacts/
        best_model.pth
```

A more descriptive naming scheme can also be useful when several experiments are stored together:

```text id="f7c2m6"
artifacts/
    baseline_best.pth
    tuned_best.pth
    experiment_03_best.pth
```

The filename itself has no special meaning to DanFlow. The important part is that the path provided to `checkpoint_path` identifies where the checkpoint should be stored.

## Common Mistakes

### Loading the Complete Checkpoint into the Model

Use:

```python id="b4k9r2"
model.load_state_dict(
    checkpoint["model_state_dict"]
)
```

rather than:

```python id="w8m3p5"
model.load_state_dict(checkpoint)
```

### Forgetting the Optimizer State

When continuing training, restoring only the model parameters does not restore the optimizer's saved internal state.

Use:

```python id="c2x7m4"
optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)
```

when optimizer continuation is required.

### Assuming the Checkpoint Is the Final Epoch

The saved checkpoint corresponds to the best validation loss observed while checkpoint saving was enabled. It is not necessarily the last epoch of training.

### Expecting Automatic Resume

The stored `epoch` value is checkpoint metadata. `Trainer.fit()` does not automatically interpret it as a resume position.

### Loading with a Different Model Architecture

`model_state_dict` can only be loaded into a compatible model architecture.

When the architecture changes, the parameter names or tensor shapes may no longer match the saved state.

## Recommended Checkpoint Workflow

A typical checkpoint workflow in DanFlow is:

```text id="e5r2k7"
Training
    Run Trainer.fit()
    Enable save_best=True

Checkpoint
    Save the best validation-loss state

Later
    Load checkpoint
    Restore model state

Optional continuation
    Restore optimizer state
    Create a Trainer
    Start a new training run
```

For evaluating a saved model, continue with the [Evaluation Guide](evaluation.md).

For configuring the training process that produces checkpoints, see the [Training Guide](training.md).