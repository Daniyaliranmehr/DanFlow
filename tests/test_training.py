# test/test_training.py

from h11 import Data

from danflow.training import (
    AverageMeter,
    Trainer
)
import torch
from torch import mode, nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import R2Score


def test_average_meter():
    meter = AverageMeter()

    meter.update(2)
    meter.update(4)

    assert abs(meter.avg - 3) < 1e-6


def test_train_epoch_returns_loss_and_none_metric():
    model = nn.Linear(2, 1)

    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    inputs = torch.randn(8, 2)
    targets = torch.randn(8, 1)

    loader = DataLoader(TensorDataset(inputs, targets), 
                        batch_size=4
    )
    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      loss_fn=loss_fn
    )

    loss, metric = trainer.train_epoch(loader)

    assert isinstance(loss, float)
    assert metric is None


def test_train_epoch_with_metric():
    model = nn.Linear(2, 1)

    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    metric = R2Score()

    inputs = torch.randn(8, 2)
    targets = torch.randn(8, 1)

    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=4
    )
    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      loss_fn=loss_fn, 
                      metric=metric
    )

    loss, r2 = trainer.train_epoch(loader)

    assert isinstance(loss, float)
    assert isinstance(r2, float)


def test_train_epoch_updates_model_parameters():
    model = nn.Linear(2, 1)

    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    inputs = torch.randn(8, 2)
    targets = torch.randn(8, 1)

    loader = DataLoader(TensorDataset(inputs, targets), 
                        batch_size=4
    )

    before = model.weight.detach().clone()

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      loss_fn=loss_fn
    )

    trainer.train_epoch(loader)

    after = model.weight.detach()

    assert not torch.equal(before, after)


def test_validate_epoch_with_metric():
    model = nn.Linear(2, 1)

    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    inputs = torch.randn(8, 2)
    targets = torch.randn(8, 1)

    loader = DataLoader(TensorDataset(inputs, targets), 
                        batch_size=4
    )
    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      loss_fn=loss_fn
    )

    loss, metric = trainer.validate_epoch(loader)

    assert isinstance(loss, float)
    assert metric is None


def test_validate_epoch_returns_loss_and_metric():
    model = nn.Linear(2, 1)

    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    metric = R2Score()

    inputs = torch.randn(8, 2)
    targets = torch.randn(8, 1)

    loader = DataLoader(TensorDataset(inputs, targets), 
                        batch_size=4
    )

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      loss_fn=loss_fn, 
                      metric=metric
    )

    loss, r2 = trainer.validate_epoch(loader)

    assert isinstance(loss, float)
    assert isinstance(r2, float)


def test_validate_epoch_does_not_update_model_parameters():
    model = nn.Linear(2, 1)

    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    inputs = torch.randn(8, 2)
    targets = torch.randn(8, 1)

    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=4
    )

    before = model.weight.detach().clone()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn
    )

    trainer.validate_epoch(loader)

    after = model.weight.detach()

    assert torch.equal(before, after)


def test_fit_runs_and_update_history():
    model = nn.Linear(2, 1)
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    inputs = torch.randn(8, 2)
    targets = torch.randn(8, 1)

    train_loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=4,
        )
    valid_loader = DataLoader(
        TensorDataset(inputs, targets), 
        batch_size=4,
        )
    
    trainer = Trainer(model, optimizer, loss_fn)
    trainer.fit(train_loader, valid_loader, epochs=2)

    assert len(trainer.loss_train_history) == 2
    assert len(trainer.loss_valid_history) == 2