# test/test_training.py

from danflow.training import (
    AverageMeter,
    Model,
    Trainer
)
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset


def test_average_meter():
    meter = AverageMeter()

    meter.update(2)
    meter.update(4)

    assert abs(meter.avg - 3) < 1e-6


def test_model():
    model = Model(in_features=4, h1=8, h2=6, out_features=1)

    x = torch.randn(10, 4) # batch_size=10 & features=4
    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (10, 1)


def test_train_epoch_returns_loss_and_metric():
    model = nn.Linear(2, 1)

    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    inputs = torch.randn(8, 2)
    targets = torch.randn(8, 1)

    loader = DataLoader(TensorDataset(inputs, targets), batch_size = 4)
    trainer = Trainer(model=model, optimizer=optimizer, loss=loss_fn)

    loss, metric = trainer.train_epoch(loader)

    assert isinstance(loss, float)
    assert metric is None
