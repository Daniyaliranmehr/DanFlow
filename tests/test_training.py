# test/test_training.py

from danflow.training import (
    AverageMeter,
    Model
)
import torch


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