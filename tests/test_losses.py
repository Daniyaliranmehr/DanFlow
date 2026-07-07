# test/test_losses.py

from danflow.losses import (
    adaptive_loss,
    log_cosh_loss
)
import torch


def test_adaptive_loss_returns_scalar():
    outputs = torch.randn(10)
    targets = torch.randn(10)

    loss = adaptive_loss(outputs, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_adaptive_loss_quadratic():
    outputs = torch.tensor([2.0, 4.0])
    targets = torch.tensor([1.0, 2.0])

    loss = adaptive_loss(outputs, targets, alpha=2)
    expected = torch.mean(0.5 * ((outputs - targets) ** 2))

    assert torch.isclose(loss, expected)


def test_adaptive_loss_cauchy():
    outputs = torch.tensor([2.0, 4.0])
    targets = torch.tensor([1.0, 2.0])

    loss = adaptive_loss(outputs, targets, alpha=0)

    expected = torch.mean(torch.log(0.5 * ((outputs - targets) ** 2) + 1))

    assert torch.isclose(loss, expected)


def test_adaptive_loss_welsch():
    outputs = torch.tensor([2.0, 4.0])
    targets = torch.tensor([1.0, 2.0])

    loss = adaptive_loss(outputs, targets, alpha=-torch.inf)

    expected = torch.mean(1 - torch.exp(-0.5 * ((outputs - targets) ** 2)))

    assert torch.isclose(loss, expected)


def test_adaptive_loss_general():
    outputs = torch.tensor([2.0, 4.0])
    targets = torch.tensor([1.0, 2.0])

    alpha = 1.0

    loss = adaptive_loss(outputs, targets, alpha=alpha)

    expected = torch.mean((abs(alpha - 2) / alpha)
                           * ((((outputs - targets) ** 2) / abs(alpha - 2) + 1)
                               ** (alpha / 2) - 1))

    assert torch.isclose(loss, expected)


def test_log_cosh_loss():
    outputs = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.0, 2.0])

    loss = log_cosh_loss(outputs, targets)
    expected = torch.mean(torch.log(torch.cosh(torch.zeros(2))))

    assert torch.isclose(loss, expected)

