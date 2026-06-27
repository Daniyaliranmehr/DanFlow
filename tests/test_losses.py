from danflow.losses import adaptive_loss
import torch


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