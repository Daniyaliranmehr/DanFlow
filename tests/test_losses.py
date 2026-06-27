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