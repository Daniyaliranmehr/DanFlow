from danflow.losses import adaptive_loss
import torch


def test_adaptive_loss_quadratic():
    outputs = torch.tensor([2.0, 4.0])
    targets = torch.tensor([1.0, 2.0])

    loss = adaptive_loss(outputs, targets, alpha=2)
    expected = torch.tensor(1.25)
    
    assert torch.isclose(loss, expected)