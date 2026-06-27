import torch


def Adaptive_loss(outputs, targets, c=1., alpha=1.):
    if alpha == 2:  # Quadratic
        loss = 0.5 * ((outputs - targets) / c) ** 2
        return torch.mean(loss)

    elif alpha == 0:  # Cauchy
        loss = torch.log((0.5 * ((outputs - targets) / c) ** 2) + 1)
        return torch.mean(loss)
    
    elif alpha == -torch.inf:  # Welsch
        loss = 1 - torch.exp(-0.5 * ((outputs - targets) / c) ** 2)
        return torch.mean(loss)
    
    else:  # Otherwise (Charbonnier & Geman-McClure)
         loss = (abs(alpha - 2) / alpha) * ((((outputs - targets) / c) ** 2 / abs(alpha - 2) + 1) ** (alpha / 2) - 1)
         return torch.mean(loss)
