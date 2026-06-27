import torch


def adaptive_loss(outputs: torch.tensor, 
                  targets: torch.tensor, 
                  c: float=1., 
                  alpha:float=1.) -> torch.Tensor:
    """
    Adaptive robust loss for regression tasks with multiple formulations controlled by alpha.
    """
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


def log_cosh_loss(outputs: torch.tensor,
                 targets: torch.tensor) -> torch.Tensor:
    error = outputs - targets
    return torch.mean(torch.log(torch.cosh(error + 1e-12)))
