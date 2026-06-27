# training/trainer.py

import torch.nn as nn


class AverageMeter:
    """
    Computes and stores the current value and running average.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Model(nn.Module):
    """
    Define the Model Architecture
    """
    def __init__(self, in_features, h1, h2, out_features):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_features)
        )

    def forward(self, x):
        return self.model(x)