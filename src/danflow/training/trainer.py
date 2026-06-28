# training/trainer.py

from torch import nn
import torch
from torch.optim import Optimizer

from typing import Optional


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
    

class Trainer:
    """
    Train and validate models
    """

    def __init__(self, 
                 model: nn.Module, 
                 optimizer: Optimizer, 
                 loss_fn,
                 metric: Optional[object] = None
                 ) -> None:
        """
        Train the model for one epoch
        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric

        self.loss_train_history = []
        self.loss_valid_history = []

        self.metric_train_history = []
        self.metric_valid_history = []


    def train_epoch(
            self,
            train_loader: torch.utils.data.DataLoader
            ) -> tuple[float, float | None]:
        
        self.model.train()

        loss_meter = AverageMeter()

        if self.metric is not None:
            self.metric.reset()

        for inputs, targets in train_loader:
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())

            if self.metric is not None:
                self.metric.update(outputs, targets)

            metric_value = None
            if self.metric is not None:
                metric_value = self.metric.compute().item()

        return loss_meter.avg, metric_value


    def validate_epoch(self,
                       valid_loader:  torch.utils.data.DataLoader,
            ) -> tuple[float, float | None]:
        """
        Evaluate the model for one epoch
        """

        self.model.eval()
        
        loss_meter = AverageMeter()

        if self.metric is not None:
            self.metric.reset()
        
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = self.model(inputs)
                
                loss = self.loss_fn(outputs, targets)

                loss_meter.update(loss.item())

                if self.metric is not None:
                    self.metric.update(outputs, targets)

        metric_value = None
        if self.metric is not None:
            metric_value = self.metric.compute().item()

        return loss_meter.avg, metric_value