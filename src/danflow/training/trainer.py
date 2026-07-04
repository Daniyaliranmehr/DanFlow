# training/trainer.py

from torch import nn
import torch
from torch.optim import Optimizer
from typing import Optional
from tqdm.auto import tqdm
from rich.table import Table
from rich.console import Console


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

        self.console = Console()


    def train_epoch(
            self,
            train_loader: torch.utils.data.DataLoader,
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


    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        save_best: bool = False,
        checkpoint_path: str = "best_model.pth",
    ) -> dict:

        best_valid_loss = float("inf")
        best_loss_epoch = 0

        best_valid_metric = float("-inf") if self.metric is not None else None
        best_metric_epoch = 0

        epoch_bar = tqdm(
            range(1, epochs + 1),
            desc="Epoch",
            unit="epoch",
        )

        for epoch in epoch_bar:

            train_loss, train_metric = self.train_epoch(train_loader)

            valid_loss, valid_metric = self.validate_epoch(valid_loader)

            self.loss_train_history.append(train_loss)
            self.loss_valid_history.append(valid_loss)

            if self.metric is not None:
                self.metric_train_history.append(train_metric)
                self.metric_valid_history.append(valid_metric)

            if save_best and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_loss_epoch = epoch

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "best_valid_loss": best_valid_loss,
                    },
                    checkpoint_path,
                )

            if (
                self.metric is not None
                and valid_metric is not None
                and valid_metric > best_valid_metric
            ):
                best_valid_metric = valid_metric
                best_metric_epoch = epoch

            postfix = (
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"Best Valid Loss: {best_valid_loss:.4f} (E{best_loss_epoch})"
            )

            if self.metric is not None:
                postfix += (
                    f" | Train Metric: {train_metric:.4f}"
                    f" | Valid Metric: {valid_metric:.4f}"
                    f" | Best Valid Metric: {best_valid_metric:.4f} (E{best_metric_epoch})"
                )

            epoch_bar.set_postfix_str(postfix)

            # -------------------------
            # Persistent table
            # -------------------------

            table = Table(
                title=f"Epoch {epoch}",
                show_header=True,
                show_lines=False,
            )

            table.add_column("Metric")
            table.add_column("Train", justify="right")
            table.add_column("Validation", justify="right")

            table.add_row(
                "Loss",
                f"{train_loss:.4f}",
                f"{valid_loss:.4f}",
            )

            if self.metric is not None:
                table.add_row(
                    "Metric",
                    f"{train_metric:.4f}",
                    f"{valid_metric:.4f}",
                )

            self.console.print(table)

        return {
            "train_loss": self.loss_train_history,
            "valid_loss": self.loss_valid_history,
            "train_metric": self.metric_train_history,
            "valid_metric": self.metric_valid_history,
            "best_valid_loss": best_valid_loss,
            "best_loss_epoch": best_loss_epoch,
            "best_valid_metric": best_valid_metric,
            "best_metric_epoch": best_metric_epoch,
        }
    