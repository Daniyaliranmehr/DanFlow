# training/trainer.py

from torch import nn
import torch
from torch.optim import Optimizer
from typing import Optional, Callable
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
    Train and validate PyTorch models.

    The Trainer class manages the training workflow, including model
    training, validation, metric computation, history tracking, and
    optional checkpoint saving.
    """

    def __init__(self, 
                 model: nn.Module, 
                 optimizer: Optimizer, 
                 loss_fn,
                 metric: Optional[object] = None
                 ) -> None:
        """
        Initialize a Trainer instance.

        Parameters
        ----------
        model
            Model to be trained and evaluated.

        optimizer
            Optimizer responsible for updating the model parameters.

        loss_fn
            Loss function used to optimize the model.

        metric
            Optional evaluation metric computed after each training
            and validation epoch.    
        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric

        self.metric_name = metric.__class__.__name__ if metric is not None else None

        self.loss_train_history = []
        self.loss_valid_history = []

        self.metric_train_history = []
        self.metric_valid_history = []

        self.console = Console()


    def train_epoch(
            self,
            train_loader: torch.utils.data.DataLoader,
            ) -> tuple[float, float | None]:
        
        """
        Train the model for a single epoch.

        The model is switched to training mode and updated by performing
        forward and backward passes over all batches in the training dataset.

        Parameters
        ----------
        train_loader
            DataLoader providing the training dataset.

        Returns
        -------
        tuple[float, float | None]
            Average training loss and the computed metric value for the
            epoch. The metric value is ``None`` if no evaluation metric is provided.

        Notes
        -----
        Model parameters are updated after every batch.
        """
        
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
        Evaluate the model for a single epoch.

        The model is evaluated in inference mode without updating its parameters.

        Parameters
        ----------
        valid_loader
            DataLoader providing the validation dataset.

        Returns
        -------
        tuple[float, float | None]
            Average validation loss and the computed metric value for the epoch.
            The metric value is ``None`` if no evaluation metric is provided.

        Notes
        -----
        Gradient computation is disabled during validation.
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
    
        """
        Train the model for multiple epochs.

        This method performs the complete training workflow by alternating
        between training and validation epochs, recording the training
        history, tracking the best validation performance, and optionally
        saving the best model checkpoint.

        Parameters
        ----------
        train_loader
            DataLoader providing the training dataset.

        valid_loader
            DataLoader providing the validation dataset.

        epochs
            Number of training epochs.

        save_best
            Whether to save the model checkpoint corresponding to the
            lowest validation loss.

        checkpoint_path
            Path where the best model checkpoint will be saved.

        Returns
        -------
        dict
            Dictionary containing the training history, validation history,
            and the best validation results obtained during training.

        Notes
        -----
        The saved checkpoint includes the model state, optimizer state,
        training epoch, and the best validation loss.
        """

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
                    f" | Train {self.metric_name}: {train_metric:.4f}"
                    f" | Valid {self.metric_name}: {valid_metric:.4f}"
                    f" | Best Valid {self.metric_name}: {best_valid_metric:.4f} (E{best_metric_epoch})"
                )

            epoch_bar.set_postfix_str(postfix)

            # ===================
            # Epoch Summary Table
            # ===================

            table = Table(
                title=f"Epoch {epoch}",
                show_header=True,
                show_lines=False,
            )

            table.add_column("Measure")
            table.add_column("Train", justify="right")
            table.add_column("Validation", justify="right")

            table.add_row(
                "Loss",
                f"{train_loss:.4f}",
                f"{valid_loss:.4f}",
            )

            if self.metric is not None:
                table.add_row(
                    self.metric_name,
                    f"{train_metric:.4f}",
                    f"{valid_metric:.4f}",
                )

            self.console.print(table)

        return {
            "train_loss": self.loss_train_history,
            "valid_loss": self.loss_valid_history,
            "train_metric": self.metric_train_history,
            "valid_metric": self.metric_valid_history,
            "metric_name": self.metric_name,
            "best_valid_loss": best_valid_loss,
            "best_loss_epoch": best_loss_epoch,
            "best_valid_metric": best_valid_metric,
            "best_metric_epoch": best_metric_epoch,
        }


class Evaluator:
    """
    Evaluate a trained PyTorch model on test data.

    The Evaluator class provides a simple interface for running inference
    on a dataset and computing optional evaluation metrics and loss
    without affecting model parameters or training state.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn=None,
        metric: Optional[Callable] = None,
    ) -> None:
        """
        Initialize an Evaluator instance.

        Parameters
        ----------
        model
            Trained PyTorch model to evaluate.

        loss_fn
            Optional loss function used to compute test loss.
            If None, loss is not computed.

        metric
            Optional callable metric function that takes
            (y_pred, y_true) and returns a scalar or tensor.
            If None, no metric is computed.
        """
        
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric

    def test(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> dict[str, float]:
        """
        Run evaluation on test data.

        The model is switched to evaluation mode and inference is
        performed without gradient computation. The original training
        state of the model is restored after evaluation.

        Parameters
        ----------
        x_test
            Input test features.

        y_test
            Ground-truth target values.

        Returns
        -------
        dict[str, float]
            Dictionary containing evaluation results. Possible keys:
            - "Loss": computed test loss (if loss_fn is provided)
            - "Metric": computed evaluation metric (if metric is provided)

        Notes
        -----
        - No gradients are computed during evaluation.
        - The model's original training/evaluation state is preserved.
        """
    
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                y_pred = self.model(x_test)

                results: dict[str, float] = {}

                if self.metric is not None:
                    value = self.metric(y_pred, y_test)
                    results["Metric"] = (
                        value.item() if torch.is_tensor(value) else float(value)
                    )

                if self.loss_fn is not None:
                    results["Loss"] = self.loss_fn(y_pred, y_test).item()

        finally:
            if was_training:
                self.model.train()

        return results
        