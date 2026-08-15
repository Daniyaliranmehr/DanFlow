# training/checker.py

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .trainer import Trainer


@dataclass
class ForwardCheckResult:
    """
    Results returned by forward_check().
    """

    num_batches: int
    average_loss: float
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass
class BackwardCheckResult:
    """
    Results returned by backward_check() and continue_backward().
    """

    initial_loss: float
    final_loss: float
    final_metric: Optional[float]
    epochs_trained: int
    target_loss: Optional[float]
    target_metric: Optional[float]
    success: Optional[bool]
    automatic_extension_used: bool


class ModelChecker:
    """
    Verify the forward and backward paths of a PyTorch model.

    The checker provides:

    - Forward-path verification before training.
    - Small-subset overfitting verification.
    - Optional metric-based and loss-based success criteria.
    - Automatic continuation when epochs are not explicitly specified.
    - Manual continuation on the same subset.
    """

    DEFAULT_EPOCHS = 500
    DEFAULT_NUM_SAMPLES = 1000
    DEFAULT_NUM_BATCHES = 5
    DEFAULT_FORWARD_BATCHES = 5

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        loss_fn,
    ) -> None:
        """
        Initialize the ModelChecker.

        Parameters
        ----------
        model
            PyTorch model to check.

        optimizer
            Optimizer used to update the model during backward checking.

        loss_fn
            Loss function used to calculate the model loss.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # State used by backward checking.
        self._backward_loader: Optional[DataLoader] = None
        self._trainer: Optional[Trainer] = None

        self._target_metric: Optional[float] = None
        self._target_loss: Optional[float] = None

        self._backward_initial_loss: Optional[float] = None
        self._backward_epochs_trained = 0

        self._backward_history: list[dict[str, Optional[float]]] = []

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dataset(dataset) -> None:
        """
        Validate that the dataset is non-empty and supports len().
        """
        try:
            dataset_length = len(dataset)
        except TypeError as exc:
            raise TypeError(
                "dataset must implement __len__ and __getitem__."
            ) from exc

        if dataset_length == 0:
            raise ValueError(
                "dataset must contain at least one sample."
            )


    @staticmethod
    def _default_batch_size(
        num_samples: int,
        num_batches: int,
    ) -> int:
        """
        Calculate a batch size that produces approximately
        `num_batches` batches.

        Example:
            1000 samples / 5 batches = 200 batch size.
        """
        return max(1, math.ceil(num_samples / num_batches))


    def _check_overfitting(
        self,
        final_loss: float,
        final_metric: Optional[float],
    ) -> Optional[bool]:
        """
        Determine whether the requested overfitting target was reached.

        Returns
        -------
        bool or None
            True if all requested targets were reached.
            False if at least one requested target was not reached.
            None if no target was provided.
        """

        if (
            self._target_loss is None
            and self._target_metric is None
        ):
            return None

        checks = []

        if self._target_loss is not None:
            checks.append(
                final_loss <= self._target_loss
            )

        if self._target_metric is not None:
            if final_metric is None:
                raise ValueError(
                    "target_metric was provided, but no metric "
                    "was supplied to backward_check()."
                )

            checks.append(
                final_metric >= self._target_metric
            )

        return all(checks)