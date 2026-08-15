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


    # ------------------------------------------------------------------
    # Forward check
    # ------------------------------------------------------------------

    def forward_check(
        self,
        train_loader: DataLoader,
        *,
        expected_output_size: Optional[int] = None,
        num_batches: int = DEFAULT_FORWARD_BATCHES,
    ) -> ForwardCheckResult:
        """
        Check the model's forward path using the provided DataLoader.

        The method:

        1. Iterates over the provided DataLoader.
        2. Passes several batches through the model.
        3. Checks input, target, and output compatibility.
        4. Checks the model output size when provided.
        5. Calculates the loss.
        6. Raises a helpful error if the loss cannot be calculated.
        7. Computes the average initial loss.

        Parameters
        ----------
        train_loader
            DataLoader used for the forward-path check.

        expected_output_size
            Expected size of the model's final output dimension.

            For example, for a 7-class classification problem:
                expected_output_size=7

        num_batches
            Maximum number of batches used to calculate the average
            initial loss.

        Returns
        -------
        ForwardCheckResult
            Information about the verified forward path.
        """

        if not isinstance(train_loader, DataLoader):
            raise TypeError(
                "train_loader must be a torch.utils.data.DataLoader."
            )

        if num_batches < 1:
            raise ValueError(
                "num_batches must be at least 1."
            )

        # Preserve the original model mode.
        was_training = self.model.training

        self.model.eval()

        try:
            losses = []

            first_input_shape = None
            first_target_shape = None
            first_output_shape = None

            for batch_index, (inputs, targets) in enumerate(train_loader):

                if batch_index >= num_batches:
                    break

                if not torch.is_tensor(inputs):
                    raise TypeError(
                        "DataLoader inputs must be torch.Tensor objects."
                    )

                if not torch.is_tensor(targets):
                    raise TypeError(
                        "DataLoader targets must be torch.Tensor objects."
                    )

                with torch.no_grad():

                    # ---------------------------
                    # Forward pass
                    # ---------------------------
                    outputs = self.model(inputs)

                    if not torch.is_tensor(outputs):
                        raise TypeError(
                            "The model output must be a torch.Tensor, "
                            f"but got {type(outputs).__name__}."
                        )

                    if outputs.ndim == 0:
                        raise ValueError(
                            "The model output does not contain "
                            "a batch dimension."
                        )

                    # ---------------------------
                    # Batch-size compatibility
                    # ---------------------------
                    if outputs.shape[0] != targets.shape[0]:
                        raise ValueError(
                            "Model output batch size does not match "
                            "target batch size.\n"
                            f"outputs.shape = {tuple(outputs.shape)}\n"
                            f"targets.shape = {tuple(targets.shape)}"
                        )

                    # ---------------------------
                    # Output-size check
                    # ---------------------------
                    if expected_output_size is not None:

                        if outputs.ndim < 2:
                            raise ValueError(
                                "expected_output_size was provided, "
                                "but the model output does not have "
                                "a class/output dimension.\n"
                                f"outputs.shape = {tuple(outputs.shape)}"
                            )

                        actual_output_size = outputs.shape[-1]

                        if actual_output_size != expected_output_size:
                            raise ValueError(
                                "Incorrect model output size.\n"
                                f"Expected output size: "
                                f"{expected_output_size}\n"
                                f"Actual output size: "
                                f"{actual_output_size}\n"
                                f"outputs.shape = "
                                f"{tuple(outputs.shape)}"
                            )

                    # ---------------------------
                    # Loss calculation
                    # ---------------------------
                    try:
                        loss = self.loss_fn(
                            outputs,
                            targets,
                        )

                    except Exception as exc:
                        raise ValueError(
                            "The loss function could not be "
                            "calculated with the model outputs "
                            "and targets.\n"
                            f"outputs.shape = "
                            f"{tuple(outputs.shape)}\n"
                            f"targets.shape = "
                            f"{tuple(targets.shape)}\n"
                            "Check the model output shape, target "
                            "shape, target dtype, and loss function."
                        ) from exc

                    # Loss must be one scalar value.
                    if (
                        not torch.is_tensor(loss)
                        or loss.numel() != 1
                    ):
                        raise ValueError(
                            "loss_fn must return a single scalar "
                            "tensor."
                        )

                    loss_value = loss.item()

                    if not math.isfinite(loss_value):
                        raise ValueError(
                            "The calculated loss is not finite: "
                            f"{loss_value}"
                        )

                losses.append(loss_value)

                if first_input_shape is None:
                    first_input_shape = tuple(inputs.shape)
                    first_target_shape = tuple(targets.shape)
                    first_output_shape = tuple(outputs.shape)

            if not losses:
                raise ValueError(
                    "The DataLoader did not produce any batches."
                )

            average_loss = sum(losses) / len(losses)

            print(
                f"Input shape:  {first_input_shape}"
            )

            print(
                f"Target shape: {first_target_shape}"
            )

            print(
                f"Output shape: {first_output_shape}"
            )

            print(
                f"Average initial loss "
                f"({len(losses)} batches): "
                f"{average_loss:.4f}"
            )

            return ForwardCheckResult(
                num_batches=len(losses),
                average_loss=average_loss,
                input_shape=first_input_shape,
                target_shape=first_target_shape,
                output_shape=first_output_shape,
            )

        finally:
            # Restore the model's original mode.
            if was_training:
                self.model.train()