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