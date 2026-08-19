from copy import deepcopy
from typing import Optional, List, Dict, Any, Callable

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm
from prettytable import PrettyTable


class LearningRateSelector:
    """
    Select a suitable learning rate by running multiple training experiments.

    The selector evaluates different learning rates by:

    - Creating independent copies of the provided model.
    - Initializing a new optimizer for each learning rate.
    - Training each model for a fixed number of epochs.
    - Comparing final loss and metric values.

    This helps identify promising learning rate ranges before performing
    more detailed hyperparameter searches.
    """

    DEFAULT_LEARNING_RATES = [
        0.1,
        0.01,
        0.001,
        0.0001
    ]

    def __init__(
        self,
        model: nn.Module,
        trainer_cls: type,
        optimizer_cls: type[Optimizer],
        loss_fn: Callable,
        metric: Optional[object] = None,
        learning_rates: Optional[List[float]] = None,
        weight_decay: float = 1e-4,
        epochs: int = 5
    ) -> None:
        """
        Initialize the learning rate selector.

        Parameters
        ----------
        model : nn.Module
            Neural network model to evaluate.

        trainer_cls : type
            Trainer class used for model training.

        optimizer_cls : type
            Optimizer class used to update model parameters.

        loss_fn : callable
            Loss function used during training.

        metric : Optional[object]
            Metric object used to evaluate model performance.
            Default: None

        learning_rates : Optional[List[float]]
            Learning rates to evaluate.
            Default:
            [0.1, 0.01, 0.001, 0.0001]

        weight_decay : float
            Weight decay value passed to the optimizer.
            Default: 1e-4

        epochs : int
            Number of epochs for each experiment.
            Default: 5
        """

        self.model = model
        self.trainer_cls = trainer_cls
        self.optimizer_cls = optimizer_cls
        self.loss_fn = loss_fn
        self.metric = metric

        self.learning_rates = (
            learning_rates
            if learning_rates is not None
            else self.DEFAULT_LEARNING_RATES
        )

        self.weight_decay = weight_decay
        self.epochs = epochs