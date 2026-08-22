from copy import deepcopy
from typing import Optional, List, Dict, Any, Callable, Type

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm
from prettytable import PrettyTable

from danflow.training import Trainer


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


    def search(
            self,
            train_loader: DataLoader
        ) -> List[Dict[str, Any]]:
            """
            Evaluate different learning rates.
    
            Parameters
            ----------
            train_loader : DataLoader
                Training data loader used for experiments.
    
            Returns
            -------
            List[Dict[str, Any]]
                Results containing learning rate, loss, and metric values.
            """
    
            results = []
    
            for learning_rate in self.learning_rates:
    
                result = self._train_with_lr(
                    train_loader,
                    learning_rate
                )
    
                results.append(result)
    
            self._print_summary(results)
    
            return results


    def _train_with_lr(
            self,
            train_loader: DataLoader,
            learning_rate: float
        ) -> Dict[str, Any]:
            """
            Train a model using a specific learning rate.
    
            A copy of the original model is created to ensure that every
            learning rate experiment starts from the same initial weights.
    
            Parameters
            ----------
            train_loader : DataLoader
                Training data loader used for model training.
    
            learning_rate : float
                Learning rate used for the optimizer.
    
            Returns
            -------
            Dict[str, Any]
                Final loss and metric values for the experiment.
            """
    
            model = deepcopy(self.model)
    
            optimizer = self.optimizer_cls(
                model.parameters(),
                lr=learning_rate,
                weight_decay=self.weight_decay
            )
    
            trainer = self.trainer_cls(
                model,
                optimizer,
                self.loss_fn,
                self.metric
            )
    
            tqdm.write(f"LR={learning_rate}")
    
            for epoch in range(self.epochs):
    
                with tqdm(
                    total=1,
                    desc=f"Epoch {epoch}",
                    unit="batch") as pbar:
    
                    loss, metric_value = trainer.train_epoch(
                        train_loader
                    )
    
                    pbar.set_postfix({
                        "metric": f"{metric_value:.4f}",
                        "loss": f"{loss:.4f}"
                    })
    
                    pbar.update(1)

            tqdm.write("")

            return {
                "learning_rate": learning_rate,
                "loss": loss,
                "metric": metric_value
            }


    def _print_summary(
            self,
            results: List[Dict[str, Any]]
        ) -> None:
            """
            Print the final comparison table.
    
            Parameters
            ----------
            results : List[Dict[str, Any]]
                Experiment results.
            """
    
            table = PrettyTable(
                [
                    "Learning Rate",
                    "Metric",
                    "Loss"
                ]
            )
    
            for result in results:
    
                table.add_row(
                    [
                        result["learning_rate"],
                        f"{result['metric']:.4f}",
                        f"{result['loss']:.4f}"
                    ]
                )
    
            print("\nFinal Results")
            print(table)
    
            best = min(
                results,
                key=lambda x: x["loss"]
            )
    
            print(
                f"\nBest learning rate: {best['learning_rate']} "
                f"(Final loss: {best['loss']:.4f})"
            )


class SmallGrid:
    """Perform a small grid search over learning rates and weight decays."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_cls: Type[Optimizer],
        loss_fn: Callable,
        metric: Callable,
        learning_rates: list[float] | None = None,
        learning_rate: float | None = None,
        weight_decays: list[float] | None = None,
        epochs: int = 5,
    ) -> None:
        """Initialize the small grid search.

        Parameters
        ----------
        model : nn.Module
            Model to train during the grid search.

        optimizer_cls : Type[Optimizer]
            Optimizer class used to train the model.

        loss_fn : Callable
            Loss function used during training.

        metric : Callable
            Metric used to evaluate model performance.

        learning_rates : list[float] | None, optional
            Learning rates to test. Defaults to None.

        learning_rate : float | None, optional
            A single learning rate to test. Defaults to None.

        weight_decays : list[float] | None, optional
            Weight decay values to test. Defaults to
            ``[0.0, 1e-4, 1e-5, 1e-6]``.
            
        epochs : int, optional
            Number of epochs used for each configuration. Defaults to 5.

        Raises
        ------
        ValueError
            If neither ``learning_rates`` nor ``learning_rate`` is provided.
        """
        self.model = model
        self.optimizer = optimizer_cls
        self.loss_fn = loss_fn
        self.metric = metric

        if learning_rate is None and learning_rates is None:
            raise ValueError(
                "At least one parameter ('learning_rates' or "
                "'learning_rate') must be provided."
            )

        self.learning_rates = (
            learning_rate if learning_rate is not None else learning_rates
        )

        self.weight_decays = (
            [0.0, 1e-4, 1e-5, 1e-6]
            if weight_decays is None
            else weight_decays
        )

        self.epochs = epochs

    def search(
        self,
        train_loader: DataLoader,
    ) -> list[dict[str, Any]]:
        """Run the grid search over learning rates and weight decays.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing the training data.

        Returns
        -------
        list[dict[str, Any]]
            Results for each learning rate and weight decay
            configuration.
        """
        results = []

        for lr in self.learning_rates:
            for wd in self.weight_decays:

                result = self._train(
                    train_loader,
                    lr,
                    wd,
                )

                results.append(result)

        self._print_summary(results)

        return results

    def _train(
        self,
        train_loader: DataLoader,
        lr: float,
        wd: float,
    ) -> dict[str, float]:
        """Train a model using a specific hyperparameter configuration.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing the training data.
        lr : float
            Learning rate for the optimizer.
        wd : float
            Weight decay for the optimizer.

        Returns
        -------
        dict[str, float]
            Final loss and metric values together with the
            corresponding learning rate and weight decay.
        """
        model = deepcopy(self.model)

        optimizer = self.optimizer(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        trainer = Trainer(
            model,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            metric=self.metric,
        )

        tqdm.write(f"LR={lr} | WD={wd}")

        for epoch in range(self.epochs):
            with tqdm(
                total=1,
                desc=f"Epoch {epoch}",
                unit="batch",
            ) as pbar:

                loss, metric_value = trainer.train_epoch(train_loader)

                pbar.set_postfix({
                    "metric": f"{metric_value:.4f}",
                    "loss": f"{loss:.4f}",
                })

                pbar.update(1)

        tqdm.write("")

        return {
            "learning_rate": lr,
            "weight_decay": wd,
            "loss": loss,
            "metric": metric_value,
        }

    def _print_summary(
        self,
        results: list[dict[str, Any]],
    ) -> None:
        """Print a summary of the grid search results.

        Parameters
        ----------
        results : list[dict[str, Any]]
            Results returned by the grid search.
        """
        table = PrettyTable([
            "Learning Rate",
            "Weight Decay",
            "Metric",
            "Loss",
        ])

        for result in results:
            table.add_row([
                result["learning_rate"],
                result["weight_decay"],
                f"{result['metric']:.4f}",
                f"{result['loss']:.4f}",
            ])

        print("\nFinal Results:")
        print(table)

        best = min(
            results,
            key=lambda x: x["loss"],
        )

        print(
            f"\nBest configuration: "
            f"Learning Rate={best['learning_rate']}, "
            f"Weight Decay={best['weight_decay']} "
            f"(Final loss: {best['loss']:.4f})"
        )