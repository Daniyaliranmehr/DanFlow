# visualization/visualizer.py

from typing import Dict, List, Optional
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    history: Dict[str, List[float]],
    name: str,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """
    Plot training and validation loss and an optional metric over epochs.

    Parameters
    ----------
    history
        Dictionary returned by Trainer.fit().
        Expected keys:
        - "train_loss"
        - "valid_loss"
        - "train_metric" (optional)
        - "valid_metric" (optional)
        - "metric_name" (optional)

    name
        Name of the experiment or model (used in title).

    save_path
        Optional path where the figure will be saved.
        If provided, the plot is saved to this location before being displayed.
        If None, the figure is not saved.

    figsize
        Size of the matplotlib figure.

    Returns
    -------
        This function does not return anything. It only displays and/or saves the plot.
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(
        epochs,
        history["train_loss"],
        color="tab:blue",
        linewidth=2,
        linestyle="-",
        label="Train Loss",
    )

    ax1.plot(
        epochs,
        history["valid_loss"],
        color="#00d9ff",
        linewidth=2,
        linestyle="--",
        label="Validation Loss",
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training History - {name}")
    ax1.grid(True, linestyle="--", alpha=0.5)

    lines, labels = ax1.get_legend_handles_labels()

    metric_name = history.get("metric_name")

    if (
        metric_name is not None
        and "train_metric" in history
        and "valid_metric" in history
    ):
        ax2 = ax1.twinx()

        ax2.plot(
            epochs,
            history["train_metric"],
            color="#8400ff",
            linewidth=2,
            linestyle="-",
            label=f"Train {metric_name}",
        )

        ax2.plot(
            epochs,
            history["valid_metric"],
            color="#a373c7",
            linewidth=2,
            linestyle="--",
            label=f"Valid {metric_name}",
        )

        ax2.set_ylabel(metric_name)

        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2

    fig.legend(
        lines,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.86, 0.91),
        frameon=True,
    )

    fig.subplots_adjust(right=0.80)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)