# visualization/data.py

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot the correlation heatmap of all numerical features.

    Parameters
    ----------
    df
        Input DataFrame.

    save_path
        Optional path where the figure will be saved.
        If provided, the plot is saved to this location before being displayed.
        If None, the figure is not saved.

    figsize
        Size of the matplotlib figure.

    Returns
    -------
        This function does not return anything.
        It only displays and/or saves the plot.
    """

    corr_matrix = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=figsize)

    image = ax.imshow(
        corr_matrix,
        cmap="coolwarm",
        interpolation="nearest",
    )

    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)

    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_yticklabels(corr_matrix.columns)

    ax.set_title("Correlation Heatmap")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    plt.close(fig)
    