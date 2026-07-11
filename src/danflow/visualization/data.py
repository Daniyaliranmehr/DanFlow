# visualization/data.py

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import math


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


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 50,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """
    Plot the histogram of a selected feature.

    Parameters
    ----------
    df
        Input DataFrame.

    column
        Name of the column to visualize.

    bins
        Number of histogram bins.

    save_path
        Optional path where the figure will be saved.
        - If a filename is provided (e.g., ``"plots/histogram.png"``), the
        figure is saved using that filename.
        - If a directory is provided (e.g., ``"plots/"`` or ``"plots"``), the
        figure is saved in that directory using an automatically generated
        filename.
        - If ``None``, the figure is not saved.

    figsize
        Size of the matplotlib figure.
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(df[column], bins=bins)

    ax.set_title(f"{column} Distribution")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)

        # Case 1: Existing directory
        if save_path.exists() and save_path.is_dir():
            save_path = save_path / f"{column}_histogram.png"

        # Case 2: Path has no extension -> treat as directory
        elif save_path.suffix == "":
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / f"{column}_histogram.png"

        # Case 3: Path is a filename
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    plt.close(fig)


def plot_multi_histograms(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
    bins: int = 50,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """
    Plot histograms for multiple features.

    Parameters
    ----------
    df
        Input DataFrame.

    columns
        List of column names to visualize.

    name
        Figure title.

    bins
        Number of histogram bins.

    save_path
        Optional path where the figure will be saved.
        If provided, the plot is saved to this location before being displayed.
        If None, the figure is not saved.

    figsize
        Size of the matplotlib figure.
        If None, the figure size is determined automatically.
    """

    n_cols = 2
    n_rows = math.ceil(len(columns) / n_cols)

    if figsize is None:
        figsize = (12, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
    )

    if len(columns) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, column in enumerate(columns):
        axes[i].hist(df[column], bins=bins)

        axes[i].set_title(f"{column} Distribution")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Frequency")

    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(name, fontsize=16)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    plt.close(fig)


def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """
    Plot a boxplot for a selected feature and optionally save it.

    Parameters
    ----------
    df
        Input dataframe.

    column
        Name of the column to visualize.

    save_path
        Optional path where the figure will be saved.
        If provided, the plot is saved before being displayed.
        If None, the figure is not saved.

    figsize
        Size of the matplotlib figure.
    """

    fig, ax = plt.subplots(figsize=figsize)

    df.boxplot(column=column, ax=ax)

    ax.set_title(f"{column} Boxplot")
    ax.set_ylabel(column)
    ax.set_xticks([])

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    plt.close(fig)


def plot_multi_boxplots(
    df: pd.DataFrame,
    columns: list[str],
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """
    Plot boxplots for multiple features and optionally save the figure.

    Parameters
    ----------
    df
        Input dataframe.

    columns
        List of column names to visualize.

    save_path
        Optional path where the figure will be saved.
        If provided, the plot is saved before being displayed.
        If None, the figure is not saved.

    figsize
        Base size of the matplotlib figure. The height is automatically
        scaled according to the number of subplot rows.
    """

    n_cols = 2
    n_rows = math.ceil(len(columns) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0], figsize[1] * n_rows),
    )

    if len(columns) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, column in enumerate(columns):
        df.boxplot(column=column, ax=axes[i])

        axes[i].set_title(f"{column} Boxplot")
        axes[i].set_ylabel(column)
        axes[i].set_xticks([])

    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    plt.close(fig)
    