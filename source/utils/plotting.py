"""Plotting functions."""
import matplotlib.pyplot as plt
import pandas as pd

from source.constants import FIGURE_LOCATION

# from mpl_toolkits.mplot3d import Axes3D


def plot_3d(data: pd.DataFrame, targets: pd.Series, save_name: str) -> None:
    """Plot data in three dimensions.

    Args:
        data (pd.DataFrame): Data to be plotted.
        targets (pd.Series): Targets of data.
        save_name (str): Name of saved image.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax_names = data.columns

    for target in set(targets):
        indices = targets == target
        indices.reset_index(drop=True, inplace=True)
        ax.scatter(
            data.loc[indices, ax_names[0]],
            data.loc[indices, ax_names[1]],
            data.loc[indices, ax_names[2]],
            label=target,
        )

    ax.set_xlabel(ax_names[0])
    ax.set_ylabel(ax_names[1])
    ax.set_zlabel(ax_names[2])
    ax.legend()

    plt.savefig(FIGURE_LOCATION + save_name)
