import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_kde_plot_nd(tensor_np, ax=None):
    """
    Create a Kernel Density Estimation (KDE) plot for n-dimensional array values.

    Parameters:
        tensor_np (numpy.ndarray): The n-dimensional numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating KDE plot for n-dimensional data")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Flatten the array for KDE
        flat_data = tensor_np.flatten()
        clean_data = flat_data[np.isfinite(flat_data)]

        if len(clean_data) == 0:
            raise ValueError("No finite values in data")

        if len(clean_data) < 3:
            raise ValueError("Too few data points for KDE (need at least 3)")

        # Create KDE plot using seaborn
        sns.kdeplot(data=clean_data, ax=ax, fill=True, alpha=0.5, linewidth=2)

        # Add rug plot at the bottom to show actual data distribution
        if len(clean_data) < 1000:  # Only for smaller datasets to avoid visual clutter
            sns.rugplot(data=clean_data, ax=ax, color="red", alpha=0.3)

        # Set plot labels and title
        ax.set_title(f"Density Estimation\n{tensor_np.shape} Array")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        # Add summary statistics
        stats_text = (
            f"Mean: {np.mean(clean_data):.2f}\n"
            f"Std: {np.std(clean_data):.2f}\n"
            f"N: {len(clean_data)}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create KDE plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"KDE Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("KDE (Error)")

    return ax
