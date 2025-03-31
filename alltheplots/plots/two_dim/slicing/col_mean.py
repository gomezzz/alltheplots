import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_col_mean_plot(tensor_np, ax=None):
    """
    Create a plot showing the mean value of each column in the 2D array.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating column mean plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Calculate mean of each column, ignoring NaN values
        col_means = np.nanmean(tensor_np, axis=0)

        # Calculate confidence intervals (std error of mean)
        col_stds = np.nanstd(tensor_np, axis=0)
        n_samples = np.sum(np.isfinite(tensor_np), axis=0)
        confidence_intervals = 1.96 * col_stds / np.sqrt(n_samples)

        # Plot mean line
        ax.plot(np.arange(len(col_means)), col_means, "b-", linewidth=1)

        # Add confidence interval
        ax.fill_between(
            np.arange(len(col_means)),
            col_means - confidence_intervals,
            col_means + confidence_intervals,
            alpha=0.2,
        )

        # Add individual points for small arrays
        if tensor_np.shape[0] <= 30:
            for i in range(tensor_np.shape[1]):
                col_data = tensor_np[np.isfinite(tensor_np[:, i]), i]
                if len(col_data) > 0:
                    ax.plot([i] * len(col_data), col_data, "o", alpha=0.2, markersize=2)

        # Set plot labels and title
        ax.set_title("Column Means")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Value")

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Show standard deviation in legend
        mean_std = np.nanmean(col_stds)
        ax.text(
            0.02,
            0.98,
            f"Mean σ: {mean_std:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    except Exception as e:
        logger.error(f"Failed to create column mean plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Column Mean Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Column Mean (Error)")

    return ax
