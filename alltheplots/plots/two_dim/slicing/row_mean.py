import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_row_mean_plot(tensor_np, ax=None):
    """
    Create a plot showing the mean value of each row in the 2D array.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating row mean plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Calculate mean of each row, ignoring NaN values
        row_means = np.nanmean(tensor_np, axis=1)

        # Calculate confidence intervals (std error of mean)
        row_stds = np.nanstd(tensor_np, axis=1)
        n_samples = np.sum(np.isfinite(tensor_np), axis=1)
        confidence_intervals = 1.96 * row_stds / np.sqrt(n_samples)

        # Plot mean line
        ax.plot(row_means, np.arange(len(row_means)), "b-", linewidth=1)

        # Add confidence interval
        ax.fill_betweenx(
            np.arange(len(row_means)),
            row_means - confidence_intervals,
            row_means + confidence_intervals,
            alpha=0.2,
        )

        # Add individual points for small arrays
        if tensor_np.shape[1] <= 30:
            for i in range(tensor_np.shape[0]):
                row_data = tensor_np[i, np.isfinite(tensor_np[i, :])]
                if len(row_data) > 0:
                    ax.plot(row_data, [i] * len(row_data), "o", alpha=0.2, markersize=2)

        # Set plot labels and title
        ax.set_title("Row Means")
        ax.set_xlabel("Value")
        ax.set_ylabel("Row Index")

        # Invert y-axis to match image orientation
        ax.invert_yaxis()

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Show standard deviation in legend
        mean_std = np.nanmean(row_stds)
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
        logger.error(f"Failed to create row mean plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Row Mean Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Row Mean (Error)")

    return ax
