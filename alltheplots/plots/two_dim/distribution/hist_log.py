import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_hist_log_plot(tensor_np, ax=None):
    """
    Create a histogram with logarithmic scale for better visualization of wide-ranging values.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating log-scale histogram")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Flatten the array and handle non-positive values
        flat_data = tensor_np.flatten()

        # Remove any non-finite values
        clean_data = flat_data[np.isfinite(flat_data)]

        if len(clean_data) == 0:
            raise ValueError("No finite values in data")

        # Handle non-positive values for log scale
        min_val = np.min(clean_data)
        if min_val <= 0:
            # Add offset to make all values positive
            offset = abs(min_val) + 1e-10
            log_data = np.log10(clean_data + offset)
            logger.debug(f"Added offset of {offset} to handle non-positive values")
        else:
            log_data = np.log10(clean_data)
            offset = 0

        # Create histogram with automatically determined bins
        n_bins = min(50, max(20, int(np.sqrt(len(log_data)))))
        sns.histplot(data=log_data, ax=ax, bins=n_bins, kde=True, stat="density", alpha=0.6)

        # Set custom x-ticks to show actual values
        log_ticks = ax.get_xticks()
        if offset > 0:
            tick_labels = [f"{10**x - offset:.1e}" for x in log_ticks]
        else:
            tick_labels = [f"{10**x:.1e}" for x in log_ticks]

        ax.set_xticklabels(tick_labels)

        # Set plot labels and title
        ax.set_title("Value Distribution (Log Scale)")
        ax.set_xlabel("Value (log scale)")
        ax.set_ylabel("Density")

        # Add note about offset if used
        if offset > 0:
            ax.text(
                0.02,
                0.98,
                f"Offset: +{offset:.1e}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        # Add grid for better readability
        ax.grid(True, alpha=0.5)

    except Exception as e:
        logger.error(f"Failed to create log-scale histogram: {e}")
        ax.text(
            0.5,
            0.5,
            f"Log Histogram Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Log Histogram (Error)")

    return ax
