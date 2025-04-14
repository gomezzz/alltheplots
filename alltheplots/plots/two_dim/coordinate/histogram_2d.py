import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_histogram_2d_plot(tensor_np, ax=None):
    """
    Create a 2D histogram plot for Nx2 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx2 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 2D histogram plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Extract x and y coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]

        # Handle very small datasets
        if len(x) < 10:
            # Just show scatter for small datasets
            ax.scatter(x, y, alpha=0.7)
            ax.text(
                0.5,
                0.95,
                "Insufficient data for 2D histogram\nshowing scatter instead",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )
        else:
            # Determine optimal number of bins
            # Use Freedman-Diaconis rule to estimate bin width
            def freedman_diaconis_bins(data):
                # IQR-based bin width estimation
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                bin_width = 2 * iqr / (len(data) ** (1 / 3)) if iqr > 0 else 1
                # Calculate number of bins based on range and width
                data_range = np.max(data) - np.min(data)
                num_bins = int(np.ceil(data_range / bin_width)) if bin_width > 0 else 10
                return max(5, min(50, num_bins))  # Constrain between 5 and 50 bins

            # Calculate bin counts for x and y separately
            if len(np.unique(x)) > 5 and len(np.unique(y)) > 5:
                # Use adaptive binning for continuous data
                bins_x = freedman_diaconis_bins(x)
                bins_y = freedman_diaconis_bins(y)
            else:
                # For more discrete data, use unique values plus padding
                bins_x = len(np.unique(x)) + 1
                bins_y = len(np.unique(y)) + 1

            # Create the 2D histogram
            hist, xedges, yedges, im = ax.hist2d(
                x,
                y,
                bins=[bins_x, bins_y],
                cmap="viridis",
                alpha=0.8,
                density=False,  # Show actual counts
            )

            # Add colorbar
            plt.colorbar(im, ax=ax, label="Count")

            # For smaller datasets, overlay scatter points for additional clarity
            if len(x) < 200:
                ax.scatter(x, y, color="red", alpha=0.3, s=10, marker=".", edgecolor="none")
                logger.debug("Added scatter overlay for small dataset")

            # Add count statistics
            total_count = np.sum(hist)
            max_count = np.max(hist)
            stats_text = f"Total: {total_count}\nMax bin: {max_count}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

        # Set plot title and labels
        ax.set_title("2D Histogram")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    except Exception as e:
        logger.error(f"Failed to create 2D histogram plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"2D Histogram Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("2D Histogram (Error)")

    return ax
