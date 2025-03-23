import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_line_scatter_logy_plot(tensor_np, ax=None, is_shared_x=False):
    """
    Create a time-domain line or scatter plot with logarithmic y-axis scale.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to plot
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        is_shared_x (bool): Whether this plot shares x-axis with other plots.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating time-domain plot with log y-axis")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    try:
        # Check if data is compatible with log scale (all positive)
        if np.any(tensor_np <= 0):
            # Offset data to make it positive for log scale
            offset = abs(min(0, np.min(tensor_np))) + 1e-10
            plot_data = tensor_np + offset
            logger.info(f"Data contains non-positive values, offsetting by {offset} for log scale")
        else:
            plot_data = tensor_np

        # Determine if we should use scatter instead of line plot
        n_points = len(tensor_np)
        unique_values = np.unique(tensor_np)
        n_unique = len(unique_values)
        zero_crossings = np.where(np.diff(np.signbit(tensor_np - np.mean(tensor_np))))[0]
        n_crossings = len(zero_crossings)

        # Use scatter if:
        # 1. Few data points (<= 50)
        # 2. Highly discrete (few unique values relative to total points)
        # 3. Many zero crossings (high frequency data)
        use_scatter = (
            n_points <= 50 or n_unique <= min(20, n_points / 5) or n_crossings > n_points / 10
        )

        # Create the appropriate plot
        if use_scatter:
            logger.debug(
                f"Using scatter plot (log y-axis) (points: {n_points}, unique values: {n_unique})"
            )
            ax.scatter(np.arange(len(plot_data)), plot_data, s=3, alpha=0.7)
        else:
            logger.debug("Using line plot for time-domain visualization (log y-axis)")
            ax.plot(np.arange(len(plot_data)), plot_data, linewidth=0.8)

        # Set log scale for y-axis
        ax.set_yscale("log")

        # Set plot labels
        ax.set_title("Time Domain (Log Y)")
        ax.set_xlabel("Index" if not is_shared_x else "")
        ax.set_ylabel("Value (Log Scale)")

    except Exception as e:
        logger.error(f"Failed to create time-domain plot with log y-axis: {e}")
        ax.text(
            0.5,
            0.5,
            f"Time Domain (Log Y) Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Time Domain (Log Y) (Error)")

    return ax
