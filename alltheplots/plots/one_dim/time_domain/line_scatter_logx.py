import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_line_scatter_logx_plot(tensor_np, ax=None, is_shared_x=False):
    """
    Create a time-domain line or scatter plot with logarithmic x-axis scale.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to plot
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        is_shared_x (bool): Whether this plot shares x-axis with other plots.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating time-domain plot with log x-axis")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    try:
        # Create a valid x-axis for log scale (can't have zeros)
        x = np.arange(1, len(tensor_np) + 1)  # Start from 1 to avoid log(0)

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
                f"Using scatter plot (log x-axis) (points: {n_points}, unique values: {n_unique})"
            )
            ax.scatter(x, tensor_np, s=3, alpha=0.7)
        else:
            logger.debug("Using line plot for time-domain visualization (log x-axis)")
            ax.plot(x, tensor_np, linewidth=0.8)

        # Set log scale for x-axis
        ax.set_xscale("log")

        # Set plot labels
        ax.set_title("Time Domain (Log X)")
        ax.set_xlabel("Index (Log Scale)" if not is_shared_x else "")
        ax.set_ylabel("Value")

    except Exception as e:
        logger.error(f"Failed to create time-domain plot with log x-axis: {e}")
        ax.text(
            0.5,
            0.5,
            f"Time Domain (Log X) Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Time Domain (Log X) (Error)")

    return ax
