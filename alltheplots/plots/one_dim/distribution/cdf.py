import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_cdf_plot(tensor_np, ax=None, is_shared_x=False):
    """
    Create a Cumulative Distribution Function (CDF) plot for 1D data.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to create CDF for
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        is_shared_x (bool): Whether this plot shares x-axis with other plots.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating CDF plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    try:
        # Get sorted data for CDF calculation
        sorted_data = np.sort(tensor_np)

        # Calculate empirical CDF (ECDF)
        # y-axis goes from 0 to 1 representing the cumulative probability
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        # Use smaller markers for scatter plot
        ax.plot(sorted_data, y, marker=".", linestyle="none", markersize=2, alpha=0.4)

        # Connect the dots with thinner lines
        ax.plot(sorted_data, y, linewidth=0.8, alpha=0.7)

        # Add horizontal lines at 0.25, 0.5, and 0.75 for quartile reference
        # Use thinner lines with less opacity
        for q in [0.25, 0.5, 0.75]:
            ax.axhline(y=q, color="r", linestyle="--", alpha=0.2, linewidth=0.5)
            # Find the x value at this quantile
            q_idx = int(q * len(sorted_data))
            if q_idx < len(sorted_data):
                q_val = sorted_data[q_idx]
                # Use smaller font for annotations
                ax.annotate(
                    f"{q:.2f}: {q_val:.2f}",
                    xy=(q_val, q),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=6,
                )

        # Set plot labels
        ax.set_title("Cumulative \n Distribution Function")
        ax.set_xlabel("Value" if not is_shared_x else "")
        ax.set_ylabel("Cumulative Probability")

        # Set y-axis limits
        ax.set_ylim(0, 1.05)

        # Add grid for better readability, but with less visual weight
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    except Exception as e:
        logger.error(f"Failed to create CDF plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"CDF Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("CDF (Error)")

    return ax
