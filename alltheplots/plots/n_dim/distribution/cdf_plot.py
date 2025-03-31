import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_cdf_plot_nd(tensor_np, ax=None):
    """
    Create a cumulative distribution function (CDF) plot for n-dimensional array values.

    Parameters:
        tensor_np (numpy.ndarray): The n-dimensional numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating CDF plot for n-dimensional data")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Flatten the array for CDF
        flat_data = tensor_np.flatten()
        clean_data = flat_data[np.isfinite(flat_data)]

        if len(clean_data) == 0:
            raise ValueError("No finite values in data")

        # Sort data for CDF
        sorted_data = np.sort(clean_data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        # Plot the CDF
        ax.plot(sorted_data, y, marker=".", linestyle="none", markersize=2, alpha=0.4)

        # Connect the dots with thinner lines
        ax.plot(sorted_data, y, linewidth=0.8, alpha=0.7)

        # Add horizontal lines at quartiles and annotate their values
        for q in [0.25, 0.5, 0.75]:
            ax.axhline(y=q, color="r", linestyle="--", alpha=0.2, linewidth=0.5)
            q_idx = int(q * len(sorted_data))
            if q_idx < len(sorted_data):
                q_val = sorted_data[q_idx]
                ax.annotate(
                    f"{q:.2f}: {q_val:.2f}",
                    xy=(q_val, q),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=6,
                )

        # Set plot labels and title
        ax.set_title(f"Cumulative Distribution\n{tensor_np.shape} Array")
        ax.set_xlabel("Value")
        ax.set_ylabel("Cumulative Probability")

        # Set y-axis limits
        ax.set_ylim(0, 1.05)

        # Add grid for better readability
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
