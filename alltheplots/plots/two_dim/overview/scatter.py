import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_scatter_plot(tensor_np, ax=None):
    """
    Create a scatter plot visualization of 2D data, showing data points as pixels.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating scatter plot visualization")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Create meshgrid for pixel coordinates
        y, x = np.mgrid[: tensor_np.shape[0], : tensor_np.shape[1]]

        # Flatten arrays for scatter plot
        x_flat = x.flatten()
        y_flat = y.flatten()
        values_flat = tensor_np.flatten()

        # Remove any non-finite values
        mask = np.isfinite(values_flat)
        x_clean = x_flat[mask]
        y_clean = y_flat[mask]
        values_clean = values_flat[mask]

        # Create scatter plot with pixel values determining color
        scatter = ax.scatter(
            x_clean,
            y_clean,
            c=values_clean,
            cmap="viridis",
            alpha=0.6,
            s=max(1, 500 / max(tensor_np.shape)),  # Adjust point size based on array dimensions
            marker="s",  # Square markers for pixel-like appearance
        )

        # Add colorbar
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

        # Set plot labels and title
        ax.set_title("Scatter Plot")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")

        # Invert y-axis to match image orientation
        ax.invert_yaxis()

        # Set aspect ratio to be equal
        ax.set_aspect("equal")

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Set reasonable axis limits
        ax.set_xlim(-0.5, tensor_np.shape[1] - 0.5)
        ax.set_ylim(tensor_np.shape[0] - 0.5, -0.5)

    except Exception as e:
        logger.error(f"Failed to create scatter plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Scatter Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Scatter Plot (Error)")

    return ax
