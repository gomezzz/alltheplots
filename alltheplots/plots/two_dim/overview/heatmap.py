import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_heatmap_plot(tensor_np, ax=None):
    """
    Create a heatmap visualization of 2D data with smart colormap selection.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating heatmap visualization")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Analyze data characteristics for colormap selection
        has_negative = np.any(tensor_np < 0)
        is_diverging = has_negative and np.any(tensor_np > 0)
        dynamic_range = (
            np.ptp(tensor_np[np.isfinite(tensor_np)]) if np.any(np.isfinite(tensor_np)) else 0
        )

        # Select appropriate colormap
        if is_diverging:
            cmap = "RdBu_r"  # Red-Blue diverging colormap
            logger.debug("Using diverging colormap for positive/negative values")
        else:
            if dynamic_range > 0:
                cmap = "viridis"  # General-purpose perceptually uniform colormap
            else:
                cmap = "gray"  # Grayscale for constant data
            logger.debug(f"Using {cmap} colormap based on data characteristics")

        # Create the heatmap
        im = ax.imshow(tensor_np, cmap=cmap, aspect="equal", interpolation="nearest")

        # Add colorbar with appropriate size and layout
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set plot labels and title
        ax.set_title("Heatmap Visualization")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")

        # Add grid for better readability if the array is small enough
        if tensor_np.shape[0] * tensor_np.shape[1] <= 400:  # Arbitrary threshold
            ax.grid(True, which="major", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_xticks(np.arange(-0.5, tensor_np.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, tensor_np.shape[0], 1), minor=True)

    except Exception as e:
        logger.error(f"Failed to create heatmap: {e}")
        ax.text(
            0.5,
            0.5,
            f"Heatmap Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Heatmap (Error)")

    return ax
