import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_std_projection_plot(tensor_np, ax=None, add_colorbar=True):
    """
    Create a heatmap of the standard deviation projection across all dimensions.

    Parameters:
        tensor_np (numpy.ndarray): The n-dimensional numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on
        add_colorbar (bool): Whether to add a colorbar to the plot

    Returns:
        matplotlib.axes.Axes: The axis with the plot
        matplotlib.image.AxesImage: The image object for colorbar creation
    """
    logger.debug("Creating standard deviation projection plot")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Get the shape of the tensor
        tensor_shape = tensor_np.shape

        if len(tensor_shape) <= 2:
            # If tensor is already 2D or less, we can't compute std projection
            # Show zeros with a message
            projection = np.zeros_like(tensor_np)
            ax.text(
                0.5,
                0.5,
                "Cannot compute std dev\nfor 2D or lower data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            title = "Standard Deviation (Not Applicable)"
        else:
            # For higher dimensions, compute std along all but first two dimensions
            axes_to_std = tuple(range(2, len(tensor_shape)))
            projection = np.nanstd(tensor_np, axis=axes_to_std)
            title = f"Std Dev Projection ({len(tensor_shape)}D → 2D)"

        # Check if projection contains valid data
        if not np.any(np.isfinite(projection)):
            raise ValueError("No finite values in projected data")

        # Create the heatmap (use plasma colormap which is good for uncertainty/variance)
        im = ax.imshow(projection, cmap="plasma", aspect="auto", interpolation="nearest")

        # Add colorbar if requested
        if add_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 0")

        # Add grid for better readability if the array is small enough
        if projection.shape[0] * projection.shape[1] <= 400:  # Arbitrary threshold
            ax.grid(True, which="major", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_xticks(np.arange(-0.5, projection.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, projection.shape[0], 1), minor=True)

    except Exception as e:
        logger.error(f"Failed to create std projection: {e}")
        ax.text(
            0.5,
            0.5,
            f"Std Dev Projection Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Std Dev Projection (Error)")
        im = None

    return ax, im
