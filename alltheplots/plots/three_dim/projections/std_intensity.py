import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_std_intensity_plot(tensor_np, ax=None, add_colorbar=True):
    """
    Create a standard deviation projection in the XY plane.

    Parameters:
        tensor_np (numpy.ndarray): The 3D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        add_colorbar (bool): Whether to add a colorbar to the plot. Default is True.

    Returns:
        matplotlib.axes.Axes: The axis with the plot and the image object
    """
    logger.debug("Creating standard deviation projection")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Compute standard deviation projection along Z axis
        std_proj = np.nanstd(tensor_np, axis=2)

        # Create the projection visualization with viridis colormap (std is always positive)
        im = ax.imshow(std_proj, cmap="viridis", aspect="equal", interpolation="nearest")

        if add_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set plot labels and title
        ax.set_title("Standard Deviation Projection (XY)")
        ax.set_xlabel("X Index")
        ax.set_ylabel("Y Index")

        # Add grid for better readability if the array is small enough
        if std_proj.shape[0] * std_proj.shape[1] <= 400:
            ax.grid(True, which="major", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_xticks(np.arange(-0.5, std_proj.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, std_proj.shape[0], 1), minor=True)

        # Add text with mean standard deviation
        mean_std = np.nanmean(std_proj)
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
        logger.error(f"Failed to create standard deviation projection: {e}")
        ax.text(
            0.5,
            0.5,
            f"Std Intensity Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Std Intensity (Error)")

    return ax, im
