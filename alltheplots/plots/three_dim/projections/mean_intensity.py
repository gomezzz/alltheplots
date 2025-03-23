import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_mean_intensity_plot(tensor_np, ax=None, add_colorbar=True):
    """
    Create a mean intensity projection in the XY plane.

    Parameters:
        tensor_np (numpy.ndarray): The 3D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        add_colorbar (bool): Whether to add a colorbar to the plot. Default is True.

    Returns:
        matplotlib.axes.Axes: The axis with the plot and the image object
    """
    logger.debug("Creating mean intensity projection")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Compute mean intensity projection along Z axis
        mean_proj = np.nanmean(tensor_np, axis=2)

        # Analyze data characteristics for colormap selection
        has_negative = np.any(mean_proj < 0)
        is_diverging = has_negative and np.any(mean_proj > 0)
        dynamic_range = (
            np.ptp(mean_proj[np.isfinite(mean_proj)]) if np.any(np.isfinite(mean_proj)) else 0
        )

        # Select appropriate colormap
        if is_diverging:
            cmap = "RdBu_r"
            logger.debug("Using diverging colormap for positive/negative values")
        else:
            if dynamic_range > 0:
                cmap = "viridis"
            else:
                cmap = "gray"
            logger.debug(f"Using {cmap} colormap based on data characteristics")

        # Create the projection visualization
        im = ax.imshow(mean_proj, cmap=cmap, aspect="equal", interpolation="nearest")

        if add_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set plot labels and title
        ax.set_title("Mean Intensity Projection (XY)")
        ax.set_xlabel("X Index")
        ax.set_ylabel("Y Index")

        # Add grid for better readability if the array is small enough
        if mean_proj.shape[0] * mean_proj.shape[1] <= 400:
            ax.grid(True, which="major", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_xticks(np.arange(-0.5, mean_proj.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, mean_proj.shape[0], 1), minor=True)

    except Exception as e:
        logger.error(f"Failed to create mean intensity projection: {e}")
        ax.text(
            0.5,
            0.5,
            f"Mean Intensity Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Mean Intensity (Error)")

    return ax, im
