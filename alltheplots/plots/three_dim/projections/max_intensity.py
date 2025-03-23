import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_max_intensity_plot(tensor_np, ax=None, add_colorbar=True):
    """
    Create a maximum intensity projection in the XY plane.

    Parameters:
        tensor_np (numpy.ndarray): The 3D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        add_colorbar (bool): Whether to add a colorbar to the plot. Default is True.

    Returns:
        matplotlib.axes.Axes: The axis with the plot and the image object
    """
    logger.debug("Creating maximum intensity projection")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Compute maximum intensity projection along Z axis
        max_proj = np.nanmax(tensor_np, axis=2)

        # Analyze data characteristics for colormap selection
        has_negative = np.any(max_proj < 0)
        is_diverging = has_negative and np.any(max_proj > 0)
        dynamic_range = (
            np.ptp(max_proj[np.isfinite(max_proj)]) if np.any(np.isfinite(max_proj)) else 0
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
        im = ax.imshow(max_proj, cmap=cmap, aspect="equal", interpolation="nearest")

        if add_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set plot labels and title
        ax.set_title("Maximum Intensity Projection (XY)")
        ax.set_xlabel("X Index")
        ax.set_ylabel("Y Index")

        # Add grid for better readability if the array is small enough
        if max_proj.shape[0] * max_proj.shape[1] <= 400:
            ax.grid(True, which="major", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_xticks(np.arange(-0.5, max_proj.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, max_proj.shape[0], 1), minor=True)

    except Exception as e:
        logger.error(f"Failed to create maximum intensity projection: {e}")
        ax.text(
            0.5,
            0.5,
            f"Max Intensity Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Max Intensity (Error)")

    return ax, im
