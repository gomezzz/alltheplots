import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_yz_slice_plot(tensor_np, ax=None, add_colorbar=True):
    """
    Create a central YZ slice visualization of 3D data.

    Parameters:
        tensor_np (numpy.ndarray): The 3D numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        add_colorbar (bool): Whether to add a colorbar to the plot. Default is True.

    Returns:
        matplotlib.axes.Axes: The axis with the plot and the image object
    """
    logger.debug("Creating YZ slice visualization")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Get central slice
        x_center = tensor_np.shape[0] // 2
        yz_slice = tensor_np[x_center, :, :]

        # Analyze data characteristics for colormap selection
        has_negative = np.any(yz_slice < 0)
        is_diverging = has_negative and np.any(yz_slice > 0)
        dynamic_range = (
            np.ptp(yz_slice[np.isfinite(yz_slice)]) if np.any(np.isfinite(yz_slice)) else 0
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

        # Create the slice visualization
        im = ax.imshow(yz_slice, cmap=cmap, aspect="equal", interpolation="nearest")

        if add_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set plot labels and title
        ax.set_title("Central YZ Slice (x={})".format(x_center))
        ax.set_xlabel("Z Index")
        ax.set_ylabel("Y Index")

        # Add grid for better readability if the array is small enough
        if yz_slice.shape[0] * yz_slice.shape[1] <= 400:  # Arbitrary threshold
            ax.grid(True, which="major", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_xticks(np.arange(-0.5, yz_slice.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, yz_slice.shape[0], 1), minor=True)

    except Exception as e:
        logger.error(f"Failed to create YZ slice: {e}")
        ax.text(
            0.5,
            0.5,
            f"YZ Slice Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("YZ Slice (Error)")

    return ax, im
