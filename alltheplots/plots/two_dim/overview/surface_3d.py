import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_surface_3d_plot(tensor_np, ax=None):
    """
    Create a 3D surface plot of 2D data with intelligent view angle selection.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib 3D axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 3D surface plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))

    try:
        # Create coordinate grids for the surface plot
        Y, X = np.mgrid[: tensor_np.shape[0], : tensor_np.shape[1]]

        # Create the surface plot
        surf = ax.plot_surface(X, Y, tensor_np, cmap="viridis", linewidth=0.5, antialiased=True)

        # Add a color bar
        plt.colorbar(surf, ax=ax, fraction=0.046, pad=0.04)

        # Analyze data characteristics for optimal viewing angle
        aspect_ratio = tensor_np.shape[1] / tensor_np.shape[0]
        value_range = (
            np.ptp(tensor_np[np.isfinite(tensor_np)]) if np.any(np.isfinite(tensor_np)) else 1
        )
        data_aspect = value_range / max(tensor_np.shape)

        # Adjust the view angle based on data characteristics
        if aspect_ratio > 2 or aspect_ratio < 0.5:
            # For very rectangular data, view from the longer side
            elev = 20
            azim = 45 if aspect_ratio > 1 else -45
        else:
            # For more square data, use standard angles
            elev = 30
            azim = -60

        ax.view_init(elev=elev, azim=azim)

        # Set reasonable axis limits
        ax.set_zlim(np.nanmin(tensor_np), np.nanmax(tensor_np))

        # Set plot labels and title
        ax.set_title("3D Surface Plot")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        ax.set_zlabel("Value")

        # Add grid lines for better depth perception
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create 3D surface plot: {e}")
        # For 3D plots, error text needs special handling
        ax.text2D(
            0.5,
            0.5,
            f"3D Surface Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("3D Surface Plot (Error)")

    return ax
