import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_3d_surface_plot(tensor_np, ax=None, axis=2, view_angle=None, add_colorbar=True):
    """
    Create a 3D surface plot with values averaged along one dimension.

    Parameters:
        tensor_np (numpy.ndarray): The 3D numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib 3D axis
        axis (int): The axis along which to average (0=X, 1=Y, 2=Z)
        view_angle (tuple): The elevation and azimuth angles for the 3D view
        add_colorbar (bool): Whether to add a colorbar

    Returns:
        matplotlib.axes.Axes: The 3D axis with the plot
    """
    logger.debug(f"Creating 3D surface plot with averaging along axis {axis}")

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))

    try:
        # Compute the mean along the specified axis
        if axis == 0:
            avg_data = np.nanmean(tensor_np, axis=0)
            title = "3D Surface (Mean along X)"
            xlabel, ylabel = "Z Index", "Y Index"
            x_grid, y_grid = np.meshgrid(
                np.arange(tensor_np.shape[2]), np.arange(tensor_np.shape[1]), indexing="ij"
            )
        elif axis == 1:
            avg_data = np.nanmean(tensor_np, axis=1)
            title = "3D Surface (Mean along Y)"
            xlabel, ylabel = "Z Index", "X Index"
            x_grid, y_grid = np.meshgrid(
                np.arange(tensor_np.shape[2]), np.arange(tensor_np.shape[0]), indexing="ij"
            )
        else:  # axis == 2
            avg_data = np.nanmean(tensor_np, axis=2)
            title = "3D Surface (Mean along Z)"
            xlabel, ylabel = "Y Index", "X Index"
            y_grid, x_grid = np.meshgrid(
                np.arange(tensor_np.shape[1]), np.arange(tensor_np.shape[0]), indexing="ij"
            )

        # Check if data contains valid values
        if not np.any(np.isfinite(avg_data)):
            raise ValueError("No finite values in averaged data")

        # Analyze data characteristics for colormap selection
        has_negative = np.any(avg_data < 0)
        is_diverging = has_negative and np.any(avg_data > 0)

        # Select appropriate colormap
        if is_diverging:
            cmap = "RdBu_r"
            logger.debug("Using diverging colormap for positive/negative values")
        else:
            cmap = "viridis"
            logger.debug("Using viridis colormap")

        # Create the surface plot
        surf = ax.plot_surface(
            x_grid, y_grid, avg_data.T, cmap=cmap, linewidth=0.5, antialiased=True, alpha=0.8
        )

        if add_colorbar:
            plt.colorbar(surf, ax=ax, shrink=0.7, aspect=20, pad=0.1, label="Value")

        # Set plot labels and title
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel("Value")

        # Set view angle if provided, otherwise use default
        if view_angle is not None:
            elev, azim = view_angle
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.view_init(elev=30, azim=-60)

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
        ax.set_title("3D Surface (Error)")

    return ax
