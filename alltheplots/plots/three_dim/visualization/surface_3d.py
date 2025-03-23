import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_3d_surface_plot(
    tensor_np,
    ax=None,
    axis=2,
    view_angle=None,
    add_colorbar=False,  # default to False if you prefer a shared colorbar
):
    """
    Create a 3D surface plot with values averaged along one dimension.
    Applies orthographic projection (if available), auto aspect, and smaller label pads
    to reduce whitespace around the subplot. Additionally, this version repositions
    the vertical (Y) axis label to appear further to the left.

    Parameters:
        tensor_np (numpy.ndarray): The 3D numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib 3D axis
        axis (int): The axis along which to average (0=X, 1=Y, 2=Z)
        view_angle (tuple): The elevation and azimuth angles for the 3D view
        add_colorbar (bool): Whether to add a colorbar (inside this axis)

    Returns:
        matplotlib.axes.Axes: The 3D axis with the plot
    """
    logger.debug(f"Creating 3D surface plot with averaging along axis {axis}")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    try:
        # Compute the mean along the specified axis
        if axis == 0:
            avg_data = np.nanmean(tensor_np, axis=0)
            title = "Surface (Mean X)"
            xlabel, ylabel = "Z Index", "Y Index"
            x_grid, y_grid = np.meshgrid(
                np.arange(tensor_np.shape[2]), np.arange(tensor_np.shape[1]), indexing="ij"
            )
        elif axis == 1:
            avg_data = np.nanmean(tensor_np, axis=1)
            title = "Surface (Mean Y)"
            xlabel, ylabel = "Z Index", "X Index"
            x_grid, y_grid = np.meshgrid(
                np.arange(tensor_np.shape[2]), np.arange(tensor_np.shape[0]), indexing="ij"
            )
        else:  # axis == 2
            avg_data = np.nanmean(tensor_np, axis=2)
            title = "Surface (Mean Z)"
            xlabel, ylabel = "Y Index", "X Index"
            y_grid, x_grid = np.meshgrid(
                np.arange(tensor_np.shape[1]), np.arange(tensor_np.shape[0]), indexing="ij"
            )

        if not np.any(np.isfinite(avg_data)):
            raise ValueError("No finite values in averaged data")

        # Choose colormap
        has_negative = np.any(avg_data < 0)
        is_diverging = has_negative and np.any(avg_data > 0)
        cmap = "RdBu_r" if is_diverging else "viridis"

        # Plot surface
        surf = ax.plot_surface(
            x_grid, y_grid, avg_data.T, cmap=cmap, linewidth=0.3, antialiased=True, alpha=0.8
        )

        if add_colorbar:
            plt.colorbar(surf, ax=ax, shrink=0.7, aspect=20, pad=0.1, label="Value")

        ax.set_title(title, pad=6)
        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_zlabel("Value", labelpad=6)

        # Switch to orthographic projection if available (Matplotlib ≥ 3.2)
        try:
            ax.set_proj_type("ortho")
        except NotImplementedError:
            pass

        ax.set_aspect("auto", adjustable="box")

        if view_angle is not None:
            elev, azim = view_angle
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.view_init(elev=30, azim=60)

        ax.set_zlabel("")

        ax.text2D(
            -0.1,  # a negative X to push it left of the axis
            0.5,  # halfway up the subplot
            "Value",  # your label text
            transform=ax.transAxes,
            rotation=90,  # rotate so it’s vertical
            va="center",
            ha="center",
        )

    except Exception as e:
        logger.error(f"Failed to create 3D surface plot: {e}")
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
