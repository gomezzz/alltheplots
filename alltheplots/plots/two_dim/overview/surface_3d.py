import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_surface_3d_plot(tensor_np, ax=None, view_angle=None):
    """
    Create a 3D surface plot of 2D data with specified view angle.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib 3D axis to plot on. If None, a new one is created.
        view_angle (tuple, optional): The (elevation, azimuth) view angle in degrees.
            If None, uses data-driven angle selection.

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

        # Check if data is constant
        unique_vals = np.unique(tensor_np[np.isfinite(tensor_np)])
        n_unique = len(unique_vals)

        if n_unique <= 1:
            # Handle constant data case
            logger.debug("Constant data detected, creating flat surface")
            if n_unique == 1:
                val = unique_vals[0]
                # Create a flat surface at the constant value
                surf = ax.plot_surface(
                    X,
                    Y,
                    np.full_like(tensor_np, val),
                    cmap="viridis",
                    linewidth=0.5,
                    antialiased=True,
                )
                # Add text annotation above the surface
                ax.text(
                    tensor_np.shape[1] / 2,
                    tensor_np.shape[0] / 2,
                    val,
                    f"Constant Value: {val}",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )
                # Set z limits with a small range around the constant value
                margin = abs(val) * 0.1 if val != 0 else 0.1
                ax.set_zlim(val - margin, val + margin)
            else:
                # No valid data case
                ax.text(
                    0.5, 0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes
                )
                ax.set_zlim(-1, 1)  # Set default z limits
        else:
            # Create the surface plot
            surf = ax.plot_surface(X, Y, tensor_np, cmap="viridis", linewidth=0.5, antialiased=True)

            # Add a color bar
            plt.colorbar(surf, ax=ax, fraction=0.046, pad=0.04)

            # Set view angle based on input or data characteristics
            if view_angle is not None:
                elev, azim = view_angle
            else:
                # Analyze data characteristics for optimal viewing angle
                aspect_ratio = tensor_np.shape[1] / tensor_np.shape[0]

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

            # Set reasonable z limits with a margin
            z_min, z_max = np.nanmin(tensor_np), np.nanmax(tensor_np)
            if z_min == z_max:  # Handle near-constant data
                margin = abs(z_min) * 0.1 if z_min != 0 else 0.1
                ax.set_zlim(z_min - margin, z_max + margin)
            else:
                margin = (z_max - z_min) * 0.1
                ax.set_zlim(z_min - margin, z_max + margin)

        # Set plot labels
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

    return ax
