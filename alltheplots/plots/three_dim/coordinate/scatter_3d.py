import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_scatter_3d_plot(tensor_np, ax=None):
    """
    Create a basic 3D scatter plot for Nx3 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx3 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib 3D axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating basic 3D scatter plot")

    # Create ax if not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    try:
        # Extract x, y, and z coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]
        z = tensor_np[:, 2]

        # Determine appropriate marker size based on number of points
        n_points = len(x)
        marker_size = max(5, min(50, 1000 / n_points))

        # Create a scatter plot with transparent markers
        _ = ax.scatter(
            x,
            y,
            z,
            s=marker_size,
            alpha=0.6,  # Transparency
            edgecolors="k",
            linewidth=0.5,
            c="dodgerblue",  # Default color
        )

        # Add labels for axes
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set equal aspect ratio for better perception
        # This is a bit tricky in 3D; we'll use a custom approach
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add title
        ax.set_title("3D Scatter Plot")

        # Add stats text
        stats_text = f"n = {n_points}"
        ax.text2D(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    except Exception as e:
        logger.error(f"Failed to create 3D scatter plot: {e}")
        ax.text2D(
            0.5,
            0.5,
            f"3D Scatter Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("3D Scatter (Error)")

    return ax
