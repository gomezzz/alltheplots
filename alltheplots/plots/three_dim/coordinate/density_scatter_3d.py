import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from ....utils.logger import logger


def create_density_scatter_3d_plot(tensor_np, ax=None):
    """
    Create a 3D scatter plot with density-based coloring for Nx3 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx3 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib 3D axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating density-colored 3D scatter plot")

    # Create ax if not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    try:
        # Extract x, y, and z coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]
        z = tensor_np[:, 2]

        n_points = len(x)

        # Determine appropriate marker size based on number of points
        marker_size = max(5, min(50, 1000 / n_points))

        # Calculate point density using KDTree for efficiency
        if n_points > 5:  # Need at least a few points to calculate density
            # Determine number of neighbors to consider based on data size
            k_neighbors = min(10, max(3, int(n_points / 10)))

            try:
                # Build KDTree for efficient nearest-neighbor search
                tree = KDTree(tensor_np)

                # Query the tree for the k nearest neighbors of each point
                # Returns distances to k nearest neighbors (including the point itself)
                distances, _ = tree.query(tensor_np, k=k_neighbors)

                # Use the mean distance to k nearest neighbors as inverse density metric
                # Smaller distance = higher density
                mean_distances = np.mean(
                    distances[:, 1:], axis=1
                )  # Skip the first distance (to self)

                # Invert and normalize to get density (higher value = higher density)
                epsilon = np.finfo(float).eps  # Small value to avoid division by zero
                density = 1.0 / (mean_distances + epsilon)
                density = (density - np.min(density)) / (
                    np.max(density) - np.min(density) + epsilon
                )

                # Use a perceptually appropriate colormap for density
                scatter = ax.scatter(
                    x,
                    y,
                    z,
                    s=marker_size,
                    c=density,  # Color by density
                    cmap="viridis",
                    alpha=0.8,
                    edgecolors="k",
                    linewidth=0.3,
                )

                # Add a colorbar to show density scale
                cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
                cbar.set_label("Density", fontsize=8)
                cbar.ax.tick_params(labelsize=7)

                # Add some stats about density
                min_density = np.min(mean_distances)
                max_density = np.max(mean_distances)
                density_ratio = max_density / (min_density + epsilon)

                density_stats = f"Density ratio: {density_ratio:.1f}x"

            except Exception as e:
                logger.warning(f"Density calculation failed: {e}. Using uniform coloring.")
                # Fall back to basic scatter plot if density calculation fails
                scatter = ax.scatter(
                    x, y, z, s=marker_size, c="dodgerblue", alpha=0.7, edgecolors="k", linewidth=0.3
                )
                density_stats = "Density calculation failed"
        else:
            # For very small datasets, just use a plain scatter plot
            scatter = ax.scatter(
                x, y, z, s=marker_size, c="dodgerblue", alpha=0.7, edgecolors="k", linewidth=0.3
            )
            density_stats = "Too few points for density"

        # Add labels for axes
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set equal aspect ratio for better perception
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add title
        ax.set_title("Density-Colored 3D Scatter")

        # Add stats text
        stats_text = f"n = {n_points}\n{density_stats}"
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
        logger.error(f"Failed to create density scatter plot: {e}")
        ax.text2D(
            0.5,
            0.5,
            f"Density Scatter Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Density Scatter (Error)")

    return ax
