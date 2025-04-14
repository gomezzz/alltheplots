import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_trajectory_3d_plot(tensor_np, ax=None, connect_points=True):
    """
    Create a 3D trajectory plot for Nx3 data, connecting points to show path through space.

    Parameters:
        tensor_np (numpy.ndarray): The Nx3 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib 3D axis to plot on. If None, a new one is created.
        connect_points (bool): Whether to connect points with lines. Default is True.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 3D trajectory plot")

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

        # Determine if this data is likely to be a trajectory
        # We'll check if the data appears to be ordered vs random
        if n_points > 5:  # Need a few points to check
            # Calculate distances between consecutive points
            diffs = np.diff(tensor_np, axis=0)
            consecutive_dists = np.linalg.norm(diffs, axis=1)
            mean_consecutive_dist = np.mean(consecutive_dists)

            # Calculate distances between random points
            import random

            random_indices = list(range(n_points))
            random.shuffle(random_indices)
            random_dists = np.linalg.norm(
                tensor_np[random_indices[:-1]] - tensor_np[random_indices[1:]], axis=1
            )
            mean_random_dist = np.mean(random_dists)

            # If consecutive points are closer than random points,
            # it's likely a trajectory
            is_likely_trajectory = mean_consecutive_dist < mean_random_dist * 0.8

            # For likely trajectories, use different colors for line segments
            # to indicate direction
            if is_likely_trajectory and connect_points:
                # Plot colored line segments with gradient to show direction

                # Create a custom colormap for the trajectory
                n_segments = n_points - 1
                cmap = plt.cm.viridis
                colors = cmap(np.linspace(0, 1, n_segments))

                # Plot each segment with its color
                for i in range(n_segments):
                    ax.plot(
                        x[i : i + 2],
                        y[i : i + 2],
                        z[i : i + 2],
                        color=colors[i],
                        linewidth=2,
                        alpha=0.8,
                    )

                # Add a colorbar to indicate direction

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_segments))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
                cbar.set_label("Trajectory Direction", fontsize=8)
                cbar.ax.tick_params(labelsize=7)

                # Mark start and end points
                ax.scatter(
                    x[0],
                    y[0],
                    z[0],
                    color="lime",
                    s=50,
                    label="Start",
                    edgecolor="black",
                    linewidth=1,
                )
                ax.scatter(
                    x[-1],
                    y[-1],
                    z[-1],
                    color="red",
                    s=50,
                    label="End",
                    edgecolor="black",
                    linewidth=1,
                )

                # Add a legend for start/end points
                ax.legend(fontsize=8, loc="upper right")

                # Also add points along the trajectory with reduced size/opacity
                ax.scatter(
                    x,
                    y,
                    z,
                    s=max(3, min(20, 300 / n_points)),
                    alpha=0.3,
                    color="gray",
                    edgecolor=None,
                )
            else:
                # If it doesn't look like a trajectory, show a scatter plot with line connecting
                _ = ax.scatter(
                    x,
                    y,
                    z,
                    s=max(5, min(30, 500 / n_points)),
                    c="dodgerblue",
                    alpha=0.7,
                    edgecolors="k",
                    linewidth=0.3,
                )

                if connect_points:
                    # Connect points with a line
                    ax.plot(x, y, z, color="gray", linestyle="-", linewidth=1, alpha=0.5)
        else:
            # For very few points, just do a basic scatter with lines
            _ = ax.scatter(x, y, z, s=30, c="dodgerblue", alpha=0.7, edgecolors="k", linewidth=0.3)

            if connect_points and n_points > 1:
                ax.plot(x, y, z, color="gray", linestyle="-", linewidth=1, alpha=0.5)

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
        ax.set_title("3D Trajectory Plot")

        # Add stats text
        path_length = np.sum(consecutive_dists) if n_points > 1 else 0
        stats_text = f"n = {n_points}\nPath length: {path_length:.2f}"
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
        logger.error(f"Failed to create 3D trajectory plot: {e}")
        ax.text2D(
            0.5,
            0.5,
            f"Trajectory Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("3D Trajectory (Error)")

    return ax
